"""Build data/inspections.csv.gz from the City of Chicago food-inspections-evaluation repository.

The repository stores its tables as R .Rds files. This script is only needed to rebuild the vendored CSV:

    git clone --depth 1 https://github.com/Chicago/food-inspections-evaluation.git
    python -m kitchenrank.ingest food-inspections-evaluation/DATA data/inspections.csv.gz

It needs the optional `rdata` package. Everything else in the project reads the CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01")
LICENCE_FLAGS = [
    "consumption_on_premises_incidental_activity",
    "tobacco_retail_over_counter",
    "package_goods",
    "limited_business_license",
    "outdoor_patio",
    "public_place_of_amusement",
    "childrens_services_facility_license",
    "tavern",
    "regulated_business_license",
    "filling_station",
    "caterers_liquor_license",
    "mobile_food_license",
]


def _read(folder: Path, name: str) -> pd.DataFrame:
    import rdata

    return rdata.conversion.convert(rdata.parser.parse_file(str(folder / name)))


def _days(series: pd.Series) -> pd.Series:
    return EPOCH + pd.to_timedelta(series.astype(float), unit="D")


def build(folder: Path) -> pd.DataFrame:
    feats = _read(folder, "23_food_insp_features.Rds")
    codes = _read(folder, "21_food_inspection_violation_matrix_nums.Rds")
    extra = _read(folder, "30_xgboost_data.Rds")
    weather = _read(folder, "17_mongo_weather_update.Rds")
    staff = _read(folder, "19_inspector_assignments.Rds")

    out = pd.DataFrame(
        {
            "inspection_id": feats["Inspection_ID"].astype(int),
            "license": feats["License"].astype(int),
            "date": _days(feats["Inspection_Date"]),
            "facility": feats["Facility_Type_Clean"].astype(str),
            "risk": feats["Risk"].astype(str).str.extract(r"Risk (\d)")[0].astype(float),
            "zip": feats["Zip"],
            "lat": feats["Latitude"],
            "lon": feats["Longitude"],
            "result": feats["Results"].astype(str),
            "critical": feats["criticalCount"].astype(int),
            "serious": feats["seriousCount"].astype(int),
            "minor": feats["minorCount"].astype(int),
            "city_past_fail": feats["pastFail"],
            "city_past_critical": feats["pastCritical"],
            "city_first_record": feats["firstRecord"],
        }
    )
    code_cols = [c for c in codes.columns if str(c) != "Inspection_ID"]
    cm = codes[["Inspection_ID", *code_cols]].copy()
    cm.columns = ["inspection_id", *[f"v{int(c):02d}" for c in code_cols]]
    cm["inspection_id"] = cm["inspection_id"].astype(int)
    for c in cm.columns[1:]:
        cm[c] = cm[c].astype(int)
    out = out.merge(cm, on="inspection_id", how="left", validate="one_to_one")

    keep = ["Inspection_ID", "ageAtInspection", "heat_burglary", "heat_garbage", "heat_sanitation", *LICENCE_FLAGS]
    ex = extra[keep].rename(
        columns={
            "Inspection_ID": "inspection_id",
            "ageAtInspection": "business_age",
            "heat_burglary": "heat_burglary",
            "heat_garbage": "heat_garbage",
            "heat_sanitation": "heat_sanitation",
        }
    )
    ex["inspection_id"] = ex["inspection_id"].astype(int)
    ex = ex.rename(columns={c: f"lic_{c}" for c in LICENCE_FLAGS})
    out = out.merge(ex, on="inspection_id", how="left", validate="one_to_one")

    w = weather.copy()
    w["date"] = _days(w["date"])
    out = out.merge(w, on="date", how="left")

    staff = staff.rename(columns={"inspectionID": "inspection_id", "sanitarian": "inspector"})
    staff["inspection_id"] = staff["inspection_id"].astype(int)
    out = out.merge(staff.drop_duplicates("inspection_id"), on="inspection_id", how="left")

    out = out.sort_values(["date", "inspection_id"]).reset_index(drop=True)
    return out


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        raise SystemExit("usage: python -m kitchenrank.ingest <DATA folder> <output csv.gz>")
    df = build(Path(argv[0]))
    df.to_csv(argv[1], index=False, float_format="%.6g")
    print(f"wrote {len(df):,} rows, {df.shape[1]} columns to {argv[1]}")
    print(df["date"].min().date(), "to", df["date"].max().date(), np.round(df["critical"].gt(0).mean(), 4))


if __name__ == "__main__":
    main()
