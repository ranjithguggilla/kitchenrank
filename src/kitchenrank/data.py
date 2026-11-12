"""Loading the inspection table and cutting it into forward-in-time blocks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_FILE, TRAIN_END, VALID_END

REQUIRED = ["inspection_id", "license", "date", "facility", "risk", "lat", "lon", "critical", "serious", "minor"]


def load_inspections(path: Path | str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"inspection table is missing columns: {missing}")
    if not df["inspection_id"].is_unique:
        raise ValueError("inspection_id must be unique")
    df = df.sort_values(["date", "inspection_id"], kind="stable").reset_index(drop=True)
    df["any_critical"] = (df["critical"] > 0).astype(int)
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    return df


def split_blocks(df: pd.DataFrame, train_end: str = TRAIN_END, valid_end: str = VALID_END) -> dict[str, pd.Index]:
    """Row labels for the three blocks. Dates compare on the day, so a block never leaks into the next."""
    d = df["date"]
    train = df.index[d <= pd.Timestamp(train_end)]
    valid = df.index[(d > pd.Timestamp(train_end)) & (d <= pd.Timestamp(valid_end))]
    test = df.index[d > pd.Timestamp(valid_end)]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("one of the time blocks is empty")
    return {"train": train, "valid": valid, "test": test}
