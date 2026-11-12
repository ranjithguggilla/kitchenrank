"""Features that are known on the morning of an inspection.

Everything here is computed "as of" the inspection date: a visit may only look at visits that happened on an
earlier day. The neighbourhood feature has a deliberately leaky twin so the leak audit can measure what peeking
at the future would have bought.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .config import (
    CRITICAL_CODES,
    NEIGHBOUR_PRIOR_STRENGTH,
    NEIGHBOUR_RADIUS_M,
    NEIGHBOUR_WINDOW_DAYS,
)

EARTH_RADIUS_M = 6_371_000.0

STATIC = ["risk", "is_restaurant", "lat", "lon"]
HISTORY = [
    "n_prior",
    "first_visit",
    "days_since_prev",
    "prev_critical",
    "prev_serious",
    "prev_minor",
    "prev_fail",
    "cum_critical_rate",
    "cum_critical_visits",
    "last3_critical_rate",
    "critical_codes_ever",
]
NEIGHBOURHOOD = ["nb_count", "nb_critical_rate", "nb_smoothed_rate"]
SEASON = ["dow", "month", "temperatureMax", "precipIntensity", "humidity", "windSpeed"]
EXTRAS = [
    "business_age",
    "heat_burglary",
    "heat_garbage",
    "heat_sanitation",
]


def code_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if len(c) == 3 and c[0] == "v" and c[1:].isdigit()]


def prev_code_columns(df: pd.DataFrame) -> list[str]:
    return [f"prev_{c}" for c in code_columns(df)]


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-licence history from strictly earlier calendar days.

    The table is sorted by date. Two visits on the same day for one licence do not see each other.
    """
    d = df.sort_values(["license", "date", "inspection_id"], kind="stable")
    g = d.groupby("license", sort=False)
    codes = code_columns(df)
    crit_codes = [f"v{c:02d}" for c in CRITICAL_CODES if f"v{c:02d}" in codes]

    out = pd.DataFrame(index=d.index)
    prev_date = g["date"].shift(1)
    same_day = prev_date.eq(d["date"])
    # when the previous row is on the same day, fall back one more row for the licence
    prev2_date = g["date"].shift(2)

    def shifted(col: str) -> pd.Series:
        s1, s2 = g[col].shift(1), g[col].shift(2)
        return s1.where(~same_day, s2)

    out["_prev_date"] = prev_date.where(~same_day, prev2_date)
    out["n_prior"] = g.cumcount() - same_day.astype(int)
    out["n_prior"] = out["n_prior"].clip(lower=0)
    out["first_visit"] = (out["n_prior"] == 0).astype(int)
    out["days_since_prev"] = (d["date"] - out["_prev_date"]).dt.days
    out["prev_critical"] = shifted("critical")
    out["prev_serious"] = shifted("serious")
    out["prev_minor"] = shifted("minor")
    out["prev_fail"] = shifted("fail_flag")

    cum_any = g["any_critical"].cumsum() - d["any_critical"]
    cum_any = cum_any - np.where(same_day, g["any_critical"].shift(1).fillna(0), 0)
    out["cum_critical_visits"] = cum_any
    with np.errstate(invalid="ignore", divide="ignore"):
        out["cum_critical_rate"] = cum_any / out["n_prior"].replace(0, np.nan)
    roll = g["any_critical"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    out["last3_critical_rate"] = roll.where(~same_day, np.nan)

    ever = pd.DataFrame(0, index=d.index, columns=crit_codes)
    for c in crit_codes:
        ever[c] = (g[c].cumsum() - d[c]).clip(upper=1)
    out["critical_codes_ever"] = ever.sum(axis=1)

    for c in codes:
        out[f"prev_{c}"] = shifted(c)
    out = out.drop(columns="_prev_date")
    # a first visit has no previous result: say so with explicit sentinels the model can read
    first = out["first_visit"] == 1
    for c in ["prev_critical", "prev_serious", "prev_minor", "prev_fail", *prev_code_columns(df)]:
        out.loc[first, c] = 0.0
    out.loc[first, "days_since_prev"] = np.nan
    return out.loc[df.index]


def _neighbour_stats(
    lat: np.ndarray,
    lon: np.ndarray,
    day: np.ndarray,
    crit: np.ndarray,
    lic: np.ndarray,
    mode: str,
    radius_m: float = NEIGHBOUR_RADIUS_M,
    window_days: int = NEIGHBOUR_WINDOW_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Count and critical-visit count of nearby inspections.

    mode "asof": only visits from earlier weeks, at most `window_days` old.
    mode "leaky": every other visit in the table, past or future. Used only by the leak audit.
    """
    n = len(lat)
    count = np.zeros(n)
    hits = np.zeros(n)
    ok = np.isfinite(lat) & np.isfinite(lon)
    rad = radius_m / EARTH_RADIUS_M
    coords = np.radians(np.c_[np.where(ok, lat, 0.0), np.where(ok, lon, 0.0)])

    if mode == "leaky":
        tree = BallTree(coords[ok], metric="haversine")
        pos = np.flatnonzero(ok)
        neigh = tree.query_radius(coords[ok], r=rad)
        for k, idx in enumerate(neigh):
            rows = pos[idx]
            rows = rows[lic[rows] != lic[pos[k]]]
            count[pos[k]] = len(rows)
            hits[pos[k]] = crit[rows].sum()
        return count, hits

    week = (day // 7).astype(int)
    for w in np.unique(week):
        cur = np.flatnonzero((week == w) & ok)
        if len(cur) == 0:
            continue
        lo, hi = w * 7 - window_days, w * 7
        pool = np.flatnonzero(ok & (day >= lo) & (day < hi))
        if len(pool) == 0:
            continue
        tree = BallTree(coords[pool], metric="haversine")
        neigh = tree.query_radius(coords[cur], r=rad)
        for k, idx in enumerate(neigh):
            rows = pool[idx]
            rows = rows[lic[rows] != lic[cur[k]]]
            count[cur[k]] = len(rows)
            hits[cur[k]] = crit[rows].sum()
    return count, hits


def neighbourhood_features(df: pd.DataFrame, city_rate: float, mode: str = "asof") -> pd.DataFrame:
    day = ((df["date"] - pd.Timestamp("2000-01-01")).dt.days).to_numpy()
    count, hits = _neighbour_stats(
        df["lat"].to_numpy(float),
        df["lon"].to_numpy(float),
        day,
        df["any_critical"].to_numpy(float),
        df["license"].to_numpy(),
        mode,
    )
    out = pd.DataFrame(index=df.index)
    out["nb_count"] = count
    with np.errstate(invalid="ignore", divide="ignore"):
        out["nb_critical_rate"] = np.where(count > 0, hits / np.where(count > 0, count, 1), np.nan)
    out["nb_smoothed_rate"] = (hits + NEIGHBOUR_PRIOR_STRENGTH * city_rate) / (count + NEIGHBOUR_PRIOR_STRENGTH)
    return out


def build_features(df: pd.DataFrame, city_rate: float, neighbour_mode: str = "asof") -> pd.DataFrame:
    """The full feature table, indexed like `df`.

    `city_rate` is the critical rate of the training block. It is a constant passed in, so the smoothing
    never reads a rate computed from rows it is scoring.
    """
    if "fail_flag" not in df.columns:
        df = df.assign(fail_flag=(df["result"] == "Fail").astype(int))
    X = pd.DataFrame(index=df.index)
    X["risk"] = df["risk"]
    X["is_restaurant"] = (df["facility"] == "Restaurant").astype(int)
    X["lat"] = df["lat"]
    X["lon"] = df["lon"]
    X = X.join(history_features(df))
    X = X.join(neighbourhood_features(df, city_rate, neighbour_mode))
    X["dow"] = df["date"].dt.dayofweek
    X["month"] = df["date"].dt.month
    for c in ["temperatureMax", "precipIntensity", "humidity", "windSpeed", *EXTRAS]:
        X[c] = df[c] if c in df.columns else np.nan
    return X


def feature_families(X: pd.DataFrame) -> dict[str, list[str]]:
    """Columns grouped the way a manager would talk about them. Used by ablations and explanations."""
    prev_codes = [c for c in X.columns if c.startswith("prev_v")]
    last_visit = ["prev_critical", "prev_serious", "prev_minor", "prev_fail", *prev_codes]
    return {
        "facility type and risk": ["risk", "is_restaurant"],
        "location": ["lat", "lon"],
        "visit history": [c for c in HISTORY if c in X.columns and c not in last_visit],
        "last visit's violations": last_visit,
        "neighbourhood": [c for c in NEIGHBOURHOOD if c in X.columns],
        "season and weather": [c for c in SEASON if c in X.columns],
    }
