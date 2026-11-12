"""Scoring one scheduled inspection and saying why in plain words."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import feature_families

BANDS = ("lower than usual", "about usual", "elevated", "high")


@dataclass
class Bundle:
    model: object
    calibrator: object
    columns: list[str]
    defaults: dict
    base_rate: float
    cutoffs: dict  # probability cut points for the bands, taken from the held-out year
    meta: dict = field(default_factory=dict)


def column_defaults(train: pd.DataFrame, columns: list[str]) -> dict:
    """The typical visit: the training median of each column (zero for sparse code flags)."""
    out = {}
    for c in columns:
        s = train[c]
        out[c] = float(s.median()) if s.notna().any() else 0.0
    return out


def expand_record(bundle: Bundle, rec: dict) -> pd.DataFrame:
    """Turn the handful of fields a person can type into the full feature row."""
    row = dict(bundle.defaults)
    n_prior = int(rec.get("n_prior", 0))
    row["n_prior"] = n_prior
    row["first_visit"] = int(n_prior == 0)
    row["risk"] = float(rec.get("risk", 1))
    row["is_restaurant"] = int(rec.get("is_restaurant", 1))
    if n_prior == 0:
        row["days_since_prev"] = np.nan
        for c in (
            "prev_critical",
            "prev_serious",
            "prev_minor",
            "prev_fail",
            "cum_critical_visits",
            "critical_codes_ever",
        ):
            row[c] = 0.0
        row["cum_critical_rate"] = np.nan
        row["last3_critical_rate"] = np.nan
        for c in bundle.columns:
            if c.startswith("prev_v"):
                row[c] = 0.0
    else:
        row["days_since_prev"] = float(rec.get("days_since_prev", 365))
        row["prev_critical"] = float(rec.get("prev_critical", 0))
        row["prev_serious"] = float(rec.get("prev_serious", 0))
        row["prev_minor"] = float(rec.get("prev_minor", 0))
        row["prev_fail"] = float(rec.get("prev_result", "Pass") == "Fail")
        crit_visits = min(int(rec.get("critical_visits", int(row["prev_critical"] > 0))), n_prior)
        row["cum_critical_visits"] = crit_visits
        row["cum_critical_rate"] = crit_visits / n_prior
        row["last3_critical_rate"] = crit_visits / n_prior
        row["critical_codes_ever"] = float(rec.get("critical_codes_ever", 1 if crit_visits > 0 else 0))
        for c in bundle.columns:
            if c.startswith("prev_v"):
                row[c] = 0.0
    if "nb_smoothed_rate" in rec:
        row["nb_smoothed_rate"] = float(rec["nb_smoothed_rate"])
    if "month" in rec:
        row["month"] = int(rec["month"])
    if "temperatureMax" in rec:
        row["temperatureMax"] = float(rec["temperatureMax"])
    return pd.DataFrame([row])[bundle.columns]


def probability(bundle: Bundle, row: pd.DataFrame) -> float:
    raw = bundle.model.predict_proba(row)[:, 1]
    return float(bundle.calibrator.predict(raw)[0])


def band_for(bundle: Bundle, p: float) -> str:
    c = bundle.cutoffs
    if p >= c["high"]:
        return BANDS[3]
    if p >= c["elevated"]:
        return BANDS[2]
    if p >= c["usual"]:
        return BANDS[1]
    return BANDS[0]


def _phrase(family: str, row: pd.Series) -> str:
    if family == "visit history":
        if row["first_visit"] == 1:
            return "this is the first visit on record"
        return (
            f"{int(row['n_prior'])} earlier visit(s), last one {int(row['days_since_prev'])} days ago, "
            f"{int(row['cum_critical_visits'])} with a critical violation"
        )
    if family == "last visit's violations":
        return (
            f"the last visit found {int(row['prev_critical'])} critical, "
            f"{int(row['prev_serious'])} serious and {int(row['prev_minor'])} minor violations"
        )
    if family == "facility type and risk":
        kind = "a restaurant" if row["is_restaurant"] == 1 else "not a restaurant"
        return f"it is {kind} in risk category {int(row['risk'])}"
    if family == "neighbourhood":
        return f"nearby businesses recently ran a smoothed critical rate of {row['nb_smoothed_rate']:.0%}"
    if family == "season and weather":
        return f"timing and weather ({row['temperatureMax']:.0f} F that day, month {int(row['month'])})"
    return "its location in the city"


def explain(bundle: Bundle, row: pd.DataFrame, k: int = 3) -> list[str]:
    """Swap each group of columns for the typical value and see how far the probability moves."""
    base = probability(bundle, row)
    fams = feature_families(row)
    effects = []
    for fam, cols in fams.items():
        cols = [c for c in cols if c in row.columns]
        alt = row.copy()
        for c in cols:
            alt[c] = bundle.defaults[c]
        effects.append((fam, base - probability(bundle, alt)))
    effects.sort(key=lambda t: -abs(t[1]))
    lines = []
    for fam, delta in effects[:k]:
        if abs(delta) < 0.002:
            continue
        direction = "raises" if delta > 0 else "lowers"
        lines.append(f"{_phrase(fam, row.iloc[0])} ({direction} the chance by {abs(delta) * 100:.1f} points)")
    return lines


def score_visit(bundle: Bundle, rec: dict) -> dict:
    row = expand_record(bundle, rec)
    p = probability(bundle, row)
    return {
        "probability": p,
        "band": band_for(bundle, p),
        "relative_to_typical": p / bundle.base_rate,
        "reasons": explain(bundle, row),
    }
