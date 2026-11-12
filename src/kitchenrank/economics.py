"""Turning days saved into money, with every assumption on the table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COSTS, Costs


def exposure_value(exposure_days: float, per_day: float) -> float:
    return float(exposure_days * per_day)


def value_table(exposure_days: float, n_inspections: int, costs: Costs = COSTS) -> pd.DataFrame:
    """Dollar value of the avoided exposure days at each assumed per-day value, per 10,000 inspections."""
    scale = 10_000 / n_inspections
    rows = [
        {
            "assumed_value_per_exposure_day": v,
            "exposure_days_avoided": exposure_days,
            "value_in_sample": exposure_value(exposure_days, v),
            "value_per_10k_inspections": exposure_value(exposure_days, v) * scale,
        }
        for v in costs.per_exposure_day
    ]
    return pd.DataFrame(rows)


def visits_to_find(y: np.ndarray, score: np.ndarray, target_share: float) -> float:
    """Share of the list that must be visited, best first, to find `target_share` of the critical cases."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(score), kind="stable")
    cum = np.cumsum(y[order]) / y.sum()
    k = int(np.searchsorted(cum, target_share) + 1)
    return k / len(y)
