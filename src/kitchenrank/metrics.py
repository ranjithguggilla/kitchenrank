"""Scores for a ranked list of inspections, plus a bootstrap that resamples whole weeks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    total = 0.0
    for idx in chunks:
        if len(idx):
            total += len(idx) / len(y) * abs(y[idx].mean() - p[idx].mean())
    return float(total)


def recall_at(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    """Share of all positives found in the top `frac` of the list."""
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def lift_at(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    base = float(np.mean(y))
    return float(np.mean(np.asarray(y)[top]) / base) if base > 0 else float("nan")


def summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "recall_top20": recall_at(y, p, 0.20),
        "lift_top20": lift_at(y, p, 0.20),
    }


def week_bootstrap(
    weeks: np.ndarray,
    fn: Callable[[np.ndarray], float],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Point estimate and 95% interval for fn(row_index), resampling calendar weeks with replacement.

    Inspections in one week share a schedule, a season and often an inspector, so rows are not independent.
    """
    weeks = np.asarray(weeks)
    uniq, inverse = np.unique(weeks, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = fn(np.arange(len(weeks)))
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        v = fn(idx)
        if np.isfinite(v):
            draws.append(v)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_week_diff(
    weeks: np.ndarray,
    fn_a: Callable[[np.ndarray], float],
    fn_b: Callable[[np.ndarray], float],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Interval for fn_a - fn_b on the same resampled weeks."""
    return week_bootstrap(weeks, lambda idx: fn_a(idx) - fn_b(idx), n_boot, seed)


def weekly_table(df: pd.DataFrame, score: np.ndarray, frac: float = 0.2) -> pd.DataFrame:
    """Per-week recall at `frac` of the list, for plotting week-to-week spread."""
    rows = []
    for wk, idx in df.groupby("week").indices.items():
        y = df["any_critical"].to_numpy()[idx]
        if y.sum() == 0 or len(idx) < 20:
            continue
        rows.append({"week": wk, "n": len(idx), "criticals": int(y.sum()), "recall": recall_at(y, score[idx], frac)})
    return pd.DataFrame(rows)
