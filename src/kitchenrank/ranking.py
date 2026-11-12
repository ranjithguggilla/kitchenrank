"""Weekly ranking quality, with each calendar week treated as one query (ranx)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ranx import Qrels, Run, evaluate

METRICS = ["ndcg@10", "precision@10", "map@50", "mrr"]


def _collect(df: pd.DataFrame, score: np.ndarray, min_size: int) -> tuple[dict, dict]:
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    y = df["any_critical"].to_numpy()
    ids = df["inspection_id"].astype(str).to_numpy()
    for wk, idx in df.groupby("week").indices.items():
        if len(idx) < min_size or y[idx].sum() == 0:
            continue
        key = str(pd.Timestamp(wk).date())
        qrels[key] = {ids[i]: int(y[i]) for i in idx}
        run[key] = {ids[i]: float(score[i]) for i in idx}
    return qrels, run


def to_qrels_run(df: pd.DataFrame, score: np.ndarray, min_size: int = 20) -> tuple[Qrels, Run]:
    qrels, run = _collect(df, score, min_size)
    return Qrels(qrels), Run(run)


def weekly_ranking_scores(df: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    raw_q, raw_r = _collect(df, score, 20)
    if not raw_q:
        return {m: float("nan") for m in METRICS}
    qrels, run = Qrels(raw_q), Run(raw_r)
    out = evaluate(qrels, run, METRICS, make_comparable=True)
    return {k: float(v) for k, v in out.items()}


def n_weeks(df: pd.DataFrame, min_size: int = 20) -> int:
    qrels, _ = _collect(df, np.zeros(len(df)), min_size)
    return len(qrels)
