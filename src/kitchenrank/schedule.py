"""Re-ordering a fixed list of inspections so the riskiest are visited first.

The department's real calendar fixes how many inspections happen on each day. A policy may only change *which*
businesses fill those slots. Two heaps do the work on every day: a max-heap on risk score for the free choice,
and a min-heap on deadline so that no business is pushed later than `max_delay` days past its real visit.

Assumption, stated plainly: every inspection in a window is on the due list `max_advance` days before its real
visit and may be taken any time after that. The real process has due dates the data does not show, so the
savings below are what a planner could reach if the due list were that flexible.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Plan:
    visit_day: np.ndarray  # new day number for every row, aligned with the input
    overflow: int  # rows forced past their deadline because a day was already full


def reorder_window(
    actual_day: np.ndarray,
    score: np.ndarray,
    max_delay: int,
    max_advance: int,
) -> Plan:
    n = len(actual_day)
    order = np.argsort(actual_day, kind="stable")
    days, counts = np.unique(actual_day, return_counts=True)
    avail_day = actual_day - max_advance
    deadline = actual_day + max_delay

    by_avail = np.argsort(avail_day, kind="stable")
    ptr = 0
    score_heap: list[tuple[float, int]] = []
    deadline_heap: list[tuple[int, int]] = []
    done = np.zeros(n, dtype=bool)
    new_day = np.full(n, -1, dtype=int)
    overflow = 0

    for day, cap in zip(days, counts, strict=True):
        while ptr < n and avail_day[by_avail[ptr]] <= day:
            i = by_avail[ptr]
            heapq.heappush(score_heap, (-float(score[i]), int(i)))
            heapq.heappush(deadline_heap, (int(deadline[i]), int(i)))
            ptr += 1
        taken = 0
        # first, anyone whose deadline has arrived
        while taken < cap and deadline_heap and deadline_heap[0][0] <= day:
            _, i = heapq.heappop(deadline_heap)
            if not done[i]:
                done[i] = True
                new_day[i] = day
                taken += 1
        # then the highest scores
        while taken < cap and score_heap:
            _, i = heapq.heappop(score_heap)
            if not done[i]:
                done[i] = True
                new_day[i] = day
                taken += 1
    # a few rows can be left over when a day filled with forced visits; they go on the last day, counted as overflow
    left = np.flatnonzero(~done)
    if len(left):
        overflow = len(left)
        new_day[left] = days[-1]
    del order
    return Plan(new_day, overflow)


def windows(day: np.ndarray, horizon: int) -> list[np.ndarray]:
    start = int(day.min())
    ids = (day - start) // horizon
    return [np.flatnonzero(ids == k) for k in np.unique(ids)]


def saved_per_row(
    df: pd.DataFrame,
    score: np.ndarray,
    horizon: int,
    max_delay: int,
    max_advance: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Days each inspection moves earlier (negative: later), the window it belongs to, and the overflow count."""
    max_advance = horizon if max_advance is None else max_advance
    day = (df["date"] - df["date"].min()).dt.days.to_numpy()
    saved = np.zeros(len(df))
    window = np.zeros(len(df), dtype=int)
    overflow = 0
    for k, idx in enumerate(windows(day, horizon)):
        plan = reorder_window(day[idx], np.asarray(score)[idx], max_delay, max_advance)
        saved[idx] = day[idx] - plan.visit_day
        window[idx] = k
        overflow += plan.overflow
    return saved, window, overflow


def simulate(
    df: pd.DataFrame,
    score: np.ndarray,
    horizon: int,
    max_delay: int,
    max_advance: int | None = None,
) -> dict[str, float]:
    """Days saved per critical finding for one policy, over every window of the test period."""
    saved, _, overflow = saved_per_row(df, score, horizon, max_delay, max_advance)
    crit = df["any_critical"].to_numpy().astype(bool)
    return {
        "days_saved_per_critical": float(saved[crit].mean()),
        "exposure_days_avoided": float(saved[crit].sum()),
        "mean_days_later_noncritical": float(-saved[~crit].mean()),
        "share_moved_later": float((saved < 0).mean()),
        "criticals": int(crit.sum()),
        "overflow": overflow,
    }
