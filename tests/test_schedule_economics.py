import numpy as np
import pandas as pd
import pytest

from kitchenrank.economics import exposure_value, value_table, visits_to_find
from kitchenrank.schedule import reorder_window, saved_per_row, simulate, windows


def week_of_days():
    day = np.repeat([0, 1, 2, 3, 4], 4)  # four inspections a day for five days
    return day


def test_capacity_per_day_is_preserved_and_everyone_is_scheduled_once():
    day = week_of_days()
    score = np.random.default_rng(0).random(len(day))
    plan = reorder_window(day, score, max_delay=3, max_advance=7)
    assert plan.overflow == 0
    assert (plan.visit_day >= 0).all()
    assert np.array_equal(np.bincount(plan.visit_day), np.bincount(day))


def test_zero_delay_cap_returns_the_real_calendar_for_everyone_due_on_time():
    day = week_of_days()
    plan = reorder_window(day, np.arange(len(day), dtype=float), max_delay=0, max_advance=0)
    assert np.array_equal(plan.visit_day, day)


def test_high_scores_move_forward_and_delay_cap_is_respected():
    day = week_of_days()
    score = np.zeros(len(day))
    score[-1] = 1.0  # the last inspection of the week is the riskiest
    plan = reorder_window(day, score, max_delay=2, max_advance=10)
    assert plan.visit_day[-1] == 0
    assert (plan.visit_day <= day + 2).all()


def test_windows_cover_every_row_once():
    day = np.arange(0, 100)
    parts = windows(day, 28)
    assert sorted(np.concatenate(parts).tolist()) == day.tolist()
    assert len(parts) == 4


def frame_for_sim(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"date": pd.Timestamp("2014-01-06") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")})
    d["any_critical"] = (rng.random(n) < 0.15).astype(int)
    return d.sort_values("date").reset_index(drop=True)


def test_oracle_beats_random_and_random_is_near_zero():
    d = frame_for_sim()
    rng = np.random.default_rng(1)
    oracle = simulate(d, d["any_critical"].to_numpy() + rng.random(len(d)) * 0.01, 28, 7)
    rand = simulate(d, rng.random(len(d)), 28, 7)
    assert oracle["days_saved_per_critical"] > 5
    assert abs(rand["days_saved_per_critical"]) < 1.5
    assert oracle["overflow"] == 0


def test_no_delay_allowed_means_no_change():
    d = frame_for_sim()
    out = simulate(d, np.random.default_rng(2).random(len(d)), 28, 0, max_advance=0)
    assert out["days_saved_per_critical"] == 0.0 and out["share_moved_later"] == 0.0


def test_saved_per_row_totals_balance_inside_each_window():
    d = frame_for_sim()
    saved, win, _ = saved_per_row(d, np.random.default_rng(3).random(len(d)), 28, 14)
    for k in np.unique(win):
        assert saved[win == k].sum() == 0  # moving visits around a fixed calendar cannot create days


def test_value_table_scales_and_validates_arithmetic():
    t = value_table(1000.0, 5000)
    assert list(t["assumed_value_per_exposure_day"]) == [0.5, 2.0, 5.0]
    assert t.loc[1, "value_in_sample"] == 2000.0
    assert t.loc[1, "value_per_10k_inspections"] == 4000.0
    assert exposure_value(10, 2.5) == 25.0


def test_visits_to_find_matches_a_hand_count():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    good = np.array([9, 8, 7, 1, 1, 1, 1, 1, 1, 1], float)
    assert visits_to_find(y, good, 0.5) == pytest.approx(0.1)
    assert visits_to_find(y, good, 1.0) == pytest.approx(0.3)
