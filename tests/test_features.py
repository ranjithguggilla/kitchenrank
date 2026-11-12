import numpy as np
import pandas as pd

from kitchenrank.features import (
    _neighbour_stats,
    build_features,
    feature_families,
    history_features,
    neighbourhood_features,
)


def tiny():
    d = pd.DataFrame(
        {
            "inspection_id": [1, 2, 3, 4, 5],
            "license": [10, 10, 10, 20, 10],
            "date": pd.to_datetime(["2012-01-02", "2012-03-05", "2012-03-05", "2012-03-06", "2012-06-04"]),
            "critical": [1, 0, 2, 0, 0],
            "serious": [0, 1, 0, 0, 0],
            "minor": [3, 2, 1, 0, 4],
            "fail_flag": [1, 0, 0, 0, 0],
            "v03": [1, 0, 1, 0, 0],
        }
    )
    d["any_critical"] = (d["critical"] > 0).astype(int)
    return d


def test_history_uses_only_earlier_days():
    h = history_features(tiny())
    assert list(h["n_prior"]) == [0, 1, 1, 0, 3]
    # rows 2 and 3 share a day: neither may see the other, both see the January visit
    assert h.loc[1, "prev_critical"] == 1 and h.loc[2, "prev_critical"] == 1
    assert h.loc[4, "days_since_prev"] == 91
    assert h.loc[4, "cum_critical_visits"] == 2
    assert h.loc[0, "first_visit"] == 1 and np.isnan(h.loc[0, "days_since_prev"])


def test_history_flags_follow_the_previous_visit():
    h = history_features(tiny())
    assert h.loc[1, "prev_v03"] == 1
    assert h.loc[0, "prev_v03"] == 0
    assert h.loc[4, "critical_codes_ever"] == 1


def test_changing_a_future_row_never_changes_a_past_feature(frame):
    f1 = history_features(frame)
    altered = frame.copy()
    last = altered.index[-1]
    altered.loc[last, ["critical", "any_critical", "minor"]] = [9, 1, 9]
    f2 = history_features(altered)
    assert f1.drop(index=last).equals(f2.drop(index=last))


def test_neighbour_asof_ignores_future_and_same_licence():
    lat = np.array([41.0, 41.0, 41.0, 41.0])
    lon = np.array([-87.0, -87.0, -87.0, -87.0])
    day = np.array([0, 20, 40, 400])
    crit = np.array([1.0, 1.0, 0.0, 1.0])
    lic = np.array([1, 2, 3, 2])
    n, h = _neighbour_stats(lat, lon, day, crit, lic, "asof", window_days=100)
    # the row at 40 days sees the rows at 0 and 20; the first row sees nothing; the row at 400 is outside the window
    assert n.tolist() == [0, 1, 2, 0]
    assert h.tolist() == [0, 1, 2, 0]
    nl, hl = _neighbour_stats(lat, lon, day, crit, lic, "leaky")
    assert nl[0] == 3 and hl[0] == 2  # the leaky mode counts the future and skips only its own licence


def test_neighbour_smoothing_pulls_empty_areas_to_the_city_rate(frame):
    nb = neighbourhood_features(frame, 0.12, "asof")
    empty = nb[nb["nb_count"] == 0]
    assert len(empty) > 0
    assert np.allclose(empty["nb_smoothed_rate"], 0.12)
    assert empty["nb_critical_rate"].isna().all()


def test_build_features_shape_and_families(frame):
    X = build_features(frame, 0.12)
    assert len(X) == len(frame) and X.index.equals(frame.index)
    fams = feature_families(X)
    flat = [c for cols in fams.values() for c in cols]
    assert len(flat) == len(set(flat))
    assert set(flat) <= set(X.columns)
    assert X["is_restaurant"].isin([0, 1]).all()
