import json

import numpy as np
import pandas as pd

from kitchenrank import evaluate, ingest, pipeline


def test_r_day_numbers_convert_to_dates():
    out = ingest._days(pd.Series([15736, 15737]))
    assert out.iloc[0] == pd.Timestamp("2013-01-31")
    assert out.iloc[1] - out.iloc[0] == pd.Timedelta(days=1)


def test_clean_rounds_and_unwraps_numpy():
    out = pipeline._clean({"a": np.float64(0.123456789), "b": [np.int64(3)], "c": (1.0000001,)})
    assert out == {"a": 0.12346, "b": [3], "c": [1.0]}
    json.dumps(out)


def test_window_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"date": pd.Timestamp("2014-01-06") + pd.to_timedelta(rng.integers(0, 150, 600), unit="D")})
    d["any_critical"] = (rng.random(600) < 0.2).astype(int)
    d = d.sort_values("date").reset_index(drop=True)
    pt, lo, hi = pipeline._window_ci(d, d["any_critical"].to_numpy() + rng.random(600) * 0.01, 28, 7, 80, 1)
    assert lo <= pt <= hi and pt > 0


def test_full_run_on_a_synthetic_table_writes_every_output(tmp_path, monkeypatch):
    from conftest import make_frame

    big = make_frame(n_lic=1400, visits=5, seed=4)
    monkeypatch.setattr(pipeline, "load_inspections", lambda: big.drop(columns=["fail_flag"]).copy())
    monkeypatch.setattr(pipeline, "weekly_rate_breaks", lambda d: evaluate.weekly_rate_breaks(d, min_week_size=5))
    out = tmp_path / "out"
    assets = tmp_path / "assets"
    m = pipeline.run(out_dir=out, asset_dir=assets, n_boot=15)
    assert (out / "metrics.json").exists() and (out / "bundle.joblib").exists()
    assert len(list(assets.glob("*.png"))) == 9
    for name in ("validation_models.csv", "family_ablation.csv", "schedule_grid.csv", "value_table.csv"):
        assert (out / "tables" / name).exists()
    assert m["selection"]["chosen"] in {"logistic", "boosting", "random forest"}
    assert m["schedule"]["headline"]["days_saved_per_critical"] > 0
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["data"]["inspections"] == len(big)


def test_profile_rates_match_a_direct_count(frame):
    from kitchenrank.features import build_features

    X = build_features(frame, 0.12)
    prof = pipeline.data_profile(frame, X)
    assert prof["overall"] == frame["any_critical"].mean()
    n = sum(v["inspections"] for v in prof["by_previous_critical_count"].values())
    assert n == len(frame)
    assert set(prof["by_month"]) <= set(range(1, 13))
