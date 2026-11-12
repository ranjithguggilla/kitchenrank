import numpy as np
import pandas as pd
import pytest

from kitchenrank import app, plots
from kitchenrank.data import split_blocks
from kitchenrank.evaluate import Calibrator
from kitchenrank.features import EXTRAS, build_features
from kitchenrank.models import make_model
from kitchenrank.score import Bundle, band_for, column_defaults, expand_record, explain, probability, score_visit
from kitchenrank.verify import main as verify_main


@pytest.fixture(scope="module")
def bundle(frame):
    blocks = split_blocks(frame, "2012-03-31", "2012-09-30")
    rate = float(frame.loc[blocks["train"], "any_critical"].mean())
    X = build_features(frame, rate)
    cols = [c for c in X.columns if c not in EXTRAS]
    model = make_model("logistic", cols).fit(X.loc[blocks["train"], cols], frame.loc[blocks["train"], "any_critical"])
    p = model.predict_proba(X.loc[blocks["test"], cols])[:, 1]
    cuts = dict(zip(["usual", "elevated", "high"], np.quantile(p, [0.4, 0.7, 0.9]), strict=True))
    return Bundle(model, Calibrator(), cols, column_defaults(X.loc[blocks["train"], cols], cols), float(p.mean()), cuts)


def test_scoring_responds_to_the_last_visit(bundle):
    clean = score_visit(bundle, {"n_prior": 2, "prev_critical": 0, "days_since_prev": 300})
    bad = score_visit(bundle, {"n_prior": 2, "prev_critical": 3, "critical_visits": 2, "prev_result": "Fail"})
    assert 0 < clean["probability"] < 1
    assert bad["probability"] > clean["probability"]
    assert bad["reasons"] and all("points" in r for r in bad["reasons"])


def test_first_visit_record_uses_sentinels(bundle):
    row = expand_record(bundle, {"n_prior": 0, "prev_critical": 5})
    assert row.loc[0, "first_visit"] == 1
    assert row.loc[0, "prev_critical"] == 0
    assert np.isnan(row.loc[0, "days_since_prev"])
    assert list(row.columns) == bundle.columns


def test_bands_are_ordered_and_cover_the_range(bundle):
    labels = [band_for(bundle, p) for p in (0.0, bundle.cutoffs["usual"], bundle.cutoffs["elevated"], 1.0)]
    assert labels == ["lower than usual", "about usual", "elevated", "high"]


def test_explain_reports_largest_groups_first(bundle):
    row = expand_record(bundle, {"n_prior": 3, "prev_critical": 4, "critical_visits": 3, "days_since_prev": 90})
    lines = explain(bundle, row, k=3)
    assert 1 <= len(lines) <= 3
    assert probability(bundle, row) > 0


def test_describe_function_formats_the_answer(bundle, monkeypatch):
    monkeypatch.setattr(app, "load_bundle", lambda path=None: bundle)
    head, why = app.describe(1, True, 2, 200, 2, 1, 4, "Fail", 1, 18.0, "Jul", 85.0)
    assert head.startswith("Chance of a critical violation:") and "Band:" in head
    assert why.startswith("- ")


def test_missing_bundle_gives_a_clear_error(tmp_path):
    app.load_bundle.cache_clear()
    with pytest.raises(FileNotFoundError, match="kitchenrank"):
        app.load_bundle(str(tmp_path / "nope.joblib"))


def test_gradio_app_builds():
    demo = app.build_app()
    assert demo is not None


def test_every_chart_renders(tmp_path):
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.15).astype(int)
    p = np.clip(0.1 + 0.2 * y + rng.normal(0, 0.05, 500), 0.01, 0.9)
    vt = pd.DataFrame({"auc": [0.55, 0.62], "log_loss": [0.38, 0.37]}, index=["static logistic", "boosting"])
    plots.model_comparison(vt, "boosting", tmp_path / "a.png")
    plots.gain_curve(y, p, rng.random(500), tmp_path / "b.png")
    plots.calibration_plot(y, p, p, tmp_path / "c.png")
    wk = pd.DataFrame({"week": pd.date_range("2014-01-06", periods=6, freq="W-MON"), "recall": [0.3] * 6})
    plots.weekly_recall(wk, tmp_path / "d.png")
    sched = pd.DataFrame(
        {
            "policy": ["model", "model", "random", "random"],
            "max_delay": [0, 7, 0, 7],
            "days_saved_per_critical": [0, 1.5, 0, 0.1],
        }
    )
    plots.delay_tradeoff(sched, tmp_path / "e.png")
    ab = pd.DataFrame(
        {
            "dropped_family": ["a", "b"],
            "auc_loss": [0.01, -0.01],
            "auc_loss_lo": [0.0, -0.02],
            "auc_loss_hi": [0.02, 0.0],
        }
    )
    plots.ablation_plot(ab, tmp_path / "f.png")
    plots.leak_plot({"as-of neighbours": 0.6, "shuffled": 0.67}, tmp_path / "g.png")
    ins = pd.DataFrame({"inspector": ["a", "b"], "obs_over_exp": [0.5, 1.5], "lo": [0.4, 1.2], "hi": [0.6, 1.8]})
    plots.inspector_plot(ins, tmp_path / "h.png")
    weekly = pd.DataFrame({"week": pd.date_range("2013-01-07", periods=10, freq="W-MON"), "rate": [0.1] * 10})
    plots.rate_over_time(weekly, [{"from": "2013-01-07", "to": "2013-03-11", "critical_rate": 0.1}], tmp_path / "i.png")
    assert len(list(tmp_path.glob("*.png"))) == 9


def test_verify_flags_a_moved_headline(tmp_path, capsys):
    import json

    base = {
        "test": {"auc": {"value": 0.61}, "recall_top20": {"value": 0.31}},
        "schedule": {"headline": {"days_saved_per_critical": 1.8}},
        "selection": {"chosen": "boosting"},
    }
    moved = json.loads(json.dumps(base))
    moved["test"]["auc"]["value"] = 0.70
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(json.dumps(base))
    b.write_text(json.dumps(base))
    c.write_text(json.dumps(moved))
    assert verify_main([str(a), str(b)]) == 0
    assert verify_main([str(a), str(c)]) == 1
    assert verify_main([str(a)]) == 2
