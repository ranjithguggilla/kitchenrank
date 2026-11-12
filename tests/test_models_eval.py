import numpy as np
import pandas as pd
import pytest

from kitchenrank.data import split_blocks
from kitchenrank.evaluate import (
    Calibrator,
    choose_calibrator,
    city_history_agreement,
    columns_without,
    compare_models,
    family_ablation,
    inspector_audit,
    leak_audit,
    pick_simplest_within_one_se,
    prune_families,
    quarterly_refit,
    report_block,
    weekly_rate_breaks,
)
from kitchenrank.features import EXTRAS, build_features
from kitchenrank.models import KINDS, make_model


@pytest.fixture(scope="module")
def setup(frame):
    blocks = split_blocks(frame, "2012-03-31", "2012-09-30")
    rate = float(frame.loc[blocks["train"], "any_critical"].mean())
    X = build_features(frame, rate)
    cols = [c for c in X.columns if c not in EXTRAS]
    return frame, blocks, rate, X, cols


@pytest.mark.parametrize("kind", KINDS)
def test_every_model_fits_and_predicts(setup, kind):
    df, blocks, _, X, cols = setup
    use = ["risk", "is_restaurant"] if kind == "static logistic" else cols
    m = make_model(kind, use).fit(X.loc[blocks["train"], use], df.loc[blocks["train"], "any_critical"])
    p = m.predict_proba(X.loc[blocks["valid"], use])[:, 1]
    assert p.shape == (len(blocks["valid"]),) and ((p >= 0) & (p <= 1)).all()


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        make_model("transformer", ["a"])


def test_models_beat_chance_on_a_planted_signal(setup):
    df, blocks, _, X, cols = setup
    y = df["any_critical"]
    fitted, preds, table = compare_models(
        X, y, blocks, df, {"logistic": cols, "boosting": cols}, ["logistic", "boosting"]
    )
    assert table.loc["logistic", "auc"] > 0.55
    assert set(fitted) == {"logistic", "boosting"}


def test_one_se_rule_takes_the_simplest_when_models_tie(setup):
    df, blocks, _, _, _ = setup
    yv = df.loc[blocks["valid"], "any_critical"].to_numpy()
    p = 0.15 + 0.05 * yv
    preds = {"logistic": p, "boosting": p.copy(), "random forest": p.copy()}
    weeks = df.loc[blocks["valid"], "week"].to_numpy()
    chosen, info = pick_simplest_within_one_se(preds, yv, weeks, ["logistic", "boosting", "random forest"], 30, 1)
    assert chosen == "logistic"
    assert info["best"] == "logistic"


def test_one_se_rule_keeps_the_complex_model_when_it_is_clearly_better(setup):
    df, blocks, _, _, _ = setup
    yv = df.loc[blocks["valid"], "any_critical"].to_numpy()
    weeks = df.loc[blocks["valid"], "week"].to_numpy()
    preds = {
        "logistic": np.full(len(yv), yv.mean()),
        "boosting": 0.05 + 0.5 * yv,
        "random forest": np.full(len(yv), 0.3),
    }
    chosen, info = pick_simplest_within_one_se(preds, yv, weeks, ["logistic", "boosting", "random forest"], 30, 1)
    assert chosen == "boosting" and info["best"] == "boosting"


def test_calibrator_identity_sigmoid_isotonic():
    rng = np.random.default_rng(0)
    true = rng.uniform(0.05, 0.4, 3000)
    y = (rng.random(3000) < true).astype(int)
    skew = np.clip(true * 0.5, 0.001, 0.999)  # model that under-predicts by half
    for method in ("sigmoid", "isotonic"):
        c = Calibrator(method).fit(skew, y)
        assert abs(c.predict(skew).mean() - y.mean()) < 0.02
    assert np.array_equal(Calibrator().predict(skew), skew)
    cal, table = choose_calibrator(skew, y)
    assert cal.method in ("sigmoid", "isotonic")
    assert set(table["method"]) == {"identity", "sigmoid", "isotonic"}


def test_choose_calibrator_keeps_identity_when_already_good():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.4, 4000)
    y = (rng.random(4000) < p).astype(int)
    cal, _ = choose_calibrator(p, y)
    assert cal.method == "identity"


def test_test_report_has_intervals(setup, dense):
    _, blocks, _, _, _ = setup
    d = dense.loc[blocks["test"]].reset_index(drop=True)
    r = report_block(d, np.random.default_rng(0).random(len(d)), 1, 30)
    assert r["auc"]["lo"] <= r["auc"]["value"] <= r["auc"]["hi"]
    assert "ndcg@10" in r["weekly_ranking"]


def test_family_ablation_and_pruning_rules(setup):
    df, blocks, _, X, cols = setup
    t = family_ablation(X, df["any_critical"], df, blocks["train"], blocks["valid"], cols, "logistic", 1, 30)
    assert {"dropped_family", "auc_loss", "auc_loss_lo", "auc_loss_hi"} <= set(t.columns)
    fake = pd.DataFrame({"dropped_family": ["a", "b"], "auc_loss_hi": [-0.01, 0.02]})
    assert prune_families(fake) == ["a"]
    left = columns_without(X, cols, ["location"])
    assert "lat" not in left and "risk" in left


def test_leak_audit_orders_the_cheats(setup):
    df, blocks, rate, _, cols = setup
    out = leak_audit(df, blocks, rate, cols, "logistic", 1)
    assert out["with the same visit's serious and minor counts"] >= out["as-of neighbours"] - 0.02
    assert set(out) >= {"as-of neighbours", "shuffled 5-fold on all years"}


def test_inspector_audit_flags_a_planted_team_effect(setup):
    df, blocks, _, _, _ = setup
    d = df.loc[blocks["test"]].reset_index(drop=True).copy()
    rng = np.random.default_rng(3)
    d["any_critical"] = (rng.random(len(d)) < np.where(d["inspector"] == "blue", 0.6, 0.1)).astype(int)
    table, test = inspector_audit(d, np.full(len(d), 0.2), 1, 30)
    assert table.iloc[-1]["inspector"] == "blue"
    assert test["p_value"] < 0.001 and test["dof"] == 2


def test_rate_breaks_find_a_level_shift():
    weeks = pd.date_range("2012-01-02", periods=60, freq="W-MON")
    rng = np.random.default_rng(0)
    rows = []
    for k, w in enumerate(weeks):
        rate = 0.1 if k < 30 else 0.3
        rows.append(pd.DataFrame({"week": w, "any_critical": (rng.random(100) < rate).astype(int)}))
    wk, segs = weekly_rate_breaks(pd.concat(rows, ignore_index=True), min_week_size=50)
    assert len(segs) >= 2
    assert segs[0]["critical_rate"] < 0.2 < segs[-1]["critical_rate"]
    assert len(wk) == 60


def test_quarterly_refit_returns_both_arms(setup):
    df, blocks, _, X, cols = setup
    out = quarterly_refit(df, X, df["any_critical"], blocks, cols, "logistic", 1)
    assert set(out) == {"frozen_auc", "refit_auc", "frozen_recall20", "refit_recall20"}


def test_history_agreement_against_city_columns(setup):
    df, _, _, X, _ = setup
    out = city_history_agreement(df, X)
    assert 0 <= out["first_visit_agreement"] <= 1 and out["rows"] == len(df)
