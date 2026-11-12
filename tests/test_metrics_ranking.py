import numpy as np

from kitchenrank.metrics import ece, lift_at, paired_week_diff, recall_at, summary, week_bootstrap, weekly_table
from kitchenrank.ranking import n_weeks, to_qrels_run, weekly_ranking_scores


def test_ece_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.2, 0.8], 500)
    y = np.r_[np.tile([0, 0, 0, 0, 1], 100), np.tile([1, 1, 1, 1, 0], 100)]
    assert ece(y, p) < 1e-9


def test_recall_and_lift_on_a_known_ranking():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert recall_at(y, p, 0.3) == 1.0
    assert recall_at(y, p, 0.1) == 0.5
    assert lift_at(y, p, 0.2) == 2.5
    assert np.isnan(recall_at(np.zeros(4), np.arange(4), 0.5))


def test_summary_keys_and_ranges(frame):
    p = np.clip(frame["any_critical"] * 0.3 + 0.1, 0, 1)
    s = summary(frame["any_critical"].to_numpy(), p.to_numpy())
    assert set(s) == {"auc", "pr_auc", "log_loss", "brier", "ece", "recall_top20", "lift_top20"}
    assert s["auc"] > 0.9


def test_week_bootstrap_is_reproducible_and_brackets_the_estimate(frame):
    weeks = frame["week"].to_numpy()
    y = frame["any_critical"].to_numpy()
    a = week_bootstrap(weeks, lambda i: y[i].mean(), 100, 1)
    b = week_bootstrap(weeks, lambda i: y[i].mean(), 100, 1)
    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_paired_difference_of_identical_functions_is_zero(frame):
    w = frame["week"].to_numpy()
    y = frame["any_critical"].to_numpy()
    pt, lo, hi = paired_week_diff(w, lambda i: y[i].mean(), lambda i: y[i].mean(), 50, 2)
    assert pt == lo == hi == 0.0


def test_weekly_table_skips_thin_weeks(dense):
    t = weekly_table(dense, np.random.default_rng(0).random(len(dense)), 0.2)
    assert (t["n"] >= 20).all()


def test_weekly_ranking_perfect_and_reversed(dense):
    frame = dense
    y = frame["any_critical"].to_numpy().astype(float)
    good = weekly_ranking_scores(frame, y + np.random.default_rng(1).random(len(y)) * 0.01)
    bad = weekly_ranking_scores(frame, -y)
    assert good["ndcg@10"] > bad["ndcg@10"]
    assert good["mrr"] == 1.0


def test_qrels_drop_weeks_without_a_positive(dense):
    f = dense.copy()
    first = f["week"].min()
    f.loc[f["week"] == first, "any_critical"] = 0
    qrels, run = to_qrels_run(f, np.zeros(len(f)), min_size=1)
    assert str(first.date()) not in qrels.qrels
    assert n_weeks(f, min_size=1) == len(qrels.qrels)


def test_no_usable_weeks_gives_nan_not_a_crash(frame):
    thin = frame.head(10)
    out = weekly_ranking_scores(thin, np.zeros(len(thin)))
    assert all(np.isnan(v) for v in out.values())
