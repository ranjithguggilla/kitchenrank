"""Run every study end to end and write metrics, tables, charts and the app bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import plots
from .config import (
    ASSET_DIR,
    COSTS,
    HEADLINE_HORIZON,
    HEADLINE_MAX_DELAY,
    HORIZONS,
    MAX_DELAYS,
    N_BOOT,
    OUT_DIR,
    SEED,
)
from .data import load_inspections, split_blocks
from .economics import value_table, visits_to_find
from .evaluate import (
    add_back,
    choose_calibrator,
    city_history_agreement,
    columns_without,
    compare_models,
    extras_study,
    family_ablation,
    inspector_ablation,
    inspector_audit,
    leak_audit,
    pick_simplest_within_one_se,
    prune_families,
    quarterly_refit,
    report_block,
    weekly_rate_breaks,
)
from .features import EXTRAS, build_features
from .metrics import summary, weekly_table
from .schedule import saved_per_row, simulate
from .score import Bundle, column_defaults


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return round(o, 5)
    return o


def _window_ci(df, score, horizon, delay, n_boot, seed):
    """Days saved per critical finding with an interval from resampling the planning windows."""
    saved, win, _ = saved_per_row(df, score, horizon, delay)
    crit = df["any_critical"].to_numpy().astype(bool)
    ids = np.unique(win)
    sums = np.array([saved[(win == k) & crit].sum() for k in ids])
    cnt = np.array([((win == k) & crit).sum() for k in ids])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        draws.append(sums[pick].sum() / cnt[pick].sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(sums.sum() / cnt.sum()), float(lo), float(hi)


def data_profile(df: pd.DataFrame, X: pd.DataFrame) -> dict:
    """Plain critical rates behind the story the model tells, all years pooled."""
    y = df["any_critical"]
    month = y.groupby(df["date"].dt.month).mean()
    prior = y.groupby(X["prev_critical"].clip(upper=2)).agg(["mean", "size"])
    return {
        "overall": float(y.mean()),
        "by_month": {int(k): float(v) for k, v in month.items()},
        "by_previous_critical_count": {
            ("2 or more" if k >= 2 else str(int(k))): {"rate": float(r["mean"]), "inspections": int(r["size"])}
            for k, r in prior.iterrows()
        },
        "by_risk": {int(k): float(v) for k, v in y.groupby(df["risk"]).mean().items()},
        "restaurant": float(y[df["facility"] == "Restaurant"].mean()),
        "other_facilities": float(y[df["facility"] != "Restaurant"].mean()),
        "first_visit": float(y[X["first_visit"] == 1].mean()),
        "repeat_visit": float(y[X["first_visit"] == 0].mean()),
    }


def run(out_dir: Path = OUT_DIR, asset_dir: Path = ASSET_DIR, n_boot: int = N_BOOT, quick: bool = False) -> dict:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    df = load_inspections()
    df["fail_flag"] = (df["result"] == "Fail").astype(int)
    blocks = split_blocks(df)
    y = df["any_critical"]
    city_rate = float(y.loc[blocks["train"]].mean())
    X = build_features(df, city_rate, "asof")
    all_columns = [c for c in X.columns if c not in EXTRAS]
    # decide on feature groups from the validation block alone, before the test year is ever scored
    prune_table = family_ablation(X, y, df, blocks["train"], blocks["valid"], all_columns, "boosting", SEED, n_boot)
    prune_table.to_csv(table_dir / "validation_family_ablation.csv", index=False)
    dropped = prune_families(prune_table)
    columns = columns_without(X, all_columns, dropped)
    static_cols = ["risk", "is_restaurant"]
    kinds = ["static logistic", "logistic", "random forest", "boosting", "boosting balanced"]
    by_kind = {k: (static_cols if k == "static logistic" else columns) for k in kinds}

    m: dict = {}
    m["data"] = {
        "inspections": len(df),
        "licences": int(df["license"].nunique()),
        "first_date": str(df["date"].min().date()),
        "last_date": str(df["date"].max().date()),
        "rows": {k: len(v) for k, v in blocks.items()},
        "critical_rate": {k: float(y.loc[v].mean()) for k, v in blocks.items()},
        "features": len(columns),
        "features_before_pruning": len(all_columns),
        "dropped_families": dropped,
        "history_agreement_with_city": city_history_agreement(df, X),
    }

    m["profile"] = data_profile(df, X)

    # ---- choose a model on the validation year-half
    fitted, vpred, vtable = compare_models(X, y, blocks, df, by_kind, kinds, SEED)
    vtable.to_csv(table_dir / "validation_models.csv")
    candidates = ["logistic", "boosting", "random forest"]
    wk_valid = df.loc[blocks["valid"], "week"].to_numpy()
    chosen, why = pick_simplest_within_one_se(
        vpred, y.loc[blocks["valid"]].to_numpy(), wk_valid, candidates, n_boot, SEED
    )
    m["selection"] = {"chosen": chosen, **why}
    model = fitted[chosen]

    cal, cal_table = choose_calibrator(vpred[chosen], y.loc[blocks["valid"]].to_numpy())
    cal_table.to_csv(table_dir / "calibration_choice.csv", index=False)
    m["calibration"] = {"method": cal.method, "options": cal_table.to_dict("records")}

    # ---- the held-out year, scored once
    te = blocks["test"]
    dte = df.loc[te].reset_index(drop=True)
    yte = y.loc[te].to_numpy()
    raw_te = model.predict_proba(X.loc[te, columns])[:, 1]
    p_te = cal.predict(raw_te)
    m["test"] = report_block(dte, p_te, SEED, n_boot)
    m["test"]["uncalibrated"] = summary(yte, raw_te)

    rule = (X["prev_critical"] + 0.5 * X["prev_fail"]).fillna(0).loc[te].to_numpy()
    rng = np.random.default_rng(SEED)
    rand = rng.random(len(te))
    m["baselines"] = {
        "last_visit_rule": report_block(dte, rule + 1e-3 * rand, SEED, n_boot),
        "random": report_block(dte, rand, SEED, n_boot),
    }
    others = {}
    for k in kinds:
        cols = by_kind[k]
        mod = fitted[k]
        others[k] = summary(yte, mod.predict_proba(X.loc[te, cols])[:, 1])
    m["test_all_models"] = others
    pd.DataFrame(others).T.to_csv(table_dir / "test_models.csv")

    # ---- bands from the held-out year
    q = np.quantile(p_te, [0.40, 0.70, 0.90])
    cut = {"usual": float(q[0]), "elevated": float(q[1]), "high": float(q[2])}
    labels = pd.cut(p_te, [-1, q[0], q[1], q[2], 2], labels=["lower than usual", "about usual", "elevated", "high"])
    band = (
        pd.DataFrame({"band": labels, "y": yte})
        .groupby("band", observed=True)
        .agg(inspections=("y", "size"), critical_rate=("y", "mean"))
    )
    band.to_csv(table_dir / "risk_bands.csv")
    m["bands"] = {"cutoffs": cut, "table": band.reset_index().to_dict("records")}

    # ---- audits and ablations
    m["leak_audit"] = leak_audit(df, blocks, city_rate, columns, chosen, SEED)
    abl = family_ablation(X, y, df, blocks["train"], te, columns, chosen, SEED, n_boot)
    abl.to_csv(table_dir / "family_ablation.csv", index=False)
    m["ablation"] = abl.to_dict("records")
    m["validation_ablation"] = prune_table.to_dict("records")
    removed_cols = [c for c in all_columns if c not in columns]
    if removed_cols:
        m["pruned_back_in"] = add_back(X, y, df, blocks["train"], te, columns, removed_cols, chosen, SEED, n_boot)
    itab, itest = inspector_audit(dte.assign(inspector=df.loc[te, "inspector"].to_numpy()), p_te, SEED, n_boot)
    itab.to_csv(table_dir / "inspector_audit.csv", index=False)
    m["inspector"] = {"table": itab.to_dict("records"), "homogeneity_test": itest}
    m["inspector_ablation"] = inspector_ablation(df, X, y, blocks, columns, chosen, SEED)
    weekly_all, segs = weekly_rate_breaks(df)
    m["rate_segments"] = segs
    m["refit"] = quarterly_refit(df, X, y, blocks, columns, chosen, SEED)
    m["extras"] = extras_study(df, X, y, blocks, columns, chosen, SEED, n_boot)

    # ---- scheduling
    policies = {"model": p_te, "last-visit rule": rule + 1e-3 * rand, "random": rand, "oracle": yte + 1e-3 * rand}
    rows = []
    for hz in HORIZONS:
        for dly in MAX_DELAYS:
            for name, sc in policies.items():
                r = simulate(dte, sc, hz, dly)
                rows.append({"policy": name, "horizon": hz, "max_delay": dly, **r})
    sched = pd.DataFrame(rows)
    sched.to_csv(table_dir / "schedule_grid.csv", index=False)
    head = sched[(sched.horizon == HEADLINE_HORIZON) & (sched.max_delay == HEADLINE_MAX_DELAY)].set_index("policy")
    pt, lo, hi = _window_ci(dte, p_te, HEADLINE_HORIZON, HEADLINE_MAX_DELAY, n_boot, SEED)
    m["schedule"] = {
        "horizon": HEADLINE_HORIZON,
        "max_delay": HEADLINE_MAX_DELAY,
        "headline": {"days_saved_per_critical": pt, "lo": lo, "hi": hi},
        "policies": head.to_dict("index"),
    }
    exposure = float(head.loc["model", "exposure_days_avoided"])
    vt = value_table(exposure, len(dte))
    vt.to_csv(table_dir / "value_table.csv", index=False)
    visit_cost = COSTS.visit_cost
    shares = {
        "model": visits_to_find(yte, p_te, 0.5),
        "last_visit_rule": visits_to_find(yte, rule + 1e-3 * rand, 0.5),
        "random": visits_to_find(yte, rand, 0.5),
    }
    saved_visits = (shares["random"] - shares["model"]) * len(dte)
    m["economics"] = {
        "visit_cost_assumed": visit_cost,
        "visits_saved_to_reach_half_of_criticals": float(saved_visits),
        "visit_saving_dollars": float(saved_visits * visit_cost),
        "criticals_per_100_visits_top30": {
            "model": float(yte[np.argsort(-p_te, kind="stable")[: int(0.3 * len(yte))]].mean() * 100),
            "random": float(yte.mean() * 100),
        },
        "exposure_days_avoided": exposure,
        "inspections": len(dte),
        "value_table": vt.to_dict("records"),
        "share_of_list_for_half_of_criticals": shares,
    }

    # ---- charts
    plots.model_comparison(vtable, chosen, asset_dir / "model_comparison.png")
    plots.gain_curve(yte, p_te, rule + 1e-3 * rand, asset_dir / "gain_curve.png")
    plots.calibration_plot(yte, raw_te, p_te, asset_dir / "calibration.png")
    plots.weekly_recall(weekly_table(dte, p_te), asset_dir / "weekly_recall.png")
    plots.delay_tradeoff(sched[sched.horizon == HEADLINE_HORIZON], asset_dir / "delay_tradeoff.png")
    plots.ablation_plot(abl, asset_dir / "ablation.png")
    plots.leak_plot(m["leak_audit"], asset_dir / "leak_audit.png")
    plots.inspector_plot(itab, asset_dir / "inspector.png")
    plots.rate_over_time(weekly_all, segs, asset_dir / "rate_over_time.png")

    # ---- app bundle
    train = X.loc[blocks["train"], columns]
    bundle = Bundle(
        model=model,
        calibrator=cal,
        columns=columns,
        defaults=column_defaults(train, columns),
        base_rate=float(np.mean(p_te)),
        cutoffs=cut,
        meta={"chosen": chosen, "city_rate": city_rate, "test_auc": m["test"]["auc"]["value"]},
    )
    joblib.dump(bundle, out_dir / "bundle.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(_clean(m), f, indent=2)
    return m


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    m = run(n_boot=60 if "--quick" in argv else N_BOOT)
    t = m["test"]
    print(f"chosen model: {m['selection']['chosen']}")
    print(f"test auc {t['auc']['value']:.3f} [{t['auc']['lo']:.3f}, {t['auc']['hi']:.3f}]")
    print(f"recall in top 20%: {t['recall_top20']['value']:.3f}")
    print(f"days earlier per critical finding: {m['schedule']['headline']['days_saved_per_critical']:.2f}")


if __name__ == "__main__":
    main()
