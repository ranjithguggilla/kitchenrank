"""The studies behind the headline numbers: model choice, calibration, leakage, ablations and audits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold

from .config import N_BOOT, SEED
from .features import EXTRAS, build_features, feature_families
from .metrics import ece, paired_week_diff, recall_at, summary, week_bootstrap
from .models import make_model
from .ranking import weekly_ranking_scores

COMPLEXITY = ["static logistic", "logistic", "boosting", "random forest"]


# ---------------------------------------------------------------- calibration
class Calibrator:
    """Maps a raw score to a probability. `method` is one of identity, sigmoid, isotonic."""

    def __init__(self, method: str = "identity"):
        self.method = method
        self._fit = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        if self.method == "sigmoid":
            self._fit = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        elif self.method == "isotonic":
            self._fit = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(np.asarray(p), y)
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "sigmoid":
            return self._fit.predict_proba(self._logit(p))[:, 1]
        if self.method == "isotonic":
            return self._fit.predict(p)
        return p


def _row_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def choose_calibrator(p_valid: np.ndarray, y_valid: np.ndarray) -> tuple[Calibrator, pd.DataFrame]:
    """Fit on the earlier half of the validation block, compare on the later half, refit the winner on all of it.

    Leaving the scores alone is the default. A calibrator is used only if it beats that by more than one standard
    error of the per-row log-loss difference.
    """
    half = len(y_valid) // 2
    y2 = np.asarray(y_valid[half:], float)
    base_loss = _row_loss(y2, p_valid[half:])
    rows, gains = [], {}
    for method in ("identity", "sigmoid", "isotonic"):
        cal = Calibrator(method).fit(p_valid[:half], y_valid[:half])
        q = np.clip(cal.predict(p_valid[half:]), 1e-6, 1 - 1e-6)
        loss = _row_loss(y2, q)
        diff = base_loss - loss
        gains[method] = (diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)))
        rows.append({"method": method, "log_loss": float(loss.mean()), "ece": ece(y2, q)})
    table = pd.DataFrame(rows)
    best = "identity"
    best_gain = 0.0
    for method in ("sigmoid", "isotonic"):
        gain, se = gains[method]
        if gain > se and gain > best_gain:
            best, best_gain = method, gain
    return Calibrator(best).fit(p_valid, y_valid), table


# ---------------------------------------------------------------- model comparison
def compare_models(X, y, blocks, df, columns_by_kind, kinds, seed=SEED):
    tr, va = blocks["train"], blocks["valid"]
    fitted, preds, rows = {}, {}, []
    for k in kinds:
        cols = columns_by_kind[k]
        m = make_model(k, cols, seed).fit(X.loc[tr, cols], y.loc[tr])
        p = m.predict_proba(X.loc[va, cols])[:, 1]
        fitted[k], preds[k] = m, p
        s = summary(y.loc[va].to_numpy(), p)
        s.update({f"weekly_{a}": b for a, b in weekly_ranking_scores(df.loc[va], p).items()})
        s["model"] = k
        rows.append(s)
    table = pd.DataFrame(rows).set_index("model")
    return fitted, preds, table


def pick_simplest_within_one_se(preds, y_valid, weeks_valid, candidates, n_boot=N_BOOT, seed=SEED):
    """Smallest model whose validation log loss is within one bootstrap SE of the best."""
    yv = np.asarray(y_valid)
    losses = {k: log_loss(yv, np.clip(preds[k], 1e-6, 1 - 1e-6)) for k in candidates}
    best = min(losses, key=losses.get)
    se = {}
    for k in candidates:
        if k == best:
            se[k] = 0.0
            continue
        pt, lo, hi = paired_week_diff(
            weeks_valid,
            lambda i, k=k: log_loss(yv[i], np.clip(preds[k][i], 1e-6, 1 - 1e-6), labels=[0, 1]),
            lambda i: log_loss(yv[i], np.clip(preds[best][i], 1e-6, 1 - 1e-6), labels=[0, 1]),
            n_boot,
            seed,
        )
        se[k] = (hi - lo) / 3.92
    tol = max(se.values()) if se else 0.0
    order = [k for k in COMPLEXITY if k in candidates]
    for k in order:
        gap = losses[k] - losses[best]
        if gap <= se.get(k, 0.0) or k == best:
            return k, {"best": best, "losses": losses, "se_of_gap": se, "tolerance": tol}
    return best, {"best": best, "losses": losses, "se_of_gap": se, "tolerance": tol}


# ---------------------------------------------------------------- test-set reporting
def report_block(df_test, p, seed=SEED, n_boot=N_BOOT):
    y = df_test["any_critical"].to_numpy()
    weeks = df_test["week"].to_numpy()
    out = {"point": summary(y, p)}
    for name, fn in {
        "auc": lambda i: roc_auc_score(y[i], p[i]),
        "recall_top10": lambda i: recall_at(y[i], p[i], 0.10),
        "recall_top20": lambda i: recall_at(y[i], p[i], 0.20),
        "recall_top30": lambda i: recall_at(y[i], p[i], 0.30),
    }.items():
        pt, lo, hi = week_bootstrap(weeks, fn, n_boot, seed)
        out[name] = {"value": pt, "lo": lo, "hi": hi}
    out["weekly_ranking"] = weekly_ranking_scores(df_test, p)
    return out


# ---------------------------------------------------------------- leakage audit
def leak_audit(df, blocks, city_rate, columns, kind, seed=SEED):
    """Three ways to fool yourself, each measured on the same held-out year."""
    tr, te = blocks["train"], blocks["test"]
    y = df["any_critical"]
    out = {}

    X_asof = build_features(df, city_rate, "asof")
    X_leaky = build_features(df, city_rate, "leaky")
    for name, X in (("as-of neighbours", X_asof), ("neighbours from the whole table", X_leaky)):
        m = make_model(kind, columns, seed).fit(X.loc[tr, columns], y.loc[tr])
        out[name] = float(roc_auc_score(y.loc[te], m.predict_proba(X.loc[te, columns])[:, 1]))

    # random folds over train + valid + test vs the forward test
    pool = df.index
    aucs = []
    for a, b in KFold(5, shuffle=True, random_state=seed).split(pool):
        m = make_model(kind, columns, seed).fit(X_asof.iloc[a][columns], y.iloc[a])
        aucs.append(roc_auc_score(y.iloc[b], m.predict_proba(X_asof.iloc[b][columns])[:, 1]))
    out["shuffled 5-fold on all years"] = float(np.mean(aucs))

    # the same-visit counts: what a careless join would hand the model
    cheat = X_asof.assign(same_visit_serious=df["serious"], same_visit_minor=df["minor"])
    cc = [*columns, "same_visit_serious", "same_visit_minor"]
    m = make_model(kind, cc, seed).fit(cheat.loc[tr, cc], y.loc[tr])
    out["with the same visit's serious and minor counts"] = float(
        roc_auc_score(y.loc[te], m.predict_proba(cheat.loc[te, cc])[:, 1])
    )
    return out


# ---------------------------------------------------------------- ablations
def _fit_predict(X, y, fit_idx, eval_idx, cols, kind, seed):
    m = make_model(kind, cols, seed).fit(X.loc[fit_idx, cols], y.loc[fit_idx])
    return m.predict_proba(X.loc[eval_idx, cols])[:, 1]


def family_ablation(X, y, df, fit_idx, eval_idx, columns, kind, seed=SEED, n_boot=N_BOOT):
    """Retrain without each feature group and report what the evaluation block loses (negative: it gained)."""
    yt = y.loc[eval_idx].to_numpy()
    weeks = df.loc[eval_idx, "week"].to_numpy()
    p_full = _fit_predict(X, y, fit_idx, eval_idx, columns, kind, seed)
    rows = []
    for fam, cols in feature_families(X).items():
        drop = set(cols)
        keep = [c for c in columns if c not in drop]
        if len(keep) == len(columns):
            continue
        p = _fit_predict(X, y, fit_idx, eval_idx, keep, kind, seed)
        d_auc = paired_week_diff(
            weeks,
            lambda i, p=p: roc_auc_score(yt[i], p_full[i]),
            lambda i, p=p: roc_auc_score(yt[i], p[i]),
            n_boot,
            seed,
        )
        d_rec = paired_week_diff(
            weeks,
            lambda i, p=p: recall_at(yt[i], p_full[i], 0.2),
            lambda i, p=p: recall_at(yt[i], p[i], 0.2),
            n_boot,
            seed,
        )
        rows.append(
            {
                "dropped_family": fam,
                "auc_without": float(roc_auc_score(yt, p)),
                "auc_loss": d_auc[0],
                "auc_loss_lo": d_auc[1],
                "auc_loss_hi": d_auc[2],
                "recall20_loss": d_rec[0],
                "recall20_lo": d_rec[1],
                "recall20_hi": d_rec[2],
            }
        )
    return pd.DataFrame(rows).sort_values("auc_loss", ascending=False).reset_index(drop=True)


def prune_families(table: pd.DataFrame) -> list[str]:
    """Families whose removal raised validation AUC with the whole interval above zero."""
    return table.loc[table["auc_loss_hi"] < 0, "dropped_family"].tolist()


def columns_without(X, columns, families: list[str]) -> list[str]:
    fams = feature_families(X)
    gone = {c for f in families for c in fams[f]}
    return [c for c in columns if c not in gone]


def add_back(X, y, df, fit_idx, eval_idx, columns, extra_cols, kind, seed=SEED, n_boot=N_BOOT):
    """Paired gain from putting columns back in (AUC with minus AUC without)."""
    yt = y.loc[eval_idx].to_numpy()
    weeks = df.loc[eval_idx, "week"].to_numpy()
    p0 = _fit_predict(X, y, fit_idx, eval_idx, columns, kind, seed)
    p1 = _fit_predict(X, y, fit_idx, eval_idx, [*columns, *extra_cols], kind, seed)
    pt, lo, hi = paired_week_diff(
        weeks, lambda i: roc_auc_score(yt[i], p1[i]), lambda i: roc_auc_score(yt[i], p0[i]), n_boot, seed
    )
    return {
        "auc_without": float(roc_auc_score(yt, p0)),
        "auc_with": float(roc_auc_score(yt, p1)),
        "gain": pt,
        "lo": lo,
        "hi": hi,
    }


# ---------------------------------------------------------------- inspector audit
def inspector_audit(df_test, p, seed=SEED, n_boot=N_BOOT):
    """Observed over expected critical rate per inspector colour, on the frozen model's probabilities."""
    d = df_test.assign(p=p).dropna(subset=["inspector"])
    rows = []
    for name, g in d.groupby("inspector"):
        idx = g.index.to_numpy()
        pos = d.index.get_indexer(idx)
        yy, pp, ww = d["any_critical"].to_numpy(), d["p"].to_numpy(), d["week"].to_numpy()
        sel = np.zeros(len(d), bool)
        sel[pos] = True

        def oe(i, sel=sel, yy=yy, pp=pp):
            m = sel[i]
            return yy[i][m].mean() / pp[i][m].mean() if m.any() else np.nan

        pt, lo, hi = week_bootstrap(ww, oe, n_boot, seed)
        rows.append(
            {
                "inspector": name,
                "inspections": len(g),
                "critical_rate": float(g["any_critical"].mean()),
                "expected_rate": float(g["p"].mean()),
                "obs_over_exp": pt,
                "lo": lo,
                "hi": hi,
            }
        )
    table = pd.DataFrame(rows).sort_values("obs_over_exp")
    ct = pd.crosstab(d["inspector"], d["any_critical"])
    chi2, pval, *_ = stats.chi2_contingency(ct)
    return table, {"chi2": float(chi2), "p_value": float(pval), "dof": int(len(ct) - 1), "rows": len(d)}


def inspector_ablation(df, X, y, blocks, columns, kind, seed=SEED):
    """Does knowing the inspector's colour improve the ranking? It does, and that is the problem."""
    tr, te = blocks["train"], blocks["test"]
    ok = df["inspector"].notna()
    dummies = pd.get_dummies(df["inspector"], prefix="insp", dtype=float)
    Xi = X.join(dummies)
    cols_i = [*columns, *dummies.columns]
    tri, tei = tr[ok.loc[tr].to_numpy()], te[ok.loc[te].to_numpy()]
    base = make_model(kind, columns, seed).fit(X.loc[tri, columns], y.loc[tri])
    withi = make_model(kind, cols_i, seed).fit(Xi.loc[tri, cols_i], y.loc[tri])
    return {
        "auc_without_inspector": float(roc_auc_score(y.loc[tei], base.predict_proba(X.loc[tei, columns])[:, 1])),
        "auc_with_inspector": float(roc_auc_score(y.loc[tei], withi.predict_proba(Xi.loc[tei, cols_i])[:, 1])),
        "rows_test": len(tei),
    }


# ---------------------------------------------------------------- drift and refits
def weekly_rate_breaks(df, min_week_size=60):
    wk = df.groupby("week").agg(n=("any_critical", "size"), rate=("any_critical", "mean")).reset_index()
    wk = wk[wk["n"] >= min_week_size].reset_index(drop=True)
    sig = wk["rate"].to_numpy()
    pen = float(np.log(len(sig)) * sig.var() * 2.0)
    algo = rpt.Pelt(model="l2", min_size=8, jump=1).fit(sig)
    ends = algo.predict(pen=pen)[:-1]
    segs, start = [], 0
    for e in [*ends, len(sig)]:
        segs.append(
            {
                "from": str(wk.loc[start, "week"].date()),
                "to": str(wk.loc[e - 1, "week"].date()),
                "weeks": int(e - start),
                "critical_rate": float(np.average(sig[start:e], weights=wk["n"].to_numpy()[start:e])),
            }
        )
        start = e
    return wk, segs


def quarterly_refit(df, X, y, blocks, columns, kind, seed=SEED):
    """Expanding-window refit at the start of every 2014 quarter, against the single frozen fit."""
    te = blocks["test"]
    frozen_tr = blocks["train"]
    quarters = df.loc[te, "date"].dt.to_period("Q")
    p_ref = pd.Series(index=te, dtype=float)
    p_frz = pd.Series(index=te, dtype=float)
    frozen = make_model(kind, columns, seed).fit(X.loc[frozen_tr, columns], y.loc[frozen_tr])
    for q in quarters.unique():
        idx = quarters.index[quarters == q]
        cutoff = df.loc[idx, "date"].min()
        past = df.index[df["date"] < cutoff]
        m = make_model(kind, columns, seed).fit(X.loc[past, columns], y.loc[past])
        p_ref.loc[idx] = m.predict_proba(X.loc[idx, columns])[:, 1]
        p_frz.loc[idx] = frozen.predict_proba(X.loc[idx, columns])[:, 1]
    yt = y.loc[te].to_numpy()
    return {
        "frozen_auc": float(roc_auc_score(yt, p_frz)),
        "refit_auc": float(roc_auc_score(yt, p_ref)),
        "frozen_recall20": recall_at(yt, p_frz.to_numpy(), 0.2),
        "refit_recall20": recall_at(yt, p_ref.to_numpy(), 0.2),
    }


def extras_study(df, X, y, blocks, columns, kind, seed=SEED, n_boot=N_BOOT):
    """Licence flags, business age and map heat exist for two thirds of rows. Do they help on those rows?"""
    have = df["business_age"].notna()
    lic = [c for c in df.columns if c.startswith("lic_")]
    Xe = X.join(df[lic])
    base = [c for c in columns if c not in EXTRAS]
    plus = [*base, *EXTRAS, *lic]
    sets = {k: v[have.loc[v].to_numpy()] for k, v in blocks.items()}
    out = {"rows": {k: len(v) for k, v in sets.items()}}
    te = sets["test"]
    p0 = _fit_predict(Xe, y, sets["train"], te, base, kind, seed)
    p1 = _fit_predict(Xe, y, sets["train"], te, plus, kind, seed)
    yt = y.loc[te].to_numpy()
    out["without extras"] = {"auc": float(roc_auc_score(yt, p0)), "recall20": recall_at(yt, p0, 0.2)}
    out["with extras"] = {"auc": float(roc_auc_score(yt, p1)), "recall20": recall_at(yt, p1, 0.2)}
    pt, lo, hi = paired_week_diff(
        df.loc[te, "week"].to_numpy(),
        lambda i: roc_auc_score(yt[i], p1[i]),
        lambda i: roc_auc_score(yt[i], p0[i]),
        n_boot,
        seed,
    )
    out["auc_gain"] = {"value": pt, "lo": lo, "hi": hi}
    return out


def city_history_agreement(df, X) -> dict[str, float]:
    """Compare the as-of history features with the columns the city shipped."""
    return {
        "first_visit_agreement": float((X["first_visit"] == df["city_first_record"]).mean()),
        "prior_fail_agreement": float((X["prev_fail"] == df["city_past_fail"]).mean()),
        "rows": len(df),
    }
