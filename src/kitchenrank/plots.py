"""README charts. One idea per figure, direct labels, the same few colours everywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1f2933"
MUTED = "#7b8794"
BLUE = "#2b6cb0"
ORANGE = "#d9822b"
GREY = "#b8c0c9"
RED = "#b83232"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
    }
)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_comparison(valid_table: pd.DataFrame, chosen: str, path: Path) -> None:
    t = valid_table.drop(index=["boosting balanced"], errors="ignore").sort_values("auc")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = [BLUE if m == chosen else GREY for m in t.index]
    axes[0].barh(t.index, t["auc"], color=colors)
    axes[0].set_xlim(0.5, max(0.7, t["auc"].max() + 0.02))
    axes[0].set_title("Validation AUC")
    axes[1].barh(t.index, t["log_loss"], color=colors)
    axes[1].set_xlim(t["log_loss"].min() - 0.01, t["log_loss"].max() + 0.004)
    axes[1].set_title("Validation log loss (lower is better)")
    axes[1].set_yticklabels([])
    for ax, col in zip(axes, ["auc", "log_loss"], strict=True):
        for y, v in enumerate(t[col]):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=9, color=INK)
    _save(fig, path)


def gain_curve(y: np.ndarray, p_model: np.ndarray, p_rule: np.ndarray, path: Path) -> None:
    def curve(score):
        order = np.argsort(-score, kind="stable")
        return np.r_[0, np.cumsum(y[order]) / y.sum()], np.linspace(0, 1, len(y) + 1)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for score, label, col in ((p_model, "model", BLUE), (p_rule, "last-visit rule", ORANGE)):
        c, x = curve(score)
        ax.plot(x, c, color=col, lw=2, label=label)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="random order")
    ax.set_xlabel("share of the list inspected, riskiest first")
    ax.set_ylabel("share of critical findings reached")
    ax.set_title("Reaching critical findings sooner (2014)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def calibration_plot(y: np.ndarray, raw: np.ndarray, cal: np.ndarray, path: Path, bins: int = 8) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for p, label, col in ((raw, "raw model", GREY), (cal, "after calibration", BLUE)):
        order = np.argsort(p)
        parts = np.array_split(order, bins)
        ax.plot([p[i].mean() for i in parts], [y[i].mean() for i in parts], marker="o", color=col, label=label)
    top = max(0.4, float(np.max(cal)))
    ax.plot([0, top], [0, top], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed critical rate")
    ax.set_title("Calibration on the 2014 test year")
    ax.legend(frameon=False)
    _save(fig, path)


def weekly_recall(weekly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(weekly["week"], weekly["recall"], color=BLUE, lw=1.5)
    ax.axhline(0.2, color=MUTED, ls="--", lw=1)
    ax.axhline(weekly["recall"].mean(), color=BLUE, lw=1, alpha=0.5)
    ax.text(weekly["week"].iloc[0], 0.02, "dashed: random order, 0.20", color=MUTED, fontsize=9, va="bottom")
    ax.text(
        weekly["week"].iloc[0], 0.09, f"thin solid: 2014 mean, {weekly['recall'].mean():.2f}", color=BLUE, fontsize=9
    )
    ax.set_ylim(0, max(0.8, weekly["recall"].max() + 0.05))
    ax.set_ylabel("critical findings in the top 20%")
    ax.set_title("Week by week, 2014")
    _save(fig, path)


def delay_tradeoff(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4))
    style = {"model": (BLUE, 2.4), "last-visit rule": (ORANGE, 1.8), "random": (MUTED, 1.4), "oracle": (RED, 1.4)}
    for pol, g in table.groupby("policy"):
        col, lw = style[pol]
        ls = "--" if pol == "oracle" else "-"
        g = g.sort_values("max_delay")
        ax.plot(g["max_delay"], g["days_saved_per_critical"], marker="o", color=col, lw=lw, ls=ls)
        ax.text(
            g["max_delay"].iloc[-1] + 0.6,
            g["days_saved_per_critical"].iloc[-1],
            pol,
            color=col,
            va="center",
            fontsize=9,
        )
    ax.set_xlim(-1, table["max_delay"].max() + 9)
    ax.set_xlabel("longest any business may be pushed back (days)")
    ax.set_ylabel("days earlier per critical finding")
    ax.set_title("What reordering buys, 28-day planning windows")
    _save(fig, path)


def ablation_plot(table: pd.DataFrame, path: Path) -> None:
    t = table.sort_values("auc_loss")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    err = np.vstack([t["auc_loss"] - t["auc_loss_lo"], t["auc_loss_hi"] - t["auc_loss"]])
    ax.barh(t["dropped_family"], t["auc_loss"], xerr=err, color=BLUE, ecolor=INK, capsize=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("AUC lost when the group is removed (95% interval, weeks resampled)")
    ax.set_title("What the model leans on")
    _save(fig, path)


def leak_plot(audit: dict, path: Path) -> None:
    names = list(audit)
    vals = [audit[n] for n in names]
    base = audit["as-of neighbours"]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.barh(names, vals, color=[BLUE if n == "as-of neighbours" else GREY for n in names])
    ax.axvline(base, color=BLUE, lw=1, ls="--")
    ax.set_xlim(0.5, max(vals) + 0.03)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("test AUC on 2014")
    ax.set_title("Leakage checks, scored on the 2014 test year")
    ax.invert_yaxis()
    _save(fig, path)


def inspector_plot(table: pd.DataFrame, path: Path) -> None:
    t = table.sort_values("obs_over_exp")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    err = np.vstack([t["obs_over_exp"] - t["lo"], t["hi"] - t["obs_over_exp"]])
    ax.errorbar(t["obs_over_exp"], t["inspector"], xerr=err, fmt="o", color=BLUE, ecolor=GREY, capsize=3)
    ax.axvline(1, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("critical findings: observed / model-expected")
    ax.set_title("Inspector teams differ")
    _save(fig, path)


def rate_over_time(weekly: pd.DataFrame, segments: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(weekly["week"], weekly["rate"], color=GREY, lw=1)
    for s in segments:
        ax.hlines(s["critical_rate"], pd.Timestamp(s["from"]), pd.Timestamp(s["to"]), color=BLUE, lw=2.4)
    ax.set_ylabel("share of inspections with a critical violation")
    ax.set_title("Weekly critical rate and the segments a change-point search finds")
    _save(fig, path)
