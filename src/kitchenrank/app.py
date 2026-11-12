"""A small Gradio app: describe a scheduled visit, get a risk band and the reasons behind it."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from .config import OUT_DIR
from .score import Bundle, score_visit

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> Bundle:
    target = Path(path) if path else OUT_DIR / "bundle.joblib"
    if not target.exists():
        raise FileNotFoundError(f"{target} not found. Run `kitchenrank` first to train and write the bundle.")
    return joblib.load(target)


def describe(
    risk_level: int,
    is_restaurant: bool,
    earlier_visits: int,
    days_since_last: int,
    last_critical: int,
    last_serious: int,
    last_minor: int,
    last_result: str,
    visits_with_critical: int,
    neighbourhood_rate_pct: float,
    month: str,
    temperature_f: float,
) -> tuple[str, str]:
    """Pure function behind the form, kept separate from Gradio so it can be tested directly."""
    bundle = load_bundle()
    rec = {
        "risk": risk_level,
        "is_restaurant": int(is_restaurant),
        "n_prior": earlier_visits,
        "days_since_prev": days_since_last,
        "prev_critical": last_critical,
        "prev_serious": last_serious,
        "prev_minor": last_minor,
        "prev_result": last_result,
        "critical_visits": visits_with_critical,
        "nb_smoothed_rate": neighbourhood_rate_pct / 100.0,
        "month": MONTHS.index(month) + 1,
        "temperatureMax": temperature_f,
    }
    out = score_visit(bundle, rec)
    headline = (
        f"Chance of a critical violation: {out['probability']:.0%} "
        f"({out['relative_to_typical']:.1f}x the typical scheduled visit). Band: {out['band']}."
    )
    reasons = "\n".join(f"- {r}" for r in out["reasons"]) or "- nothing here moves the estimate by more than 0.2 points"
    return headline, reasons


def build_app():
    import gradio as gr

    with gr.Blocks(title="kitchenrank") as demo:
        gr.Markdown(
            "# kitchenrank\nEnter what is known about a scheduled inspection. "
            "The estimate comes from a model trained on Chicago canvass inspections from 2011 to mid-2013."
        )
        with gr.Row():
            with gr.Column():
                risk = gr.Radio([1, 2, 3], value=1, label="City risk category (1 is highest)")
                rest = gr.Checkbox(value=True, label="Restaurant")
                prior = gr.Slider(0, 9, value=1, step=1, label="Earlier inspections on record")
                days = gr.Slider(1, 1200, value=365, step=1, label="Days since the last inspection")
                lc = gr.Slider(0, 7, value=0, step=1, label="Critical violations at the last visit")
                ls = gr.Slider(0, 5, value=0, step=1, label="Serious violations at the last visit")
                lm = gr.Slider(0, 15, value=2, step=1, label="Minor violations at the last visit")
                res = gr.Radio(["Pass", "Pass w/ Conditions", "Fail"], value="Pass", label="Result of the last visit")
                cv = gr.Slider(0, 4, value=0, step=1, label="Earlier visits that had a critical violation")
                nb = gr.Slider(0, 40, value=13, step=1, label="Recent critical rate nearby (%)")
                mo = gr.Dropdown(MONTHS, value="Jul", label="Month of the visit")
                tf = gr.Slider(10, 100, value=75, step=1, label="High temperature that day (F)")
                go = gr.Button("Score this visit", variant="primary")
            with gr.Column():
                head = gr.Textbox(label="Estimate", lines=2)
                why = gr.Markdown()
        go.click(describe, [risk, rest, prior, days, lc, ls, lm, res, cv, nb, mo, tf], [head, why])
    return demo


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
