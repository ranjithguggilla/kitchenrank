"""Model constructors. Imputing and scaling live inside the pipeline so they are fitted on training rows only."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED

KINDS = ("static logistic", "logistic", "random forest", "boosting", "boosting balanced")


def make_model(kind: str, columns: list[str], seed: int = SEED) -> Pipeline:
    if kind in ("static logistic", "logistic"):
        prep = ColumnTransformer(
            [("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), columns)]
        )
        return Pipeline([("prep", prep), ("clf", LogisticRegression(C=0.3, max_iter=3000))])
    if kind == "random forest":
        prep = ColumnTransformer([("num", SimpleImputer(strategy="median", add_indicator=True), columns)])
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=25, max_features=0.4, n_jobs=-1, random_state=seed
        )
        return Pipeline([("prep", prep), ("clf", clf)])
    if kind in ("boosting", "boosting balanced"):
        prep = ColumnTransformer([("num", "passthrough", columns)])
        clf = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=400,
            max_depth=4,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            class_weight="balanced" if kind.endswith("balanced") else None,
            random_state=seed,
        )
        return Pipeline([("prep", prep), ("clf", clf)])
    raise ValueError(f"unknown model kind: {kind}")


def predict_proba(model: Pipeline, X) -> np.ndarray:  # noqa: F821
    return model.predict_proba(X)[:, 1]
