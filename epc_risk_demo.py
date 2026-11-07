"""Small, synthetic EPC activity-risk demo for interview practice.

Run with Python 3.10+ after installing requirements.txt. The data is fictional;
this is a learning scaffold, not McDermott data or a production model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)
CLASSIFICATION_THRESHOLD = 0.5


def make_demo_data(n=2400, seed=17):
    """Create fictional activity rows with a known risk-generating pattern."""
    rng = np.random.default_rng(seed)
    disciplines = rng.choice(["piping", "electrical", "civil", "instrumentation"], n)
    projects = rng.choice([f"P{i:02d}" for i in range(1, 13)], n)
    material_ready = rng.beta(5, 2, n)
    procurement_late = rng.binomial(1, 0.24, n)
    schedule_variance_days = rng.normal(0, 9, n) + procurement_late * 8
    fabrication_progress = rng.uniform(0, 1, n)
    planned_progress = rng.uniform(0.2, 0.95, n)
    # Hidden rule used only to make a plausible label for this synthetic exercise.
    log_odds = (-2.0 + 2.0 * procurement_late + 2.2 * (1 - material_ready)
                + 0.11 * np.maximum(schedule_variance_days, 0)
                + 1.7 * np.maximum(planned_progress - fabrication_progress, 0))
    probability = 1 / (1 + np.exp(-log_odds))
    risk = rng.binomial(1, probability)
    return pd.DataFrame({
        "activity_id": [f"A{i:05d}" for i in range(n)],
        "project_id": projects,
        "discipline": disciplines,
        "material_ready_pct": material_ready,
        "procurement_late": procurement_late,
        "schedule_variance_days": schedule_variance_days.round(1),
        "fabrication_progress_pct": fabrication_progress,
        "planned_progress_pct": planned_progress,
        "high_risk_next_30d": risk,
    })


def validate_data(df):
    """Fail early when required identifiers, labels, or ranges are invalid."""
    required = {
        "activity_id", "project_id", "discipline", "material_ready_pct",
        "procurement_late", "schedule_variance_days",
        "fabrication_progress_pct", "planned_progress_pct",
        "high_risk_next_30d",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["activity_id"].isna().any() or not df["activity_id"].is_unique:
        raise ValueError("activity_id values must be present and unique")
    if df["project_id"].isna().any():
        raise ValueError("project_id values must be present")

    fraction_columns = ["material_ready_pct", "fabrication_progress_pct",
                        "planned_progress_pct"]
    for column in fraction_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError(f"{column} must contain numeric values from 0 to 1")

    for column in ["procurement_late", "high_risk_next_30d"]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1")
    if not np.isfinite(pd.to_numeric(df["schedule_variance_days"], errors="coerce")).all():
        raise ValueError("schedule_variance_days must contain finite numeric values")


def main():
    df = make_demo_data()
    validate_data(df)
    df.to_csv(OUT / "synthetic_activities.csv", index=False)

    # Inputs available before the prediction window; exclude IDs and the answer label.
    target = "high_risk_next_30d"
    feature_cols = ["discipline", "material_ready_pct", "procurement_late",
                    "schedule_variance_days", "fabrication_progress_pct",
                    "planned_progress_pct"]
    X, y = df[feature_cols], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    numeric = [c for c in feature_cols if c != "discipline"]
    categorical = ["discipline"]
    prep = ColumnTransformer([
        ("numbers", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categories", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5,
                                                 class_weight="balanced", random_state=17),
    }
    scores = []
    fitted = {}
    for name, estimator in models.items():
        model = Pipeline([("prepare", prep), ("model", estimator)])
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        predicted = probability >= CLASSIFICATION_THRESHOLD
        scores.append({
            "model": name,
            "threshold": CLASSIFICATION_THRESHOLD,
            "roc_auc": roc_auc_score(y_test, probability),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
            "f1": f1_score(y_test, predicted, zero_division=0),
        })
        fitted[name] = model
        print(
            f"\n{name}: ROC-AUC={scores[-1]['roc_auc']:.3f}, "
            f"precision={scores[-1]['precision']:.3f}, "
            f"recall={scores[-1]['recall']:.3f}, "
            f"F1={scores[-1]['f1']:.3f} "
            f"(threshold={CLASSIFICATION_THRESHOLD:.2f})"
        )
        print(classification_report(y_test, predicted, zero_division=0))

    pd.DataFrame(scores).to_csv(OUT / "model_comparison.csv", index=False)
    winner_name = max(scores, key=lambda row: row["roc_auc"])["model"]
    winner = fitted[winner_name]

    # Score all rows as a demonstration of a batch scoring step.
    df["risk_score"] = winner.predict_proba(X)[:, 1]
    df["risk_band"] = pd.cut(df["risk_score"], bins=[-0.001, 0.35, 0.65, 1.0],
                             labels=["low", "medium", "high"])
    df.sort_values("risk_score", ascending=False).to_csv(OUT / "activity_risk_scores.csv", index=False)
    summary = (df.groupby("project_id", observed=False)
                 .agg(activities=("activity_id", "count"),
                      average_risk=("risk_score", "mean"),
                      high_risk_activities=("risk_band", lambda s: (s == "high").sum()))
                 .reset_index().sort_values("average_risk", ascending=False))
    summary.to_csv(OUT / "project_risk_summary.csv", index=False)
    print(f"\nSelected model: {winner_name}")
    print(f"Wrote demo outputs to: {OUT.resolve()}")
    print("Reminder: synthetic labels and metrics are for practice only.")


if __name__ == "__main__":
    main()
