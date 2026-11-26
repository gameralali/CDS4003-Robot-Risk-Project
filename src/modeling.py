"""Leakage-safe model training, tuning, threshold selection, and evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES, FALSE_NEGATIVE_COST_AED, FALSE_POSITIVE_COST_AED,
    METADATA_PATH, MODEL_PATH, NUMERIC_FEATURES, PERMUTATION_SAMPLE_SIZE,
    RANDOM_STATE, TABLE_DIR, TUNING_SAMPLE_SIZE,
)


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_pipeline(model_name: str) -> Pipeline:
    if model_name == "dummy_prior":
        model = DummyClassifier(strategy="prior")
        scale = False
    elif model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1200, solver="liblinear", random_state=RANDOM_STATE)
        scale = True
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
        scale = False
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", make_preprocessor(scale)), ("model", model)])


def classification_metrics(y_true, probability, threshold: float) -> dict:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    cost = int(fp * FALSE_POSITIVE_COST_AED + fn * FALSE_NEGATIVE_COST_AED)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_cost_aed": int(fp * FALSE_POSITIVE_COST_AED),
        "false_negative_cost_aed": int(fn * FALSE_NEGATIVE_COST_AED),
        "total_error_cost_aed": cost,
    }


def optimise_threshold(y_true, probability) -> tuple[dict, pd.DataFrame]:
    rows = [classification_metrics(y_true, probability, t) for t in np.arange(0.05, 0.951, 0.01)]
    curve = pd.DataFrame(rows)
    # Business cost first; recall and precision break exact cost ties.
    best = curve.sort_values(
        ["total_error_cost_aed", "recall", "precision", "threshold"],
        ascending=[True, False, False, True],
    ).iloc[0]
    return best.to_dict(), curve


def _tuning_sample(X, y):
    if len(X) <= TUNING_SAMPLE_SIZE:
        return X, y
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=TUNING_SAMPLE_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_sample, y_sample


def tune_and_validate(X_train, y_train, X_val, y_val) -> tuple[dict, pd.DataFrame, dict]:
    specifications = {
        "dummy_prior": None,
        "logistic_regression": {
            "model__C": loguniform(0.03, 20.0),
            "model__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "model__n_estimators": [100, 180],
            "model__max_depth": [12, 24],
            "model__min_samples_leaf": [1, 5],
            "model__max_features": ["sqrt"],
            "model__class_weight": [None, "balanced"],
        },
    }
    n_iterations = {"logistic_regression": 5, "random_forest": 4}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    X_sample, y_sample = _tuning_sample(X_train, y_train)
    fitted = {}
    validation_rows = []
    tuning_histories = {}

    for name, distributions in specifications.items():
        start = time.time()
        pipeline = make_pipeline(name)
        if distributions is None:
            best = pipeline
            history = pd.DataFrame({"note": ["Prior-probability baseline; no hyperparameters tuned."]})
        else:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=distributions,
                n_iter=n_iterations[name],
                scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                refit="roc_auc",
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(X_sample, y_sample)
            best = clone(search.best_estimator_)
            history = pd.DataFrame(search.cv_results_).sort_values("rank_test_roc_auc")
            tuning_histories[name] = history
            history.to_csv(TABLE_DIR / f"tuning_history_{name}.csv", index=False)

        # Refit on all training data after tuning on a stratified training-only sample.
        if name == "random_forest":
            best.set_params(model__n_jobs=-1)
        best.fit(X_train, y_train)
        probability = best.predict_proba(X_val)[:, 1]
        default_metrics = classification_metrics(y_val, probability, 0.5)
        best_threshold, curve = optimise_threshold(y_val, probability)
        fitted[name] = {
            "pipeline": best,
            "validation_probability": probability,
            "best_threshold": best_threshold,
            "threshold_curve": curve,
            "best_params": best.get_params(),
        }
        validation_rows.append({
            "model": name,
            "tuning_seconds": round(time.time() - start, 2),
            "default_threshold_accuracy": default_metrics["accuracy"],
            "default_threshold_recall": default_metrics["recall"],
            "default_threshold_f1": default_metrics["f1"],
            **{f"validation_{k}": v for k, v in best_threshold.items()},
        })

    comparison = pd.DataFrame(validation_rows).sort_values("validation_total_error_cost_aed")
    return fitted, comparison, tuning_histories


def train_final_model(X, y) -> dict:
    """70/15/15 train/validation/test split; final test remains untouched until selection."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    fitted, comparison, _ = tune_and_validate(X_train, y_train, X_val, y_val)
    selected_name = str(comparison.iloc[0]["model"])
    selected = fitted[selected_name]
    threshold = float(selected["best_threshold"]["threshold"])
    balanced_row = selected["threshold_curve"].loc[selected["threshold_curve"]["f1"].idxmax()]
    balanced_threshold = float(balanced_row["threshold"])

    # Retrain selected configuration on train + validation; thresholds stay validation-derived.
    final_pipeline = clone(selected["pipeline"])
    if selected_name == "random_forest":
        final_pipeline.set_params(model__n_jobs=-1)
    final_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_probability = final_pipeline.predict_proba(X_test)[:, 1]
    test_default = classification_metrics(y_test, test_probability, 0.5)
    test_optimised = classification_metrics(y_test, test_probability, threshold)
    test_balanced = classification_metrics(y_test, test_probability, balanced_threshold)

    permutation_sample = min(PERMUTATION_SAMPLE_SIZE, len(X_test))
    X_perm, _, y_perm, _ = train_test_split(
        X_test, y_test, train_size=permutation_sample, stratify=y_test,
        random_state=RANDOM_STATE,
    )
    importance = permutation_importance(
        final_pipeline, X_perm, y_perm, scoring="average_precision", n_repeats=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    # Pipeline-level permutation importance is measured per original input column.
    feature_names = X_test.columns
    importance_table = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(TABLE_DIR / "model_comparison_validation.csv", index=False)
    selected["threshold_curve"].to_csv(TABLE_DIR / "selected_model_validation_threshold_curve.csv", index=False)
    pd.DataFrame([
        {"evaluation": "test_default_0.5", "model": selected_name, **test_default},
        {"evaluation": "test_balanced_f1_threshold", "model": selected_name, **test_balanced},
        {"evaluation": "test_validation_selected_threshold", "model": selected_name, **test_optimised},
    ]).to_csv(TABLE_DIR / "final_test_metrics.csv", index=False)
    importance_table.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)
    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "failure_probability": test_probability,
        "predicted_failure": (test_probability >= threshold).astype(int),
    }).to_csv(TABLE_DIR / "final_test_predictions.csv", index=False)

    metadata = {
        "selected_model": selected_name,
        "decision_threshold": threshold,
        "balanced_f1_threshold": balanced_threshold,
        "false_negative_cost_aed": FALSE_NEGATIVE_COST_AED,
        "false_positive_cost_aed": FALSE_POSITIVE_COST_AED,
        "split": {
            "train_n": int(len(X_train)), "validation_n": int(len(X_val)),
            "test_n": int(len(X_test)), "train_plus_validation_n": int(len(X_train) + len(X_val)),
        },
        "class_balance": {"overall_failure_rate": float(y.mean())},
        "validation_metrics_at_selected_threshold": selected["best_threshold"],
        "validation_metrics_at_balanced_threshold": balanced_row.to_dict(),
        "test_metrics_default_threshold": test_default,
        "test_metrics_balanced_threshold": test_balanced,
        "test_metrics_selected_threshold": test_optimised,
        "best_parameters": {
            key: value for key, value in final_pipeline.named_steps["model"].get_params().items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "excluded_columns": ["task_id", "task_success", "failure_probability_simulated",
                             "completion_time_sec", "operator_intervention"],
        "features": list(X.columns),
        "business_cost_note": "Illustrative scenario assumptions, not observed financial data.",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": final_pipeline,
        "metadata": metadata,
        "comparison": comparison,
        "importance": importance_table,
        "X_test": X_test,
        "y_test": y_test,
        "test_probability": test_probability,
        "threshold_curve": selected["threshold_curve"],
    }
