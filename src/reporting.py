"""Reproducible tables and figures for EDA and evaluation."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.config import FIGURE_DIR, TABLE_DIR, TARGET
from src.data_processing import add_engineered_features, group_failure_summary

PALETTE = ["#22d3ee", "#fbbf24", "#f87171", "#34d399"]


def _save(fig, name: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_eda(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    labeled = clean.loc[clean[TARGET].isin([0, 1])].copy()
    labeled[TARGET] = labeled[TARGET].astype(int)

    missing = (raw.isna().mean() * 100).sort_values(ascending=False).head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    missing.plot.barh(ax=ax, color=PALETTE[0])
    ax.set(title="Highest missing-value rates before cleaning", xlabel="Missing values (%)", ylabel="")
    _save(fig, "01_missingness_before_cleaning.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    counts = labeled[TARGET].value_counts().sort_index()
    ax.bar(["No failure (0)", "Failure (1)"], counts.values, color=[PALETTE[3], PALETTE[2]])
    ax.set(title="Target distribution after target cleaning", ylabel="Tasks")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}\n({value/len(labeled):.1%})", ha="center", va="bottom")
    ax.set_ylim(0, max(counts.values) * 1.16)
    _save(fig, "02_target_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, column in zip(axes, ["task_type", "shift", "warehouse_zone"]):
        table = group_failure_summary(clean, column)
        table.to_csv(TABLE_DIR / f"failure_rate_by_{column}.csv", index=False)
        ax.barh(table[column].astype(str), table["failure_rate_pct"], color=PALETTE[0])
        ax.set(title=f"Failure rate by {column.replace('_', ' ')}", xlabel="Failure rate (%)", ylabel="")
        ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "03_failure_rates_by_category.png")

    plot_columns = ["battery_level_pct", "vision_confidence", "maintenance_hours_since_service",
                    "motor_temperature_c", "travel_distance_m", "object_weight_kg"]
    sample = labeled.sample(min(25_000, len(labeled)), random_state=4003)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, column in zip(axes.flat, plot_columns):
        for value, label, color in [(0, "No failure", PALETTE[3]), (1, "Failure", PALETTE[2])]:
            sns.histplot(sample.loc[sample[TARGET] == value, column].dropna(), bins=35,
                         stat="density", element="step", fill=False, ax=ax, label=label, color=color)
        ax.set(title=column.replace("_", " "), xlabel="", ylabel="Density")
    axes.flat[0].legend()
    fig.tight_layout()
    _save(fig, "04_numeric_distributions_by_outcome.png")

    battery = labeled.dropna(subset=["battery_level_pct"]).copy()
    battery["battery_band"] = pd.cut(battery["battery_level_pct"], [0, 20, 40, 60, 80, 100],
                                     include_lowest=True)
    maintenance = labeled.dropna(subset=["maintenance_hours_since_service"]).copy()
    maintenance["maintenance_band"] = pd.qcut(maintenance["maintenance_hours_since_service"], 5, duplicates="drop")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, frame, band, title in [
        (axes[0], battery, "battery_band", "Failure rate by battery band"),
        (axes[1], maintenance, "maintenance_band", "Failure rate by maintenance quintile"),
    ]:
        table = frame.groupby(band, observed=True)[TARGET].agg(["count", "mean"]).reset_index()
        table.to_csv(TABLE_DIR / f"{title.lower().replace(' ', '_')}.csv", index=False)
        ax.bar(range(len(table)), table["mean"] * 100, color=PALETTE[1])
        ax.set_xticks(range(len(table)), table[band].astype(str), rotation=25, ha="right")
        ax.set(title=title, ylabel="Failure rate (%)", xlabel="")
    fig.tight_layout()
    _save(fig, "05_failure_rates_battery_and_maintenance.png")

    engineered = add_engineered_features(labeled)
    corr_columns = [TARGET, "battery_level_pct", "vision_confidence", "motor_temperature_c",
                    "joint_load_pct", "maintenance_hours_since_service", "travel_distance_m",
                    "object_weight_kg", "load_distance_index", "thermal_load_index",
                    "obstacles_per_10m", "battery_deficit"]
    corr = engineered[corr_columns].corr(numeric_only=True)
    corr.to_csv(TABLE_DIR / "selected_feature_correlations.csv")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, annot=False)
    ax.set_title("Selected feature correlations (association, not causation)")
    _save(fig, "06_correlation_heatmap.png")

    descriptive = labeled.describe(include="all").T
    descriptive.to_csv(TABLE_DIR / "cleaned_descriptive_statistics.csv")
    return {
        "labeled_rows": int(len(labeled)),
        "failure_rate": float(labeled[TARGET].mean()),
        "figures": 6,
    }


def generate_evaluation_figures(results: dict) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    y_test = results["y_test"]
    probability = results["test_probability"]
    threshold = results["metadata"]["decision_threshold"]
    predicted = (probability >= threshold).astype(int)
    model_name = results["metadata"]["selected_model"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    RocCurveDisplay.from_predictions(y_test, probability, ax=axes[0], color=PALETTE[0])
    axes[0].set_title(f"ROC curve — {model_name}")
    PrecisionRecallDisplay.from_predictions(y_test, probability, ax=axes[1], color=PALETTE[1])
    axes[1].set_title(f"Precision–recall curve — {model_name}")
    fig.tight_layout()
    _save(fig, "07_test_roc_and_pr_curves.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=["No failure", "Failure"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Untouched test set — threshold {threshold:.2f}")
    _save(fig, "08_test_confusion_matrix.png")

    curve = results["threshold_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(curve["threshold"], curve["total_error_cost_aed"], color=PALETTE[2], lw=2)
    axes[0].axvline(threshold, color=PALETTE[0], ls="--", label=f"Selected {threshold:.2f}")
    axes[0].set(title="Validation error cost by threshold", xlabel="Threshold", ylabel="Illustrative cost (AED)")
    axes[0].legend()
    for metric, color in [("recall", PALETTE[3]), ("precision", PALETTE[1]), ("f1", PALETTE[0])]:
        axes[1].plot(curve["threshold"], curve[metric], label=metric, color=color)
    axes[1].axvline(threshold, color="#545454", ls="--")
    axes[1].set(title="Validation metric trade-offs", xlabel="Threshold", ylabel="Score", ylim=(0, 1.02))
    axes[1].legend()
    fig.tight_layout()
    _save(fig, "09_threshold_cost_and_metrics.png")

    top = results["importance"].head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color=PALETTE[0])
    ax.set(title="Permutation importance on a test-set sample", xlabel="Decrease in average precision", ylabel="")
    _save(fig, "10_permutation_importance.png")
