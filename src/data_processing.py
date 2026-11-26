"""Deterministic cleaning, auditing, and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES, EXCLUDED_COLUMNS,
    MODEL_FEATURES, TARGET, VALID_RANGES,
)

BINARY_MAP = {
    "0": 0.0, "0.0": 0.0, "no": 0.0, "false": 0.0, "n": 0.0,
    "1": 1.0, "1.0": 1.0, "yes": 1.0, "true": 1.0, "y": 1.0,
}
CATEGORY_REPLACEMENTS = {
    "warehouse_zone": {"shippping": "shipping"},
}


def _normalise_binary(series: pd.Series) -> pd.Series:
    """Map mixed binary labels to 0/1; unknown and other labels become missing."""
    cleaned = series.astype("string").str.strip().str.lower()
    return cleaned.map(BINARY_MAP).astype(float)


def _normalise_category(series: pd.Series, column: str) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.lower()
    cleaned = cleaned.str.replace(r"[\s\-]+", "_", regex=True)
    cleaned = cleaned.replace(CATEGORY_REPLACEMENTS.get(column, {}))
    # scikit-learn expects np.nan rather than pandas.NA in object arrays.
    return pd.Series(cleaned.to_numpy(dtype=object, na_value=np.nan), index=series.index)


def audit_data(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Return a column-level quality audit without modifying the input."""
    return pd.DataFrame({
        "stage": stage,
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_n": df.isna().sum().values,
        "missing_pct": (df.isna().mean().values * 100).round(3),
        "unique_n": df.nunique(dropna=True).values,
    })


def invalid_value_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, (lower, upper) in VALID_RANGES.items():
        if column not in df:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.notna() & ((values < lower) | (values > upper))
        rows.append({
            "column": column,
            "invalid_n": int(invalid.sum()),
            "valid_min": lower,
            "valid_max": upper,
        })
    return pd.DataFrame(rows).sort_values("invalid_n", ascending=False)


def clean_dataset(
    raw: pd.DataFrame, *, drop_structural_rows: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Clean deterministic issues; imputation is deliberately left to model pipelines.

    Training removes blank and duplicate records. Batch scoring disables structural
    row removal so every uploaded record receives exactly one prediction.
    """
    df = raw.copy()
    rows_raw = len(df)
    blank_rows = int(df.isna().all(axis=1).sum())
    if drop_structural_rows:
        df = df.dropna(how="all").copy()

    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["task_id"].duplicated().sum()) if "task_id" in df else 0
    if drop_structural_rows:
        df = df.drop_duplicates().copy()
        if "task_id" in df:
            df = df.drop_duplicates(subset=["task_id"], keep="first").copy()

    for column in CATEGORICAL_FEATURES:
        if column in df:
            df[column] = _normalise_category(df[column], column)
    for column in ["human_nearby", "operator_intervention", "task_failure", "task_success"]:
        if column in df:
            df[column] = _normalise_binary(df[column])

    non_numeric = set(CATEGORICAL_FEATURES)
    for column in df.columns:
        if column not in non_numeric:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_before = invalid_value_counts(df)
    invalid_total = 0
    for column, (lower, upper) in VALID_RANGES.items():
        if column not in df:
            continue
        invalid = df[column].notna() & ((df[column] < lower) | (df[column] > upper))
        invalid_total += int(invalid.sum())
        df.loc[invalid, column] = np.nan

    summary = {
        "rows_raw": int(rows_raw),
        "blank_rows_removed": blank_rows if drop_structural_rows else 0,
        "duplicate_rows_removed": duplicate_rows if drop_structural_rows else 0,
        "duplicate_task_ids_removed": duplicate_ids if drop_structural_rows else 0,
        "domain_invalid_values_set_missing": int(invalid_total),
        "rows_after_structural_cleaning": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells_raw": int(raw.isna().sum().sum()),
        "missing_cells_after_domain_validation": int(df.isna().sum().sum()),
        "invalid_by_column": invalid_before.to_dict(orient="records"),
    }
    return df, summary


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic business features available before task execution."""
    out = df.copy()
    out["load_distance_index"] = out["object_weight_kg"] * out["travel_distance_m"]
    out["thermal_load_index"] = out["motor_temperature_c"] * out["joint_load_pct"] / 100.0
    out["obstacles_per_10m"] = 10.0 * out["obstacles_count"] / (out["travel_distance_m"] + 1.0)
    out["battery_deficit"] = 100.0 - out["battery_level_pct"]
    out["maintenance_failure_interaction"] = (
        out["maintenance_hours_since_service"] * (out["previous_task_failures"] + 1.0)
    )
    out["low_battery_flag"] = np.where(
        out["battery_level_pct"].notna(), (out["battery_level_pct"] < 20.0).astype(float), np.nan
    )
    out["low_vision_flag"] = np.where(
        out["vision_confidence"].notna(), (out["vision_confidence"] < 0.70).astype(float), np.nan
    )
    return out


def prepare_model_data(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop unknown targets, remove leakage/post-task fields, and engineer features."""
    labeled = clean.loc[clean[TARGET].isin([0.0, 1.0])].copy()
    labeled[TARGET] = labeled[TARGET].astype(int)
    labeled = add_engineered_features(labeled)
    X = labeled[MODEL_FEATURES].copy()
    y = labeled[TARGET].copy()
    assert not any(column in X.columns for column in EXCLUDED_COLUMNS)
    assert X.shape[1] == len(MODEL_FEATURES)
    return X, y


def group_failure_summary(clean: pd.DataFrame, column: str) -> pd.DataFrame:
    labeled = clean.loc[clean[TARGET].isin([0.0, 1.0])].copy()
    result = (labeled.groupby(column, dropna=False)[TARGET]
              .agg(records="count", failures="sum", failure_rate="mean")
              .reset_index())
    result["failure_rate_pct"] = (100 * result["failure_rate"]).round(2)
    return result.sort_values("failure_rate", ascending=False)
