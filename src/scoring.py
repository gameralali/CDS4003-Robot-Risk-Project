"""Shared row-preserving inference helpers for single and batch scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES
from src.data_processing import add_engineered_features, clean_dataset


def prepare_scoring_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw inputs without dropping or reordering any records."""
    if frame.empty:
        raise ValueError("The input contains no data rows.")
    required = [*CATEGORICAL_FEATURES, *BASE_NUMERIC_FEATURES]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    cleaned, _ = clean_dataset(frame.copy(), drop_structural_rows=False)
    cleaned = add_engineered_features(cleaned)
    prepared = cleaned.reindex(columns=MODEL_FEATURES)
    assert len(prepared) == len(frame)
    return prepared


def score_frame(model, frame: pd.DataFrame) -> np.ndarray:
    """Return one task-failure probability per input row, in the original order."""
    return model.predict_proba(prepare_scoring_frame(frame))[:, 1]
