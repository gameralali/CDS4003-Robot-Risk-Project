"""Smoke and regression tests for deterministic cleaning and deployment scoring."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import EXCLUDED_COLUMNS, MODEL_FEATURES, ROOT
from src.data_processing import add_engineered_features, clean_dataset
from src.scoring import prepare_scoring_frame, score_frame


class ProjectPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data" / "sample_humanoid_robot_tasks.csv")
        cls.model = joblib.load(ROOT / "models" / "best_model.joblib")
        cls.meta = json.loads((ROOT / "models" / "model_metadata.json").read_text())

    def test_training_cleaning_removes_structural_rows(self):
        frame = pd.concat([self.raw.head(1), self.raw.head(1)], ignore_index=True)
        blank = pd.DataFrame([{column: np.nan for column in frame.columns}])
        frame = pd.concat([frame, blank], ignore_index=True)
        cleaned, summary = clean_dataset(frame)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(summary["blank_rows_removed"], 1)
        self.assertGreaterEqual(summary["duplicate_rows_removed"], 1)

    def test_category_binary_and_domain_cleaning(self):
        frame = self.raw.head(1).copy()
        frame.loc[0, "warehouse_zone"] = " Shippping "
        frame["human_nearby"] = frame["human_nearby"].astype(object)
        frame.loc[0, "human_nearby"] = "YES"
        frame.loc[0, "battery_level_pct"] = 999
        cleaned, _ = clean_dataset(frame)
        self.assertEqual(cleaned.loc[0, "warehouse_zone"], "shipping")
        self.assertEqual(cleaned.loc[0, "human_nearby"], 1.0)
        self.assertTrue(np.isnan(cleaned.loc[0, "battery_level_pct"]))

    def test_feature_engineering_formula(self):
        frame = self.raw.head(1).copy()
        featured = add_engineered_features(frame)
        expected = frame.loc[0, "object_weight_kg"] * frame.loc[0, "travel_distance_m"]
        self.assertAlmostEqual(featured.loc[0, "load_distance_index"], expected)
        self.assertAlmostEqual(
            featured.loc[0, "battery_deficit"], 100 - frame.loc[0, "battery_level_pct"]
        )

    def test_scoring_preserves_blank_duplicate_rows_and_order(self):
        frame = pd.concat(
            [self.raw.head(6), self.raw.head(1),
             pd.DataFrame([{column: np.nan for column in self.raw.columns}])],
            ignore_index=True,
        )
        probabilities = score_frame(self.model, frame)
        self.assertEqual(len(probabilities), len(frame))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertAlmostEqual(probabilities[0], probabilities[6])

    def test_scoring_frame_has_exact_model_features(self):
        prepared = prepare_scoring_frame(self.raw.head(3))
        self.assertEqual(list(prepared.columns), MODEL_FEATURES)
        self.assertEqual(len(prepared), 3)

    def test_missing_required_column_is_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "battery_level_pct"):
            prepare_scoring_frame(self.raw.head(1).drop(columns=["battery_level_pct"]))

    def test_empty_input_is_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "no data rows"):
            prepare_scoring_frame(self.raw.head(0))

    def test_artifact_excludes_leakage_and_post_task_fields(self):
        self.assertFalse(any(column in self.meta["features"] for column in EXCLUDED_COLUMNS))
        self.assertEqual(self.meta["features"], MODEL_FEATURES)


if __name__ == "__main__":
    unittest.main()
