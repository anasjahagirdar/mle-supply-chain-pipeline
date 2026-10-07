"""Unit tests for src/model_training/model_training.py.

First function under test: DemandModelTrainer.split_features_target
(separates the input columns X from the target column y).
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

# Make the folder src/model_training importable from the tests folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "model_training"))

from model_training import DemandModelTrainer, FEATURE_COLUMNS, TARGET_COLUMN


class TestSplitFeaturesTarget(unittest.TestCase):
    """Tests for DemandModelTrainer.split_features_target."""

    def setUp(self):
        """Runs before every test: build a trainer and a tiny sample table."""
        # spark is not used by split_features_target, so None is enough here
        self.trainer = DemandModelTrainer(
            spark=None,
            experiment_name="test_experiment",
            model_params={},
            model_name="test_model",
        )
        self.sample_pdf = pd.DataFrame({
            "product_name": ["Product A", "Product B"],
            "destination_city": ["Pune", "Mumbai"],
            "demand_1": [10.0, 20.0],
            "demand_7": [11.0, 21.0],
            "rolling_7_day_avg": [12.0, 22.0],
            "month": [1, 2],
            "day_of_week": [3, 4],
            "extra_column": ["x", "y"],       # not a feature, must be dropped
            "total_demand": [100.0, 200.0],   # the target
        })

    def test_x_has_only_feature_columns(self):
        x, _ = self.trainer.split_features_target(self.sample_pdf)
        self.assertEqual(list(x.columns), FEATURE_COLUMNS)

    def test_x_does_not_contain_target(self):
        x, _ = self.trainer.split_features_target(self.sample_pdf)
        self.assertNotIn(TARGET_COLUMN, x.columns)

    def test_y_is_target_column(self):
        _, y = self.trainer.split_features_target(self.sample_pdf)
        self.assertEqual(y.name, TARGET_COLUMN)
        self.assertEqual(y.tolist(), [100.0, 200.0])

    def test_row_count_is_unchanged(self):
        x, y = self.trainer.split_features_target(self.sample_pdf)
        self.assertEqual(len(x), len(self.sample_pdf))
        self.assertEqual(len(y), len(self.sample_pdf))

    def test_missing_feature_column_raises_key_error(self):
        broken_pdf = self.sample_pdf.drop(columns=["demand_1"])
        with self.assertRaises(KeyError):
            self.trainer.split_features_target(broken_pdf)


if __name__ == "__main__":
    unittest.main()