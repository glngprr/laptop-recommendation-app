"""Unit tests for the integrated zero-database service layer and application (Stage 3.3)."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import service as S
from core.preprocessor import (
    CatalogPreprocessor,
    ELIGIBILITY_MIN_RAM_GB,
    ELIGIBILITY_MIN_STORAGE_GB,
)
from core.recommenders import WeightedScoringRecommender


class TestZeroDatabaseServiceLayer(unittest.TestCase):
    """Test suite verifying standalone zero-database operation."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.ref = S.prepare_catalog()

    def test_prepare_catalog_loads_from_csv(self):
        """Verify catalog loads directly from reference CSV without database."""
        self.assertEqual(len(self.df), 636)
        self.assertIn("preprocessor", self.ref)
        self.assertIn("recommender", self.ref)
        self.assertIsInstance(self.ref["preprocessor"], CatalogPreprocessor)
        self.assertIsInstance(self.ref["recommender"], WeightedScoringRecommender)

    def test_run_query_valid_output_zero_db(self):
        """Verify recommendations work purely in-memory using Weighted Scoring."""
        params = {
            "budget": 15_000_000,
            "target_ram": 16,
            "min_ram": 16,
            "target_storage": 512,
            "min_storage": 256,
            "cpu_level": "sedang",
        }
        result = S.run_query(self.df, self.ref, params)

        self.assertIn("recs", result)
        self.assertIn("summary", result)
        recs = result["recs"]
        self.assertEqual(len(recs), 5)

        # Check candidate eligibility
        for r in recs:
            self.assertLessEqual(r["price"], 15_000_000)
            self.assertGreaterEqual(float(r["ram"].replace(".", "")), 16)
            self.assertGreaterEqual(float(r["storage"].replace(".", "")), 256)
            self.assertIn("score", r)
            self.assertGreater(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)
            self.assertIn("explain", r)
            self.assertTrue(len(r["explain"]["points"]) >= 4)

    def test_empty_candidate_pool(self):
        """Verify graceful handling when no laptops meet budget/eligibility."""
        params = {
            "budget": 3_000_000,  # Unrealistic budget for RAM >= 16GB
            "target_ram": 16,
            "min_ram": 16,
            "target_storage": 512,
            "min_storage": 256,
            "cpu_level": "sedang",
        }
        result = S.run_query(self.df, self.ref, params)
        self.assertEqual(len(result["recs"]), 0)
        self.assertTrue(len(result["empty_reason"]) > 0)
        self.assertEqual(result["summary"]["n_after_basic_filter"], 0)

    def test_invalid_target_validation(self):
        """Verify that invalid targets outside catalog bounds raise ValueError."""
        params = {
            "budget": 20_000_000,
            "target_ram": 128,  # Out of catalog bounds [4, 64]
            "min_ram": 16,
            "target_storage": 512,
            "min_storage": 256,
            "cpu_level": "sedang",
        }
        with self.assertRaises(ValueError):
            S.run_query(self.df, self.ref, params)

    def test_no_database_modules_in_active_app(self):
        """Verify that db module is not imported by service or streamlit_app."""
        import sys
        self.assertNotIn("db", sys.modules)
        
        # Check requirements.txt does not require pymysql
        req_path = Path("requirements.txt")
        self.assertTrue(req_path.exists())
        req_text = req_path.read_text(encoding="utf-8").lower()
        self.assertNotIn("pymysql", req_text)
        self.assertNotIn("sqlite", req_text)

    def test_evaluation_artifacts_available(self):
        """Verify static evaluation artifacts are loaded correctly."""
        from streamlit_app import load_evaluation_artifacts
        summary_df, winner_data, metadata, benchmark_df = load_evaluation_artifacts()
        self.assertIsNotNone(summary_df)
        self.assertEqual(len(summary_df), 3)
        self.assertEqual(winner_data["winner"], "weighted_scoring")
        self.assertEqual(len(benchmark_df), 216)


if __name__ == "__main__":
    unittest.main()
