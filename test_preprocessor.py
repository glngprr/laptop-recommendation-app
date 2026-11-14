"""Unit tests for core/preprocessor.py verifying locked specification v2.1.0."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from core.preprocessor import (
    CatalogPreprocessor,
    PreprocessingParameters,
    RANKING_FEATURES,
    ELIGIBILITY_MIN_RAM_GB,
    ELIGIBILITY_MIN_STORAGE_GB,
    EXPECTED_TOTAL_CATALOG_ROWS,
    EXPECTED_VALID_CATALOG_ROWS,
    EXPECTED_EXCLUDED_ROWS,
)


class TestCatalogPreprocessor(unittest.TestCase):
    """Test suite for CatalogPreprocessor and PreprocessingParameters."""

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = CatalogPreprocessor()

    def test_raw_and_reference_catalog_counts(self):
        """1 & 2: Verify total catalog is 648 and valid reference catalog is exactly 636."""
        raw = self.preprocessor.raw_catalog
        ref = self.preprocessor.reference_catalog

        self.assertEqual(len(raw), EXPECTED_TOTAL_CATALOG_ROWS)
        self.assertEqual(len(ref), EXPECTED_VALID_CATALOG_ROWS)

        # 12 records without valid CPU PassMark excluded
        missing_cpu_count = raw["cpu_passmark_mark"].isnull().sum()
        self.assertEqual(missing_cpu_count, EXPECTED_EXCLUDED_ROWS)
        self.assertEqual(len(raw) - len(ref), EXPECTED_EXCLUDED_ROWS)
        self.assertTrue(ref["cpu_passmark_mark"].notnull().all())

    def test_original_index_preserved(self):
        """7: Verify original row indices remain available and uncorrupted."""
        raw = self.preprocessor.raw_catalog
        ref = self.preprocessor.reference_catalog

        self.assertIn("original_index", raw.columns)
        self.assertIn("original_index", ref.columns)
        self.assertEqual(raw["original_index"].iloc[0], 0)
        self.assertEqual(raw["original_index"].iloc[-1], len(raw) - 1)

        # Reference indices must match raw row indices
        first_ref_idx = ref["original_index"].iloc[0]
        self.assertEqual(ref["name"].iloc[0], raw["name"].iloc[first_ref_idx])

    def test_standard_scaler_fitted_on_all_reference_records(self):
        """3: Verify StandardScaler is fitted once on all 636 reference records."""
        params = self.preprocessor.parameters
        ref = self.preprocessor.reference_catalog

        scaler = params.scaler
        self.assertEqual(scaler.n_samples_seen_, EXPECTED_VALID_CATALOG_ROWS)
        self.assertEqual(len(scaler.mean_), len(RANKING_FEATURES))

        # Check means match reference catalog means
        expected_means = ref[RANKING_FEATURES].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)

        # Check standard deviation (scale_)
        expected_stds = ref[RANKING_FEATURES].std(ddof=0).to_numpy()
        np.testing.assert_allclose(scaler.scale_, expected_stds, rtol=1e-5)

    def test_min_max_bounds_calculated_from_all_reference_records(self):
        """4: Verify Min-Max bounds are calculated from all 636 reference records."""
        params = self.preprocessor.parameters
        ref = self.preprocessor.reference_catalog

        for feat in RANKING_FEATURES:
            self.assertEqual(params.min_bounds[feat], float(ref[feat].min()))
            self.assertEqual(params.max_bounds[feat], float(ref[feat].max()))

        # Specific known values
        self.assertEqual(params.min_bounds["ram_gb"], 4.0)
        self.assertEqual(params.max_bounds["ram_gb"], 64.0)
        self.assertEqual(params.min_bounds["storage_gb"], 256.0)
        self.assertEqual(params.max_bounds["storage_gb"], 4096.0)
        self.assertEqual(params.min_bounds["cpu_passmark_mark"], 1763.0)
        self.assertEqual(params.max_bounds["cpu_passmark_mark"], 62164.0)

    def test_candidate_counts_across_budget_tiers(self):
        """5: Verify candidate counts match locked specification across budget tiers."""
        expected_counts = {
            12_000_000: 56,
            18_000_000: 192,
            25_000_000: 329,
            40_000_000: 428,
        }

        for budget, expected_n in expected_counts.items():
            pool = self.preprocessor.get_candidate_pool(budget)
            self.assertEqual(
                len(pool),
                expected_n,
                f"Candidate count for budget {budget} was {len(pool)}, expected {expected_n}",
            )
            # Verify all candidates satisfy eligibility
            self.assertTrue((pool["price_numeric"] <= budget).all())
            self.assertTrue((pool["ram_gb"] >= ELIGIBILITY_MIN_RAM_GB).all())
            self.assertTrue((pool["storage_gb"] >= ELIGIBILITY_MIN_STORAGE_GB).all())
            self.assertIn("original_index", pool.columns)
            self.assertIn("price_numeric", pool.columns)

    def test_candidate_filtering_independent_of_targets(self):
        """6: Verify candidate pool does not depend on target preferences."""
        pool_base = self.preprocessor.get_candidate_pool(18_000_000)

        # Filtering must only use budget, min_ram, min_storage
        # Laptops with RAM 16 GB must be present even if someone might prefer 32 GB
        ram_16_count = (pool_base["ram_gb"] == 16).sum()
        self.assertGreater(ram_16_count, 0)
        self.assertEqual(len(pool_base), 192)

    def test_target_validation(self):
        """8: Verify target validation accepts valid targets and rejects out-of-range targets."""
        # Valid targets
        valid = self.preprocessor.validate_targets(
            target_ram=16,
            target_storage=512,
            target_cpu=18871.0,
        )
        self.assertEqual(valid["ram_gb"], 16.0)
        self.assertEqual(valid["storage_gb"], 512.0)
        self.assertEqual(valid["cpu_passmark_mark"], 18871.0)

        # Out of range RAM
        with self.assertRaises(ValueError) as ctx:
            self.preprocessor.validate_targets(target_ram=128, target_storage=512, target_cpu=18871)
        self.assertIn("Target RAM (128.0 GB) di luar rentang", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.preprocessor.validate_targets(target_ram=2, target_storage=512, target_cpu=18871)
        self.assertIn("Target RAM (2.0 GB) di luar rentang", str(ctx.exception))

        # Out of range Storage
        with self.assertRaises(ValueError) as ctx:
            self.preprocessor.validate_targets(target_ram=16, target_storage=128, target_cpu=18871)
        self.assertIn("Target Storage (128.0 GB) di luar rentang", str(ctx.exception))

        # Out of range CPU
        with self.assertRaises(ValueError) as ctx:
            self.preprocessor.validate_targets(target_ram=16, target_storage=512, target_cpu=999999)
        self.assertIn("Target CPU PassMark (999999) di luar rentang", str(ctx.exception))

    def test_empty_candidate_pool_handling(self):
        """9: Verify empty candidate pool is handled explicitly without crashing."""
        # Unrealistic low budget (e.g. Rp 1,000,000)
        empty_pool = self.preprocessor.get_candidate_pool(1_000_000)
        self.assertIsInstance(empty_pool, pd.DataFrame)
        self.assertEqual(len(empty_pool), 0)
        self.assertIn("original_index", empty_pool.columns)
        self.assertIn("price_numeric", empty_pool.columns)

    def test_transform_functions_knn_and_cosine(self):
        """Verify transform helpers for KNN and Cosine behave predictably."""
        params = self.preprocessor.parameters
        ref = self.preprocessor.reference_catalog

        # KNN transform
        knn_features = params.transform_knn(ref.head(5))
        self.assertEqual(knn_features.shape, (5, 3))

        target_dict = {"ram_gb": 16.0, "storage_gb": 512.0, "cpu_passmark_mark": 18871.0}
        knn_target = params.transform_target_knn(target_dict)
        self.assertEqual(knn_target.shape, (1, 3))

        # Cosine transform (must be in [0, 1])
        cos_features = params.transform_cosine(ref)
        self.assertTrue(np.all(cos_features >= 0.0))
        self.assertTrue(np.all(cos_features <= 1.0))

        cos_target = params.transform_target_cosine(target_dict)
        self.assertTrue(np.all(cos_target >= 0.0))
        self.assertTrue(np.all(cos_target <= 1.0))


if __name__ == "__main__":
    unittest.main()

