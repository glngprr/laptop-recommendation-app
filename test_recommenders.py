"""Unit and integration tests for core/recommenders.py according to docs/EVALUATION_SPEC.md v2.1.0."""

import unittest
import numpy as np
import pandas as pd

from core.preprocessor import CatalogPreprocessor
from core.recommenders import (
    KNNRecommender,
    CosineRecommender,
    WeightedScoringRecommender,
    _assign_tie_groups,
    _apply_tie_breaking,
    SCORE_TIE_TOLERANCE,
    WEIGHT_RAM,
    WEIGHT_CPU,
    WEIGHT_STORAGE,
)


class TestRecommenders(unittest.TestCase):
    """Test suite verifying all three recommendation algorithms and tie-breaking."""

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = CatalogPreprocessor()
        cls.knn = KNNRecommender(cls.preprocessor)
        cls.cosine = CosineRecommender(cls.preprocessor)
        cls.weighted = WeightedScoringRecommender(cls.preprocessor)

        # Realistic Android Dev Target
        cls.target_q = {
            "ram_gb": 16,
            "storage_gb": 512,
            "cpu_passmark_mark": 18871.0,
        }

    def test_recommenders_process_same_candidate_pool(self):
        """1: Verify all three recommenders process the same candidate pool."""
        pool = self.preprocessor.get_candidate_pool(18_000_000)
        self.assertEqual(len(pool), 192)

        res_knn = self.knn.recommend(pool, self.target_q, top_n=5)
        res_cos = self.cosine.recommend(pool, self.target_q, top_n=5)
        res_wt = self.weighted.recommend(pool, self.target_q, top_n=5)

        for res in [res_knn, res_cos, res_wt]:
            self.assertEqual(len(res), 5)
            self.assertIn("rank_position", res.columns)
            self.assertIn("score", res.columns)
            self.assertIn("name", res.columns)
            self.assertIn("price_numeric", res.columns)
            self.assertIn("original_index", res.columns)
            self.assertEqual(list(res["rank_position"]), [1, 2, 3, 4, 5])

    def test_knn_ranking_is_ascending_by_distance(self):
        """2: Verify KNN ranking is sorted ascending by Euclidean distance."""
        pool = self.preprocessor.get_candidate_pool(25_000_000)
        res = self.knn.recommend(pool, self.target_q, top_n=20)

        scores = res["score"].to_numpy()
        # Verify strictly non-decreasing
        self.assertTrue(np.all(np.diff(scores) >= -1e-9), f"KNN scores not ascending: {scores}")

    def test_cosine_ranking_is_descending_by_similarity(self):
        """3: Verify Cosine ranking is sorted descending by similarity."""
        pool = self.preprocessor.get_candidate_pool(25_000_000)
        res = self.cosine.recommend(pool, self.target_q, top_n=20)

        scores = res["score"].to_numpy()
        # Verify strictly non-increasing
        self.assertTrue(np.all(np.diff(scores) <= 1e-9), f"Cosine scores not descending: {scores}")
        # Cosine similarity on non-negative vectors must be between 0.0 and 1.0
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0 + 1e-9))

    def test_cosine_zero_norm_target_returns_zero_similarity(self):
        """4: Verify Cosine zero-norm target returns 0.0 similarity for every candidate."""
        pool = self.preprocessor.get_candidate_pool(12_000_000)
        bounds = self.preprocessor.parameters.min_bounds

        # Target exactly matching minimum bounds produces zero-norm in Min-Max space
        zero_norm_target = {
            "ram_gb": bounds["ram_gb"],
            "storage_gb": bounds["storage_gb"],
            "cpu_passmark_mark": bounds["cpu_passmark_mark"],
        }

        res = self.cosine.recommend(pool, zero_norm_target, top_n=10)
        self.assertEqual(len(res), 10)
        # All scores must be exactly 0.0
        self.assertTrue(np.all(res["score"] == 0.0))
        # Ranking should be resolved deterministically by price then original_index
        prices = res["price_numeric"].to_numpy()
        self.assertTrue(np.all(np.diff(prices) >= 0), "Ties not resolved by price ascending")

    def test_weighted_utility_piecewise_formula(self):
        """5, 6, 7: Verify piecewise utility u(x, target) for shortfall, exact, and surplus."""
        target_val = 16.0
        catalog_max = 64.0

        # Below target (shortfall): x = 8 -> u = 0.8 * (8 / 16) = 0.4
        u_below = WeightedScoringRecommender._calculate_feature_utility(
            np.array([8.0]), target_val=target_val, catalog_max=catalog_max
        )
        self.assertAlmostEqual(u_below[0], 0.4, places=6)

        # Exact target: x = 16 -> u = 0.8 + 0.2 * 0 = 0.8
        u_exact = WeightedScoringRecommender._calculate_feature_utility(
            np.array([16.0]), target_val=target_val, catalog_max=catalog_max
        )
        self.assertAlmostEqual(u_exact[0], 0.8, places=6)

        # Above target (surplus): x = 40
        # g = (40 - 16) / (64 - 16) = 24 / 48 = 0.5
        # u = 0.8 + 0.2 * 0.5 = 0.9
        u_above = WeightedScoringRecommender._calculate_feature_utility(
            np.array([40.0]), target_val=target_val, catalog_max=catalog_max
        )
        self.assertAlmostEqual(u_above[0], 0.9, places=6)

        # Max out (x = catalog_max = 64): g = 1.0 -> u = 1.0
        u_max = WeightedScoringRecommender._calculate_feature_utility(
            np.array([64.0]), target_val=target_val, catalog_max=catalog_max
        )
        self.assertAlmostEqual(u_max[0], 1.0, places=6)

        # Beyond catalog max (x = 80): g capped at 1.0 -> u = 1.0
        u_beyond = WeightedScoringRecommender._calculate_feature_utility(
            np.array([80.0]), target_val=target_val, catalog_max=catalog_max
        )
        self.assertAlmostEqual(u_beyond[0], 1.0, places=6)

        # Target equals catalog max: target_val = 64, catalog_max = 64
        # g = 0 -> u = 0.8
        u_target_max = WeightedScoringRecommender._calculate_feature_utility(
            np.array([64.0]), target_val=64.0, catalog_max=64.0
        )
        self.assertAlmostEqual(u_target_max[0], 0.8, places=6)

    def test_weighted_scoring_weights(self):
        """8: Verify Weighted Scoring uses exact weights 0.45 RAM, 0.35 CPU, 0.20 Storage."""
        self.assertEqual(WEIGHT_RAM, 0.45)
        self.assertEqual(WEIGHT_CPU, 0.35)
        self.assertEqual(WEIGHT_STORAGE, 0.20)
        self.assertAlmostEqual(WEIGHT_RAM + WEIGHT_CPU + WEIGHT_STORAGE, 1.0)

        # Synthetic fixture with single candidate exactly matching targets
        fixture_df = pd.DataFrame([{
            "name": "Laptop Exact",
            "price": "Rp 15.000.000",
            "price_numeric": 15000000,
            "ram_gb": 16,
            "storage_gb": 512,
            "cpu_passmark_mark": 18871.0,
            "original_index": 0,
        }])

        res = self.weighted.recommend(fixture_df, self.target_q, top_n=1)
        # When all features match target, each u_j = 0.8
        # total score = 0.45 * 0.8 + 0.35 * 0.8 + 0.20 * 0.8 = 0.80
        self.assertAlmostEqual(res["score"].iloc[0], 0.80, places=6)

    def test_shared_preprocessing_parameters_used(self):
        """9: Verify all methods share the same CatalogPreprocessor instance parameters."""
        self.assertIs(self.knn.preprocessor, self.preprocessor)
        self.assertIs(self.cosine.preprocessor, self.preprocessor)
        self.assertIs(self.weighted.preprocessor, self.preprocessor)

    def test_deterministic_rankings_across_repeated_executions(self):
        """10: Verify repeated executions produce 100% deterministic rankings."""
        pool = self.preprocessor.get_candidate_pool(18_000_000)

        for rec in [self.knn, self.cosine, self.weighted]:
            run1 = rec.recommend(pool, self.target_q, top_n=10)
            run2 = rec.recommend(pool, self.target_q, top_n=10)
            pd.testing.assert_frame_equal(run1, run2)

    def test_empty_candidate_pool_handling(self):
        """11: Verify empty candidate pool is handled cleanly without exceptions."""
        empty_pool = self.preprocessor.get_candidate_pool(1_000_000)
        self.assertEqual(len(empty_pool), 0)

        for rec in [self.knn, self.cosine, self.weighted]:
            res = rec.recommend(empty_pool, self.target_q, top_n=5)
            self.assertEqual(len(res), 0)
            self.assertIn("rank_position", res.columns)
            self.assertIn("score", res.columns)

    def test_invalid_targets_rejected(self):
        """12: Verify out-of-range targets are explicitly rejected across all methods."""
        pool = self.preprocessor.get_candidate_pool(18_000_000)

        invalid_targets = {
            "ram_gb": 128,  # Max is 64
            "storage_gb": 512,
            "cpu_passmark_mark": 18871.0,
        }

        for rec in [self.knn, self.cosine, self.weighted]:
            with self.assertRaises(ValueError) as ctx:
                rec.recommend(pool, invalid_targets)
            self.assertIn("di luar rentang", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Strict Boundary Tests for Tie Tolerance 1e-7 (Exact, Smaller, Greater)
    # -------------------------------------------------------------------------
    def test_tie_tolerance_exact_boundary(self):
        """Verify scores differing by EXACTLY 1e-7 are tied in both directions."""
        base = 0.5000000
        exact_tie = 0.5000001  # |0.5000001 - 0.5000000| == 1e-7
        scores = np.array([base, exact_tie])

        # Ascending direction
        groups_asc = _assign_tie_groups(scores, ascending=True, tol=SCORE_TIE_TOLERANCE)
        self.assertEqual(groups_asc[0], groups_asc[1], "Scores with diff == 1e-7 failed to tie (ascending)")

        # Descending direction
        groups_desc = _assign_tie_groups(scores, ascending=False, tol=SCORE_TIE_TOLERANCE)
        self.assertEqual(groups_desc[0], groups_desc[1], "Scores with diff == 1e-7 failed to tie (descending)")

    def test_tie_tolerance_slightly_smaller(self):
        """Verify scores differing by slightly less than 1e-7 (e.g. 0.99e-7) are tied."""
        base = 0.50000000
        smaller_diff = base + 0.99e-7  # diff = 0.99e-7 < 1e-7
        scores = np.array([base, smaller_diff])

        # Ascending direction
        groups_asc = _assign_tie_groups(scores, ascending=True, tol=SCORE_TIE_TOLERANCE)
        self.assertEqual(groups_asc[0], groups_asc[1], "Scores with diff < 1e-7 failed to tie (ascending)")

        # Descending direction
        groups_desc = _assign_tie_groups(scores, ascending=False, tol=SCORE_TIE_TOLERANCE)
        self.assertEqual(groups_desc[0], groups_desc[1], "Scores with diff < 1e-7 failed to tie (descending)")

    def test_tie_tolerance_slightly_greater(self):
        """Verify scores differing by slightly more than 1e-7 (e.g. 1.01e-7) are NOT tied."""
        base = 0.50000000
        greater_diff = base + 1.01e-7  # diff = 1.01e-7 > 1e-7
        scores = np.array([base, greater_diff])

        # Ascending direction: base (smaller) should be in group 0, greater_diff in group 1
        groups_asc = _assign_tie_groups(scores, ascending=True, tol=SCORE_TIE_TOLERANCE)
        self.assertNotEqual(groups_asc[0], groups_asc[1], "Scores with diff > 1e-7 were incorrectly tied (ascending)")
        self.assertEqual(groups_asc[0], 0)
        self.assertEqual(groups_asc[1], 1)

        # Descending direction: greater_diff (larger) should be in group 0, base in group 1
        groups_desc = _assign_tie_groups(scores, ascending=False, tol=SCORE_TIE_TOLERANCE)
        self.assertNotEqual(groups_desc[0], groups_desc[1], "Scores with diff > 1e-7 were incorrectly tied (descending)")
        self.assertEqual(groups_desc[1], 0)
        self.assertEqual(groups_desc[0], 1)

    def test_tie_breaking_opposite_rounding_boundaries(self):
        """Verify values on opposite rounding boundaries with diff <= 1e-7 are tied."""
        fixture = pd.DataFrame([
            {"name": "Laptop_B", "score": 0.10000006, "price_numeric": 14000000, "original_index": 2},
            {"name": "Laptop_A", "score": 0.10000004, "price_numeric": 18000000, "original_index": 1},
        ])

        # Ascending direction (e.g. KNN)
        res_asc = _apply_tie_breaking(fixture, score_col="score", ascending=True)
        # Because scores tie (|0.10000006 - 0.10000004| = 2e-8 <= 1e-7), lower price Laptop_B (14M) wins rank 1
        self.assertEqual(res_asc["name"].iloc[0], "Laptop_B")
        self.assertEqual(res_asc["name"].iloc[1], "Laptop_A")

        # Descending direction (e.g. Cosine / Weighted)
        res_desc = _apply_tie_breaking(fixture, score_col="score", ascending=False)
        self.assertEqual(res_desc["name"].iloc[0], "Laptop_B")
        self.assertEqual(res_desc["name"].iloc[1], "Laptop_A")

    def test_tie_breaking_hierarchical_price_and_original_index(self):
        """Verify hierarchical tie-breaking (Price first, then original_index)."""
        fixture = pd.DataFrame([
            {"name": "L_P16_Idx20", "score": 0.80000001, "price_numeric": 16000000, "original_index": 20},
            {"name": "L_P14_Idx10", "score": 0.80000003, "price_numeric": 14000000, "original_index": 10},
            {"name": "L_P14_Idx05", "score": 0.80000002, "price_numeric": 14000000, "original_index": 5},
        ])

        # All scores within 2e-8 of each other -> all tied on score
        # Tie-breaker 1: price_numeric (14M before 16M)
        # Tie-breaker 2: original_index (Idx05 before Idx10)
        expected_order = ["L_P14_Idx05", "L_P14_Idx10", "L_P16_Idx20"]

        res_asc = _apply_tie_breaking(fixture, score_col="score", ascending=True)
        self.assertEqual(list(res_asc["name"]), expected_order)

        res_desc = _apply_tie_breaking(fixture, score_col="score", ascending=False)
        self.assertEqual(list(res_desc["name"]), expected_order)


if __name__ == "__main__":
    unittest.main()
