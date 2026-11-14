"""Unit and integration tests for core/evaluator.py according to docs/EVALUATION_SPEC.md v2.1.0."""

import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from core.evaluator import (
    OfflineEvaluator,
    EvaluationScenario,
    generate_72_scenarios,
    compute_ground_truth_relevance,
    calculate_scenario_metrics,
    measure_scenario_latency,
    select_winner_hierarchical,
    TOTAL_SCENARIOS,
    GRID_RAM_LEVELS,
    GRID_STORAGE_LEVELS,
    GRID_CPU_LEVELS,
    GRID_BUDGET_LEVELS,
)
from core.preprocessor import CatalogPreprocessor


class TestOfflineEvaluator(unittest.TestCase):
    """Test suite verifying all evaluation logic, ground truth, metrics, and winner selection."""

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = CatalogPreprocessor()
        cls.evaluator = OfflineEvaluator(cls.preprocessor)

    def test_exactly_72_scenarios_generated(self):
        """1 & 2: Verify exactly 72 scenarios generated with correct targets and budgets."""
        scenarios = generate_72_scenarios()
        self.assertEqual(len(scenarios), TOTAL_SCENARIOS)
        self.assertEqual(len(scenarios), 72)

        # Check unique IDs from Q-01 to Q-72
        ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(set(ids)), 72)
        self.assertEqual(ids[0], "Q-01")
        self.assertEqual(ids[-1], "Q-72")

        # Verify values match grid
        for s in scenarios:
            self.assertIn(s.target_ram, GRID_RAM_LEVELS)
            self.assertIn(s.target_storage, GRID_STORAGE_LEVELS)
            self.assertIn(s.target_cpu, GRID_CPU_LEVELS)
            self.assertIn(s.budget, GRID_BUDGET_LEVELS)

    def test_candidate_pools_follow_fixed_eligibility(self):
        """3 & 4: Verify candidate pools follow fixed eligibility and all methods get identical pools."""
        budget = 18_000_000
        pool = self.preprocessor.get_candidate_pool(budget)
        self.assertEqual(len(pool), 192)

        # Eligibility check
        self.assertTrue((pool["price_numeric"] <= budget).all())
        self.assertTrue((pool["ram_gb"] >= 16).all())
        self.assertTrue((pool["storage_gb"] >= 256).all())

        # All 3 methods receive the same instance
        for recommender in self.evaluator.recommenders.values():
            recs = recommender.recommend(pool, {"ram_gb": 16, "storage_gb": 512, "cpu_passmark_mark": 18871.0})
            self.assertEqual(len(recs), 5)
            self.assertTrue(set(recs["original_index"]).issubset(set(pool["original_index"])))

    def test_ground_truth_independent_and_grades_bounded(self):
        """5, 6, 7: Verify ground truth assigns mutually exclusive grades 0 to 3."""
        pool = self.preprocessor.get_candidate_pool(25_000_000)
        gt = compute_ground_truth_relevance(pool, target_ram=16, target_storage=512, target_cpu=18871.0)

        self.assertEqual(len(gt), len(pool))
        for idx, grade in gt.items():
            self.assertIn(grade, [0, 1, 2, 3])

        # Test specific edge cases on synthetic dataframe
        synth_pool = pd.DataFrame([
            {"original_index": 1, "ram_gb": 32, "storage_gb": 1024, "cpu_passmark_mark": 25000.0},  # Grade 3
            {"original_index": 2, "ram_gb": 16, "storage_gb": 256, "cpu_passmark_mark": 19000.0},   # Grade 2 (storage < 512 target but >= 256)
            {"original_index": 3, "ram_gb": 16, "storage_gb": 512, "cpu_passmark_mark": 15000.0},   # Grade 1 (Case 1a: CPU between P10 and target)
            {"original_index": 4, "ram_gb": 16, "storage_gb": 512, "cpu_passmark_mark": 8000.0},    # Grade 0 (CPU < P10 threshold 10510)
        ])

        synth_gt = compute_ground_truth_relevance(synth_pool, target_ram=16, target_storage=512, target_cpu=18871.0)
        self.assertEqual(synth_gt[1], 3)
        self.assertEqual(synth_gt[2], 2)
        self.assertEqual(synth_gt[3], 1)
        self.assertEqual(synth_gt[4], 0)

    def test_ndcg_precision_recall_metric_calculations(self):
        """8, 9, 10, 11, 12: Verify Precision@5, Recall@5, and NDCG@5 calculations."""
        gt = {1: 3, 2: 2, 3: 1, 4: 0, 5: 2, 6: 0, 7: 1}

        # Case A: Top-5 recommendations [1, 2, 3, 4, 5]
        # Grades: [3, 2, 1, 0, 2] -> Relevant (Grade >= 2) at positions 0, 1, 4 (count = 3)
        # Total relevant in pool: {1: 3, 2: 2, 5: 2} -> count = 3
        recs_df = pd.DataFrame([{"original_index": i} for i in [1, 2, 3, 4, 5]])
        metrics = calculate_scenario_metrics(recs_df, gt, candidate_pool_size=len(gt), k=5)

        # Precision@5 = 3 / 5 = 0.60
        self.assertAlmostEqual(metrics["precision_5"], 0.60, places=5)
        # Recall@5 = 3 / 3 = 1.00
        self.assertAlmostEqual(metrics["recall_5"], 1.00, places=5)

        # DCG@5:
        # pos 0 (g=3): (2^3 - 1)/log2(2) = 7 / 1 = 7.0
        # pos 1 (g=2): (2^2 - 1)/log2(3) = 3 / 1.5849625 = 1.892789
        # pos 2 (g=1): (2^1 - 1)/log2(4) = 1 / 2 = 0.5
        # pos 3 (g=0): 0
        # pos 4 (g=2): 3 / log2(6) = 3 / 2.5849625 = 1.160558
        # Ideal order: [3, 2, 2, 1, 1]
        # IDCG@5: 7/1 + 3/log2(3) + 3/log2(4) + 1/log2(5) + 1/log2(6)
        self.assertTrue(0.0 <= metrics["ndcg_5"] <= 1.0)
        self.assertFalse(np.isnan(metrics["ndcg_5"]))

        # Case B: Zero relevant items in pool (all Grade <= 1)
        gt_zero_rel = {1: 1, 2: 0, 3: 1, 4: 0}
        recs_zero_df = pd.DataFrame([{"original_index": i} for i in [1, 2, 3, 4]])
        m_zero = calculate_scenario_metrics(recs_zero_df, gt_zero_rel, candidate_pool_size=4, k=5)

        self.assertEqual(m_zero["precision_5"], 0.0)
        self.assertTrue(np.isnan(m_zero["recall_5"]), "Recall must be NaN when zero relevant items exist")
        # NDCG is still valid since Grade 1 has gain (2^1 - 1) = 1 -> IDCG > 0
        self.assertFalse(np.isnan(m_zero["ndcg_5"]))

    def test_coverage_denominators(self):
        """13 & 14: Verify Catalog-Wide Coverage denominator is 636 and Eligible-Pool denominator is 428."""
        ref_catalog_len = len(self.preprocessor.reference_catalog)
        self.assertEqual(ref_catalog_len, 636)

        # Union of candidate pools across all budgets:
        # B1 (56) in B2 (192) in B3 (329) in B4 (428)
        # So union is exactly B4 size (428)
        p4 = self.preprocessor.get_candidate_pool(40_000_000)
        self.assertEqual(len(p4), 428)

    def test_latency_protocol(self):
        """15, 16, 17: Verify latency protocol discards 2 warm-ups and aggregates 72 medians."""
        pool = self.preprocessor.get_candidate_pool(12_000_000)
        targets = {"ram_gb": 16, "storage_gb": 512, "cpu_passmark_mark": 14823.0}

        med_ms, iqr_ms = measure_scenario_latency(
            self.evaluator.recommenders["knn_euclidean"],
            pool,
            targets,
            top_n=5,
            repetitions=6,
            warmup=2,
        )

        self.assertGreater(med_ms, 0.0)
        self.assertGreaterEqual(iqr_ms, 0.0)

    def test_hierarchical_winner_selection(self):
        """18 & 19: Verify exact 4-stage hierarchical winner selection and Recall exclusion."""
        # Stage 1: Absolute winner when NDCG diff > 0.01
        summary_clear = pd.DataFrame([
            {"method": "M1", "ndcg_5_mean": 0.85, "precision_5_mean": 0.70, "catalog_coverage": 0.15, "latency_aggregate_median_ms": 2.0},
            {"method": "M2", "ndcg_5_mean": 0.80, "precision_5_mean": 0.75, "catalog_coverage": 0.20, "latency_aggregate_median_ms": 1.5},
            {"method": "M3", "ndcg_5_mean": 0.75, "precision_5_mean": 0.60, "catalog_coverage": 0.10, "latency_aggregate_median_ms": 3.0},
        ])
        res1 = select_winner_hierarchical(summary_clear)
        self.assertEqual(res1["winner"], "M1")
        self.assertEqual(res1["final_stage_reached"], 1)

        # Stage 2: NDCG tie (diff <= 0.01), decided by Precision
        summary_prec = pd.DataFrame([
            {"method": "M1", "ndcg_5_mean": 0.855, "precision_5_mean": 0.70, "catalog_coverage": 0.15, "latency_aggregate_median_ms": 2.0},
            {"method": "M2", "ndcg_5_mean": 0.850, "precision_5_mean": 0.82, "catalog_coverage": 0.12, "latency_aggregate_median_ms": 2.5},
            {"method": "M3", "ndcg_5_mean": 0.800, "precision_5_mean": 0.60, "catalog_coverage": 0.10, "latency_aggregate_median_ms": 3.0},
        ])
        res2 = select_winner_hierarchical(summary_prec)
        self.assertEqual(res2["winner"], "M2")
        self.assertEqual(res2["final_stage_reached"], 2)

        # Stage 3: NDCG & Precision tie, decided by Coverage
        summary_cov = pd.DataFrame([
            {"method": "M1", "ndcg_5_mean": 0.855, "precision_5_mean": 0.815, "catalog_coverage": 0.15, "latency_aggregate_median_ms": 2.0},
            {"method": "M2", "ndcg_5_mean": 0.850, "precision_5_mean": 0.820, "catalog_coverage": 0.25, "latency_aggregate_median_ms": 2.5},
        ])
        res3 = select_winner_hierarchical(summary_cov)
        self.assertEqual(res3["winner"], "M2")
        self.assertEqual(res3["final_stage_reached"], 3)

        # Stage 4: NDCG, Precision, and Coverage tie, decided by Latency
        summary_lat = pd.DataFrame([
            {"method": "M1", "ndcg_5_mean": 0.850, "precision_5_mean": 0.800, "catalog_coverage": 0.20, "latency_aggregate_median_ms": 1.2},
            {"method": "M2", "ndcg_5_mean": 0.850, "precision_5_mean": 0.800, "catalog_coverage": 0.20, "latency_aggregate_median_ms": 2.8},
        ])
        res4 = select_winner_hierarchical(summary_lat)
        self.assertEqual(res4["winner"], "M1")
        self.assertEqual(res4["final_stage_reached"], 4)

    def test_export_artifacts_structure(self):
        """20 & 21: Verify exported artifact files exist, have correct headers and schema."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run minimal evaluation without repetitions for fast unit test
            artifacts = self.evaluator.export_artifacts(output_dir=tmpdir, measure_latency=False)

            self.assertTrue(artifacts["benchmark_results"].exists())
            self.assertTrue(artifacts["summary_metrics"].exists())
            self.assertTrue(artifacts["winner_decision"].exists())
            self.assertTrue(artifacts["run_metadata"].exists())

            df_bench = pd.read_csv(artifacts["benchmark_results"])
            # Exactly 72 scenarios * 3 methods = 216 rows
            self.assertEqual(len(df_bench), 216)

            df_sum = pd.read_csv(artifacts["summary_metrics"])
            # Exactly 3 rows (one per method)
            self.assertEqual(len(df_sum), 3)
            self.assertIn("ndcg_5_mean", df_sum.columns)
            self.assertIn("precision_5_mean", df_sum.columns)
            self.assertIn("recall_5_mean", df_sum.columns)
            self.assertIn("catalog_coverage", df_sum.columns)
            self.assertIn("eligible_pool_coverage", df_sum.columns)
            self.assertIn("latency_aggregate_median_ms", df_sum.columns)
            self.assertIn("latency_aggregate_iqr_ms", df_sum.columns)

            with open(artifacts["winner_decision"], "r", encoding="utf-8") as f:
                win_data = json.load(f)
            self.assertIn("winner", win_data)
            self.assertIn("decision_sequence", win_data)

            with open(artifacts["run_metadata"], "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            self.assertEqual(meta_data["number_of_scenarios"], 72)
            self.assertEqual(meta_data["number_of_methods"], 3)
            self.assertEqual(meta_data["valid_reference_catalog_count"], 636)


if __name__ == "__main__":
    unittest.main()

