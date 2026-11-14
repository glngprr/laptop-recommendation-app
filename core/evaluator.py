"""Offline Evaluation Engine for Laptop Recommendation System.

Implements the locked methodology in docs/EVALUATION_SPEC.md v2.1.0:
- Fixed 72-scenario evaluation grid.
- Independent heuristic ground-truth relevance labeling (Grades 0 to 3).
- Evaluation of KNN, Cosine Similarity, and Weighted Scoring (MAUT).
- Precision@5, Recall@5, NDCG@5, Catalog-Wide Coverage, Eligible-Pool Coverage.
- Latency measurement protocol (10 runs, 2 warm-ups discarded, per-scenario median/IQR,
  aggregate median/IQR across the 72 scenarios).
- Exact 4-stage hierarchical winner selection procedure.
- Export of reproducible evaluation artifacts to evaluation/.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import sklearn

from core.preprocessor import (
    CatalogPreprocessor,
    ELIGIBILITY_MIN_RAM_GB,
    ELIGIBILITY_MIN_STORAGE_GB,
    get_project_root,
)
from core.recommenders import (
    BaseRecommender,
    CosineRecommender,
    KNNRecommender,
    WeightedScoringRecommender,
)

# Locked specification constants (docs/EVALUATION_SPEC.md v2.1.0)
GRID_RAM_LEVELS: Tuple[int, ...] = (16, 24, 32)
GRID_STORAGE_LEVELS: Tuple[int, ...] = (512, 1024)
GRID_CPU_LEVELS: Tuple[float, ...] = (14823.0, 18871.0, 24969.0)
GRID_BUDGET_LEVELS: Tuple[int, ...] = (12_000_000, 18_000_000, 25_000_000, 40_000_000)

TOTAL_SCENARIOS: int = len(GRID_RAM_LEVELS) * len(GRID_STORAGE_LEVELS) * len(GRID_CPU_LEVELS) * len(GRID_BUDGET_LEVELS)  # 72
CPU_P10_THRESHOLD: float = 10510.0
STORAGE_SAFE_MIN_GB: int = 256
EVAL_TOP_K: int = 5

LATENCY_REPETITIONS: int = 10
LATENCY_WARMUP_RUNS: int = 2


@dataclass(frozen=True)
class EvaluationScenario:
    """Fixed controlled evaluation scenario definition."""

    scenario_id: str
    target_ram: int
    target_storage: int
    target_cpu: float
    budget: int

    @property
    def target_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "ram_gb": self.target_ram,
            "storage_gb": self.target_storage,
            "cpu_passmark_mark": self.target_cpu,
        }


def generate_72_scenarios() -> List[EvaluationScenario]:
    """Generate the exact 72 orthogonal evaluation scenarios in deterministic order."""
    scenarios: List[EvaluationScenario] = []
    idx = 1
    for r in GRID_RAM_LEVELS:
        for s in GRID_STORAGE_LEVELS:
            for c in GRID_CPU_LEVELS:
                for b in GRID_BUDGET_LEVELS:
                    scenarios.append(
                        EvaluationScenario(
                            scenario_id=f"Q-{idx:02d}",
                            target_ram=r,
                            target_storage=s,
                            target_cpu=c,
                            budget=b,
                        )
                    )
                    idx += 1
    return scenarios


def compute_ground_truth_relevance(
    candidate_pool: pd.DataFrame,
    target_ram: float,
    target_storage: float,
    target_cpu: float,
) -> Dict[int, int]:
    """Assign independent heuristic relevance grades (0 to 3) to every candidate.

    Follows docs/EVALUATION_SPEC.md Section 5.1 mutually exclusive and exhaustive hierarchy:
    - Grade 3 (Fully Meets Defined Targets):
        ram >= target_ram AND storage >= target_storage AND cpu >= target_cpu
    - Grade 2 (Mostly Meets Defined Targets):
        not G3, but ram >= target_ram AND cpu >= target_cpu AND storage >= 256
    - Grade 1 (Partially Meets Defined Targets):
        not G3/G2, and meets any of:
        1a) ram >= target_ram AND 10510 <= cpu < target_cpu AND storage >= 256
        1b) 16 <= ram < target_ram AND cpu >= target_cpu AND storage >= 256
        1c) 16 <= ram < target_ram AND 10510 <= cpu < target_cpu AND storage >= 256
    - Grade 0 (Does Not Meet Defined Targets):
        all remaining candidates in candidate pool.

    Returns:
        dict mapping original_index -> integer grade (0, 1, 2, or 3).
    """
    if len(candidate_pool) == 0:
        return {}

    grades: Dict[int, int] = {}
    for _, row in candidate_pool.iterrows():
        idx = int(row["original_index"])
        ram = float(row["ram_gb"])
        storage = float(row["storage_gb"])
        cpu = float(row["cpu_passmark_mark"])

        # Check Grade 3
        if ram >= target_ram and storage >= target_storage and cpu >= target_cpu:
            grades[idx] = 3
        # Check Grade 2
        elif ram >= target_ram and cpu >= target_cpu and storage >= STORAGE_SAFE_MIN_GB:
            grades[idx] = 2
        # Check Grade 1
        elif (
            (ram >= target_ram and CPU_P10_THRESHOLD <= cpu < target_cpu and storage >= STORAGE_SAFE_MIN_GB)
            or (ELIGIBILITY_MIN_RAM_GB <= ram < target_ram and cpu >= target_cpu and storage >= STORAGE_SAFE_MIN_GB)
            or (ELIGIBILITY_MIN_RAM_GB <= ram < target_ram and CPU_P10_THRESHOLD <= cpu < target_cpu and storage >= STORAGE_SAFE_MIN_GB)
        ):
            grades[idx] = 1
        # Check Grade 0
        else:
            grades[idx] = 0

    return grades


def calculate_scenario_metrics(
    recommendations: pd.DataFrame,
    ground_truth: Dict[int, int],
    candidate_pool_size: int,
    k: int = EVAL_TOP_K,
) -> Dict[str, Any]:
    """Calculate Precision@K, Recall@K, and NDCG@K for a single scenario.

    Precision@K:
        Numerator: count of recommended items in top-K with Grade >= 2.
        Denominator: fixed k (5), even if fewer recommendations exist.

    Recall@K:
        Numerator: count of recommended items in top-K with Grade >= 2.
        Denominator: total items in candidate pool with Grade >= 2.
        If total relevant items == 0: Recall is np.nan (undefined).

    NDCG@K:
        Discounted Cumulative Gain with graded gain formula: 2^rel - 1.
        Discount: log2(rank + 1).
        IDCG@K: maximum possible DCG@K from the candidate pool.
        If IDCG@K == 0: NDCG is np.nan (undefined).
    """
    # Extract top-K recommendations
    top_recs = recommendations.head(k)
    rec_indices = top_recs["original_index"].tolist() if "original_index" in top_recs.columns else []

    # Get relevance grades for recommendations
    rec_grades = [ground_truth.get(idx, 0) for idx in rec_indices]

    # Pad with grade 0 if fewer than k recommendations
    while len(rec_grades) < k:
        rec_grades.append(0)

    # Relevant count in top-K (Grade >= 2)
    rel_in_top_k = sum(1 for g in rec_grades[:k] if g >= 2)

    # Precision@K
    precision_k = float(rel_in_top_k) / float(k)

    # Total relevant in candidate pool (Grade >= 2)
    all_grades = list(ground_truth.values())
    total_relevant_in_pool = sum(1 for g in all_grades if g >= 2)

    # Recall@K
    if total_relevant_in_pool > 0:
        recall_k = float(rel_in_top_k) / float(total_relevant_in_pool)
    else:
        recall_k = np.nan

    # DCG@K
    dcg_k = sum((2 ** g - 1) / np.log2(rank + 2) for rank, g in enumerate(rec_grades[:k]))

    # IDCG@K: sort all grades in pool descending
    sorted_pool_grades = sorted(all_grades, reverse=True)[:k]
    idcg_k = sum((2 ** g - 1) / np.log2(rank + 2) for rank, g in enumerate(sorted_pool_grades))

    if idcg_k > 0:
        ndcg_k = float(dcg_k) / float(idcg_k)
    else:
        ndcg_k = np.nan

    return {
        "precision_5": precision_k,
        "recall_5": recall_k,
        "ndcg_5": ndcg_k,
        "total_relevant_pool": total_relevant_in_pool,
        "rec_indices": rec_indices,
    }


def measure_scenario_latency(
    recommender: BaseRecommender,
    candidate_pool: pd.DataFrame,
    targets: Dict[str, Union[int, float]],
    top_n: int = EVAL_TOP_K,
    repetitions: int = LATENCY_REPETITIONS,
    warmup: int = LATENCY_WARMUP_RUNS,
) -> Tuple[float, float]:
    """Execute recommendation timing protocol.

    - Exactly 10 repetitions per scenario.
    - 2 warm-up runs discarded.
    - Measures recommendation execution only (excludes precomputation, CSV export, GT).
    - Returns (median_ms, iqr_ms) from the 8 measured runs.
    """
    times_ms: List[float] = []

    for i in range(repetitions):
        start = time.perf_counter()
        _ = recommender.recommend(candidate_pool, targets, top_n=top_n)
        duration_ms = (time.perf_counter() - start) * 1000.0

        if i >= warmup:
            times_ms.append(duration_ms)

    times_arr = np.array(times_ms, dtype=float)
    median_val = float(np.median(times_arr))
    q75, q25 = np.percentile(times_arr, [75, 25])
    iqr_val = float(q75 - q25)

    return median_val, iqr_val


def select_winner_hierarchical(summary_df: pd.DataFrame) -> Dict[str, Any]:
    """Execute exact 4-stage hierarchical winner selection procedure from Section 8.

    Stage 1: Find methods whose Mean NDCG@5 is within 0.01 of the maximum Mean NDCG@5.
    Stage 2: If multiple methods remain, retain those whose Mean Precision@5 is within
             0.01 of the maximum Mean Precision@5 among the remaining methods.
    Stage 3: If multiple methods remain, choose method with highest Catalog-Wide Coverage.
    Stage 4: If still tied, choose method with lowest aggregate median latency.

    Recall@5 must NOT be used as a tie-breaker.
    """
    history: List[Dict[str, Any]] = []

    # Map summary dataframe rows to dict
    methods_data = summary_df.set_index("method").to_dict(orient="index")
    current_candidates = list(methods_data.keys())

    # --- Stage 1: Mean NDCG@5 ---
    ndcg_scores = {m: methods_data[m]["ndcg_5_mean"] for m in current_candidates}
    max_ndcg = max(ndcg_scores.values())
    stage1_candidates = [m for m in current_candidates if ndcg_scores[m] >= max_ndcg - 0.01]

    history.append({
        "stage": 1,
        "criterion": "Mean NDCG@5 within 0.01 of max",
        "scores": ndcg_scores,
        "max_score": max_ndcg,
        "retained_candidates": stage1_candidates,
    })

    if len(stage1_candidates) == 1:
        winner = stage1_candidates[0]
        return {
            "winner": winner,
            "decision_sequence": history,
            "final_stage_reached": 1,
            "rationale": f"Metode '{winner}' memenangkan seleksi mutlak pada Tahap 1 dengan Mean NDCG@5 tertinggi ({max_ndcg:.4f}).",
        }

    # --- Stage 2: Mean Precision@5 ---
    current_candidates = stage1_candidates
    prec_scores = {m: methods_data[m]["precision_5_mean"] for m in current_candidates}
    max_prec = max(prec_scores.values())
    stage2_candidates = [m for m in current_candidates if prec_scores[m] >= max_prec - 0.01]

    history.append({
        "stage": 2,
        "criterion": "Mean Precision@5 within 0.01 of max among Stage 1 finalists",
        "scores": prec_scores,
        "max_score": max_prec,
        "retained_candidates": stage2_candidates,
    })

    if len(stage2_candidates) == 1:
        winner = stage2_candidates[0]
        return {
            "winner": winner,
            "decision_sequence": history,
            "final_stage_reached": 2,
            "rationale": f"Metode '{winner}' memenangkan seleksi pada Tahap 2 dengan Mean Precision@5 tertinggi ({max_prec:.4f}) di antara finalis Tahap 1.",
        }

    # --- Stage 3: Catalog-Wide Coverage ---
    current_candidates = stage2_candidates
    cov_scores = {m: methods_data[m]["catalog_coverage"] for m in current_candidates}
    max_cov = max(cov_scores.values())
    stage3_candidates = [m for m in current_candidates if cov_scores[m] == max_cov]

    history.append({
        "stage": 3,
        "criterion": "Highest Catalog-Wide Coverage among Stage 2 finalists",
        "scores": cov_scores,
        "max_score": max_cov,
        "retained_candidates": stage3_candidates,
    })

    if len(stage3_candidates) == 1:
        winner = stage3_candidates[0]
        return {
            "winner": winner,
            "decision_sequence": history,
            "final_stage_reached": 3,
            "rationale": f"Metode '{winner}' memenangkan seleksi pada Tahap 3 dengan Catalog-Wide Coverage tertinggi ({max_cov:.4f}).",
        }

    # --- Stage 4: Aggregate Median Latency ---
    current_candidates = stage3_candidates
    lat_scores = {m: methods_data[m]["latency_aggregate_median_ms"] for m in current_candidates}
    min_lat = min(lat_scores.values())
    stage4_candidates = [m for m in current_candidates if lat_scores[m] == min_lat]

    history.append({
        "stage": 4,
        "criterion": "Lowest aggregate median latency among Stage 3 finalists",
        "scores": lat_scores,
        "min_score": min_lat,
        "retained_candidates": stage4_candidates,
    })

    winner = stage4_candidates[0]
    return {
        "winner": winner,
        "decision_sequence": history,
        "final_stage_reached": 4,
        "rationale": f"Metode '{winner}' memenangkan seleksi pada Tahap 4 dengan latensi agregat terendah ({min_lat:.3f} ms).",
    }


class OfflineEvaluator:
    """Comprehensive offline evaluation orchestrator."""

    def __init__(self, preprocessor: Optional[CatalogPreprocessor] = None) -> None:
        self.preprocessor = preprocessor or CatalogPreprocessor()
        self.scenarios = generate_72_scenarios()
        self.recommenders: Dict[str, BaseRecommender] = {
            "knn_euclidean": KNNRecommender(self.preprocessor),
            "cosine_similarity": CosineRecommender(self.preprocessor),
            "weighted_scoring": WeightedScoringRecommender(self.preprocessor),
        }

    def run_benchmark(self, measure_latency: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Execute the full 72-scenario benchmark across all 3 methods."""
        detailed_records: List[Dict[str, Any]] = []
        method_rec_sets: Dict[str, Set[int]] = {m: set() for m in self.recommenders}
        method_latencies: Dict[str, List[float]] = {m: [] for m in self.recommenders}

        catalog_valid_count = len(self.preprocessor.reference_catalog)
        all_eligible_indices: Set[int] = set()

        # Cache candidate pools and ground truths per budget
        pools_cache: Dict[int, pd.DataFrame] = {}
        for b in GRID_BUDGET_LEVELS:
            pool = self.preprocessor.get_candidate_pool(b)
            pools_cache[b] = pool
            all_eligible_indices.update(pool["original_index"].tolist())

        total_eligible_across_all = len(all_eligible_indices)

        for scenario in self.scenarios:
            pool = pools_cache[scenario.budget]
            targets = scenario.target_dict

            # 1. Independent Ground Truth
            ground_truth = compute_ground_truth_relevance(
                pool,
                target_ram=scenario.target_ram,
                target_storage=scenario.target_storage,
                target_cpu=scenario.target_cpu,
            )

            # 2. Evaluate all 3 methods on the exact same pool and targets
            for m_key, recommender in self.recommenders.items():
                # Perform recommendation
                recs = recommender.recommend(pool, targets, top_n=EVAL_TOP_K)

                # Calculate metrics
                metrics = calculate_scenario_metrics(
                    recs,
                    ground_truth,
                    candidate_pool_size=len(pool),
                    k=EVAL_TOP_K,
                )

                # Track recommended laptop indices for coverage
                rec_idx_list = metrics["rec_indices"]
                method_rec_sets[m_key].update(rec_idx_list)

                # Measure latency according to protocol
                if measure_latency:
                    med_lat, iqr_lat = measure_scenario_latency(
                        recommender,
                        pool,
                        targets,
                        top_n=EVAL_TOP_K,
                    )
                else:
                    med_lat, iqr_lat = 0.0, 0.0

                method_latencies[m_key].append(med_lat)

                detailed_records.append({
                    "scenario_id": scenario.scenario_id,
                    "target_ram": scenario.target_ram,
                    "target_storage": scenario.target_storage,
                    "target_cpu": scenario.target_cpu,
                    "budget": scenario.budget,
                    "method": m_key,
                    "ndcg_5": metrics["ndcg_5"],
                    "precision_5": metrics["precision_5"],
                    "recall_5": metrics["recall_5"],
                    "eligible_pool_size": len(pool),
                    "relevant_pool_count": metrics["total_relevant_pool"],
                    "recommended_original_indices": json.dumps(rec_idx_list),
                    "latency_scenario_median_ms": med_lat,
                    "latency_scenario_iqr_ms": iqr_lat,
                })

        detailed_df = pd.DataFrame(detailed_records)

        # 3. Compute Summary Metrics
        summary_records: List[Dict[str, Any]] = []
        for m_key in self.recommenders:
            sub = detailed_df[detailed_df["method"] == m_key]

            # NDCG@5 stats (exclude NaN if IDCG=0)
            valid_ndcg = sub["ndcg_5"].dropna()
            ndcg_mean = float(valid_ndcg.mean())
            ndcg_std = float(valid_ndcg.std())
            ndcg_med = float(valid_ndcg.median())
            ndcg_valid_n = int(len(valid_ndcg))

            # Precision@5 stats
            valid_prec = sub["precision_5"].dropna()
            prec_mean = float(valid_prec.mean())
            prec_valid_n = int(len(valid_prec))

            # Recall@5 stats (exclude NaN when relevant pool count == 0)
            valid_rec = sub["recall_5"].dropna()
            rec_mean = float(valid_rec.mean())
            rec_valid_n = int(len(valid_rec))
            rec_excluded_n = int(sub["recall_5"].isnull().sum())

            # Coverage
            unique_recs = len(method_rec_sets[m_key])
            catalog_cov = float(unique_recs) / float(catalog_valid_count)
            pool_cov = float(unique_recs) / float(total_eligible_across_all)

            # Aggregate Latency: median and IQR of the 72 scenario medians
            scenario_medians = np.array(method_latencies[m_key], dtype=float)
            agg_med_lat = float(np.median(scenario_medians))
            q75, q25 = np.percentile(scenario_medians, [75, 25])
            agg_iqr_lat = float(q75 - q25)

            summary_records.append({
                "method": m_key,
                "ndcg_5_mean": ndcg_mean,
                "ndcg_5_std": ndcg_std,
                "ndcg_5_median": ndcg_med,
                "ndcg_valid_queries": ndcg_valid_n,
                "precision_5_mean": prec_mean,
                "precision_valid_queries": prec_valid_n,
                "recall_5_mean": rec_mean,
                "recall_valid_queries": rec_valid_n,
                "recall_excluded_queries": rec_excluded_n,
                "catalog_coverage": catalog_cov,
                "eligible_pool_coverage": pool_cov,
                "latency_aggregate_median_ms": agg_med_lat,
                "latency_aggregate_iqr_ms": agg_iqr_lat,
            })

        summary_df = pd.DataFrame(summary_records)

        # 4. Winner Selection
        winner_result = select_winner_hierarchical(summary_df)

        return detailed_df, summary_df, winner_result

    def export_artifacts(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        measure_latency: bool = True,
    ) -> Dict[str, Path]:
        """Run full evaluation and export all required reproducible artifacts."""
        root = get_project_root()
        out_path = Path(output_dir).resolve() if output_dir else (root / "evaluation")
        out_path.mkdir(parents=True, exist_ok=True)

        detailed_df, summary_df, winner_result = self.run_benchmark(measure_latency=measure_latency)

        benchmark_csv_path = out_path / "benchmark_results.csv"
        summary_csv_path = out_path / "summary_metrics.csv"
        winner_json_path = out_path / "winner_decision.json"
        metadata_json_path = out_path / "run_metadata.json"

        # A. benchmark_results.csv
        detailed_df.to_csv(benchmark_csv_path, index=False)

        # B. summary_metrics.csv
        summary_df.to_csv(summary_csv_path, index=False)

        # C. winner_decision.json
        with open(winner_json_path, "w", encoding="utf-8") as f:
            json.dump(winner_result, f, indent=2, ensure_ascii=False)

        # D. run_metadata.json
        metadata: Dict[str, Any] = {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dataset_filename": "laptops_benchmark_enriched_resolved.csv",
            "dataset_row_count": len(self.preprocessor.raw_catalog),
            "valid_reference_catalog_count": len(self.preprocessor.reference_catalog),
            "number_of_scenarios": len(self.scenarios),
            "number_of_methods": len(self.recommenders),
            "number_of_latency_repetitions": LATENCY_REPETITIONS,
            "number_of_warmup_runs_discarded": LATENCY_WARMUP_RUNS,
            "specification_version": "2.1.0",
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return {
            "benchmark_results": benchmark_csv_path,
            "summary_metrics": summary_csv_path,
            "winner_decision": winner_json_path,
            "run_metadata": metadata_json_path,
        }


if __name__ == "__main__":
    evaluator = OfflineEvaluator()
    print("Menjalankan benchmark offline 72 skenario...")
    artifacts = evaluator.export_artifacts()
    print("Artefak berhasil diekspor:")
    for k, p in artifacts.items():
        print(f"  - {k}: {p}")

