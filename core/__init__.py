"""Core package for Laptop Recommendation System."""

from core.preprocessor import (
    CatalogPreprocessor,
    PreprocessingParameters,
    RANKING_FEATURES,
    ELIGIBILITY_MIN_RAM_GB,
    ELIGIBILITY_MIN_STORAGE_GB,
    get_project_root,
)
from core.recommenders import (
    BaseRecommender,
    KNNRecommender,
    CosineRecommender,
    WeightedScoringRecommender,
    WEIGHT_RAM,
    WEIGHT_CPU,
    WEIGHT_STORAGE,
)
from core.evaluator import (
    OfflineEvaluator,
    EvaluationScenario,
    generate_72_scenarios,
    compute_ground_truth_relevance,
    calculate_scenario_metrics,
    select_winner_hierarchical,
)

__all__ = [
    "CatalogPreprocessor",
    "PreprocessingParameters",
    "RANKING_FEATURES",
    "ELIGIBILITY_MIN_RAM_GB",
    "ELIGIBILITY_MIN_STORAGE_GB",
    "get_project_root",
    "BaseRecommender",
    "KNNRecommender",
    "CosineRecommender",
    "WeightedScoringRecommender",
    "WEIGHT_RAM",
    "WEIGHT_CPU",
    "WEIGHT_STORAGE",
    "OfflineEvaluator",
    "EvaluationScenario",
    "generate_72_scenarios",
    "compute_ground_truth_relevance",
    "calculate_scenario_metrics",
    "select_winner_hierarchical",
]
