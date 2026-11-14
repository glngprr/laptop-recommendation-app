"""Recommendation algorithms module for Laptop Recommendation System.

Implements three independent recommendation methods according to
docs/EVALUATION_SPEC.md v2.1.0:
1. KNN-based Euclidean Ranking
2. Cosine Similarity
3. Weighted Scoring (MAUT)

All methods use the same candidate pool and shared preprocessing parameters
supplied by CatalogPreprocessor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from core.preprocessor import CatalogPreprocessor, RANKING_FEATURES

# Tie-breaking numerical tolerance specified in docs/EVALUATION_SPEC.md Section 4.4
SCORE_TIE_TOLERANCE: float = 1e-7

# Standard output columns
OUTPUT_COLUMNS: List[str] = [
    "rank_position",
    "score",
    "name",
    "price",
    "price_numeric",
    "ram_gb",
    "storage_gb",
    "cpu_passmark_mark",
    "original_index",
]

# Weighted Scoring feature weights (docs/EVALUATION_SPEC.md Section 4.3)
WEIGHT_RAM: float = 0.45
WEIGHT_CPU: float = 0.35
WEIGHT_STORAGE: float = 0.20


def _assign_tie_groups(
    scores: np.ndarray,
    ascending: bool,
    tol: float = SCORE_TIE_TOLERANCE,
) -> np.ndarray:
    """Group scores that differ by <= tol into identical integer group IDs.

    Mathematical explanation:
    Given scores sorted by the primary ranking objective (ascending for KNN,
    descending for Cosine/Weighted), consecutive scores falling within the exact
    absolute tolerance `tol` (measured from the cluster's leader/anchor) are assigned to
    the same tie group. This prevents the boundary-splitting artifact of fixed-precision
    rounding while adhering strictly to the exact specified tolerance boundary:
    |score - anchor| <= tol.

    Returns:
        np.ndarray of integer group IDs matching the input scores' original order,
        where lower group ID corresponds to better primary ranking rank.
    """
    n = len(scores)
    if n <= 1:
        return np.zeros(n, dtype=int)

    # Sort indices by primary score direction (stable mergesort)
    order = np.argsort(scores, kind="mergesort")
    if not ascending:
        order = order[::-1]

    sorted_scores = scores[order]

    # Cluster sorted scores using anchor-based tolerance matching
    sorted_groups = np.zeros(n, dtype=int)
    curr_group = 0
    anchor_idx = 0

    for i in range(1, n):
        diff = abs(sorted_scores[i] - sorted_scores[anchor_idx])
        # Exact specified tolerance comparison without extra padding
        if diff <= tol:
            sorted_groups[i] = curr_group
        else:
            curr_group += 1
            sorted_groups[i] = curr_group
            anchor_idx = i

    # Map group IDs back to the original row order
    groups = np.empty(n, dtype=int)
    groups[order] = sorted_groups
    return groups


def _apply_tie_breaking(
    cand: pd.DataFrame,
    score_col: str,
    ascending: bool,
    top_n: Optional[int] = 5,
) -> pd.DataFrame:
    """Apply standardized tie-breaking policy from docs/EVALUATION_SPEC.md Section 4.4.

    1. Primary ranking score:
       - ascending for distance (KNN)
       - descending for similarity / utility (Cosine, Weighted)
       Scores with absolute difference <= 1e-7 belong to the same tie group.
    2. Ties within tolerance 1e-7 -> price_numeric ascending (lower price first).
    3. Further ties -> original_index ascending (original catalog order).
    """
    if len(cand) == 0:
        empty_res = cand.copy()
        for col in ["rank_position", "score"]:
            if col not in empty_res.columns:
                empty_res[col] = []
        return empty_res

    res = cand.copy()
    scores = res[score_col].to_numpy(dtype=float)

    # Assign integer tie groups according to exact absolute tolerance 1e-7
    res["_tie_group"] = _assign_tie_groups(scores, ascending=ascending, tol=SCORE_TIE_TOLERANCE)

    # Sort with hierarchical keys:
    # 1. _tie_group ascending (since group 0 contains the best scores)
    # 2. price_numeric ascending (tie-breaker 1)
    # 3. original_index ascending (tie-breaker 2)
    sort_cols = ["_tie_group", "price_numeric", "original_index"]
    sort_ascending = [True, True, True]

    res = res.sort_values(by=sort_cols, ascending=sort_ascending)

    if top_n is not None and top_n > 0:
        res = res.head(top_n)

    res["rank_position"] = np.arange(1, len(res) + 1, dtype=int)
    res["score"] = res[score_col].astype(float)

    # Select standard output columns plus any existing metadata
    available_output_cols = [c for c in OUTPUT_COLUMNS if c in res.columns]
    other_cols = [c for c in res.columns if c not in OUTPUT_COLUMNS and not c.startswith("_")]
    final_cols = available_output_cols + other_cols

    return res[final_cols].reset_index(drop=True)


class BaseRecommender(ABC):
    """Abstract base class for recommendation methods."""

    def __init__(self, preprocessor: CatalogPreprocessor) -> None:
        self.preprocessor = preprocessor

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Name of the recommendation method."""
        pass

    @property
    @abstractmethod
    def score_column_name(self) -> str:
        """Name of the primary score column."""
        pass

    @abstractmethod
    def recommend(
        self,
        candidate_pool: pd.DataFrame,
        targets: Dict[str, Union[int, float]],
        top_n: Optional[int] = 5,
    ) -> pd.DataFrame:
        """Generate top-N recommendations for the candidate pool and targets."""
        pass


class KNNRecommender(BaseRecommender):
    """Method 1: KNN-based Euclidean Ranking (docs/EVALUATION_SPEC.md Section 4.1)."""

    @property
    def method_name(self) -> str:
        return "knn_euclidean"

    @property
    def score_column_name(self) -> str:
        return "knn_distance"

    def recommend(
        self,
        candidate_pool: pd.DataFrame,
        targets: Dict[str, Union[int, float]],
        top_n: Optional[int] = 5,
    ) -> pd.DataFrame:
        """Rank candidates by ascending standardized Euclidean distance."""
        validated_targets = self.preprocessor.validate_targets(
            target_ram=targets["ram_gb"],
            target_storage=targets["storage_gb"],
            target_cpu=targets["cpu_passmark_mark"],
        )

        if len(candidate_pool) == 0:
            return _apply_tie_breaking(
                candidate_pool,
                score_col=self.score_column_name,
                ascending=True,
                top_n=top_n,
            )

        params = self.preprocessor.parameters

        # Transform candidate features using fixed reference StandardScaler
        X_std = params.transform_knn(candidate_pool)

        # Transform target preferences using the same scaler
        t_std = params.transform_target_knn(validated_targets)

        # Calculate Euclidean distance across ranking features
        diff = X_std - t_std
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        res = candidate_pool.copy()
        res[self.score_column_name] = distances

        return _apply_tie_breaking(
            res,
            score_col=self.score_column_name,
            ascending=True,
            top_n=top_n,
        )


class CosineRecommender(BaseRecommender):
    """Method 2: Cosine Similarity (docs/EVALUATION_SPEC.md Section 4.2)."""

    @property
    def method_name(self) -> str:
        return "cosine_similarity"

    @property
    def score_column_name(self) -> str:
        return "cosine_similarity"

    def recommend(
        self,
        candidate_pool: pd.DataFrame,
        targets: Dict[str, Union[int, float]],
        top_n: Optional[int] = 5,
    ) -> pd.DataFrame:
        """Rank candidates by descending cosine similarity in Min-Max space."""
        validated_targets = self.preprocessor.validate_targets(
            target_ram=targets["ram_gb"],
            target_storage=targets["storage_gb"],
            target_cpu=targets["cpu_passmark_mark"],
        )

        if len(candidate_pool) == 0:
            return _apply_tie_breaking(
                candidate_pool,
                score_col=self.score_column_name,
                ascending=False,
                top_n=top_n,
            )

        params = self.preprocessor.parameters

        # Min-Max scale candidate features to [0, 1]
        X_norm = params.transform_cosine(candidate_pool)

        # Min-Max scale target preferences to [0, 1]
        t_norm = params.transform_target_cosine(validated_targets)

        t_vec = t_norm.flatten()
        t_norm_val = float(np.linalg.norm(t_vec))

        # Explicit zero-norm target handling
        # If target norm is zero, assign similarity 0.0 to every candidate
        if t_norm_val == 0.0:
            similarities = np.zeros(len(candidate_pool), dtype=float)
        else:
            X_norms = np.linalg.norm(X_norm, axis=1)
            # Safe division: if any candidate vector has norm zero
            denom = X_norms * t_norm_val
            dot_prods = np.dot(X_norm, t_vec)
            similarities = np.where(denom > 0.0, dot_prods / denom, 0.0)

        res = candidate_pool.copy()
        res[self.score_column_name] = similarities

        return _apply_tie_breaking(
            res,
            score_col=self.score_column_name,
            ascending=False,
            top_n=top_n,
        )


class WeightedScoringRecommender(BaseRecommender):
    """Method 3: Weighted Scoring (MAUT) (docs/EVALUATION_SPEC.md Section 4.3)."""

    @property
    def method_name(self) -> str:
        return "weighted_scoring"

    @property
    def score_column_name(self) -> str:
        return "weighted_score"

    def recommend(
        self,
        candidate_pool: pd.DataFrame,
        targets: Dict[str, Union[int, float]],
        top_n: Optional[int] = 5,
    ) -> pd.DataFrame:
        """Rank candidates by descending multi-attribute utility score."""
        validated_targets = self.preprocessor.validate_targets(
            target_ram=targets["ram_gb"],
            target_storage=targets["storage_gb"],
            target_cpu=targets["cpu_passmark_mark"],
        )

        if len(candidate_pool) == 0:
            return _apply_tie_breaking(
                candidate_pool,
                score_col=self.score_column_name,
                ascending=False,
                top_n=top_n,
            )

        params = self.preprocessor.parameters
        max_bounds = params.max_bounds

        # Vectorized utility calculation
        u_ram = self._calculate_feature_utility(
            candidate_pool["ram_gb"].to_numpy(dtype=float),
            target_val=validated_targets["ram_gb"],
            catalog_max=max_bounds["ram_gb"],
        )

        u_cpu = self._calculate_feature_utility(
            candidate_pool["cpu_passmark_mark"].to_numpy(dtype=float),
            target_val=validated_targets["cpu_passmark_mark"],
            catalog_max=max_bounds["cpu_passmark_mark"],
        )

        u_storage = self._calculate_feature_utility(
            candidate_pool["storage_gb"].to_numpy(dtype=float),
            target_val=validated_targets["storage_gb"],
            catalog_max=max_bounds["storage_gb"],
        )

        # Total score with locked weights: 0.45 RAM, 0.35 CPU, 0.20 Storage
        scores = (
            WEIGHT_RAM * u_ram
            + WEIGHT_CPU * u_cpu
            + WEIGHT_STORAGE * u_storage
        )

        res = candidate_pool.copy()
        res[self.score_column_name] = scores

        return _apply_tie_breaking(
            res,
            score_col=self.score_column_name,
            ascending=False,
            top_n=top_n,
        )

    @staticmethod
    def _calculate_feature_utility(
        x: np.ndarray,
        target_val: float,
        catalog_max: float,
    ) -> np.ndarray:
        """Calculate piecewise utility u(x, target) for a feature.

        If x < target:
            u = 0.8 * x / target
        If x >= target:
            u = 0.8 + 0.2 * g(x, target)
        Where g(x, target):
            if target >= catalog_max:
                g = 0.0
            else:
                g = min(1.0, (x - target) / (catalog_max - target))
        """
        u = np.empty_like(x, dtype=float)

        # Condition 1: shortfall (x < target)
        shortfall_mask = x < target_val
        if target_val > 0.0:
            u[shortfall_mask] = 0.8 * (x[shortfall_mask] / target_val)
        else:
            u[shortfall_mask] = 0.8

        # Condition 2: target fulfilled or surplus (x >= target)
        surplus_mask = ~shortfall_mask

        if target_val >= catalog_max:
            g = np.zeros(np.sum(surplus_mask), dtype=float)
        else:
            denom = catalog_max - target_val
            diff = x[surplus_mask] - target_val
            g = np.minimum(1.0, diff / denom)

        u[surplus_mask] = 0.8 + 0.2 * g
        return u
