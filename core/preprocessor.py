"""Preprocessor module for Laptop Recommendation System.

Implements catalog loading, validation, shared preprocessing parameters,
candidate pool extraction, and target preference validation according to
docs/EVALUATION_SPEC.md v2.1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_RELATIVE_PATH = Path("dataset") / "laptops_benchmark_enriched_resolved.csv"

# Locked specification constants (docs/EVALUATION_SPEC.md v2.1.0)
EXPECTED_TOTAL_CATALOG_ROWS = 648
EXPECTED_VALID_CATALOG_ROWS = 636
EXPECTED_EXCLUDED_ROWS = 12

RANKING_FEATURES: List[str] = ["ram_gb", "storage_gb", "cpu_passmark_mark"]
REQUIRED_NUMERICAL_COLUMNS: List[str] = [
    "price_numeric",
    "ram_gb",
    "storage_gb",
    "cpu_passmark_mark",
]

# Mandatory candidate eligibility constants
ELIGIBILITY_MIN_RAM_GB = 16
ELIGIBILITY_MIN_STORAGE_GB = 256


def get_project_root() -> Path:
    """Resolve project root directory robustly."""
    current = Path(__file__).resolve().parent
    # Check parent directory (since preprocessor.py is in core/)
    if (current.parent / DATASET_RELATIVE_PATH).exists():
        return current.parent
    # Fallback checking current or parents
    for p in [current] + list(current.parents):
        if (p / DATASET_RELATIVE_PATH).exists():
            return p
    # Final fallback to parent of core/
    return current.parent


@dataclass(frozen=True)
class PreprocessingParameters:
    """Encapsulates fixed preprocessing parameters fitted on the reference catalog."""

    scaler: StandardScaler
    feature_names: Tuple[str, ...]
    min_bounds: Dict[str, float]
    max_bounds: Dict[str, float]

    def transform_knn(self, df_features: pd.DataFrame) -> np.ndarray:
        """Standardize features for KNN using the fixed reference StandardScaler."""
        X = df_features[list(self.feature_names)].to_numpy(dtype=float)
        return self.scaler.transform(X)

    def transform_target_knn(self, targets: Dict[str, float]) -> np.ndarray:
        """Standardize target vector for KNN using the fixed reference StandardScaler."""
        vec = np.array([[targets[f] for f in self.feature_names]], dtype=float)
        return self.scaler.transform(vec)

    def transform_cosine(self, df_features: pd.DataFrame) -> np.ndarray:
        """Min-Max scale features to [0, 1] using fixed reference bounds."""
        X = df_features[list(self.feature_names)].to_numpy(dtype=float)
        mins = np.array([self.min_bounds[f] for f in self.feature_names], dtype=float)
        maxs = np.array([self.max_bounds[f] for f in self.feature_names], dtype=float)
        diff = maxs - mins
        diff = np.where(diff == 0.0, 1.0, diff)
        return (X - mins) / diff

    def transform_target_cosine(self, targets: Dict[str, float]) -> np.ndarray:
        """Min-Max scale target vector to [0, 1] using fixed reference bounds."""
        vec = np.array([[targets[f] for f in self.feature_names]], dtype=float)
        mins = np.array([self.min_bounds[f] for f in self.feature_names], dtype=float)
        maxs = np.array([self.max_bounds[f] for f in self.feature_names], dtype=float)
        diff = maxs - mins
        diff = np.where(diff == 0.0, 1.0, diff)
        return (vec - mins) / diff


class CatalogPreprocessor:
    """Preprocessor for loading catalog, fitting shared parameters, and generating pools."""

    def __init__(self, dataset_path: Optional[Union[str, Path]] = None) -> None:
        if dataset_path is None:
            self.dataset_path = get_project_root() / DATASET_RELATIVE_PATH
        else:
            self.dataset_path = Path(dataset_path).resolve()

        self._raw_df: pd.DataFrame = self._load_and_validate_raw()
        self._reference_df: pd.DataFrame = self._build_reference_catalog()
        self._params: PreprocessingParameters = self._fit_preprocessing_parameters()

    @property
    def raw_catalog(self) -> pd.DataFrame:
        """Original raw catalog with original_index preserved (read-only copy)."""
        return self._raw_df.copy()

    @property
    def reference_catalog(self) -> pd.DataFrame:
        """Benchmark-complete 636-row reference catalog (read-only copy)."""
        return self._reference_df.copy()

    @property
    def parameters(self) -> PreprocessingParameters:
        """Shared preprocessing parameters (StandardScaler and Min-Max bounds)."""
        return self._params

    def _load_and_validate_raw(self) -> pd.DataFrame:
        """Load CSV and validate existence, schema, and original index."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan pada: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)

        if len(df) != EXPECTED_TOTAL_CATALOG_ROWS:
            raise ValueError(
                f"Katalog awal tidak sesuai spesifikasi. Ditemukan {len(df)} baris, "
                f"diharapkan tepat {EXPECTED_TOTAL_CATALOG_ROWS} baris."
            )

        # Check required columns
        for col in REQUIRED_NUMERICAL_COLUMNS:
            if col not in df.columns:
                raise KeyError(f"Kolom wajib '{col}' tidak ditemukan dalam dataset.")

        # Assign original row index before any filtering
        df = df.copy()
        df.insert(0, "original_index", np.arange(len(df), dtype=int))
        return df

    def _build_reference_catalog(self) -> pd.DataFrame:
        """Filter out rows missing valid finite CPU PassMark score without imputation."""
        df = self._raw_df.copy()

        # Check cpu_passmark_mark validity
        cpu_col = df["cpu_passmark_mark"]
        valid_mask = cpu_col.notnull() & np.isfinite(cpu_col)
        excluded_count = (~valid_mask).sum()

        if excluded_count != EXPECTED_EXCLUDED_ROWS:
            raise ValueError(
                f"Jumlah baris tanpa CPU PassMark valid ({excluded_count}) tidak sesuai spesifikasi "
                f"({EXPECTED_EXCLUDED_ROWS} baris)."
            )

        ref_df = df[valid_mask].copy()

        if len(ref_df) != EXPECTED_VALID_CATALOG_ROWS:
            raise ValueError(
                f"Katalog referensi benchmark-complete memiliki {len(ref_df)} baris, "
                f"diharapkan tepat {EXPECTED_VALID_CATALOG_ROWS} baris."
            )

        # Validate finiteness of numerical columns in reference catalog
        for col in REQUIRED_NUMERICAL_COLUMNS:
            vals = ref_df[col].to_numpy(dtype=float)
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"Kolom '{col}' pada katalog referensi mengandung nilai tidak berhingga atau NaN.")

        # Ensure correct numeric types
        ref_df["price_numeric"] = ref_df["price_numeric"].astype(np.int64)
        ref_df["ram_gb"] = ref_df["ram_gb"].astype(int)
        ref_df["storage_gb"] = ref_df["storage_gb"].astype(int)
        ref_df["cpu_passmark_mark"] = ref_df["cpu_passmark_mark"].astype(float)

        return ref_df

    def _fit_preprocessing_parameters(self) -> PreprocessingParameters:
        """Fit StandardScaler and compute Min-Max bounds once on the reference catalog."""
        ref = self._reference_df
        X_ref = ref[RANKING_FEATURES].to_numpy(dtype=float)

        scaler = StandardScaler()
        scaler.fit(X_ref)

        min_bounds: Dict[str, float] = {}
        max_bounds: Dict[str, float] = {}

        for feat in RANKING_FEATURES:
            vals = ref[feat].to_numpy(dtype=float)
            min_bounds[feat] = float(np.min(vals))
            max_bounds[feat] = float(np.max(vals))

        return PreprocessingParameters(
            scaler=scaler,
            feature_names=tuple(RANKING_FEATURES),
            min_bounds=min_bounds,
            max_bounds=max_bounds,
        )

    def get_candidate_pool(
        self,
        budget: Union[int, float],
        min_ram: int = ELIGIBILITY_MIN_RAM_GB,
        min_storage: int = ELIGIBILITY_MIN_STORAGE_GB,
    ) -> pd.DataFrame:
        """Extract candidate pool according to locked eligibility constraints.

        Eligibility rule:
            price_numeric <= budget AND ram_gb >= min_ram AND storage_gb >= min_storage

        Does NOT filter on target preferences.
        Preserves original_index and price_numeric.
        """
        try:
            b_val = float(budget)
        except (TypeError, ValueError):
            raise ValueError(f"Budget tidak valid: {budget}")

        if not np.isfinite(b_val) or b_val <= 0:
            raise ValueError(f"Budget harus berupa angka positif: {budget}")

        ref = self._reference_df

        mask = (
            (ref["price_numeric"] <= b_val)
            & (ref["ram_gb"] >= min_ram)
            & (ref["storage_gb"] >= min_storage)
        )

        cand = ref[mask].copy()
        return cand

    def validate_targets(
        self,
        target_ram: Union[int, float],
        target_storage: Union[int, float],
        target_cpu: Union[int, float],
    ) -> Dict[str, float]:
        """Validate user preference targets against fixed catalog bounds.

        Explicitly rejects invalid or out-of-range targets (no silent clipping).
        """
        errors = []

        # Validate RAM
        try:
            ram_val = float(target_ram)
            min_ram = self._params.min_bounds["ram_gb"]
            max_ram = self._params.max_bounds["ram_gb"]
            if not (min_ram <= ram_val <= max_ram):
                errors.append(
                    f"Target RAM ({ram_val} GB) di luar rentang katalog [{min_ram:.0f}, {max_ram:.0f} GB]."
                )
        except (TypeError, ValueError):
            errors.append(f"Target RAM tidak valid: {target_ram}")
            ram_val = None

        # Validate Storage
        try:
            storage_val = float(target_storage)
            min_storage = self._params.min_bounds["storage_gb"]
            max_storage = self._params.max_bounds["storage_gb"]
            if not (min_storage <= storage_val <= max_storage):
                errors.append(
                    f"Target Storage ({storage_val} GB) di luar rentang katalog [{min_storage:.0f}, {max_storage:.0f} GB]."
                )
        except (TypeError, ValueError):
            errors.append(f"Target Storage tidak valid: {target_storage}")
            storage_val = None

        # Validate CPU
        try:
            cpu_val = float(target_cpu)
            min_cpu = self._params.min_bounds["cpu_passmark_mark"]
            max_cpu = self._params.max_bounds["cpu_passmark_mark"]
            if not (min_cpu <= cpu_val <= max_cpu):
                errors.append(
                    f"Target CPU PassMark ({cpu_val:.0f}) di luar rentang katalog [{min_cpu:.0f}, {max_cpu:.0f}]."
                )
        except (TypeError, ValueError):
            errors.append(f"Target CPU PassMark tidak valid: {target_cpu}")
            cpu_val = None

        if errors:
            raise ValueError("; ".join(errors))

        return {
            "ram_gb": float(ram_val),
            "storage_gb": float(storage_val),
            "cpu_passmark_mark": float(cpu_val),
        }

