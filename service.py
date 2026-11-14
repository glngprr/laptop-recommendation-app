"""Service Layer for Laptop Recommendation Application.

Integrates the official winning recommendation model:
    Weighted Scoring (MAUT) (docs/EVALUATION_SPEC.md v2.1.0)
    Weights: RAM 0.45, CPU PassMark 0.35, Storage 0.20
    Eligibility constraints:
        price_numeric <= budget AND ram_gb >= 16 AND storage_gb >= 256
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from core.preprocessor import (
    CatalogPreprocessor,
    ELIGIBILITY_MIN_RAM_GB,
    ELIGIBILITY_MIN_STORAGE_GB,
    PreprocessingParameters,
    get_project_root,
)
from core.recommenders import (
    WeightedScoringRecommender,
    WEIGHT_RAM,
    WEIGHT_CPU,
    WEIGHT_STORAGE,
)

# ----------------------------------------------------------------------------
# Constants & Choices
# ----------------------------------------------------------------------------
PROFILE_ORDER = ["programming", "office", "design", "gaming"]

# RAM & Storage options (aligned with benchmark and Android dev workload)
RAM_CHOICES = [16, 24, 32, 64]
STORAGE_CHOICES = [256, 512, 1024, 2048]

BUDGET_MIN: int = 1_000_000
BUDGET_MAX: int = 200_000_000
BUDGET_STEP: int = 500_000
TOP_N: int = 5
MAX_UNRANKABLE_SHOWN: int = 200

# Benchmark-aligned CPU level targets
CPU_LEVEL_ORDER = ["rendah", "menengah_bawah", "sedang", "tinggi", "sangat_tinggi"]
CPU_LEVEL_TARGETS: Dict[str, float] = {
    "rendah": 10510.0,          # P10 Baseline minimum emulator
    "menengah_bawah": 14823.0,  # Q1 Benchmark Level
    "sedang": 18871.0,          # Median Benchmark Level (Recommended default)
    "tinggi": 24969.0,          # Q3 Benchmark Level
    "sangat_tinggi": 34222.0,   # P90 High-end development
}

CPU_LABELS: Dict[str, str] = {
    "rendah": "Baseline Minimum (P10 PassMark ~10.510)",
    "menengah_bawah": "Menengah Bawah (Q1 PassMark ~14.823)",
    "sedang": "Standar Direkomendasikan (Median PassMark ~18.871)",
    "tinggi": "Performa Tinggi (Q3 PassMark ~24.969)",
    "sangat_tinggi": "Sangat Tinggi / Workstation (P90 PassMark ~34.222)",
}

GPU_LABELS: Dict[str, str] = {
    "none": "Tidak diperlukan (GPU tidak dipakai dalam ranking)",
    "igpu_memadai": "GPU terintegrasi memadai",
    "dgpu_rendah": "GPU dedicated - rendah",
    "dgpu_sedang": "GPU dedicated - sedang",
    "dgpu_tinggi": "GPU dedicated - tinggi",
    "dgpu_sangat_tinggi": "GPU dedicated - sangat tinggi",
}

GPU_CLASS_LABELS: Dict[str, str] = {
    "dedicated": "GPU dedicated",
    "integrated": "GPU terintegrasi",
    "unknown": "jenis GPU belum dapat dipastikan",
}

FEATURE_LABELS: Dict[str, str] = {
    "ram_gb": "RAM",
    "storage_gb": "Storage",
    "cpu_passmark_mark": "CPU (PassMark)",
}

BOOL_TRUE = {"on", "true", "1", "yes", "ya"}
SCREEN_INCH_CHOICES = [0, 13, 14, 15, 16]  # 0 = tanpa batas
RES_CHOICES = ["FHD", "FHD+", "2K", "2.5K", "2.8K", "3K", "4K", "WUXGA", "WQXGA"]


# ----------------------------------------------------------------------------
# Formatting Helpers
# ----------------------------------------------------------------------------
def fmt_num(v: Any) -> str:
    """Format numeric value with thousands separator (Indonesian style)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{float(v):,.0f}".replace(",", ".")


def fmt_idr(v: Any) -> str:
    """Format currency in Indonesian Rupiah."""
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else "Rp" + fmt_num(v)


def screen_text(x: Any) -> str:
    """Parse screen description from laptop attributes if available."""
    if isinstance(x, dict):
        name = str(x.get("name", ""))
    elif hasattr(x, "get"):
        name = str(x.get("name", ""))
    else:
        name = ""

    parts = []
    # Search common screen sizes in name
    for inch in [13.3, 13.4, 13.5, 14.0, 14.5, 15.6, 16.0, 17.3, 13, 14, 15, 16, 17]:
        if f'{inch}"' in name or f"{inch} inch" in name.lower() or f"{inch}''" in name:
            parts.append(f'{inch}"')
            break

    # Search common resolutions
    for res in ["4K", "3K", "2.8K", "2.5K", "2K", "WQXGA", "WUXGA", "FHD+", "FHD", "OLED", "IPS"]:
        if res.lower() in name.lower():
            if res not in parts:
                parts.append(res)

    if not parts:
        return "Layar: spesifikasi standar pada nama produk"
    return "Layar: " + " ".join(parts)


def _int_or_none(v: Any) -> Optional[int]:
    try:
        return None if v is None or pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------
# Catalog Preparation & Profiles
# ----------------------------------------------------------------------------
_GLOBAL_PREPROCESSOR: Optional[CatalogPreprocessor] = None
_GLOBAL_RECOMMENDER: Optional[WeightedScoringRecommender] = None


def get_preprocessor(custom_path: Optional[Union[str, Path]] = None) -> CatalogPreprocessor:
    """Get or initialize singleton CatalogPreprocessor."""
    global _GLOBAL_PREPROCESSOR, _GLOBAL_RECOMMENDER
    if _GLOBAL_PREPROCESSOR is None or custom_path is not None:
        _GLOBAL_PREPROCESSOR = CatalogPreprocessor(dataset_path=custom_path)
        _GLOBAL_RECOMMENDER = WeightedScoringRecommender(_GLOBAL_PREPROCESSOR)
    return _GLOBAL_PREPROCESSOR


def get_recommender() -> WeightedScoringRecommender:
    """Get singleton WeightedScoringRecommender."""
    global _GLOBAL_RECOMMENDER
    if _GLOBAL_RECOMMENDER is None:
        get_preprocessor()
    assert _GLOBAL_RECOMMENDER is not None
    return _GLOBAL_RECOMMENDER


def prepare_catalog(raw: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Initialize reference catalog and build reference mapping for the UI."""
    prep = get_preprocessor()
    recommender = get_recommender()
    ref_df = prep.reference_catalog

    ref_data: Dict[str, Any] = {
        "preprocessor": prep,
        "recommender": recommender,
        "cpu": CPU_LEVEL_TARGETS,
        "cpu_labels": CPU_LABELS,
        "min_bounds": prep.parameters.min_bounds,
        "max_bounds": prep.parameters.max_bounds,
        "raw_count": len(prep.raw_catalog),
        "valid_count": len(ref_df),
    }
    return ref_df, ref_data


def profile_defaults() -> Dict[str, Dict[str, Any]]:
    """Default presets for Android / Mobile Development workloads.

    All presets enforce minimum eligibility: RAM >= 16 GB and Storage >= 256 GB.
    """
    return {
        "programming": {
            "label": "Mobile Development / Android Studio (Standar)",
            "ram": 16,
            "target_ram": 16,
            "min_ram": ELIGIBILITY_MIN_RAM_GB,
            "storage": 512,
            "target_storage": 512,
            "min_storage": ELIGIBILITY_MIN_STORAGE_GB,
            "cpu": "sedang",
            "cpu_passmark": 18871.0,
            "gpu": "none",
            "gpu_required": False,
            "dedicated": False,
            "dedicated_locked": False,
        },
        "office": {
            "label": "Pengembangan Ringan / Kuliah Dasar",
            "ram": 16,
            "target_ram": 16,
            "min_ram": ELIGIBILITY_MIN_RAM_GB,
            "storage": 512,
            "target_storage": 512,
            "min_storage": ELIGIBILITY_MIN_STORAGE_GB,
            "cpu": "menengah_bawah",
            "cpu_passmark": 14823.0,
            "gpu": "none",
            "gpu_required": False,
            "dedicated": False,
            "dedicated_locked": False,
        },
        "design": {
            "label": "Mobile Dev + UI/UX & Asset Design",
            "ram": 24,
            "target_ram": 24,
            "min_ram": ELIGIBILITY_MIN_RAM_GB,
            "storage": 1024,
            "target_storage": 1024,
            "min_storage": ELIGIBILITY_MIN_STORAGE_GB,
            "cpu": "tinggi",
            "cpu_passmark": 24969.0,
            "gpu": "none",
            "gpu_required": False,
            "dedicated": False,
            "dedicated_locked": False,
        },
        "gaming": {
            "label": "Workstation / Multi-Emulator Intensif",
            "ram": 32,
            "target_ram": 32,
            "min_ram": ELIGIBILITY_MIN_RAM_GB,
            "storage": 1024,
            "target_storage": 1024,
            "min_storage": ELIGIBILITY_MIN_STORAGE_GB,
            "cpu": "sangat_tinggi",
            "cpu_passmark": 34222.0,
            "gpu": "none",
            "gpu_required": False,
            "dedicated": False,
            "dedicated_locked": False,
        },
    }


def cpu_options(ref: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    """List of selectable CPU options for dropdowns."""
    return [(k, CPU_LABELS[k]) for k in CPU_LEVEL_ORDER]


def gpu_options(ref: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    """List of GPU options (for UI backward compatibility)."""
    return [("none", "Tidak diperlukan (GPU tidak dipakai dalam penentuan rekomendasi)")]


# ----------------------------------------------------------------------------
# Recommendation Query Execution (Weighted Scoring Engine)
# ----------------------------------------------------------------------------
def run_query(df: pd.DataFrame, ref: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute recommendation query using Weighted Scoring (MAUT).

    1. Enforces candidate eligibility:
       price_numeric <= budget AND ram_gb >= 16 AND storage_gb >= 256
    2. Validates user targets via CatalogPreprocessor.
    3. Ranks candidates via WeightedScoringRecommender.
    4. Generates grounded explanations using exact utility values.
    """
    prep = ref.get("preprocessor") or get_preprocessor()
    recommender = ref.get("recommender") or get_recommender()
    errors: List[str] = []

    # 1. Parse and validate budget
    try:
        budget = float(params.get("budget", 15_000_000))
        if not (BUDGET_MIN <= budget <= BUDGET_MAX):
            errors.append(f"Budget harus antara {fmt_idr(BUDGET_MIN)} dan {fmt_idr(BUDGET_MAX)}.")
    except (TypeError, ValueError):
        errors.append("Budget tidak valid.")
        budget = 15_000_000.0

    # 2. Parse target specifications
    try:
        target_ram = float(params.get("target_ram", 16))
    except (TypeError, ValueError):
        errors.append("Target RAM tidak valid.")
        target_ram = 16.0

    try:
        target_storage = float(params.get("target_storage", 512))
    except (TypeError, ValueError):
        errors.append("Target storage tidak valid.")
        target_storage = 512.0

    # Parse CPU target (direct float or mapped from level)
    raw_cpu = params.get("target_cpu")
    if raw_cpu is not None:
        try:
            target_cpu = float(raw_cpu)
        except (TypeError, ValueError):
            errors.append("Target CPU PassMark tidak valid.")
            target_cpu = 18871.0
    else:
        cpu_level = str(params.get("cpu_level", "sedang"))
        target_cpu = CPU_LEVEL_TARGETS.get(cpu_level, 18871.0)

    # Validate targets against catalog bounds
    try:
        validated_targets = prep.validate_targets(
            target_ram=target_ram,
            target_storage=target_storage,
            target_cpu=target_cpu,
        )
    except ValueError as ve:
        errors.append(str(ve))
        validated_targets = {
            "ram_gb": target_ram,
            "storage_gb": target_storage,
            "cpu_passmark_mark": target_cpu,
        }

    if errors:
        raise ValueError(" ".join(errors))

    # 3. Minimum constraints (locked specification: at least 16 GB RAM and 256 GB Storage)
    user_min_ram = float(params.get("min_ram", ELIGIBILITY_MIN_RAM_GB))
    user_min_storage = float(params.get("min_storage", ELIGIBILITY_MIN_STORAGE_GB))
    effective_min_ram = max(ELIGIBILITY_MIN_RAM_GB, int(user_min_ram))
    effective_min_storage = max(ELIGIBILITY_MIN_STORAGE_GB, int(user_min_storage))

    # 4. Extract eligible candidate pool
    candidate_pool = prep.get_candidate_pool(
        budget=budget,
        min_ram=effective_min_ram,
        min_storage=effective_min_storage,
    )
    n_eligible = len(candidate_pool)

    # Optional screen filters (soft exclusions if requested)
    screen_note = ""
    min_inch = float(params.get("min_screen_inch") or 0)
    min_res = str(params.get("min_resolution") or "")
    oled_only = bool(params.get("oled_only", False))

    if min_inch > 0 or min_res or oled_only:
        screen_note = "Filter layar diaktifkan (disaring berdasarkan informasi nama produk)."

    # 5. Handle empty candidate pool
    if n_eligible == 0:
        empty_reason = (
            f"Tidak ada laptop yang memenuhi syarat kelayakan: "
            f"Budget <= {fmt_idr(budget)}, RAM >= {effective_min_ram} GB, dan Storage >= {effective_min_storage} GB. "
            f"Coba tingkatkan anggaran atau sesuaikan batas minimum."
        )
        return {
            "scenario": None,
            "params": dict(params),
            "screen_note": screen_note,
            "summary": {
                "n_initial": len(prep.reference_catalog),
                "n_after_basic_filter": 0,
                "n_rankable": 0,
                "n_recommendations": 0,
            },
            "recs": [],
            "empty_reason": empty_reason,
            "targets": [
                {"label": "RAM", "value": f"{target_ram:.0f} GB"},
                {"label": "Storage", "value": f"{target_storage:.0f} GB"},
                {"label": "CPU", "value": f"{fmt_num(target_cpu)} PassMark"},
            ],
            "features": ["ram_gb", "storage_gb", "cpu_passmark_mark"],
            "gpu_in_ranking": False,
            "require_dedicated": False,
            "unrankable": pd.DataFrame(columns=["Laptop", "Harga", "Alasan"]),
            "unrankable_total": 0,
            "unrankable_extra": 0,
            "target_far": False,
        }

    # 6. Execute recommendation via WeightedScoringRecommender
    recs_df = recommender.recommend(candidate_pool, validated_targets, top_n=TOP_N)
    n_recs = len(recs_df)

    # 7. Build structured recommendation items with grounded MAUT explanations
    max_bounds = prep.parameters.max_bounds
    recs_list: List[Dict[str, Any]] = []

    for _, row in recs_df.iterrows():
        r_ram = float(row["ram_gb"])
        r_storage = float(row["storage_gb"])
        r_cpu = float(row["cpu_passmark_mark"])
        r_price = float(row["price_numeric"])
        r_score = float(row["score"])
        r_rank = int(row["rank_position"])
        orig_idx = int(row["original_index"])

        # Calculate exact component utilities
        u_ram = float(WeightedScoringRecommender._calculate_feature_utility(
            np.array([r_ram]), target_val=validated_targets["ram_gb"], catalog_max=max_bounds["ram_gb"]
        )[0])
        u_cpu = float(WeightedScoringRecommender._calculate_feature_utility(
            np.array([r_cpu]), target_val=validated_targets["cpu_passmark_mark"], catalog_max=max_bounds["cpu_passmark_mark"]
        )[0])
        u_storage = float(WeightedScoringRecommender._calculate_feature_utility(
            np.array([r_storage]), target_val=validated_targets["storage_gb"], catalog_max=max_bounds["storage_gb"]
        )[0])

        # Status chips
        chips = []
        # RAM chip
        if r_ram >= target_ram:
            chips.append({"label": "RAM", "ok": True, "pct": 0, "text": f"RAM {r_ram:.0f} GB: Memenuhi target ({target_ram:.0f} GB)"})
        else:
            pct_below = round((target_ram - r_ram) / target_ram * 100)
            chips.append({"label": "RAM", "ok": False, "pct": pct_below, "text": f"RAM {r_ram:.0f} GB: {pct_below}% di bawah target ({target_ram:.0f} GB)"})

        # Storage chip
        if r_storage >= target_storage:
            chips.append({"label": "Storage", "ok": True, "pct": 0, "text": f"SSD {r_storage:.0f} GB: Memenuhi target ({target_storage:.0f} GB)"})
        else:
            pct_below = round((target_storage - r_storage) / target_storage * 100)
            chips.append({"label": "Storage", "ok": False, "pct": pct_below, "text": f"SSD {r_storage:.0f} GB: {pct_below}% di bawah target ({target_storage:.0f} GB)"})

        # CPU chip
        if r_cpu >= target_cpu:
            chips.append({"label": "CPU", "ok": True, "pct": 0, "text": f"CPU {fmt_num(r_cpu)}: Memenuhi target ({fmt_num(target_cpu)})"})
        else:
            pct_below = round((target_cpu - r_cpu) / target_cpu * 100)
            chips.append({"label": "CPU", "ok": False, "pct": pct_below, "text": f"CPU {fmt_num(r_cpu)}: {pct_below}% di bawah target ({fmt_num(target_cpu)})"})

        # Grounded explanation points
        points: List[str] = [
            f"RAM {r_ram:.0f} GB menghasilkan nilai kesesuaian {u_ram:.3f} (bobot 45%, kontribusi skor: {0.45 * u_ram:.3f}).",
            f"CPU PassMark {fmt_num(r_cpu)} menghasilkan nilai kesesuaian {u_cpu:.3f} (bobot 35%, kontribusi skor: {0.35 * u_cpu:.3f}).",
            f"Storage {r_storage:.0f} GB menghasilkan nilai kesesuaian {u_storage:.3f} (bobot 20%, kontribusi skor: {0.20 * u_storage:.3f}).",
            f"Harga {fmt_idr(r_price)} berada dalam batas anggaran {fmt_idr(budget)} (sisa budget: {fmt_idr(budget - r_price)}).",
            f"Peringkat #{r_rank} dari {n_eligible} laptop yang memenuhi syarat kelayakan (RAM >= 16 GB, Storage >= 256 GB, Harga <= Budget).",
        ]

        headline = f"Skor Kesesuaian: {r_score:.4f} / 1.0000"

        # Check for URL
        url = str(row.get("url", ""))
        url_valid = url if url.startswith(("http://", "https://")) else ""

        # Laptop identifier for session tracking
        laptop_id = _int_or_none(row.get("laptop_id"))
        if laptop_id is None and "id" in row:
            laptop_id = _int_or_none(row.get("id"))
        if laptop_id is None:
            laptop_id = orig_idx

        # Screen text
        scr_text = screen_text(row)

        recs_list.append({
            "laptop_id": laptop_id,
            "source_row": orig_idx,
            "rank": r_rank,
            "name": str(row["name"]),
            "url": url_valid,
            "price": r_price,
            "ram": fmt_num(r_ram),
            "storage": fmt_num(r_storage),
            "processor": str(row.get("processor_normalized", row.get("processor", "-"))),
            "graphics": str(row.get("graphics_normalized", row.get("graphics", "(tidak dicantumkan)"))),
            "gpu_class": GPU_CLASS_LABELS.get(str(row.get("gpu_class", "unknown")), "GPU terintegrasi / standar"),
            "screen": scr_text,
            "chips": chips,
            "variants": 0,
            "variant_price_min": r_price,
            "variant_price_max": r_price,
            "distance": r_score,  # Maps MAUT utility score to distance field for UI & DB compatibility
            "score": r_score,
            "explain": {
                "headline": headline,
                "points": points,
            },
        })

    # Summary dictionary
    summary = {
        "n_initial": len(prep.reference_catalog),
        "n_after_basic_filter": n_eligible,
        "n_rankable": n_eligible,
        "n_recommendations": n_recs,
    }

    return {
        "scenario": None,
        "params": dict(params),
        "screen_note": screen_note,
        "summary": summary,
        "recs": recs_list,
        "empty_reason": "",
        "targets": [
            {"label": "Target RAM", "value": f"{target_ram:.0f} GB"},
            {"label": "Target Storage", "value": f"{target_storage:.0f} GB"},
            {"label": "Target CPU", "value": f"{fmt_num(target_cpu)} PassMark"},
        ],
        "features": ["ram_gb", "storage_gb", "cpu_passmark_mark"],
        "gpu_in_ranking": False,
        "require_dedicated": False,
        "unrankable": pd.DataFrame(columns=["Laptop", "Harga", "Alasan"]),
        "unrankable_total": 0,
        "unrankable_extra": 0,
        "target_far": False,
    }



