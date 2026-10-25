"""
KNN v8: profile-based laptop recommendation (hard filter -> profile-to-feature
mapping -> KNN ranking -> evaluation).

Run from the project root (same convention as v7):

    # 1) Evaluation suite: 8 illustrative scenarios (2 per profile)
    python knn_v8_profile_recommender.py

    # 2) One custom query
    python knn_v8_profile_recommender.py --profile programming ^
        --budget 15000000 --min-ram 16 --min-storage 512

Input:
    dataset/laptops_benchmark_enriched_resolved.csv   (override with --input)

Outputs (folder inspection/):
    knn_v8_gpu_classification_audit.csv   unique GPU names + class + basis (to verify manually)
    knn_v8_reference_targets.csv          CPU/GPU level -> benchmark value (fixed reference)
    knn_v8_scenario_summary.csv           counts, constraint checks, diagnostics, sensitivity
    knn_v8_recommendations.csv            top-N per scenario
    knn_v8_full_ranking.csv               every rankable candidate, incl. equivalent variants
    knn_v8_unrankable_candidates.csv      pass hard filters but cannot be ranked, with reason
    knn_v8_evaluation.md                  human-readable report
(custom mode writes the same files with prefix knn_v8_custom_*)

Methodological summary (details are written into the report):
  A. Hard filters: budget, min RAM, min storage (+ dedicated GPU for Gaming).
  B. Targets come from an explicit profile table. CPU/GPU levels are percentiles
     of a FIXED reference (whole catalog), never medians of the filtered pool.
     These percentiles are operational assumptions, NOT industry standards.
  C. KNN: StandardScaler (fitted on the reference catalog, not the filtered pool)
     + Euclidean distance. Surplus above target is down-weighted by sqrt(lambda)
     (asymmetric distance); shortfall keeps full weight.
  D. Missing benchmarks are never imputed. Candidates that pass the hard filters
     but lack a required score are reported, not silently dropped.
  E. Evaluation checks constraints, vector diversity, ties, and sensitivity.
     No accuracy claim is made: there is no ground truth in the dataset.
"""

from __future__ import annotations

import argparse
import re
import sys
import dataclasses
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DATASET_FILENAME = "laptops_benchmark_enriched_resolved.csv"


def detect_root() -> Path:
    """Find the project root: the folder that contains dataset/<csv>.

    The script lives in the project root, so its own folder is tried first, then
    its parent folder, then the current working directory. Falls back to the
    script's own folder. Outputs always go to <root>/inspection/.
    """
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent, Path.cwd().resolve()):
        if (cand / "dataset" / DATASET_FILENAME).exists():
            return cand
    return here


ROOT = detect_root()
DEFAULT_INPUT = ROOT / "dataset" / DATASET_FILENAME
OUT_DIR = ROOT / "inspection"

REQUIRED_COLUMNS = [
    "name", "url", "price_numeric", "ram_gb", "storage_gb",
    "processor_normalized", "graphics_normalized",
    "cpu_passmark_mark", "gpu_passmark_g3d_mark", "gpu_passmark_name",
]
NUMERIC_COLUMNS = ["price_numeric", "ram_gb", "storage_gb",
                   "cpu_passmark_mark", "gpu_passmark_g3d_mark"]

COL_RAM, COL_STORAGE = "ram_gb", "storage_gb"
COL_CPU, COL_GPU = "cpu_passmark_mark", "gpu_passmark_g3d_mark"

# ----------------------------------------------------------------------------
# Profile table (explicit design decisions; see evaluation report)
# ----------------------------------------------------------------------------
CPU_LEVELS = {"rendah": 0.25, "sedang": 0.50, "tinggi": 0.75, "sangat_tinggi": 0.90}
CPU_LEVEL_ORDER = list(CPU_LEVELS)

# level -> (reference population by GPU class, percentile over UNIQUE GPU MODELS)
GPU_LEVELS = {
    "igpu_memadai": ("integrated", 0.75),
    "dgpu_rendah": ("dedicated", 0.25),
    "dgpu_sedang": ("dedicated", 0.50),
    "dgpu_tinggi": ("dedicated", 0.75),
    "dgpu_sangat_tinggi": ("dedicated", 0.90),
}
GPU_LEVEL_ORDER = list(GPU_LEVELS)

PROFILES = {
    "office": dict(label="Office dan kuliah", ram_rec=8, storage_rec=512,
                   cpu_level="rendah", gpu_level=None,
                   gpu_in_features=False, require_dedicated=False),
    "programming": dict(label="Programming dan web development", ram_rec=16, storage_rec=512,
                        cpu_level="sedang", gpu_level=None,
                        gpu_in_features=False, require_dedicated=False),
    "design": dict(label="Desain grafis dan editing video", ram_rec=16, storage_rec=512,
                   cpu_level="tinggi", gpu_level="dgpu_sedang",
                   gpu_in_features=True, require_dedicated=False),
    "gaming": dict(label="Gaming", ram_rec=16, storage_rec=512,
                   cpu_level="tinggi", gpu_level="dgpu_sedang",
                   gpu_in_features=True, require_dedicated=True),
}

DEFAULT_LAMBDA = 0.25
DEFAULT_TOP_N = 5

# ----------------------------------------------------------------------------
# GPU classification rules (documented; unknown instead of guessing)
# ----------------------------------------------------------------------------
APPLE_CPU_RE = re.compile(r"^(A\d{1,2}|M\d)(\s+(Pro|Max|Ultra))?$", re.I)
DEDICATED_RE = re.compile(
    r"GeForce|\bRTX\b|\bGTX\b|Radeon\s+RX|Radeon\s+Pro|Quadro|\bArc\s+A\d{3}", re.I)
DEDICATED_NAME_TOKEN_RE = re.compile(r"\bRTX\b|\bGTX\b|Radeon\s+RX", re.I)
INTEL_IGPU_RE = re.compile(r"Intel\s+UHD|Iris\s+Xe|^Intel\s+Graphics$|Adreno", re.I)
ARC_VT_IGPU_RE = re.compile(
    r"^Intel\s+Arc\s+(Graphics\s+)?1\d{2}[VT](\s+Graphics|\s+GPU)?$", re.I)
RADEON_IGPU_RE = re.compile(
    r"^AMD\s+Radeon\s+(Graphic\s+)?(\d{3}M|8060S)(\s+Graphics)?$", re.I)
GENERIC_NAMES = {"amd radeon graphics", "intel arc graphics", "amd radeon"}
# Generic iGPU names are only trusted as 'integrated' when paired with a CPU family whose
# graphics are always on-package (AMD Ryzen/Athlon APUs; Intel Core Ultra). The product name
# must not mention a discrete GPU (checked earlier in classify_gpu).
AMD_APU_CPU_RE = re.compile(r"^AMD\s+(Ryzen|Athlon|Octa\s+Core\s+Ryzen)", re.I)
INTEL_ULTRA_CPU_RE = re.compile(r"^Intel\s+Core\s+Ultra", re.I)

# Cases checked against manufacturer / reliable sources during design.
VALIDATED_BY_SOURCE = {
    "Intel Arc B390": "Intel ARK + Notebookcheck: iGPU of Core Ultra X7 358H "
                      "(https://www.intel.com/content/www/us/en/products/sku/245527/)",
    "AMD Radeon 8060S": "Notebookcheck: iGPU of Ryzen AI Max+ 395 "
                        "(https://www.notebookcheck.net/AMD-Ryzen-AI-Max-395-Processor-Benchmarks-and-Specs.942323.0.html)",
    "NVIDIA GeForce MX570A 2GB": "Notebookcheck: MX570/MX570 A is a discrete mobile GPU (GA107, own 2 GB GDDR6) "
                                 "(https://www.notebookcheck.net/NVIDIA-GeForce-MX570-GPU-Benchmarks-and-Specs.586934.0.html)",
}
VALIDATED_BY_SOURCE.update({
    "Intel Graphics": "PassMark CPU page (Core Ultra 5 125U): 'Intel Graphics' is the CPU's integrated graphics "
                      "(https://www.cpubenchmark.net/compare/7383vs5840/AMD-Ryzen-AI-7-345-vs-Intel-Ultra-5-125U)",
    "Intel Iris Xe Graphics": "PassMark CPU page (i7-1165G7): Intel Iris Xe Graphics is integrated "
                              "(https://www.cpubenchmark.net/compare/4104vs3814/Apple-M1-8-3200-MHz-vs-Intel-i7-1165G7)",
    "AMD Radeon 840M Graphics": "PassMark: Ryzen AI 7 345 'with Radeon 840M' (integrated, part of the APU) "
                                "(https://www.cpubenchmark.net/compare/7383vs5840/AMD-Ryzen-AI-7-345-vs-Intel-Ultra-5-125U)",
    "AMD Radeon 680M Graphics": "Notebookcheck: Radeon 680M is the iGPU of Ryzen 7 7735HS "
                                "(https://www.notebookcheck.net/R7-7735HS-vs-R7-7730U_14952_14956.247596.0.html)",
    "AMD Radeon 680M": "Notebookcheck: Radeon 680M is the iGPU of Ryzen 7 7735HS "
                       "(https://www.notebookcheck.net/R7-7735HS-vs-R7-7730U_14952_14956.247596.0.html)",
    "AMD Radeon Graphic 680M": "Notebookcheck: Radeon 680M is the iGPU of Ryzen 7 7735HS "
                               "(https://www.notebookcheck.net/R7-7735HS-vs-R7-7730U_14952_14956.247596.0.html)",
})
# Validation of a whole rule (applies to every row classified by that basis).
VALIDATED_BASIS = {
    "rule_generic_intel_arc_with_core_ultra":
        "Notebookcheck (Ultra 5 125H vs Ultra 7 155H): Intel Arc 7/8-core iGPU on Core Ultra H-series "
        "(https://www.notebookcheck.com/Ultra-5-125H-vs-Ultra-7-155H_16911_16906.247552.0.html); "
        "PassMark CPU page lists 'Intel Arc graphics' for Core Ultra 5 125H; pair rule only, "
        "rows whose product name mentions a discrete GPU stay unknown",
    "rule_generic_radeon_with_ryzen_apu":
        "Notebookcheck: Ryzen 7 7735HS integrates Radeon 680M "
        "(https://www.notebookcheck.net/R7-7735HS-vs-R7-7730U_14952_14956.247596.0.html); "
        "PassMark lists Ryzen AI 7 345 'with Radeon 840M'; pair rule only (generic name + AMD Ryzen/Athlon CPU)",
}


def classify_gpu(name, graphics, processor) -> tuple[str, str]:
    """Return (gpu_class, basis). gpu_class in {dedicated, integrated, unknown}."""
    g = "" if pd.isna(graphics) else str(graphics).strip()
    n = "" if pd.isna(name) else str(name)
    cpu = "" if pd.isna(processor) else str(processor).strip()

    if g == "":
        if APPLE_CPU_RE.match(cpu):
            return "integrated", "apple_soc"
        return "unknown", "no_graphics_field"

    g_dedicated = bool(DEDICATED_RE.search(g))
    if not g_dedicated and DEDICATED_NAME_TOKEN_RE.search(n):
        return "unknown", "conflict_name_vs_graphics_field"
    if g_dedicated:
        return "dedicated", "rule_discrete_vendor_name"
    if g.lower() == "intel arc b390" and cpu.startswith("Intel Core Ultra X"):
        return "integrated", "pair_rule_validated_source"
    if INTEL_IGPU_RE.search(g):
        return "integrated", "rule_intel_uhd_iris_adreno"
    if ARC_VT_IGPU_RE.match(g):
        return "integrated", "rule_intel_arc_v_t_series"
    if RADEON_IGPU_RE.match(g):
        return "integrated", "rule_radeon_m_series_or_8060s"
    if g.lower() in ("amd radeon graphics", "amd radeon") and AMD_APU_CPU_RE.match(cpu):
        return "integrated", "rule_generic_radeon_with_ryzen_apu"
    if g.lower() == "intel arc graphics" and INTEL_ULTRA_CPU_RE.match(cpu):
        return "integrated", "rule_generic_intel_arc_with_core_ultra"
    if g.lower() in GENERIC_NAMES or g.lower().startswith("intel arc b"):
        return "unknown", "generic_or_ambiguous_name"
    return "unknown", "unmatched_by_rules"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
OVERRIDES_FILENAME = "cpu_mark_overrides.csv"
OVERRIDE_COLUMNS = ["processor_normalized", "action", "cpu_passmark_mark", "reason", "source"]


def apply_cpu_overrides(df: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """Apply documented CPU benchmark corrections WITHOUT touching the source data.

    dataset/cpu_mark_overrides.csv (optional) has one row per processor_normalized:
      action = exclude -> score treated as unavailable (NaN); the candidate is reported as
                          unrankable with the reason, never silently ranked on a suspect score.
      action = set     -> replace the score with the value in cpu_passmark_mark (needs a source).
    Nothing is imputed: 'set' only accepts a value you looked up and cited in `source`.
    """
    df["cpu_override_note"] = ""
    df.attrs["cpu_overrides_applied"] = []
    if path is None or not Path(path).exists():
        return df
    ov = pd.read_csv(path)
    missing = [c for c in OVERRIDE_COLUMNS if c not in ov.columns]
    if missing:
        raise ValueError(f"{Path(path).name}: kolom tidak ada: {missing}")
    for _, r in ov.iterrows():
        action = str(r["action"]).strip().lower()
        if action not in ("exclude", "set"):
            raise ValueError(f"{Path(path).name}: action harus 'exclude' atau 'set', bukan '{r['action']}'")
        mask = df["processor_normalized"] == r["processor_normalized"]
        n = int(mask.sum())
        val = None
        if action == "set":
            val = pd.to_numeric(r["cpu_passmark_mark"], errors="coerce")
            if pd.isna(val) or val <= 0 or pd.isna(r["source"]) or not str(r["source"]).strip():
                raise ValueError(f"{Path(path).name}: action 'set' memerlukan nilai > 0 dan kolom source terisi "
                                 f"({r['processor_normalized']}).")
            df.loc[mask, "cpu_passmark_mark"] = float(val)
        else:
            df.loc[mask, "cpu_passmark_mark"] = np.nan
        df.loc[mask, "cpu_override_note"] = f"{action}: {str(r['reason']).strip()}"
        df.attrs["cpu_overrides_applied"].append(dict(
            processor=str(r["processor_normalized"]), action=action, rows=n,
            value=None if val is None else float(val), reason=str(r["reason"]).strip(),
            source=str(r["source"]).strip()))
    return df


# ----------------------------------------------------------------------------
# Screen attributes parsed from the product NAME (the catalog has no screen columns)
# ----------------------------------------------------------------------------
RES_ORDER = {"HD": 1, "FHD": 2, "WUXGA": 3, "QHD/2K": 4, "2.5K-3K": 5, "4K": 6}
SCREEN_SIZE_RE = re.compile(r"(\d{2}(?:[.,]\d{1,2})?)\s*(?:\u2033|\"|\u201d|-?\s?inch|in\b)", re.I)


def parse_screen(name, is_apple: bool = False) -> dict:
    """Return screen size/resolution class/OLED/IPS/touch read from the product name.

    Positive evidence only: a missing OLED tag does not prove the panel is not OLED, and a
    missing resolution term leaves the resolution unknown (never guessed). Apple names carry no
    resolution term, so Apple resolution stays unknown.
    """
    out = dict(screen_inch=np.nan, screen_res=None, screen_res_rank=np.nan,
               screen_oled=False, screen_ips=False, screen_touch=False)
    text = "" if pd.isna(name) else str(name)
    # OLED is positive evidence anywhere in the name (model names such as 'Vivobook S14 OLED' count).
    out["screen_oled"] = bool(re.search(r"\bOLED\b", text, re.I))
    m = SCREEN_SIZE_RE.search(text)
    if not m:
        return out
    out["screen_inch"] = float(m.group(1).replace(",", "."))
    seg = text[m.end():].split("/")[0].upper()
    out["screen_ips"] = "IPS" in seg
    out["screen_touch"] = "TOUCH" in seg
    if is_apple:
        return out
    if re.search(r"\b4K\b|\bUHD\b|3840", seg):
        res = "4K"
    elif re.search(r"\b2\.[5-9]K\b|\b3K\b|\b3\.\dK\b|2880|3000", seg):
        res = "2.5K-3K"
    elif re.search(r"\bQHD\+?\b|\bWQHD\b|\b2K\b|WQXGA|2560", seg):
        res = "QHD/2K"
    elif re.search(r"WUXGA|1920\s*X\s*1200", seg):
        res = "WUXGA"
    elif re.search(r"\bFHD\b|1920\s*X\s*1080", seg):
        res = "FHD"
    elif re.search(r"\bHD\b|WXGA|1366", seg):
        res = "HD"
    else:
        res = None
    out["screen_res"] = res
    out["screen_res_rank"] = float(RES_ORDER[res]) if res else np.nan
    return out


def prepare_dataframe(df: pd.DataFrame, overrides_path=...) -> pd.DataFrame:
    """Validate columns, coerce numerics, add source_row and GPU class.

    Works on a DataFrame from any source (CSV or MySQL). Missing benchmark
    values stay NaN; nothing is imputed.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}\nKolom tersedia: {list(df.columns)}")
    df = df.copy()
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.reset_index(drop=True)
    df["source_row"] = df.index
    cls = [classify_gpu(n, g, p) for n, g, p in
           zip(df["name"], df["graphics_normalized"], df["processor_normalized"])]
    df["gpu_class"] = [c[0] for c in cls]
    df["gpu_class_basis"] = [c[1] for c in cls]
    scr = [parse_screen(n, bool(APPLE_CPU_RE.match(str(p).strip()))) for n, p in
           zip(df["name"], df["processor_normalized"])]
    for col in ("screen_inch", "screen_res", "screen_res_rank", "screen_oled", "screen_ips", "screen_touch"):
        df[col] = [r[col] for r in scr]
    if overrides_path is ...:
        overrides_path = ROOT / "dataset" / OVERRIDES_FILENAME
    return apply_cpu_overrides(df, overrides_path)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Jalankan dari root project yang memiliki folder dataset/, "
            "atau berikan --input <path-csv>.")
    return prepare_dataframe(pd.read_csv(path))


def build_reference(df: pd.DataFrame) -> dict:
    """Fixed reference targets (independent of any user filter)."""
    cpu_vals = df[COL_CPU].dropna().to_numpy(float)
    if len(cpu_vals) == 0:
        raise ValueError("Tidak ada skor CPU PassMark pada dataset.")
    ref = {"cpu": {lvl: float(np.quantile(cpu_vals, q)) for lvl, q in CPU_LEVELS.items()},
           "gpu": {}, "meta": []}
    ref["meta"].append(dict(kind="cpu", level="(populasi)", percentile="",
                            value="", n=len(cpu_vals),
                            population="semua laptop dengan skor CPU"))
    for lvl, q in CPU_LEVELS.items():
        ref["meta"].append(dict(kind="cpu", level=lvl, percentile=q,
                                value=ref["cpu"][lvl], n=len(cpu_vals),
                                population="semua laptop dengan skor CPU"))
    scored = df.dropna(subset=[COL_GPU, "gpu_passmark_name"])
    for lvl, (cls, q) in GPU_LEVELS.items():
        models = (scored[scored["gpu_class"] == cls]
                  .drop_duplicates(subset=["gpu_passmark_name"])[COL_GPU].to_numpy(float))
        if len(models) == 0:
            ref["gpu"][lvl] = float("nan")
        else:
            ref["gpu"][lvl] = float(np.quantile(models, q))
        ref["meta"].append(dict(kind="gpu", level=lvl, percentile=q,
                                value=ref["gpu"][lvl], n=len(models),
                                population=f"model GPU unik, kelas {cls}, berskor G3D"))
    return ref


# ----------------------------------------------------------------------------
# Scenario specification
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    scenario_id: str
    profile: str
    budget: float
    min_ram: float
    min_storage: float
    cpu_level: Optional[str] = None
    gpu_level: Optional[str] = None
    require_dedicated: Optional[bool] = None
    lam: float = DEFAULT_LAMBDA
    top_n: int = DEFAULT_TOP_N
    note: str = ""
    min_screen_inch: float = 0.0     # 0 = no screen-size filter
    min_resolution: str = ""         # "" = no resolution filter; else a key of RES_ORDER
    oled_only: bool = False          # only laptops whose name says OLED
    target_ram: Optional[float] = None
    target_storage: Optional[float] = None


def validate_scenario(s: Scenario) -> None:
    if s.profile not in PROFILES:
        raise ValueError(f"Profil tidak dikenal: {s.profile}. Pilihan: {list(PROFILES)}")
    if not (s.budget > 0):
        raise ValueError("Budget harus lebih besar dari 0.")
    if s.min_ram < 0 or s.min_storage < 0:
        raise ValueError("RAM dan storage minimum tidak boleh negatif.")
    if s.target_ram is not None:
        if s.target_ram < 0:
            raise ValueError("Target RAM tidak boleh negatif.")
        if s.target_ram < s.min_ram:
            raise ValueError(f"Target RAM ({s.target_ram:g} GB) tidak boleh lebih kecil dari RAM minimum ({s.min_ram:g} GB).")
    if s.target_storage is not None:
        if s.target_storage < 0:
            raise ValueError("Target storage tidak boleh negatif.")
        if s.target_storage < s.min_storage:
            raise ValueError(f"Target storage ({s.target_storage:g} GB) tidak boleh lebih kecil dari storage minimum ({s.min_storage:g} GB).")
    if s.cpu_level is not None and s.cpu_level not in CPU_LEVELS:
        raise ValueError(f"Level CPU tidak dikenal: {s.cpu_level}. Pilihan: {CPU_LEVEL_ORDER}")
    if s.gpu_level is not None and s.gpu_level not in GPU_LEVELS:
        raise ValueError(f"Level GPU tidak dikenal: {s.gpu_level}. Pilihan: {GPU_LEVEL_ORDER}")
    if not (0.0 <= s.lam <= 1.0):
        raise ValueError("lambda-surplus harus berada pada rentang 0..1.")
    if s.min_screen_inch < 0:
        raise ValueError("Ukuran layar minimum tidak boleh negatif.")
    if s.min_resolution and s.min_resolution not in RES_ORDER:
        raise ValueError(f"Resolusi minimum tidak dikenal: {s.min_resolution}. Pilihan: {list(RES_ORDER)}")
    if not (1 <= s.top_n <= 50):
        raise ValueError("top-n harus berada pada rentang 1..50.")
    if s.profile == "gaming" and s.require_dedicated is False:
        raise ValueError("Profil Gaming mewajibkan GPU dedicated (keputusan desain v8).")


# ----------------------------------------------------------------------------
# Core ranking
# ----------------------------------------------------------------------------
def knn_distances(cand: pd.DataFrame, features: list[str], targets: dict,
                  scaler: StandardScaler, lam: float) -> np.ndarray:
    """Euclidean KNN distance to the target in standardized space.

    Surplus above the target (positive standardized difference) is multiplied by
    sqrt(lam) so that lam=1 is the plain symmetric baseline and lam=0 ignores surplus.
    The scaler is fitted on the reference catalog, NOT on the filtered pool.
    """
    X = scaler.transform(cand[features].to_numpy(float))
    t = scaler.transform(np.array([[targets[f] for f in features]], dtype=float))
    diff = X - t
    adjusted = np.where(diff > 0, diff * np.sqrt(lam), diff)
    X_adj = t + adjusted
    nn = NearestNeighbors(n_neighbors=len(cand), metric="euclidean")
    nn.fit(X_adj)
    d, idx = nn.kneighbors(t)
    dist = np.empty(len(cand), dtype=float)
    dist[idx[0]] = d[0]
    return dist


def rank_groups(cand: pd.DataFrame, features: list[str], targets: dict,
                scaler: StandardScaler, lam: float):
    cand = cand.copy()
    cand["knn_distance"] = knn_distances(cand, features, targets, scaler, lam)
    # Tie policy: equal distance (to 1e-9) -> lower price -> original row order.
    cand["_dist_key"] = np.round(cand["knn_distance"], 9)
    cand = cand.sort_values(["_dist_key", "price_numeric", "source_row"],
                            kind="mergesort").reset_index(drop=True)
    cand["rank_all"] = np.arange(1, len(cand) + 1)
    # Identical feature vectors form one group (variants of an equivalent spec).
    cand["group_id"] = cand.groupby(features, sort=False).ngroup() + 1
    g = cand.groupby("group_id")
    cand["group_size"] = g["group_id"].transform("size")
    cand["group_price_min"] = g["price_numeric"].transform("min")
    cand["group_price_max"] = g["price_numeric"].transform("max")
    cand["group_gpu_classes"] = g["gpu_class"].transform(lambda s: "/".join(sorted(set(s))))
    cand["is_group_representative"] = ~cand.duplicated("group_id")
    for f in features:
        cand[f"below_target_{f}"] = cand[f] < (targets[f] - 1e-9)
    for f in features:
        gap = (targets[f] - cand[f]).clip(lower=0)
        cand[f"shortfall_{f}"] = gap
        cand[f"shortfall_pct_{f}"] = gap / targets[f] * 100.0
    cand["n_features_below_target"] = cand[[f"below_target_{f}" for f in features]].sum(axis=1)
    reps = cand[cand["is_group_representative"]].copy()
    reps["recommendation_rank"] = np.arange(1, len(reps) + 1)
    return cand, reps


def vector_set(reps: pd.DataFrame, features: list[str], n: int) -> set:
    return {tuple(float(v) for v in r) for r in reps.head(n)[features].to_numpy()}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def shifted(order: list[str], level: Optional[str], step: int) -> Optional[str]:
    if level is None or level not in order:
        return None
    i = order.index(level) + step
    return order[i] if 0 <= i < len(order) else None


def run_scenario(df: pd.DataFrame, ref: dict, s: Scenario) -> dict:
    validate_scenario(s)
    p = PROFILES[s.profile]
    cpu_level = s.cpu_level or p["cpu_level"]
    gpu_level = s.gpu_level or p["gpu_level"]
    use_gpu = p["gpu_in_features"] or (s.gpu_level is not None)
    if use_gpu and gpu_level is None:
        gpu_level = "dgpu_sedang"
    require_dedicated = p["require_dedicated"] if s.require_dedicated is None else s.require_dedicated

    features = [COL_RAM, COL_STORAGE, COL_CPU] + ([COL_GPU] if use_gpu else [])
    target_ram = float(s.target_ram) if s.target_ram is not None else float(max(s.min_ram, p["ram_rec"]))
    target_storage = float(s.target_storage) if s.target_storage is not None else float(max(s.min_storage, p["storage_rec"]))
    targets = {
        COL_RAM: target_ram,
        COL_STORAGE: target_storage,
        COL_CPU: ref["cpu"][cpu_level],
    }
    if use_gpu:
        targets[COL_GPU] = ref["gpu"][gpu_level]
        if not np.isfinite(targets[COL_GPU]):
            raise ValueError(f"Target GPU untuk level '{gpu_level}' tidak dapat dihitung dari dataset.")

    n_initial = len(df)
    # ---- A. hard filters -----------------------------------------------------
    basic = ((df["price_numeric"].notna() & (df["price_numeric"] <= s.budget)) &
             (df[COL_RAM].notna() & (df[COL_RAM] >= s.min_ram)) &
             (df[COL_STORAGE].notna() & (df[COL_STORAGE] >= s.min_storage)))
    pool_basic = df[basic].copy()
    class_counts = pool_basic["gpu_class"].value_counts().to_dict()

    # optional screen filters (user choices). Unknown attributes are never guessed: such laptops
    # are excluded AND reported with a reason.
    screen_active = s.min_screen_inch > 0 or bool(s.min_resolution) or s.oled_only
    screen_unknown = pool_basic.iloc[0:0].copy()
    n_screen_nomatch = 0
    pool_screen = pool_basic
    if screen_active:
        idx = pool_basic.index
        false = pd.Series(False, index=idx)
        size_fail = (pool_basic["screen_inch"].notna() & (pool_basic["screen_inch"] < s.min_screen_inch)
                     if s.min_screen_inch > 0 else false)
        size_unk = pool_basic["screen_inch"].isna() if s.min_screen_inch > 0 else false
        res_fail = (pool_basic["screen_res_rank"].notna() &
                    (pool_basic["screen_res_rank"] < RES_ORDER.get(s.min_resolution, 0))
                    if s.min_resolution else false)
        res_unk = pool_basic["screen_res_rank"].isna() if s.min_resolution else false
        oled_fail = ~pool_basic["screen_oled"].astype(bool) if s.oled_only else false
        definite_fail = size_fail | res_fail | oled_fail
        unknown_excl = ~definite_fail & (size_unk | res_unk)
        pool_screen = pool_basic[~definite_fail & ~unknown_excl].copy()
        n_screen_nomatch = int(definite_fail.sum())
        screen_unknown = pool_basic[unknown_excl].copy()
        if len(screen_unknown):
            reasons_s = []
            for i in screen_unknown.index:
                r_ = []
                if size_unk.loc[i]:
                    r_.append("ukuran layar tidak terbaca dari nama produk (filter ukuran aktif)")
                if res_unk.loc[i]:
                    r_.append("resolusi layar tidak tertulis pada nama produk (filter resolusi aktif)")
                reasons_s.append("; ".join(r_))
            screen_unknown["unrankable_reason"] = reasons_s

    if require_dedicated:
        pool = pool_screen[pool_screen["gpu_class"] == "dedicated"].copy()
        unconfirmed = pool_screen[pool_screen["gpu_class"] == "unknown"].copy()
        n_integrated_excluded = int((pool_screen["gpu_class"] == "integrated").sum())
    else:
        pool = pool_screen
        unconfirmed = pool_screen.iloc[0:0]
        n_integrated_excluded = 0

    # ---- D. missing data (no imputation) ----------------------------------------
    cpu_missing = pool[COL_CPU].isna()
    gpu_missing = pool[COL_GPU].isna() if use_gpu else pd.Series(False, index=pool.index)
    rankable_mask = ~(cpu_missing | gpu_missing)
    cand = pool[rankable_mask].copy()

    unr = pool[~rankable_mask].copy()
    reasons = []
    for idx in unr.index:
        r = []
        if cpu_missing.loc[idx]:
            note = str(unr.loc[idx, "cpu_override_note"]) if "cpu_override_note" in unr.columns else ""
            r.append("skor CPU PassMark tidak tersedia" + (f" ({note})" if note else ""))
        if gpu_missing.loc[idx]:
            r.append(f"skor GPU G3D tidak tersedia (kelas={unr.loc[idx, 'gpu_class']}, "
                     f"basis={unr.loc[idx, 'gpu_class_basis']})")
        reasons.append("; ".join(r))
    unr["unrankable_reason"] = reasons
    if len(unconfirmed):
        unconfirmed = unconfirmed.copy()
        unconfirmed["unrankable_reason"] = ("kelas GPU unknown; tidak dapat dipastikan dedicated "
                                            f"(basis={'/'.join(sorted(set(unconfirmed['gpu_class_basis'])))})")
        unr = pd.concat([unr, unconfirmed], ignore_index=True)
    if len(screen_unknown):
        unr = pd.concat([unr, screen_unknown], ignore_index=True)
    unr.insert(0, "scenario_id", s.scenario_id)

    summary = dict(
        scenario_id=s.scenario_id, profile=s.profile, profile_label=p["label"],
        note=s.note, budget_idr=s.budget, min_ram_gb=s.min_ram, min_storage_gb=s.min_storage,
        features="+".join(features), gpu_used_in_ranking=use_gpu,
        cpu_level=cpu_level, cpu_target=targets[COL_CPU],
        gpu_level=gpu_level if use_gpu else "", gpu_target=targets.get(COL_GPU, np.nan),
        ram_target=targets[COL_RAM], storage_target=targets[COL_STORAGE],
        require_dedicated=require_dedicated, lambda_surplus=s.lam, top_n=s.top_n,
        n_initial=n_initial, n_after_basic_filter=len(pool_basic),
        min_screen_inch=s.min_screen_inch, min_resolution=s.min_resolution, oled_only=s.oled_only,
        n_after_screen_filter=len(pool_screen), n_screen_excluded_nomatch=n_screen_nomatch,
        n_screen_excluded_unknown=len(screen_unknown),
        basic_dedicated=class_counts.get("dedicated", 0),
        basic_integrated=class_counts.get("integrated", 0),
        basic_unknown=class_counts.get("unknown", 0),
        n_after_gpu_requirement=len(pool),
        n_integrated_excluded_by_dedicated_rule=n_integrated_excluded,
        n_unknown_excluded_by_dedicated_rule=len(unconfirmed),
        n_rankable=len(cand),
        n_excluded_cpu_missing=int(cpu_missing.sum()),
        n_excluded_gpu_missing_cpu_present=int((gpu_missing & ~cpu_missing).sum()),
        n_excluded_gpu_missing_apple=int((gpu_missing & (pool["gpu_class_basis"] == "apple_soc")).sum()),
        n_unrankable_total=int((~rankable_mask).sum()),
    )

    if cand.empty:
        summary.update(n_recommendations=0, n_rankable_apple=0, n_unique_vectors_rankable=0,
                       checks_all_constraints_pass=True)
        return dict(summary=summary, recs=pd.DataFrame(), full=pd.DataFrame(),
                    unrankable=unr, targets=targets, features=features)

    # ---- C. KNN ranking -----------------------------------------------------------
    ref_rows = df.dropna(subset=features)
    if len(ref_rows) < 2:
        raise ValueError("Populasi referensi untuk StandardScaler terlalu kecil.")
    scaler = StandardScaler().fit(ref_rows[features].to_numpy(float))
    full, reps = rank_groups(cand, features, targets, scaler, s.lam)
    recs = reps.head(s.top_n).copy()

    # ---- E. evaluation --------------------------------------------------------------
    chk_budget = bool((recs["price_numeric"] <= s.budget).all())
    chk_ram = bool((recs[COL_RAM] >= s.min_ram).all())
    chk_storage = bool((recs[COL_STORAGE] >= s.min_storage).all())
    chk_ded = bool((recs["gpu_class"] == "dedicated").all()) if require_dedicated else True
    best = full["_dist_key"].min()
    tie_rows = full[full["_dist_key"] == best]
    summary.update(
        n_recommendations=len(recs),
        n_rankable_apple=int((full["gpu_class_basis"] == "apple_soc").sum()),
        n_unique_vectors_rankable=int(full["group_id"].nunique()),
        n_unique_vectors_in_recs=int(recs[features].drop_duplicates().shape[0]),
        recs_duplicate_vectors=int(len(recs) - recs[features].drop_duplicates().shape[0]),
        check_within_budget=chk_budget, check_min_ram=chk_ram, check_min_storage=chk_storage,
        check_dedicated_if_required=chk_ded,
        checks_all_constraints_pass=bool(chk_budget and chk_ram and chk_storage and chk_ded),
        best_distance=float(full["knn_distance"].min()),
        n_rows_tied_at_best_distance=int(len(tie_rows)),
        n_vectors_tied_at_best_distance=int(tie_rows["group_id"].nunique()),
        top_distance_min=float(recs["knn_distance"].min()),
        top_distance_max=float(recs["knn_distance"].max()),
        recs_with_any_feature_below_target=int((recs["n_features_below_target"] > 0).sum()),
        pool_pct_meeting_all_targets=float((full["n_features_below_target"] == 0).mean() * 100.0),
        pool_pct_gpu_meets_target=(float((~full[f"below_target_{COL_GPU}"]).mean() * 100.0)
                                   if use_gpu else float("nan")),
        recs_apple=int((recs["gpu_class_basis"] == "apple_soc").sum()),
        price_min_recs=float(recs["price_numeric"].min()),
        price_max_recs=float(recs["price_numeric"].max()),
    )

    # sensitivity (Jaccard overlap of top-N vector sets vs the baseline)
    base_set = vector_set(reps, features, s.top_n)
    for lam2, key in [(1.0, "sens_jaccard_lambda_1"), (0.0, "sens_jaccard_lambda_0")]:
        _, r2 = rank_groups(cand, features, targets, scaler, lam2)
        summary[key] = jaccard(base_set, vector_set(r2, features, s.top_n))
    for step, key in [(-1, "sens_jaccard_cpu_level_lower"), (1, "sens_jaccard_cpu_level_higher")]:
        lv = shifted(CPU_LEVEL_ORDER, cpu_level, step)
        if lv is None:
            summary[key] = float("nan")
            continue
        t2 = dict(targets); t2[COL_CPU] = ref["cpu"][lv]
        _, r2 = rank_groups(cand, features, t2, scaler, s.lam)
        summary[key] = jaccard(base_set, vector_set(r2, features, s.top_n))
    for step, key in [(-1, "sens_jaccard_gpu_level_lower"), (1, "sens_jaccard_gpu_level_higher")]:
        lv = shifted(GPU_LEVEL_ORDER, gpu_level, step) if use_gpu else None
        if lv is None or not np.isfinite(ref["gpu"].get(lv, np.nan)):
            summary[key] = float("nan")
            continue
        t2 = dict(targets); t2[COL_GPU] = ref["gpu"][lv]
        _, r2 = rank_groups(cand, features, t2, scaler, s.lam)
        summary[key] = jaccard(base_set, vector_set(r2, features, s.top_n))

    # decorate output frames
    for frame in (recs, full):
        frame.insert(0, "scenario_id", s.scenario_id)
        frame.insert(1, "profile", s.profile)
        frame["ram_target"] = targets[COL_RAM]
        frame["storage_target"] = targets[COL_STORAGE]
        frame["cpu_target"] = targets[COL_CPU]
        frame["gpu_target"] = targets.get(COL_GPU, np.nan)
        frame["gpu_used_in_ranking"] = use_gpu
    recs["equivalent_variants_hidden"] = recs["group_size"] - 1
    return dict(summary=summary, recs=recs, full=full, unrankable=unr,
                targets=targets, features=features)


# ----------------------------------------------------------------------------
# Similar laptops (item-to-item KNN)
# ----------------------------------------------------------------------------
def similar_laptops(df: pd.DataFrame, ref: dict, s: Scenario, source_row: int, top_n: int = 5) -> dict:
    """Laptops whose profile features are closest to those of ONE given laptop.

    Differences from the profile ranking: the query is the laptop's own feature vector (not the
    profile target), the distance is the plain symmetric Euclidean distance (similarity, not target
    attainment), and the BUDGET is deliberately not enforced. All other hard filters of the scenario
    (minimum RAM/storage, dedicated GPU if the profile requires it, optional screen filters) still
    apply because the scenario is re-run through run_scenario with only the budget relaxed.
    Laptops with a feature vector identical to the source are reported as variants, not as neighbours.
    """
    relaxed = dataclasses.replace(s, budget=1e12, top_n=1)
    res = run_scenario(df, ref, relaxed)
    features, full = res["features"], res["full"]
    src_rows = df[df["source_row"] == source_row]
    out = dict(features=features, source=None, items=pd.DataFrame(), identical=pd.DataFrame(),
               n_pool=0, message="")
    if src_rows.empty:
        out["message"] = "Laptop acuan tidak ditemukan pada katalog."
        return out
    src = src_rows.iloc[0]
    out["source"] = src
    missing = [f for f in features if pd.isna(src[f])]
    if missing:
        out["message"] = ("Laptop acuan tidak memiliki skor benchmark yang dibutuhkan profil ini, "
                          "sehingga kemiripan tidak dapat dihitung.")
        return out
    cand = full[full["source_row"] != source_row].copy()
    src_vec = np.array([float(src[f]) for f in features])
    same = (cand[features].astype(float).to_numpy() == src_vec).all(axis=1) if len(cand) else np.array([], bool)
    out["identical"] = cand[same].copy()
    cand = cand[~same].copy()
    out["n_pool"] = int(len(cand))
    if cand.empty:
        out["message"] = "Tidak ada laptop lain yang lolos filter wajib untuk dibandingkan."
        return out
    ref_rows = df.dropna(subset=features)
    scaler = StandardScaler().fit(ref_rows[features].to_numpy(float))
    X = scaler.transform(cand[features].to_numpy(float))
    q = scaler.transform(src_vec.reshape(1, -1))
    nn = NearestNeighbors(n_neighbors=len(cand), metric="euclidean").fit(X)
    d, idx = nn.kneighbors(q)
    dist = np.empty(len(cand)); dist[idx[0]] = d[0]
    cand["sim_distance"] = dist
    cand["_dist_key"] = np.round(dist, 9)
    cand = cand.sort_values(["_dist_key", "price_numeric", "source_row"], kind="mergesort").reset_index(drop=True)
    cand["group_id"] = cand.groupby(features, sort=False).ngroup() + 1
    g = cand.groupby("group_id")
    cand["group_size"] = g["group_id"].transform("size")
    cand["group_price_min"] = g["price_numeric"].transform("min")
    cand["group_price_max"] = g["price_numeric"].transform("max")
    reps = cand[~cand.duplicated("group_id")].head(top_n).copy()
    reps["similar_rank"] = np.arange(1, len(reps) + 1)
    out["items"] = reps
    return out


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
REC_COLUMNS = [
    "scenario_id", "profile", "recommendation_rank", "name", "url", "price_numeric",
    "ram_gb", "storage_gb", "cpu_passmark_mark", "gpu_passmark_g3d_mark",
    "gpu_used_in_ranking", "gpu_class", "gpu_class_basis", "processor_normalized",
    "graphics_normalized", "knn_distance", "ram_target", "storage_target", "cpu_target",
    "gpu_target", "n_features_below_target", "equivalent_variants_hidden",
    "group_price_min", "group_price_max", "group_gpu_classes",
]
FULL_COLUMNS = [
    "scenario_id", "profile", "rank_all", "group_id", "is_group_representative",
    "group_size", "name", "url", "price_numeric", "ram_gb", "storage_gb",
    "cpu_passmark_mark", "gpu_passmark_g3d_mark", "gpu_class", "gpu_class_basis",
    "knn_distance", "n_features_below_target",
    "screen_inch", "screen_res", "screen_oled", "screen_ips", "screen_touch",
]
UNR_COLUMNS = [
    "scenario_id", "name", "url", "price_numeric", "ram_gb", "storage_gb",
    "processor_normalized", "graphics_normalized", "gpu_class", "gpu_class_basis",
    "cpu_passmark_mark", "gpu_passmark_g3d_mark", "unrankable_reason",
]


def keep_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=cols)
    extra = [c for c in df.columns if c.startswith(("below_target_", "shortfall_"))]
    return df[[c for c in cols if c in df.columns] + extra]


def gpu_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["graphics_label"] = work["graphics_normalized"].fillna("(kosong)")
    rows = []
    for (label, cls, basis), g in work.groupby(["graphics_label", "gpu_class", "gpu_class_basis"]):
        scored = g[COL_GPU].dropna()
        rows.append(dict(
            graphics_normalized=label, gpu_class=cls, gpu_class_basis=basis,
            n_laptops=len(g), n_with_g3d=len(scored),
            g3d_min=scored.min() if len(scored) else np.nan,
            g3d_max=scored.max() if len(scored) else np.nan,
            example_processors="; ".join(sorted(set(g["processor_normalized"].astype(str)))[:3]),
            validation_status=("divalidasi sumber: " + VALIDATED_BY_SOURCE[label])
            if label in VALIDATED_BY_SOURCE else (
                ("divalidasi sumber (aturan pasangan): " + VALIDATED_BASIS[basis])
                if basis in VALIDATED_BASIS else "BELUM divalidasi manual"),
        ))
    return pd.DataFrame(rows).sort_values(["gpu_class", "n_laptops"], ascending=[True, False])


def md_table(rows: list[dict], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return out


SHORT = {COL_RAM: "RAM", COL_STORAGE: "storage", COL_CPU: "CPU", COL_GPU: "GPU"}


def shortfall_text(row, features) -> str:
    parts = []
    for f in features:
        pct = row.get(f"shortfall_pct_{f}", 0.0)
        if pd.notna(pct) and pct > 1e-9:
            parts.append(f"{SHORT[f]} -{pct:.0f}%")
    return ", ".join(parts) if parts else "-"


def idr(v) -> str:
    return f"Rp{float(v):,.0f}".replace(",", ".")


def fnum(v, nd=0) -> str:
    return "n/a" if pd.isna(v) else f"{float(v):,.{nd}f}".replace(",", ".")


def build_report(results: list[dict], ref: dict, df: pd.DataFrame, input_path: Path,
                 audit: pd.DataFrame, title: str) -> str:
    L = [f"# {title}", "",
         "Laporan dibuat otomatis oleh `knn_v8_profile_recommender.py`. "
         "Seluruh angka berasal dari eksekusi script terhadap file input di bawah ini.", "",
         f"- Input: `{input_path.name}` ({len(df)} baris)",
         f"- Klasifikasi GPU: dedicated={int((df.gpu_class == 'dedicated').sum())}, "
         f"integrated={int((df.gpu_class == 'integrated').sum())}, "
         f"unknown={int((df.gpu_class == 'unknown').sum())} "
         "(aturan terdokumentasi; belum seluruhnya divalidasi manual, lihat "
         "`knn_v8_gpu_classification_audit.csv`)", "",
         "## 1. Metodologi singkat", "",
         "1. **Hard filter**: harga <= budget, RAM >= minimum, storage >= minimum; "
         "Gaming juga mewajibkan `gpu_class = dedicated`.",
         "2. **Target**: tabel profil eksplisit. RAM/storage target = max(minimum pengguna, rekomendasi profil). "
         "Target CPU/GPU = persentil referensi **tetap** (bukan median pool terfilter).",
         "3. **KNN**: StandardScaler dipasang pada seluruh katalog (populasi referensi), "
         "jarak Euclidean ke target; kelebihan di atas target dikalikan sqrt(lambda), "
         "kekurangan berbobot penuh. Harga bukan fitur; harga hanya batas, pemutus seri, dan kolom tampilan.",
         "4. **Seri/duplikasi**: jarak sama (1e-9) -> harga terendah. Laptop dengan vektor fitur identik "
         "dikelompokkan; satu perwakilan (termurah) ditampilkan, varian lain tetap ada di `knn_v8_full_ranking.csv`.",
         "5. **Data hilang**: tidak ada imputasi. Kandidat yang lolos hard filter tetapi tidak punya skor yang dibutuhkan "
         "dicatat di `knn_v8_unrankable_candidates.csv`.", "",
         "## 2. Target referensi (asumsi operasional, bukan standar industri)", ""]
    meta_rows = [dict(jenis=m["kind"], level=m["level"], persentil=m["percentile"],
                      nilai=fnum(m["value"]) if m["value"] != "" else "",
                      n=m["n"], populasi=m["population"]) for m in ref["meta"]]
    L += md_table(meta_rows, ["jenis", "level", "persentil", "nilai", "n", "populasi"])
    L += ["", "Catatan: persentil bersifat relatif terhadap katalog Plaza IT pada tanggal data ini dan "
          "akan berubah bila katalog berubah. Level GPU dihitung atas model GPU unik agar tidak ditentukan stok toko.", ""]

    applied = df.attrs.get("cpu_overrides_applied", []) if hasattr(df, "attrs") else []
    if applied:
        L += ["### Koreksi data CPU yang diterapkan (dataset asli tidak diubah)", ""]
        L += md_table([dict(prosesor=a["processor"], tindakan=a["action"], baris=a["rows"],
                            nilai=fnum(a["value"]) if a["value"] is not None else "-", alasan=a["reason"])
                       for a in applied], ["prosesor", "tindakan", "baris", "nilai", "alasan"])
        L += ["", "Prosesor dengan tindakan `exclude` dicatat sebagai tidak dapat diranking; skornya tidak diganti dengan perkiraan.", ""]
    L += ["## 3. Jumlah kandidat per skenario", "",
          "Skenario bersifat ilustrasi eksplorasi; bukan hasil rekomendasi final.", ""]
    rows = []
    for r in results:
        sm = r["summary"]
        rows.append(dict(
            skenario=sm["scenario_id"], budget=idr(sm["budget_idr"]),
            RAM_min=sm["min_ram_gb"], storage_min=sm["min_storage_gb"],
            awal=sm["n_initial"], lolos_filter_dasar=sm["n_after_basic_filter"],
            lolos_syarat_GPU=sm["n_after_gpu_requirement"], dapat_diranking=sm["n_rankable"],
            keluar_CPU_hilang=sm["n_excluded_cpu_missing"],
            keluar_GPU_hilang=sm["n_excluded_gpu_missing_cpu_present"],
            tidak_terkonfirmasi_dedicated=sm["n_unknown_excluded_by_dedicated_rule"],
            rekomendasi=sm["n_recommendations"]))
    L += md_table(rows, list(rows[0].keys()))
    L += ["", "Klasifikasi GPU pada kandidat yang lolos filter dasar:", ""]
    rows = [dict(skenario=r["summary"]["scenario_id"], dedicated=r["summary"]["basic_dedicated"],
                 integrated=r["summary"]["basic_integrated"], unknown=r["summary"]["basic_unknown"],
                 fitur=r["summary"]["features"]) for r in results]
    L += md_table(rows, list(rows[0].keys()))

    L += ["", "## 4. Pemeriksaan constraint dan keragaman", ""]
    rows = []
    for r in results:
        sm = r["summary"]
        if sm["n_recommendations"] == 0:
            rows.append(dict(skenario=sm["scenario_id"], hasil="tidak ada rekomendasi"))
            continue
        rows.append(dict(
            skenario=sm["scenario_id"], rekomendasi=sm["n_recommendations"],
            budget_ok=sm["check_within_budget"], RAM_ok=sm["check_min_ram"],
            storage_ok=sm["check_min_storage"], dedicated_ok=sm["check_dedicated_if_required"],
            vektor_unik_pool=sm["n_unique_vectors_rankable"],
            vektor_duplikat_di_rekomendasi=sm["recs_duplicate_vectors"],
            seri_pada_jarak_terbaik=f"{sm['n_rows_tied_at_best_distance']} baris / "
                                    f"{sm['n_vectors_tied_at_best_distance']} vektor",
            rekomendasi_dgn_fitur_di_bawah_target=sm["recs_with_any_feature_below_target"],
            pool_mencapai_semua_target=f"{sm['pool_pct_meeting_all_targets']:.1f}%",
            pool_GPU_mencapai_target=("n/a" if pd.isna(sm["pool_pct_gpu_meets_target"])
                                      else f"{sm['pool_pct_gpu_meets_target']:.1f}%")))
    L += md_table(rows, list(rows[0].keys()) if len(rows) else [])
    L += ["", "Catatan membaca jarak: jarak adalah ukuran kedekatan terhadap target, bukan mutu rekomendasi. "
          "Bila `pool_mencapai_semua_target` rendah atau 0%, target berada di luar jangkauan pool dan jarak yang besar "
          "tidak berarti rekomendasinya buruk; ranking menjadi 'terdekat dari bawah'. "
          "Gunakan kolom `kekurangan_terhadap_target` (persen di bawah target per fitur) untuk antarmuka pengguna.", "",
          "## 5. Sensitivitas (tumpang tindih Jaccard himpunan vektor top-N terhadap baseline)", "",
          "Nilai 1,00 = himpunan identik; nilai rendah = ranking sensitif terhadap asumsi tersebut.", ""]
    rows = []
    for r in results:
        sm = r["summary"]
        if sm["n_recommendations"] == 0:
            continue
        rows.append(dict(
            skenario=sm["scenario_id"],
            lambda_1=fnum(sm["sens_jaccard_lambda_1"], 2), lambda_0=fnum(sm["sens_jaccard_lambda_0"], 2),
            CPU_level_turun=fnum(sm["sens_jaccard_cpu_level_lower"], 2),
            CPU_level_naik=fnum(sm["sens_jaccard_cpu_level_higher"], 2),
            GPU_level_turun=fnum(sm["sens_jaccard_gpu_level_lower"], 2),
            GPU_level_naik=fnum(sm["sens_jaccard_gpu_level_higher"], 2)))
    if rows:
        L += md_table(rows, list(rows[0].keys()))

    L += ["", "## 6. Contoh audit ranking per skenario", ""]
    for r in results:
        sm = r["summary"]
        L += [f"### {sm['scenario_id']} - {sm['profile_label']}", "",
              f"- Fitur ranking: {sm['features']}; lambda={sm['lambda_surplus']}",
              f"- Target: RAM {fnum(sm['ram_target'])} GB, storage {fnum(sm['storage_target'])} GB, "
              f"CPU {fnum(sm['cpu_target'])} (level {sm['cpu_level']})"
              + (f", GPU G3D {fnum(sm['gpu_target'])} (level {sm['gpu_level']})" if sm["gpu_used_in_ranking"] else
                 ", GPU tidak dipakai dalam ranking"), ""]
        recs = r["recs"]
        if recs.empty:
            L += ["Tidak ada rekomendasi (tidak ada kandidat yang dapat diranking).", ""]
            continue
        rows = []
        for _, x in recs.iterrows():
            rows.append(dict(
                rank=int(x["recommendation_rank"]), produk=str(x["name"])[:60],
                harga=idr(x["price_numeric"]), RAM=fnum(x["ram_gb"]), storage=fnum(x["storage_gb"]),
                CPU=fnum(x["cpu_passmark_mark"]),
                GPU_G3D=f"{fnum(x['gpu_passmark_g3d_mark'])} ({x['gpu_class']})",
                jarak=f"{x['knn_distance']:.4f}", fitur_di_bawah_target=int(x["n_features_below_target"]),
                kekurangan_terhadap_target=shortfall_text(x, r["features"]),
                varian_setara=int(x["equivalent_variants_hidden"])))
        L += md_table(rows, list(rows[0].keys())) + [""]

    L += ["## 7. Keterbatasan (wajib dibaca)", "",
          "- Tidak ada ground truth, penilaian ahli, atau evaluasi pengguna: laporan ini **tidak** membuktikan akurasi rekomendasi. "
          "Jarak kecil hanya berarti dekat dengan target numerik.",
          "- Default Gaming diubah dari dgpu_tinggi ke dgpu_sedang setelah audit menunjukkan target tinggi tidak tercapai oleh kandidat mana pun pada budget uji; pilihan ini tetap asumsi.",
          "- Persentil CPU/GPU, level default per profil, rekomendasi RAM/storage, dan lambda adalah asumsi operasional yang perlu diuji.",
          "- Office dan Programming memakai fitur yang sama dan hanya berbeda pada target; perbedaan itu belum tervalidasi.",
          "- Skor G3D bersifat per nama GPU dan sering generik (mis. satu skor untuk semua 'Intel Graphics'), sehingga "
          "tidak membedakan iGPU dengan baik; karena itu GPU tidak dipakai pada Office/Programming secara default.",
          "- Jenis GPU (dedicated/integrated) dan tingkat performa (G3D) adalah dua hal berbeda "
          "(contoh: Radeon 8060S adalah iGPU dengan G3D tinggi). Kelas `unknown` tidak ditebak.",
          "- Laptop Apple tanpa skor CPU/GPU tidak dapat diranking dan dicatat di `knn_v8_unrankable_candidates.csv`. "
          "Kesebandingan PassMark CPU Apple dan x86 adalah asumsi. Skor CPU M1 dan M3 identik pada dataset; perlu dicek manual.",
          "- Pengelompokan vektor identik menyembunyikan perbedaan yang tidak ada di fitur (layar, baterai, garansi, merek).",
          "- Dataset adalah snapshot katalog satu toko; hasil tidak berlaku umum.", ""]
    return "\n".join(L)


def write_outputs(prefix: str, results: list[dict], ref: dict, df: pd.DataFrame,
                  input_path: Path, title: str, audit: pd.DataFrame) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["audit"] = OUT_DIR / "knn_v8_gpu_classification_audit.csv"
    audit.to_csv(paths["audit"], index=False)
    paths["ref"] = OUT_DIR / "knn_v8_reference_targets.csv"
    pd.DataFrame(ref["meta"]).to_csv(paths["ref"], index=False)

    paths["summary"] = OUT_DIR / f"{prefix}_scenario_summary.csv"
    pd.DataFrame([r["summary"] for r in results]).to_csv(paths["summary"], index=False)

    def cat(key, cols):
        frames = [keep_cols(r[key], cols) for r in results if not r[key].empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)

    paths["recs"] = OUT_DIR / f"{prefix}_recommendations.csv"
    cat("recs", REC_COLUMNS).to_csv(paths["recs"], index=False)
    paths["full"] = OUT_DIR / f"{prefix}_full_ranking.csv"
    cat("full", FULL_COLUMNS).to_csv(paths["full"], index=False)
    paths["unr"] = OUT_DIR / f"{prefix}_unrankable_candidates.csv"
    cat("unrankable", UNR_COLUMNS).to_csv(paths["unr"], index=False)
    paths["md"] = OUT_DIR / f"{prefix}_evaluation.md"
    paths["md"].write_text(build_report(results, ref, df, input_path, audit, title), encoding="utf-8")
    return list(paths.values())


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
SUITE = [
    Scenario("office_A", "office", 10_000_000, 8, 256),
    Scenario("office_B", "office", 15_000_000, 8, 512),
    Scenario("programming_A", "programming", 14_999_994, 16, 512,
             note="budget/RAM/storage sama dengan profil uji v7"),
    Scenario("programming_B", "programming", 20_000_000, 16, 512),
    Scenario("design_A", "design", 25_000_000, 16, 512),
    Scenario("design_B", "design", 35_000_000, 32, 1024),
    Scenario("gaming_A", "gaming", 20_000_000, 16, 512),
    Scenario("gaming_B", "gaming", 30_000_000, 16, 1024),
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="KNN v8: rekomendasi laptop berbasis profil kebutuhan. "
                    "Tanpa --profile: jalankan 8 skenario evaluasi.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--profile", choices=list(PROFILES))
    ap.add_argument("--budget", type=float, help="budget maksimum (IDR)")
    ap.add_argument("--min-ram", type=float, help="RAM minimum (GB)")
    ap.add_argument("--min-storage", type=float, help="storage minimum (GB)")
    ap.add_argument("--cpu-level", choices=CPU_LEVEL_ORDER)
    ap.add_argument("--gpu-level", choices=GPU_LEVEL_ORDER,
                    help="level GPU; pada Office/Programming opsi ini memasukkan GPU ke ranking")
    ap.add_argument("--require-dedicated", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--lambda-surplus", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--target-ram", type=float, default=None, help="target RAM acuan KNN (GB)")
    ap.add_argument("--target-storage", type=float, default=None, help="target storage acuan KNN (GB)")
    a = ap.parse_args(argv)
    if a.profile:
        for flag, val in [("--budget", a.budget), ("--min-ram", a.min_ram), ("--min-storage", a.min_storage)]:
            if val is None:
                ap.error(f"{flag} wajib diisi bila --profile digunakan.")
    return a


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args(argv)
    try:
        df = load_dataset(args.input)
        ref = build_reference(df)
        audit = gpu_audit_table(df)
        if args.profile:
            scenarios = [Scenario("custom_" + args.profile, args.profile, args.budget, args.min_ram,
                                  args.min_storage, args.cpu_level, args.gpu_level,
                                  args.require_dedicated, args.lambda_surplus, args.top_n,
                                  target_ram=args.target_ram, target_storage=args.target_storage)]
            prefix, title = "knn_v8_custom", "KNN v8 - Query Kustom"
        else:
            scenarios = SUITE
            prefix, title = "knn_v8", "KNN v8 - Evaluasi Profil Kebutuhan (skenario ilustrasi)"
        results = [run_scenario(df, ref, s) for s in scenarios]
        paths = write_outputs(prefix, results, ref, df, args.input, title, audit)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("=== KNN v8: rekomendasi berbasis profil ===")
    print(f"Input rows: {len(df)}")
    for r in results:
        sm = r["summary"]
        print(f"\n[{sm['scenario_id']}] {sm['profile_label']} | fitur: {sm['features']}")
        print(f"  awal={sm['n_initial']} | lolos filter dasar={sm['n_after_basic_filter']} | "
              f"lolos syarat GPU={sm['n_after_gpu_requirement']} | dapat diranking={sm['n_rankable']} | "
              f"keluar (CPU hilang/GPU hilang)={sm['n_excluded_cpu_missing']}/"
              f"{sm['n_excluded_gpu_missing_cpu_present']} | rekomendasi={sm['n_recommendations']}")
        if sm["n_recommendations"] == 0:
            print("  Tidak ada rekomendasi: tidak ada kandidat yang lolos filter dan dapat diranking.")
            continue
        print(f"  Constraint semua lolos: {sm['checks_all_constraints_pass']} | "
              f"vektor duplikat di rekomendasi: {sm['recs_duplicate_vectors']}")
        if args.profile:
            for _, x in r["recs"].iterrows():
                print(f"  {int(x['recommendation_rank'])}. {str(x['name'])[:70]} | {idr(x['price_numeric'])} | "
                      f"RAM {fnum(x['ram_gb'])} | SSD {fnum(x['storage_gb'])} | CPU {fnum(x['cpu_passmark_mark'])} | "
                      f"G3D {fnum(x['gpu_passmark_g3d_mark'])} ({x['gpu_class']}) | jarak {x['knn_distance']:.4f} | "
                      f"varian setara {int(x['equivalent_variants_hidden'])}")
    print("\nFile output:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
