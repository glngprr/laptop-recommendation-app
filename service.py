from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knn_v8_profile_recommender as v8  # noqa: E402

PROFILE_ORDER = list(v8.PROFILES)
RAM_CHOICES = [4, 8, 12, 16, 24, 32, 64]
STORAGE_CHOICES = [128, 256, 512, 1024, 2048]
BUDGET_MIN, BUDGET_MAX, BUDGET_STEP = 1_000_000, 200_000_000, 500_000
TOP_N = 5
MAX_UNRANKABLE_SHOWN = 200

CPU_LABELS = {"rendah": "Rendah", "sedang": "Sedang", "tinggi": "Tinggi", "sangat_tinggi": "Sangat tinggi"}
GPU_LABELS = {"igpu_memadai": "GPU terintegrasi memadai", "dgpu_rendah": "GPU dedicated - rendah",
              "dgpu_sedang": "GPU dedicated - sedang", "dgpu_tinggi": "GPU dedicated - tinggi",
              "dgpu_sangat_tinggi": "GPU dedicated - sangat tinggi"}
GPU_CLASS_LABELS = {"dedicated": "GPU dedicated", "integrated": "GPU terintegrasi",
                    "unknown": "jenis GPU belum dapat dipastikan"}
FEATURE_LABELS = {v8.COL_RAM: "RAM", v8.COL_STORAGE: "Storage", v8.COL_CPU: "CPU", v8.COL_GPU: "GPU"}
BOOL_TRUE = {"on", "true", "1", "yes", "ya"}
SCREEN_INCH_CHOICES = [0, 13, 14, 15, 16]                       # 0 = tanpa batas
RES_CHOICES = [k for k in v8.RES_ORDER if k != "HD"]            # urut dari rendah ke tinggi


def fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{float(v):,.0f}".replace(",", ".")


def fmt_idr(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else "Rp" + fmt_num(v)


def prepare_catalog(raw: pd.DataFrame):
    """Raw catalog (from CSV or MySQL) -> (prepared DataFrame, reference targets)."""
    df = v8.prepare_dataframe(raw)
    return df, v8.build_reference(df)


def profile_defaults() -> dict:
    out = {}
    for key, p in v8.PROFILES.items():
        out[key] = dict(
            label=p["label"], ram=p["ram_rec"], target_ram=p["ram_rec"], min_ram=p["ram_rec"],
            storage=p["storage_rec"], target_storage=p["storage_rec"], min_storage=p["storage_rec"],
            cpu=p["cpu_level"],
            gpu=p["gpu_level"] if p["gpu_in_features"] else "none",
            gpu_required=bool(p["gpu_in_features"]), dedicated=bool(p["require_dedicated"]),
            dedicated_locked=(key == "gaming"))
    return out


def cpu_options(ref: dict) -> list:
    return [(k, f"{CPU_LABELS[k]} (target PassMark sekitar {fmt_num(ref['cpu'][k])})")
            for k in v8.CPU_LEVEL_ORDER]


def gpu_options(ref: dict) -> list:
    opts = [("none", "Tidak diperlukan (GPU tidak dipakai)")]
    opts += [(k, f"{GPU_LABELS[k]} (target G3D sekitar {fmt_num(ref['gpu'][k])})")
             for k in v8.GPU_LEVEL_ORDER if np.isfinite(ref["gpu"].get(k, np.nan))]
    return opts


def make_scenario(params: dict) -> "v8.Scenario":
    """Validate user input and build a v8 Scenario. Raises ValueError with all problems."""
    errors = []
    profile = str(params.get("profile", ""))
    if profile not in v8.PROFILES:
        raise ValueError("Profil tidak dikenal.")
    p = v8.PROFILES[profile]
    d = profile_defaults()[profile]

    try:
        budget = float(params.get("budget"))
        if not (BUDGET_MIN <= budget <= BUDGET_MAX):
            errors.append(f"Budget harus antara {fmt_idr(BUDGET_MIN)} dan {fmt_idr(BUDGET_MAX)}.")
    except (TypeError, ValueError):
        errors.append("Budget tidak valid.")
        budget = None

    def pick(key, label, choices, default):
        try:
            val = int(float(params.get(key, default)))
        except (TypeError, ValueError):
            errors.append(f"{label} tidak valid.")
            return default
        if val not in choices:
            errors.append(f"{label} harus salah satu dari {choices}.")
        return val

    target_ram_raw = params.get("target_ram")
    min_ram_raw = params.get("min_ram")
    if target_ram_raw is not None:
        target_ram = pick("target_ram", "Target RAM", RAM_CHOICES, d["ram"])
    else:
        target_ram = None

    if min_ram_raw is not None:
        min_ram = pick("min_ram", "RAM minimum", RAM_CHOICES, target_ram if target_ram is not None else d["ram"])
    else:
        min_ram = target_ram if target_ram is not None else d["ram"]

    if target_ram is not None and min_ram > target_ram:
        errors.append(f"Target RAM ({target_ram} GB) tidak boleh lebih kecil dari RAM minimum ({min_ram} GB).")

    target_storage_raw = params.get("target_storage")
    min_storage_raw = params.get("min_storage")
    if target_storage_raw is not None:
        target_storage = pick("target_storage", "Target storage", STORAGE_CHOICES, d["storage"])
    else:
        target_storage = None

    if min_storage_raw is not None:
        min_storage = pick("min_storage", "Storage minimum", STORAGE_CHOICES, target_storage if target_storage is not None else d["storage"])
    else:
        min_storage = target_storage if target_storage is not None else d["storage"]

    if target_storage is not None and min_storage > target_storage:
        errors.append(f"Target storage ({target_storage} GB) tidak boleh lebih kecil dari storage minimum ({min_storage} GB).")

    cpu_level = str(params.get("cpu_level") or d["cpu"])
    if cpu_level not in v8.CPU_LEVELS:
        errors.append("Level performa CPU tidak dikenal.")

    gpu_raw = str(params.get("gpu_level") or "")
    gpu_level = None
    if p["gpu_in_features"]:
        gpu_raw = gpu_raw or d["gpu"]
        if gpu_raw == "none":
            errors.append("Profil ini memerlukan skor GPU; pilih level GPU.")
        elif gpu_raw not in v8.GPU_LEVELS:
            errors.append("Level performa GPU tidak dikenal.")
        else:
            gpu_level = gpu_raw
    elif gpu_raw not in ("", "none"):
        if gpu_raw in v8.GPU_LEVELS:
            gpu_level = gpu_raw
        else:
            errors.append("Level performa GPU tidak dikenal.")

    raw_ded = params.get("require_dedicated", False)
    require_dedicated = True if profile == "gaming" else (
        raw_ded is True or str(raw_ded).strip().lower() in BOOL_TRUE)

    try:
        min_inch = float(params.get("min_screen_inch") or 0)
    except (TypeError, ValueError):
        errors.append("Ukuran layar minimum tidak valid.")
        min_inch = 0.0
    if min_inch not in SCREEN_INCH_CHOICES:
        errors.append(f"Ukuran layar minimum harus salah satu dari {SCREEN_INCH_CHOICES}.")
    min_res = str(params.get("min_resolution") or "")
    if min_res and min_res not in RES_CHOICES:
        errors.append("Resolusi minimum tidak dikenal.")
    raw_oled = params.get("oled_only", False)
    oled_only = raw_oled is True or str(raw_oled).strip().lower() in BOOL_TRUE

    if errors:
        raise ValueError(" ".join(errors))
    return v8.Scenario("web", profile, budget, float(min_ram), float(min_storage), cpu_level,
                       gpu_level, require_dedicated, v8.DEFAULT_LAMBDA, TOP_N,
                       min_screen_inch=min_inch, min_resolution=min_res, oled_only=oled_only,
                       target_ram=float(target_ram) if target_ram is not None else None,
                       target_storage=float(target_storage) if target_storage is not None else None)


def screen_text(x) -> str:
    """Screen description read from the product name (positive evidence only)."""
    size, res = x.get("screen_inch"), x.get("screen_res")
    parts = []
    if size is not None and pd.notna(size):
        parts.append(f"{float(size):g}\u2033")
    has_res = res is not None and pd.notna(res)
    if has_res:
        parts.append(str(res))
    if bool(x.get("screen_oled")):
        parts.append("OLED")
    if bool(x.get("screen_ips")):
        parts.append("IPS")
    if bool(x.get("screen_touch")):
        parts.append("touchscreen")
    if not parts:
        return "Layar: tidak tertulis pada nama produk"
    text = "Layar: " + " ".join(parts)
    return text if has_res else text + " (resolusi tidak tertulis pada nama produk)"


def _int_or_none(v):
    try:
        return None if v is None or pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


EXPLAIN_ORDER = [v8.COL_CPU, v8.COL_GPU, v8.COL_RAM, v8.COL_STORAGE]
EXPLAIN_NAMES = {v8.COL_CPU: "CPU (PassMark)", v8.COL_GPU: "GPU (G3D Mark)",
                 v8.COL_RAM: "RAM", v8.COL_STORAGE: "Storage"}


def _pct_text(pct: float) -> str:
    r = round(pct)
    return "kurang dari 1%" if r < 1 else f"{r}%"


def explain_rec(x, features: list, targets: dict, budget: float, prices, n_rankable: int,
                gpu_in_ranking: bool, variants: int, vmin: float, vmax: float) -> dict:
    """Plain-language reasons built ONLY from numbers the ranking already uses.

    It describes how close the laptop is to the profile targets, its price position and rank.
    It makes no claim about overall quality.
    """
    below, points, any_surplus = [], [], False
    for f in [c for c in EXPLAIN_ORDER if c in features]:
        val, tgt = float(x[f]), float(targets[f])
        unit = " GB" if f in (v8.COL_RAM, v8.COL_STORAGE) else ""
        name = EXPLAIN_NAMES[f]
        if val < tgt - 1e-9:
            below.append(name.split(" ")[0])
            points.append(f"{name} {fmt_num(val)}{unit}, {_pct_text((tgt - val) / tgt * 100)} di bawah target "
                          f"({fmt_num(tgt)}{unit}).")
        elif abs(val - tgt) < 1e-9:
            points.append(f"{name} {fmt_num(val)}{unit}, sama dengan target ({fmt_num(tgt)}{unit}).")
        else:
            pct = (val - tgt) / tgt * 100
            any_surplus = any_surplus or round(pct) >= 1
            points.append(f"{name} {fmt_num(val)}{unit}, {_pct_text(pct)} di atas target ({fmt_num(tgt)}{unit}).")
    n = len(features)
    headline = (f"Memenuhi semua target profil ({n} dari {n})." if not below else
                f"Mencapai {n - len(below)} dari {n} target; di bawah target: {', '.join(below)}.")

    price = float(x["price_numeric"])
    gap = float(budget) - price
    price_text = (f"Harga {fmt_idr(price)}, {fmt_idr(gap)} di bawah budget." if gap > 0.5 else
                  f"Harga {fmt_idr(price)}, tepat pada batas budget.")
    others = len(prices) - 1
    if others > 0:
        cheaper = int((prices > price).sum())        # others that cost more than this laptop
        pricier = int((prices < price).sum())        # others that cost less
        if cheaper >= pricier:
            price_text += f" Lebih murah daripada {round(100 * cheaper / others)}% laptop lain yang dapat diranking."
        else:
            price_text += f" Lebih mahal daripada {round(100 * pricier / others)}% laptop lain yang dapat diranking."
    points.append(price_text)
    points.append(screen_text(x) + ".")

    if not gpu_in_ranking:
        g = x["graphics_normalized"] if pd.notna(x["graphics_normalized"]) else "tidak dicantumkan"
        if x["gpu_class"] == "dedicated":
            points.append(f"Memiliki GPU dedicated ({g}); GPU tidak dipakai dalam ranking profil ini, sehingga "
                          "tidak menaikkan atau menurunkan peringkat.")
        elif x["gpu_class"] == "integrated":
            points.append("GPU terintegrasi; GPU tidak dipakai dalam ranking profil ini.")
        else:
            points.append("Jenis GPU belum dapat dipastikan; GPU tidak dipakai dalam ranking profil ini.")
    if any_surplus:
        points.append("Kelebihan di atas target dihukum lebih ringan daripada kekurangan dalam ranking.")
    points.append(f"Peringkat {int(x['recommendation_rank'])} dari {int(n_rankable)} laptop yang lolos syarat dan "
                  "dapat diranking.")
    if variants > 0:
        points.append(f"Ada {variants} varian lain dengan spesifikasi setara pada fitur yang dipakai "
                      f"({fmt_idr(vmin)} - {fmt_idr(vmax)}); yang ditampilkan adalah yang termurah.")
    return dict(headline=headline, points=points)


def run_query(df: pd.DataFrame, ref: dict, params: dict) -> dict:
    """Run one recommendation query and return a display-ready dict."""
    sc = make_scenario(params)
    res = v8.run_scenario(df, ref, sc)
    sm = res["summary"]
    features, targets = res["features"], res["targets"]

    prices = res["full"]["price_numeric"].to_numpy(float) if len(res["full"]) else np.array([])
    recs = []
    for _, x in res["recs"].iterrows():
        chips = []
        for f in features:
            pct = float(x.get(f"shortfall_pct_{f}", 0.0) or 0.0)
            unit = " GB" if f in (v8.COL_RAM, v8.COL_STORAGE) else ""
            ok = pct <= 1e-9
            text = (f"{FEATURE_LABELS[f]} {fmt_num(x[f])}{unit}: memenuhi target" if ok else
                    f"{FEATURE_LABELS[f]} {fmt_num(x[f])}{unit}: {round(pct)}% di bawah target "
                    f"({fmt_num(targets[f])}{unit})")
            chips.append(dict(label=FEATURE_LABELS[f], ok=ok, pct=round(pct), text=text))
        url = str(x["url"]) if pd.notna(x["url"]) else ""
        recs.append(dict(
            laptop_id=_int_or_none(x["laptop_id"]) if "laptop_id" in x.index else None,
            source_row=int(x["source_row"]), screen=screen_text(x),
            rank=int(x["recommendation_rank"]), name=str(x["name"]),
            url=url if url.startswith(("http://", "https://")) else "",
            price=float(x["price_numeric"]), ram=fmt_num(x["ram_gb"]), storage=fmt_num(x["storage_gb"]),
            processor=str(x["processor_normalized"]) if pd.notna(x["processor_normalized"]) else "-",
            graphics=str(x["graphics_normalized"]) if pd.notna(x["graphics_normalized"]) else "(tidak dicantumkan)",
            gpu_class=GPU_CLASS_LABELS.get(x["gpu_class"], str(x["gpu_class"])), chips=chips,
            variants=int(x["equivalent_variants_hidden"]),
            variant_price_min=float(x["group_price_min"]), variant_price_max=float(x["group_price_max"]),
            distance=float(x["knn_distance"]),
            explain=explain_rec(x, features, targets, sc.budget, prices, sm["n_rankable"],
                                bool(sm["gpu_used_in_ranking"]), int(x["equivalent_variants_hidden"]),
                                float(x["group_price_min"]), float(x["group_price_max"]))))

    unr = res["unrankable"]
    unr_df = pd.DataFrame({
        "Laptop": unr["name"].astype(str).to_numpy(),
        "Harga": [fmt_idr(v) for v in unr["price_numeric"]],
        "Alasan": unr["unrankable_reason"].astype(str).to_numpy(),
    }).head(MAX_UNRANKABLE_SHOWN) if len(unr) else pd.DataFrame(columns=["Laptop", "Harga", "Alasan"])

    empty_reason = ""
    if not recs:
        if sm["n_after_basic_filter"] == 0:
            empty_reason = ("Tidak ada laptop yang memenuhi budget, RAM minimum, dan storage minimum. "
                            "Coba naikkan budget atau turunkan RAM/storage minimum.")
        elif sm["n_after_gpu_requirement"] == 0:
            empty_reason = ("Ada laptop yang memenuhi budget, RAM, dan storage, tetapi tidak ada yang "
                            "terkonfirmasi memiliki GPU dedicated. Coba naikkan budget atau matikan syarat "
                            "GPU dedicated (jika profil mengizinkan).")
        else:
            empty_reason = ("Ada laptop yang lolos syarat wajib, tetapi tidak satu pun memiliki skor benchmark "
                            "yang dibutuhkan untuk diranking. Lihat daftar di bagian bawah.")

    screen_note = ""
    if sc.min_screen_inch > 0 or sc.min_resolution or sc.oled_only:
        screen_note = (f"Filter layar aktif: {sm['n_screen_excluded_nomatch']} laptop tidak cocok dan "
                       f"{sm['n_screen_excluded_unknown']} tidak dapat dipastikan karena informasi layar tidak "
                       "tertulis pada nama produk (dicantumkan pada daftar belum dapat diranking).")
        if sc.oled_only:
            screen_note += " Filter OLED hanya mengenali laptop yang menuliskan OLED pada nama produk."
    pool_pct = sm.get("pool_pct_meeting_all_targets", float("nan"))
    return dict(
        scenario=sc, params=dict(params), screen_note=screen_note,
        summary={k: v for k, v in sm.items()}, recs=recs, empty_reason=empty_reason,
        targets=[dict(label=FEATURE_LABELS[f],
                      value=fmt_num(targets[f]) + (" GB" if f in (v8.COL_RAM, v8.COL_STORAGE) else ""))
                 for f in features],
        features=features, gpu_in_ranking=bool(sm["gpu_used_in_ranking"]),
        require_dedicated=bool(sm["require_dedicated"]),
        unrankable=unr_df, unrankable_total=int(len(unr)),
        unrankable_extra=max(0, int(len(unr)) - MAX_UNRANKABLE_SHOWN),
        target_far=bool(recs) and pool_pct is not None and not pd.isna(pool_pct) and pool_pct < 5,
    )


def session_record(result: dict, source: str) -> dict:
    """Fields to log for one query (anonymous; no personal data)."""
    sc, sm = result["scenario"], result["summary"]
    return dict(
        profile=sc.profile, budget_idr=int(sc.budget), min_ram_gb=int(sc.min_ram),
        min_storage_gb=int(sc.min_storage), cpu_level=sm["cpu_level"],
        gpu_level=(sm["gpu_level"] or None) if sm["gpu_used_in_ranking"] else None,
        require_dedicated=bool(sm["require_dedicated"]), lambda_surplus=float(sm["lambda_surplus"]),
        catalog_source=source, n_after_filter=int(sm["n_after_gpu_requirement"]),
        n_rankable=int(sm["n_rankable"]), n_recommendations=int(sm["n_recommendations"]))


def session_items(result: dict) -> list:
    return [dict(rank=r["rank"], laptop_id=r["laptop_id"], distance=r["distance"])
            for r in result["recs"] if r["laptop_id"] is not None]


def _diff_vs_source(x, src, features: list) -> str:
    parts = []
    for f in [c for c in EXPLAIN_ORDER if c in features]:
        a, b = float(x[f]), float(src[f])
        label = FEATURE_LABELS[f]
        if f in (v8.COL_CPU, v8.COL_GPU):
            pct = (a - b) / b * 100 if b else 0.0
            parts.append(f"{label} sama" if a == b else
                         f"{label} hampir sama" if abs(pct) < 1 else
                         f"{label} {round(abs(pct))}% lebih {'tinggi' if pct > 0 else 'rendah'}")
        else:
            parts.append(f"{label} sama" if a == b else
                         f"{label} {abs(a - b):g} GB lebih {'besar' if a > b else 'kecil'}")
    dp = float(x["price_numeric"]) - float(src["price_numeric"])
    parts.append("harga sama" if abs(dp) < 0.5 else
                 f"harga {fmt_idr(abs(dp))} lebih {'mahal' if dp > 0 else 'murah'}")
    return ", ".join(parts)


def similar_query(df: pd.DataFrame, ref: dict, params: dict, source_row: int, top_n: int = 5) -> dict:
    """Laptops most similar to one laptop. The budget is NOT enforced (user decision)."""
    sc = make_scenario(params)
    r = v8.similar_laptops(df, ref, sc, int(source_row), top_n)
    src = r["source"]
    out = dict(message=r["message"], items=[], identical_n=0, identical_range=None, n_pool=r["n_pool"],
               features=r["features"], budget=sc.budget, source_name="", source_screen="")
    if src is None:
        return out
    out["source_name"] = str(src["name"])
    out["source_screen"] = screen_text(src)
    ident = r["identical"]
    if len(ident):
        out["identical_n"] = int(len(ident))
        out["identical_range"] = (float(ident["price_numeric"].min()), float(ident["price_numeric"].max()))
    for _, x in r["items"].iterrows():
        price = float(x["price_numeric"])
        out["items"].append(dict(
            rank=int(x["similar_rank"]), name=str(x["name"]),
            url=str(x["url"]) if pd.notna(x["url"]) and str(x["url"]).startswith(("http://", "https://")) else "",
            price=price, over_budget=max(0.0, price - sc.budget), screen=screen_text(x),
            gpu_class=GPU_CLASS_LABELS.get(x["gpu_class"], str(x["gpu_class"])),
            graphics=str(x["graphics_normalized"]) if pd.notna(x["graphics_normalized"]) else "(tidak dicantumkan)",
            processor=str(x["processor_normalized"]) if pd.notna(x["processor_normalized"]) else "-",
            ram=fmt_num(x["ram_gb"]), storage=fmt_num(x["storage_gb"]),
            diff=_diff_vs_source(x, src, r["features"]), variants=int(x["group_size"]) - 1,
            vmin=float(x["group_price_min"]), vmax=float(x["group_price_max"]),
            distance=float(x["sim_distance"])))
    return out
