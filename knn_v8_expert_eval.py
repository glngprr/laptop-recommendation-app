"""
KNN v8: expert evaluation tooling (blind rating sheet + scoring).

Why: the dataset has no ground truth. The only way to say anything about
recommendation quality is human judgement. This tool creates a BLIND sheet
(raters do not see method, rank, distance, or benchmark scores) that mixes:

    v8            top-N of the v8 recommender (lambda = 0.25)
    v8_symmetric  top-N with lambda = 1 (plain symmetric Euclidean), to test the asymmetry choice
    baseline_cheapest  the N cheapest candidates passing the hard filters
    baseline_random    N random candidates passing the hard filters (seeded)

Run from the project root (place this file in the root, next to knn_v8_profile_recommender.py):

    # 1) create the sheet + hidden key  (outputs in inspection/expert_eval/)
    python knn_v8_expert_eval.py make-sheet

    # 2) give knn_v8_rating_sheet.csv to each rater; each fills `rating` (1-5), `rater_id`
    #    save as e.g. inspection/expert_eval/filled_rater1.csv

    # 3) score all filled sheets
    python knn_v8_expert_eval.py score --ratings inspection/expert_eval/filled_rater1.csv inspection/expert_eval/filled_rater2.csv inspection/expert_eval/filled_rater3.csv

Do NOT show knn_v8_rating_key.csv to raters. Results are only meaningful with
real raters; with few raters/items they are indicative, not conclusive.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    sys.path.insert(0, str(_p))
import knn_v8_profile_recommender as v8  # noqa: E402

EVAL_DIR = v8.OUT_DIR / "expert_eval"
SHEET_PATH = EVAL_DIR / "knn_v8_rating_sheet.csv"
KEY_PATH = EVAL_DIR / "knn_v8_rating_key.csv"
RATING_MIN, RATING_MAX = 1, 5
ACCEPTABLE = 4
METHODS = ["v8", "v8_symmetric", "baseline_cheapest", "baseline_random"]

SCALE = ("1=sangat tidak sesuai (spesifikasi jelas tidak memadai atau tidak masuk akal untuk profil), "
         "2=kurang sesuai, 3=cukup/dapat diterima dengan kompromi, 4=sesuai, 5=sangat sesuai")


# ----------------------------------------------------------------------------
# make-sheet
# ----------------------------------------------------------------------------
def make_sheet(args) -> int:
    df = v8.load_dataset(args.input)
    ref = v8.build_reference(df)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows, key_rows = [], []
    for i, sc in enumerate(v8.SUITE):
        sc = dataclasses.replace(sc, top_n=args.n)
        base = v8.run_scenario(df, ref, sc)
        sym = v8.run_scenario(df, ref, dataclasses.replace(sc, lam=1.0))
        full = base["full"]
        if full.empty:
            continue
        picks = {}  # source_row -> list of (method, rank)

        def add(method, frame):
            for r, (_, x) in enumerate(frame.iterrows(), start=1):
                picks.setdefault(int(x["source_row"]), []).append((method, r))

        add("v8", base["recs"])
        add("v8_symmetric", sym["recs"])
        add("baseline_cheapest", full.sort_values(["price_numeric", "source_row"]).head(args.n))
        rng = np.random.default_rng(args.seed + i)
        add("baseline_random", full.iloc[rng.choice(len(full), size=min(args.n, len(full)), replace=False)])

        p = v8.PROFILES[sc.profile]
        sm = base["summary"]
        meta = dict(
            scenario_id=sc.scenario_id, profil=p["label"], budget_maks_idr=int(sc.budget),
            ram_min_gb=int(sc.min_ram), storage_min_gb=int(sc.min_storage),
            preferensi_cpu=sm["cpu_level"],
            preferensi_gpu=sm["gpu_level"] if sm["gpu_used_in_ranking"] else "tidak dipakai",
            gpu_dedicated_wajib="ya" if sm["require_dedicated"] else "tidak")
        order = list(picks)
        rng2 = np.random.default_rng(args.seed + 1000 + i)
        rng2.shuffle(order)
        for src in order:
            x = full[full["source_row"] == src].iloc[0]
            item_id = f"{sc.scenario_id}-{len(rows) + 1:03d}"
            rows.append(dict(item_id=item_id, **meta, name=x["name"], url=x["url"],
                             price_idr=int(x["price_numeric"]), ram_gb=int(x["ram_gb"]),
                             storage_gb=int(x["storage_gb"]), processor=x["processor_normalized"],
                             graphics=x["graphics_normalized"] if pd.notna(x["graphics_normalized"]) else "(tidak dicantumkan)",
                             rating="", catatan="", rater_id=""))
            for method, rank in picks[src]:
                key_rows.append(dict(item_id=item_id, scenario_id=sc.scenario_id, profile=sc.profile,
                                     method=method, rank_in_method=rank))
    sheet = pd.DataFrame(rows)
    sheet.to_csv(SHEET_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(key_rows).to_csv(KEY_PATH, index=False)
    print(f"Lembar penilaian: {SHEET_PATH}  ({len(sheet)} item untuk dinilai)")
    print(f"Kunci metode (JANGAN dibagikan ke penilai): {KEY_PATH}")
    print(f"Skala: {SCALE}")
    print("Item per skenario:", sheet.groupby("scenario_id").size().to_dict())
    return 0


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def qwk(a: np.ndarray, b: np.ndarray, k: int = RATING_MAX - RATING_MIN + 1) -> float:
    """Quadratic weighted kappa for integer ratings in 1..5."""
    a = a.astype(int) - RATING_MIN
    b = b.astype(int) - RATING_MIN
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[x, y] += 1
    n = obs.sum()
    if n == 0:
        return float("nan")
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    den = (w * exp).sum()
    return float("nan") if den == 0 else 1.0 - (w * obs).sum() / den


def load_ratings(paths: list[Path], valid_ids: set[str]) -> pd.DataFrame:
    frames, problems = [], []
    for pth in paths:
        if not pth.exists():
            raise FileNotFoundError(f"File penilaian tidak ditemukan: {pth}")
        d = pd.read_csv(pth, encoding="utf-8-sig")
        for c in ("item_id", "rating"):
            if c not in d.columns:
                raise ValueError(f"{pth.name}: kolom '{c}' tidak ada.")
        rid = d["rater_id"].astype(str).str.strip() if "rater_id" in d.columns else pd.Series("", index=d.index)
        rid = rid.where(~rid.isin(["", "nan"]), pth.stem)
        d = d.assign(rater_id=rid)
        unknown = sorted(set(d["item_id"]) - valid_ids)
        if unknown:
            raise ValueError(f"{pth.name}: item_id tidak dikenal: {unknown[:5]}")
        r = pd.to_numeric(d["rating"], errors="coerce")
        blank = int(r.isna().sum())
        rv = r.dropna()
        bad = int(((rv < RATING_MIN) | (rv > RATING_MAX) | (rv != rv.round())).sum())
        if bad:
            raise ValueError(f"{pth.name}: {bad} rating di luar 1..5 atau bukan bilangan bulat.")
        if blank:
            problems.append(f"{pth.name}: {blank} item belum dinilai (diabaikan).")
        frames.append(pd.DataFrame({"item_id": d["item_id"], "rater_id": d["rater_id"],
                                    "rating": r}).dropna())
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(["item_id", "rater_id"]).sum()
    if dup:
        raise ValueError(f"Ada {dup} penilaian ganda untuk pasangan (item, penilai); periksa rater_id.")
    for pr in problems:
        print("PERINGATAN:", pr)
    return out


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n=5000, seed=7):
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.mean(a) - np.mean(b)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def score(args) -> int:
    if not KEY_PATH.exists():
        raise FileNotFoundError(f"{KEY_PATH} tidak ada. Jalankan 'make-sheet' lebih dulu.")
    key = pd.read_csv(KEY_PATH)
    ratings = load_ratings([Path(p) for p in args.ratings], set(key["item_id"]))
    raters = sorted(ratings["rater_id"].unique())
    item_mean = ratings.groupby("item_id")["rating"].agg(["mean", "count"]).rename(
        columns={"mean": "mean_rating", "count": "n_raters"})
    m = key.merge(item_mean, left_on="item_id", right_index=True, how="inner")
    m = m.merge(ratings.assign(acceptable=lambda x: x["rating"] >= ACCEPTABLE)[["item_id", "acceptable"]]
                .groupby("item_id")["acceptable"].mean().rename("share_raters_acceptable"),
                left_on="item_id", right_index=True)

    L = ["# KNN v8 - Hasil Evaluasi Ahli", "",
         f"- Penilai: {len(raters)} ({', '.join(raters)})",
         f"- Item bernilai: {ratings['item_id'].nunique()} dari {key['item_id'].nunique()}",
         f"- Skala: {SCALE}", f"- 'Sesuai' = rating >= {ACCEPTABLE}", ""]
    if len(raters) < 3:
        L += ["> PERINGATAN: kurang dari 3 penilai. Kesepakatan antar-penilai dan selisih antar-metode "
              "hanya indikatif.", ""]

    # per method
    rows, store = [], {}
    for meth in METHODS:
        sub = m[m["method"] == meth]
        if sub.empty:
            continue
        store[meth] = sub["mean_rating"].to_numpy(float)
        cells = ratings[ratings["item_id"].isin(sub["item_id"])]
        rows.append(dict(metode=meth, item=sub["item_id"].nunique(),
                         rata2_rating=f"{sub['mean_rating'].mean():.2f}",
                         item_dengan_rata2_ge_4=f"{(sub['mean_rating'] >= ACCEPTABLE).mean():.0%}",
                         sel_rating_ge_4=f"{(cells['rating'] >= ACCEPTABLE).mean():.0%}"))
    L += ["## Mutu per metode", ""] + v8.md_table(rows, list(rows[0].keys()))

    # per profile for v8 vs baselines
    rows = []
    for prof in sorted(m["profile"].unique()):
        r = dict(profil=prof)
        for meth in METHODS:
            s = m[(m["profile"] == prof) & (m["method"] == meth)]
            r[meth] = f"{s['mean_rating'].mean():.2f} (n={len(s)})" if len(s) else "-"
        rows.append(r)
    L += ["", "## Rata-rata rating per profil", ""] + v8.md_table(rows, list(rows[0].keys()))

    # paired contrasts
    rows = []
    for other in ("baseline_cheapest", "baseline_random", "v8_symmetric"):
        if "v8" in store and other in store:
            d, lo, hi = bootstrap_diff(store["v8"], store[other])
            rows.append(dict(kontras=f"v8 - {other}", selisih_rata2=f"{d:+.2f}",
                             CI95_bootstrap=f"[{lo:+.2f}; {hi:+.2f}]",
                             tafsir="CI tidak memuat 0" if (lo > 0 or hi < 0) else "CI memuat 0: belum terbukti berbeda"))
    if rows:
        L += ["", "## Selisih v8 terhadap pembanding (bootstrap atas item, 5000 resampel)", ""] + \
             v8.md_table(rows, list(rows[0].keys()))

    # rank validity
    v = m[m["method"] == "v8"]
    if len(v) >= 5:
        rho = v["rank_in_method"].corr(v["mean_rating"], method="spearman")
        L += ["", "## Apakah urutan rank v8 sejalan dengan penilaian?", "",
              f"Korelasi Spearman antara rank v8 (1=terbaik) dan rata-rata rating: {rho:.2f} "
              "(negatif berarti rank kecil dinilai lebih baik, sesuai harapan).", ""]

    # agreement
    wide = ratings.pivot(index="item_id", columns="rater_id", values="rating")
    ks = []
    for a, b in itertools.combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        if len(both) >= 5:
            ks.append(dict(pasangan=f"{a} vs {b}", item_bersama=len(both),
                           QWK=f"{qwk(both[a].to_numpy(), both[b].to_numpy()):.2f}"))
    L += ["", "## Kesepakatan antar-penilai (quadratic weighted kappa)", ""]
    if ks:
        L += v8.md_table(ks, list(ks[0].keys()))
        mean_k = np.nanmean([float(x["QWK"]) for x in ks])
        L += ["", f"Rata-rata QWK: {mean_k:.2f}. Kesepakatan rendah berarti skala/kriteria belum seragam; "
              "hasil per metode kurang dapat dipercaya."]
    else:
        L += ["Perlu minimal 2 penilai dengan >= 5 item bersama."]
    L += ["", "## Keterbatasan", "",
          "- Penilaian ahli bersifat subjektif dan bergantung pada kriteria yang disepakati.",
          "- Jumlah item kecil (maks. sekitar 20 per skenario); gunakan hasil sebagai indikasi.",
          "- Hasil hanya berlaku untuk 8 skenario, katalog satu toko, dan snapshot tanggal data ini.", ""]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_md = EVAL_DIR / "knn_v8_expert_eval_results.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    m.to_csv(EVAL_DIR / "knn_v8_expert_eval_item_scores.csv", index=False)
    print("\n".join(L))
    print(f"\nFile: {out_md}")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Alat evaluasi ahli KNN v8.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make-sheet")
    a.add_argument("--input", type=Path, default=v8.DEFAULT_INPUT)
    a.add_argument("--n", type=int, default=5, help="item per metode per skenario")
    a.add_argument("--seed", type=int, default=2026)
    b = sub.add_parser("score")
    b.add_argument("--ratings", nargs="+", required=True)
    args = ap.parse_args(argv)
    try:
        return make_sheet(args) if args.cmd == "make-sheet" else score(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
