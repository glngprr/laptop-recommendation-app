from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

import service as S

st.set_page_config(
    page_title="Perbandingan Metode Sistem Rekomendasi Laptop untuk Android Developer",
    layout="wide"
)

PROFILES = S.profile_defaults()
FORM_KEYS = ("budget", "target_ram", "target_storage", "cpu_level")

PROFILE_DESCRIPTIONS = {
    "programming": (
        "Direkomendasikan untuk pengembangan aplikasi Android standar menggunakan Android Studio. "
        "Membutuhkan RAM yang cukup untuk menjalankan Gradle build dan satu emulator Android lokal."
    ),
    "office": (
        "Untuk aktivitas perkuliahan dasar dan pemrograman ringan (modul sederhana). "
        "Mengutamakan efisiensi anggaran dengan batas kelayakan pengembangan aplikasi."
    ),
    "design": (
        "Untuk pengembangan aplikasi yang dipadukan dengan desain antarmuka intensif (Figma, rendering aset). "
        "Membutuhkan kapasitas memori dan penyimpanan lebih besar untuk multitasking lancar."
    ),
    "gaming": (
        "Untuk beban kerja berat atau workstation: menjalankan beberapa emulator secara simultan dan proyek berskala besar. "
        "Membutuhkan kapasitas RAM dan performa prosesor tertinggi."
    ),
}

CUSTOM_CSS = """
<style>
/* Base spacing & responsive layout */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3.5rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1440px;
    margin-left: auto;
    margin-right: auto;
}

@media (max-width: 1200px) {
    .block-container {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

/* Ensure selectboxes behave as fixed choice dropdowns and prevent text typing or deletion */
div[data-baseweb="select"] input {
    pointer-events: none !important;
    caret-color: transparent !important;
    user-select: none !important;
}
div[data-baseweb="select"] {
    cursor: pointer !important;
}

/* Price callout */
.price-tag {
    font-size: 1.35rem;
    font-weight: 700;
    color: #1d4ed8;
    margin-top: 0.2rem;
    margin-bottom: 0.45rem;
}
@media (prefers-color-scheme: dark) {
    .price-tag {
        color: #60a5fa;
    }
}

/* Specs pills container */
.specs-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.35rem;
    margin-bottom: 0.55rem;
}
.spec-pill {
    display: inline-flex;
    align-items: center;
    font-size: 0.82rem;
    padding: 0.25rem 0.55rem;
    border-radius: 6px;
    background: #f1f5f9;
    color: #334155;
    font-weight: 500;
    border: 1px solid #e2e8f0;
}
@media (prefers-color-scheme: dark) {
    .spec-pill {
        background: #1e293b;
        color: #cbd5e1;
        border-color: #334155;
    }
}

/* Target fulfillment chips */
.chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.65rem;
}
.chip-ok {
    font-size: 0.78rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    background: #dcfce7;
    color: #166534;
    font-weight: 500;
    border: 1px solid #bbf7d0;
}
.chip-warn {
    font-size: 0.78rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    background: #fef3c7;
    color: #92400e;
    font-weight: 500;
    border: 1px solid #fde68a;
}
@media (prefers-color-scheme: dark) {
    .chip-ok {
        background: #064e3b;
        color: #a7f3d0;
        border-color: #047857;
    }
    .chip-warn {
        background: #78350f;
        color: #fde68a;
        border-color: #b45309;
    }
}
</style>
"""

# ----------------------------------------------------------------------------
# data & artifact loading (standalone file-based, zero-database)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat katalog...")
def load_catalog_cached():
    """Load reference catalog directly from dataset CSV."""
    df, ref = S.prepare_catalog()
    return df, ref


@st.cache_data
def load_evaluation_artifacts():
    """Load precomputed offline evaluation artifacts from evaluation/ directory."""
    eval_dir = Path(__file__).resolve().parent / "evaluation"
    summary_path = eval_dir / "summary_metrics.csv"
    winner_path = eval_dir / "winner_decision.json"
    metadata_path = eval_dir / "run_metadata.json"
    benchmark_path = eval_dir / "benchmark_results.csv"

    if not summary_path.exists():
        return None, None, None, None

    summary_df = pd.read_csv(summary_path)
    with open(winner_path, "r", encoding="utf-8") as f:
        winner_data = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    benchmark_df = pd.read_csv(benchmark_path)

    return summary_df, winner_data, metadata, benchmark_df


# ----------------------------------------------------------------------------
# form state management
# ----------------------------------------------------------------------------
def on_profile_change():
    """Update target preferences when preset profile changes, preserving budget."""
    new_prof = st.session_state.get("profile")
    if new_prof not in S.PROFILE_ORDER:
        new_prof = "programming"
        st.session_state["profile"] = new_prof

    d = PROFILES[new_prof]
    st.session_state["last_profile"] = new_prof
    st.session_state["target_ram"] = d["target_ram"]
    st.session_state["target_storage"] = d["target_storage"]
    st.session_state["cpu_level"] = d["cpu"]
    st.session_state["profile_notice"] = (
        f"Target spesifikasi diperbarui ke preset {d['label']}. "
        f"Anggaran Anda tetap dipertahankan."
    )


def init_state():
    if "profile" not in st.session_state or st.session_state["profile"] not in S.PROFILE_ORDER:
        st.session_state["profile"] = "programming"
    if "last_profile" not in st.session_state:
        st.session_state["last_profile"] = st.session_state["profile"]

    if "budget" not in st.session_state or not isinstance(st.session_state["budget"], (int, float)):
        st.session_state["budget"] = 15_000_000

    d = PROFILES[st.session_state["profile"]]
    st.session_state.setdefault("target_ram", d["target_ram"])
    st.session_state.setdefault("target_storage", d["target_storage"])
    st.session_state.setdefault("cpu_level", d["cpu"])
    st.session_state.setdefault("screen_inch", 0)
    st.session_state.setdefault("screen_res", "")
    st.session_state.setdefault("screen_oled", False)


def md_escape(text: str) -> str:
    out = str(text)
    for ch in ("\\", "`", "*", "_", "[", "]", "$", "<", ">", "#", "|", "~"):
        out = out.replace(ch, "\\" + ch)
    return out


def render_result(stored: dict):
    data = stored["data"]
    sm = data["summary"]
    st.subheader("Hasil Rekomendasi Laptop")
    c1, c2, c3 = st.columns(3)
    c1.metric("Katalog Referensi", S.fmt_num(sm["n_initial"]))
    c2.metric("Lolos Syarat Kelayakan", S.fmt_num(sm["n_after_basic_filter"]))
    c3.metric("Rekomendasi Terpilih", S.fmt_num(sm["n_recommendations"]))

    # Summary box
    target_parts = [f"**{t['label']}**: {t['value']}" for t in data["targets"]]

    st.info(
        f"**Target Spesifikasi**: {' · '.join(target_parts)}\n\n"
        f"**Syarat Kelayakan Otomatis**: Anggaran $\\le$ {S.fmt_idr(data['params']['budget'])} · "
        f"RAM $\\ge$ 16 GB · "
        f"Penyimpanan SSD $\\ge$ 256 GB"
    )

    if data["screen_note"]:
        st.info(data["screen_note"])

    if not data["recs"]:
        st.warning(data["empty_reason"])
    else:
        for r in data["recs"]:
            with st.container(border=True):
                # 1. Nomor peringkat dan nama laptop
                header_html = (
                    f'<div style="margin-bottom:0.35rem; line-height:1.4;">'
                    f'<span style="font-weight:700; color:#64748b; font-size:1.05rem; margin-right:0.4rem;">#{r["rank"]}</span>'
                    f'<strong style="font-size:1.05rem;">{md_escape(r["name"])}</strong>'
                    f'</div>'
                )
                st.markdown(header_html, unsafe_allow_html=True)

                # 2. Harga
                st.markdown(f'<div class="price-tag">{S.fmt_idr(r["price"])}</div>', unsafe_allow_html=True)

                # 3. Spesifikasi utama dalam format pills
                pills_html = (
                    f'<div class="specs-pills">'
                    f'<span class="spec-pill">RAM: {r["ram"]} GB</span>'
                    f'<span class="spec-pill">SSD: {r["storage"]} GB</span>'
                    f'<span class="spec-pill">CPU: {md_escape(r["processor"])}</span>'
                    f'<span class="spec-pill">GPU: {md_escape(r["graphics"])}</span>'
                    f'<span class="spec-pill">{md_escape(r["screen"])}</span>'
                    f'</div>'
                )
                st.markdown(pills_html, unsafe_allow_html=True)

                # 4. Status kesesuaian target dalam bentuk chips
                chips_html = '<div class="chips-container">' + "".join(
                    f'<span class="{"chip-ok" if c["ok"] else "chip-warn"}">{md_escape(c["text"])}</span>'
                    for c in r["chips"]
                ) + '</div>'
                st.markdown(chips_html, unsafe_allow_html=True)

                # 5. Penjelasan rekomendasi
                with st.expander("Mengapa laptop ini direkomendasikan?"):
                    st.markdown(f"**{md_escape(r['explain']['headline'])}**")
                    for p in r["explain"]["points"]:
                        st.markdown(f"- {md_escape(p)}")
                    st.caption("Penjelasan ini dihitung berdasarkan kesesuaian spesifikasi RAM (bobot 45%), performa prosesor (bobot 35%), dan kapasitas penyimpanan (bobot 20%).")

                # 6. Skor Kesesuaian
                st.caption(f"Skor Kesesuaian: {r['score']:.4f} menunjukkan tingkat kecocokan spesifikasi laptop terhadap kebutuhan yang dimasukkan (rentang 0.0 s/d 1.0). Skor yang lebih tinggi menandakan kecocokan yang lebih baik.")

                # 7. Tombol tautan ke toko
                if r["url"]:
                    st.link_button("Lihat di Toko", r["url"], use_container_width=True)

    with st.expander("Cara membaca hasil dan urutan rekomendasi"):
        st.markdown(
            "- **Penyaringan Laptop**: Laptop dengan harga di atas anggaran, RAM di bawah 16 GB, atau kapasitas penyimpanan di bawah 256 GB otomatis dieliminasi dari daftar pencarian.\n"
            "- **Metode Rekomendasi**: Sistem menggunakan metode **Weighted Scoring** sebagai algoritma aktif terpilih dari hasil evaluasi komparatif offline 72 skenario.\n"
            "- **Bobot Atribut**: RAM (45%), performa prosesor (35%), dan kapasitas penyimpanan (20%).\n"
            "- **Penilaian Kesesuaian**: Laptop yang memenuhi atau melampaui target spesifikasi menerima nilai kesesuaian tinggi (hingga 1.0), sedangkan laptop di bawah target menerima pengurangan nilai secara proporsional.\n"
            "- **Urutan Rekomendasi**: Laptop diurutkan dari skor kesesuaian tertinggi ke terendah. Jika skor sama (seri dalam batas toleransi $10^{-7}$), laptop dengan harga lebih terjangkau diprioritaskan."
        )


def render_model_evaluation_tab():
    st.subheader("Evaluasi Model: Hasil Uji Komparatif")
    st.caption(
        "Evaluasi kinerja tiga metode rekomendasi laptop menggunakan 72 skenario pengujian."
    )

    summary_df, winner_data, metadata, bm_df = load_evaluation_artifacts()

    if summary_df is None:
        st.warning("Data hasil evaluasi belum ditemukan.")
        return

    # =========================================================================
    # A. Hasil Perbandingan
    # =========================================================================
    st.markdown("### Hasil Perbandingan")

    # 1. Winning Model Banner
    winner_name = winner_data.get("winner", "weighted_scoring")
    winner_label = "Weighted Scoring" if winner_name == "weighted_scoring" else winner_name
    winner_rationale = winner_data.get("rationale", "")

    st.success(
        f"#### Metode Terpilih: {winner_label}\n\n"
        f"**Rata-rata Kualitas Urutan Rekomendasi (Mean NDCG@5): 0,9786**\n\n"
        f"**Alasan Pemilihan:**\n\n"
        f"Berdasarkan hasil pengujian, metode Weighted Scoring berhasil unggul "
        f"pada Tahap 1, yaitu kriteria utama dalam prosedur pemilihan metode, "
        f"dengan memperoleh nilai NDCG@5 tertinggi dibandingkan KNN Euclidean "
        f"dan Cosine Similarity. Nilainya mencapai 0,9786, "
        f"lebih tinggi 0,2052 dibandingkan KNN Euclidean yang berada di posisi kedua, "
        f"jauh di atas toleransi 0,01.\n\n"
        f"Dengan hasil tersebut, Weighted Scoring ditetapkan sebagai metode "
        f"dengan kualitas urutan rekomendasi terbaik dalam pengujian ini."
    )

    # 2. Key Metrics Cards
    k1, k2, k3, k4 = st.columns(4)
    ws_row = summary_df[summary_df["method"] == "weighted_scoring"].iloc[0]
    k1.metric("Kualitas Urutan (NDCG@5)", f"{ws_row['ndcg_5_mean']:.4f}")
    k2.metric("Ketepatan 5 Rekomendasi", f"{ws_row['precision_5_mean']:.2%}")
    k3.metric("Cakupan Laptop Sesuai", f"{ws_row['recall_5_mean']:.2%}")
    k4.metric("Waktu Pemrosesan", f"{ws_row['latency_aggregate_median_ms']:.2f} ms")

    st.markdown("---")

    # 3. Overall Comparison Table
    st.markdown("#### Tabel Perbandingan Ketiga Metode")

    name_map = {
        "knn_euclidean": "KNN Euclidean",
        "cosine_similarity": "Cosine Similarity",
        "weighted_scoring": "Weighted Scoring [Pemenang]",
    }
    short_name_map = {
        "knn_euclidean": "KNN Euclidean",
        "cosine_similarity": "Cosine Similarity",
        "weighted_scoring": "Weighted Scoring",
    }

    display_df = summary_df.copy()
    display_df["Metode"] = display_df["method"].map(name_map)
    display_df["Kualitas Urutan (NDCG@5)"] = display_df["ndcg_5_mean"].apply(lambda v: f"{v:.4f}")
    display_df["Std NDCG@5"] = display_df["ndcg_5_std"].apply(lambda v: f"{v:.4f}")
    display_df["Median NDCG@5"] = display_df["ndcg_5_median"].apply(lambda v: f"{v:.4f}")
    display_df["Ketepatan 5 Rekomendasi (Precision@5)"] = display_df["precision_5_mean"].apply(lambda v: f"{v:.2%}")
    display_df["Cakupan Laptop Sesuai (Recall@5)"] = display_df["recall_5_mean"].apply(lambda v: f"{v:.2%}")
    display_df["Cakupan Seluruh Katalog"] = display_df["catalog_coverage"].apply(lambda v: f"{v:.2%}")
    display_df["Cakupan Kumpulan Syarat"] = display_df["eligible_pool_coverage"].apply(lambda v: f"{v:.2%}")
    display_df["Waktu Median (ms)"] = display_df["latency_aggregate_median_ms"].apply(lambda v: f"{v:.2f}")
    display_df["Variasi Waktu (IQR ms)"] = display_df["latency_aggregate_iqr_ms"].apply(lambda v: f"{v:.2f}")

    table_cols = [
        "Metode", "Kualitas Urutan (NDCG@5)", "Std NDCG@5", "Median NDCG@5",
        "Ketepatan 5 Rekomendasi (Precision@5)", "Cakupan Laptop Sesuai (Recall@5)",
        "Cakupan Seluruh Katalog", "Cakupan Kumpulan Syarat",
        "Waktu Median (ms)", "Variasi Waktu (IQR ms)"
    ]
    st.dataframe(display_df[table_cols], hide_index=True, width="stretch")

    st.caption(
        "*Catatan Pengujian:* Metrik Kualitas Urutan (NDCG@5) dan Ketepatan 5 Rekomendasi (Precision@5) dievaluasi pada seluruh 72 skenario (N=72). "
        "Metrik Cakupan Laptop Sesuai (Recall@5) dievaluasi pada 56 skenario (N=56, karena terdapat 16 skenario dengan anggaran ketat di mana tidak ada laptop di pasar yang memenuhi kriteria relevansi). "
        "Cakupan Seluruh Katalog dihitung terhadap total 636 laptop referensi."
    )

    st.markdown("---")

    # 4. Seven Visual Comparison Charts
    st.markdown("#### Grafik Perbandingan Hasil Evaluasi")

    chart_base = summary_df.copy()
    chart_base["Metode"] = chart_base["method"].map(short_name_map)

    # Kelompok 1: Kualitas Urutan dan Akurasi
    st.markdown("##### 1. Kualitas Urutan Rekomendasi dan Akurasi")
    g1_c1, g1_c2, g1_c3 = st.columns(3)

    with g1_c1:
        st.markdown("**Chart 1: Kualitas Urutan Rekomendasi (NDCG@5)**")
        df_ndcg = pd.DataFrame({
            "Metode": chart_base["Metode"],
            "Skor NDCG@5": chart_base["ndcg_5_mean"]
        })
        st.bar_chart(df_ndcg, x="Metode", y="Skor NDCG@5", color="Metode")
        st.caption("Menilai apakah laptop yang paling sesuai dengan kebutuhan pengguna sudah ditempatkan pada urutan teratas. Semakin tinggi nilainya, semakin baik susunan rekomendasi yang dihasilkan (skala 0.0 s/d 1.0).")

    with g1_c2:
        st.markdown("**Chart 2: Ketepatan 5 Rekomendasi (Precision@5)**")
        df_prec = pd.DataFrame({
            "Metode": chart_base["Metode"],
            "Ketepatan Rekomendasi": chart_base["precision_5_mean"]
        })
        st.bar_chart(df_prec, x="Metode", y="Ketepatan Rekomendasi", color="Metode")
        st.caption("Menghitung berapa banyak dari lima laptop yang direkomendasikan memenuhi kriteria relevansi penelitian. Semakin banyak laptop yang sesuai, semakin tinggi nilai Precision@5.")

    with g1_c3:
        st.markdown("**Chart 3: Cakupan Laptop yang Sesuai (Recall@5)**")
        df_rec = pd.DataFrame({
            "Metode": chart_base["Metode"],
            "Cakupan Sesuai": chart_base["recall_5_mean"]
        })
        st.bar_chart(df_rec, x="Metode", y="Cakupan Sesuai", color="Metode")
        st.caption("Menunjukkan seberapa banyak pilihan laptop yang sebenarnya relevan berhasil ditemukan oleh metode. Semakin banyak laptop relevan yang masuk ke lima posisi teratas, semakin tinggi nilai Recall@5.")

    # Kelompok 2: Cakupan Rekomendasi (Coverage)
    st.markdown("##### 2. Cakupan Rekomendasi (Coverage)")
    g2_c1, g2_c2 = st.columns(2)

    with g2_c1:
        st.markdown("**Chart 4: Cakupan Seluruh Katalog (Catalog-Wide Coverage)**")
        df_cat = pd.DataFrame({
            "Metode": chart_base["Metode"],
            "Cakupan Katalog": chart_base["catalog_coverage"]
        })
        st.bar_chart(df_cat, x="Metode", y="Cakupan Katalog", color="Metode")
        st.caption("Menunjukkan seberapa luas metode menggunakan pilihan laptop yang tersedia dalam katalog. Semakin banyak laptop berbeda yang pernah direkomendasikan, semakin tinggi nilai cakupannya.")

    with g2_c2:
        st.markdown("**Chart 5: Cakupan Kumpulan Laptop yang Memenuhi Syarat (Eligible Pool Coverage)**")
        df_pool = pd.DataFrame({
            "Metode": chart_base["Metode"],
            "Cakupan Kumpulan": chart_base["eligible_pool_coverage"]
        })
        st.bar_chart(df_pool, x="Metode", y="Cakupan Kumpulan", color="Metode")
        st.caption("Menggambarkan seberapa banyak pilihan yang tersedia setelah proses filter berhasil digunakan oleh metode rekomendasi. Metrik ini membantu melihat apakah metode hanya berulang kali memilih laptop tertentu atau memberikan pilihan yang lebih beragam.")

    # Kelompok 3: Waktu Pemrosesan (Latensi)
    st.markdown("##### 3. Kecepatan dan Stabilitas Pemrosesan (Latency)")
    g3_c1, g3_c2 = st.columns(2)

    with g3_c1:
        st.markdown("**Chart 6: Waktu Pemrosesan Rata-rata (Median Latency)**")
        df_lat_med = pd.DataFrame({
            "Metode": chart_base["Metode"],
            "Waktu Pemrosesan (ms)": chart_base["latency_aggregate_median_ms"]
        })
        st.bar_chart(df_lat_med, x="Metode", y="Waktu Pemrosesan (ms)", color="Metode")
        st.caption("Menunjukkan seberapa cepat metode menghasilkan lima rekomendasi untuk satu skenario pengujian. Nilai ditampilkan dalam milidetik (ms), dengan nilai lebih rendah menunjukkan waktu pemrosesan yang lebih singkat.")

    with g3_c2:
        st.markdown("**Chart 7: Variasi Waktu Pemrosesan (Latency IQR)**")
        df_lat_iqr = pd.DataFrame({
            "Metode": chart_base["Metode"],
            "Variasi Waktu (ms)": chart_base["latency_aggregate_iqr_ms"]
        })
        st.bar_chart(df_lat_iqr, x="Metode", y="Variasi Waktu (ms)", color="Metode")
        st.caption("Mengukur seberapa besar perubahan waktu pemrosesan dalam beberapa kali pengujian. Nilai yang lebih kecil menunjukkan bahwa waktu pemrosesan metode cenderung lebih konsisten.")

    st.markdown("---")

    # =========================================================================
    # B. Bagaimana Pengujian Dilakukan?
    # =========================================================================
    st.markdown("### Bagaimana Pengujian Dilakukan?")
    with st.container(border=True):
        st.markdown(
            "Pengujian performa sistem rekomendasi dilakukan secara objektif melalui tahapan berikut:\n\n"
            "1. **Katalog Data Laptop**: Evaluasi menggunakan 636 laptop yang memiliki data performa prosesor (CPU PassMark) terverifikasi dari katalog toko komputer.\n"
            "2. **72 Skenario Pengujian Terkontrol**: Disusun 72 skenario pengujian terstruktur yang mengombinasikan berbagai variasi anggaran, target RAM, target kapasitas penyimpanan, dan target performa prosesor.\n"
            "3. **Kombinasi Kebutuhan Pengguna**: Skenario tersebut mencakup berbagai profil kebutuhan pengembangan aplikasi, mulai dari kebutuhan perkuliahan dasar hingga workstation tingkat lanjut.\n"
            "4. **Penetapan Tingkat Relevansi (Relevance Grade)**: Setiap laptop kandidat dinilai kesesuaiannya dan diberi label relevansi bergradasi dari 0 hingga 3 (0 = Tidak Relevan, 1 = Cukup Relevan, 2 = Relevan, 3 = Sangat Relevan) berdasarkan kriteria penelitian yang telah ditentukan sebelumnya.\n"
            "5. **Kondisi Pengujian yang Identik**: Ketiga metode rekomendasi (KNN Euclidean, Cosine Similarity, dan Weighted Scoring) menerima skenario pengujian dan kumpulan kandidat laptop yang sama persis untuk memastikan perbandingan yang adil (*apple-to-apple*).\n"
            "6. **Perbandingan Urutan Rekomendasi**: Daftar urutan rekomendasi dari ketiga metode diuji dan dibandingkan menggunakan metrik evaluasi baku (NDCG@5, Precision@5, Recall@5, Coverage, dan Waktu Pemrosesan).\n"
            "7. **Pemilihan Metode Terbaik**: Metode pemenang dipilih menggunakan prosedur keputusan bertingkat (*hierarchical decision procedure*) formal yang telah ditentukan sebelum pengujian dimulai.\n\n"
            "**Catatan Penting Penelitian:**\n"
            "- **Aturan Relevansi Penelitian**: Tingkat relevansi (grade 0–3) dihasilkan secara otomatis berdasarkan aturan penelitian terdefinisi. Nilai ini **bukan** survei kepuasan subjektif pengguna nyata atau penilaian langsung pakar untuk setiap unit laptop.\n"
            "- **Skenario Terkontrol**: Sebanyak 72 skenario pengujian merupakan kasus uji terkontrol yang dirancang peneliti untuk mensimulasikan variasi kebutuhan, **bukan** 72 pengguna nyata."
        )

    st.markdown("---")

    # =========================================================================
    # C. Apa Arti Hasil Pengujian?
    # =========================================================================
    st.markdown("### Apa Arti Hasil Pengujian?")
    with st.container(border=True):
        st.markdown(
            "Setiap metrik evaluasi memiliki makna spesifik dalam menilai keandalan sistem rekomendasi:\n\n"
            "- **Kualitas Urutan Rekomendasi (NDCG@5)**: Mengukur apakah laptop yang paling sesuai muncul di bagian paling atas daftar rekomendasi. Nilai mendekati 1.0 menunjukkan susunan rekomendasi yang sangat ideal.\n"
            "- **Ketepatan 5 Rekomendasi (Precision@5)**: Mengukur berapa banyak dari 5 laptop yang direkomendasikan memenuhi kriteria relevansi penelitian.\n"
            "- **Cakupan Laptop yang Sesuai (Recall@5)**: Mengukur seberapa banyak laptop relevan yang berhasil ditemukan dan masuk ke dalam 5 rekomendasi teratas dari seluruh pilihan yang tersedia di pasar.\n"
            "- **Cakupan Seluruh Katalog (Catalog-wide Coverage)**: Mengukur seberapa banyak variasi laptop di seluruh katalog (636 laptop) yang pernah direkomendasikan. Metrik ini menggambarkan keberagaman alternatif produk, bukan akurasi rekomendasi.\n"
            "- **Cakupan Kumpulan Laptop Memenuhi Syarat (Eligible-pool Coverage)**: Mengukur seberapa banyak laptop dari kumpulan yang lolos syarat batas anggaran dan kelayakan yang pernah direkomendasikan.\n"
            "- **Waktu Pemrosesan Rata-rata (Median Latency)**: Mengukur waktu komputasi tipikal yang dibutuhkan algoritma untuk menghasilkan rekomendasi (dalam milidetik). **Nilai yang lebih rendah lebih baik** karena menunjukkan sistem yang lebih cepat dan responsif.\n"
            "- **Variasi Waktu Pemrosesan (Latency IQR)**: Menggambarkan konsistensi kecepatan pemrosesan antar-pencarian. Nilai yang lebih kecil berarti performa sistem lebih stabil.\n\n"
            "**Catatan Interpretasi Nilai Metrik:**\n"
            "- Nilai yang lebih tinggi **tidak selalu lebih baik** untuk semua metrik. Sebagai contoh, waktu pemrosesan justru lebih baik jika bernilai rendah. Cakupan katalog mencerminkan variasi sebaran rekomendasi dan bukan ukuran ketepatan ranking.\n\n"
            "**Penjelasan 16 Skenario Tanpa Recall@5:**\n"
            "- Terdapat **16 skenario yang tidak diikutsertakan** dalam perhitungan rata-rata Recall@5. Pada 16 skenario tersebut (kombinasi anggaran terbatas dengan target spesifikasi tinggi), tidak ada satu pun laptop di katalog yang memenuhi kriteria relevansi minimum (jumlah laptop relevan = 0). Oleh karena itu, metrik Recall tidak dapat dihitung secara matematis (terjadi pembagian dengan nol) dan dikecualikan agar tidak membiaskan nilai rata-rata."
        )

    st.markdown("---")

    # =========================================================================
    # D. Keterbatasan Pengujian
    # =========================================================================
    st.markdown("### Keterbatasan Pengujian")
    with st.container(border=True):
        st.markdown(
            "Hasil evaluasi ini perlu dipahami dengan memperhatikan beberapa batasan penelitian:\n\n"
            "1. **Sumber Data Katalog Tunggal**: Data laptop diperoleh dari satu snapshot inventaris katalog toko komputer (Plaza IT). Karakteristik harga dan ketersediaan stok mencerminkan kondisi toko tersebut dan belum tentu mewakili pasar laptop secara global.\n"
            "2. **Skenario Terkontrol dan Sintetis**: Pengujian menggunakan 72 skenario terstruktur buatan peneliti untuk keperluan pengujian algoritma, bukan sampel acak dari survei perilaku pengguna nyata.\n"
            "3. **Relevansi Berbasis Aturan Logis**: Derajat relevansi dihitung menggunakan aturan kriteria penelitian terdefinisi, sehingga tidak mencerminkan preferensi personal atau kepuasan subjektif pengguna perorangan.\n"
            "4. **CPU PassMark sebagai Proksi Estimasi**: Skor benchmark CPU PassMark merupakan proksi estimasi kemampuan komputasi prosesor secara umum, bukan pengukuran langsung durasi kompilasi proyek (*clean build*) atau rendering emulator di Android Studio.\n"
            "5. **Bukan Jaminan Kesesuaian Mutlak**: Kemenangan metode pada benchmark ini membuktikan keunggulan kualitas pemeringkatan algoritma pada kondisi uji terkontrol, namun tidak memberikan jaminan kepuasan mutlak bagi setiap individu atau kesesuaian universal di luar kondisi pengujian."
        )

    with st.expander("Tabel Lengkap Data Hasil Pengujian (216 Baris)"):
        st.caption("Data mentah hasil evaluasi dari 72 skenario pengujian pada 3 metode rekomendasi.")
        st.dataframe(bm_df, hide_index=True, width="stretch")


def render_basis_tab():
    st.subheader("Sistem Rekomendasi")
    st.caption("Penjelasan alur kerja, logika penentuan spesifikasi, dan formulasi metode rekomendasi.")

    # A. Memahami Kebutuhan Pengguna
    with st.container(border=True):
        st.markdown(
            "#### A. Memahami Kebutuhan Pengguna\n"
            "Sistem menyediakan 4 pilihan preset kebutuhan yang menggambarkan aktivitas pengembangan aplikasi Android yang berbeda:\n\n"
            "1. **Pengembangan Ringan / Kuliah Dasar**\n"
            "   - Ditujukan untuk mahasiswa atau pemula yang baru mempelajari pemrograman dasar, membuat aplikasi berskala kecil (satu modul), "
            "dan melakukan pengujian aplikasi langsung pada smartphone fisik (USB debugging) tanpa emulator berat.\n"
            "   - Target spesifikasi: RAM 16 GB, Kapasitas Penyimpanan SSD 512 GB, dan Prosesor Tingkat Cukup.\n\n"
            "2. **Mobile Development / Android Studio (Standar)**\n"
            "   - Ditujukan untuk alur kerja harian pengembang aplikasi: menjalankan Android Studio, satu emulator Android (AVD) lokal, "
            "serta peramban web untuk membaca dokumentasi secara bersamaan.\n"
            "   - Target spesifikasi: RAM 16 GB, Kapasitas Penyimpanan SSD 512 GB, dan Prosesor Tingkat Sedang.\n\n"
            "3. **Mobile Dev + UI/UX & Asset Design**\n"
            "   - Ditujukan untuk alur kerja gabungan: pemrograman aplikasi sekaligus perancangan visual antarmuka menggunakan aplikasi desain grafis "
            "(seperti Figma atau editor grafis lainnya) dan rendering aset gambar resolusi tinggi.\n"
            "   - Target spesifikasi: RAM 24 GB, Kapasitas Penyimpanan SSD 1 TB, dan Prosesor Tingkat Tinggi.\n\n"
            "4. **Workstation / Multi-Emulator Intensif**\n"
            "   - Ditujukan untuk beban kerja berat berskala profesional: menjalankan beberapa emulator Android secara bersamaan "
            "(misalnya menguji tampilan tablet dan smartphone sekaligus), pengujian otomatis, atau kompilasi basis kode proyek besar secara berulang.\n"
            "   - Target spesifikasi: RAM 32 GB, Kapasitas Penyimpanan SSD 1 TB, dan Prosesor Tingkat Sangat Tinggi (P90, ~34.222 PassMark)."
        )

    # B. Menentukan Spesifikasi Laptop
    with st.container(border=True):
        st.markdown(
            "#### B. Menentukan Spesifikasi Laptop\n"
            "Tiga komponen utama berikut menjadi penentu utama kelayakan laptop untuk pengembangan Android dalam model penelitian ini:\n\n"
            "1. **Kapasitas RAM (Memori)**\n"
            "   - Android Studio berjalan di atas Java Virtual Machine (JVM) dan menjalankan proses kompilasi Gradle di latar belakang. "
            "Selain itu, emulator Android (AVD) mengalokasikan RAM fisik tersendiri dari komputer induk. "
            "Dalam model penelitian ini, kapasitas RAM 16 GB diadopsi sebagai batas kelayakan dasar untuk alur kerja IDE dan emulator lokal yang stabil, "
            "meskipun pengembangan Android yang sangat ringan tanpa emulator terkadang masih dapat berjalan pada spesifikasi di bawahnya.\n\n"
            "2. **Performa Prosesor (CPU)**\n"
            "   - Proses kompilasi kode sumber Kotlin/Java dan pembangunan aplikasi melalui Gradle dapat memanfaatkan kemampuan multi-core dan multi-threading prosesor. "
            "Karena itu, performa CPU menjadi salah satu faktor yang berkontribusi terhadap waktu pembangunan aplikasi. "
            "Dalam penelitian ini, skor CPU PassMark digunakan sebagai proksi kemampuan komputasi prosesor, bukan pengukuran langsung durasi kompilasi Android Studio.\n\n"
            "3. **Kapasitas Penyimpanan (Storage SSD)**\n"
            "   - File instalasi Android Studio, Android SDK, image emulator, direktori dependensi, serta cache Gradle membutuhkan ruang penyimpanan yang besar. "
            "Sistem mengevaluasi kapasitas penyimpanan dalam gigabyte (GB) untuk menjamin kelapangan ruang penampungan file, bukan mengukur kecepatan transfer data baca-tulis (*read/write speed*) SSD secara langsung.\n\n"
            "*(Catatan: Target performa prosesor diselaraskan secara empiris dengan sebaran data inventaris katalog laptop yang tersedia)*."
        )

        st.markdown(
            "**Tabel Acuan Target CPU PassMark per Tingkat Prosesor:**\n\n"
            "| Tingkat Prosesor | Label Deskriptif | Target PassMark |\n"
            "|:---|:---|---:|\n"
            "| Rendah | Baseline Minimum (P10) | ~10.510 |\n"
            "| Menengah Bawah | Menengah Bawah (Q1) | ~14.823 |\n"
            "| Sedang | Standar Direkomendasikan (Median) | ~18.871 |\n"
            "| Tinggi | Performa Tinggi (Q3) | ~24.969 |\n"
            "| Sangat Tinggi | Sangat Tinggi / Workstation (P90) | ~34.222 |\n\n"
            "*Nilai target bersumber dari sebaran persentil skor CPU PassMark pada katalog laptop yang diteliti, dan digunakan untuk menormalisasi skor kesesuaian prosesor dalam model penelitian. Tabel ini menampilkan seluruh skala referensi yang tersedia; setiap preset beban kerja memilih satu tingkat target tertentu dari skala tersebut.*"
        )

    # C. Menyaring Laptop
    with st.container(border=True):
        st.markdown(
            "#### C. Menyaring Laptop\n"
            "Sebelum penilaian dan pemeringkatan dilakukan, sistem menerapkan penyaringan otomatis yang bersifat mutlak (*hard eligibility rules*):\n\n"
            "1. **Batas Anggaran**: $\\text{Harga Laptop} \\le \\text{Anggaran Pengguna}$\n"
            "2. **Batas Kelayakan RAM**: $\\text{RAM} \\ge 16\\text{ GB}$\n"
            "3. **Batas Kelayakan Penyimpanan**: $\\text{Kapasitas SSD} \\ge 256\\text{ GB}$\n\n"
            "Laptop yang tidak memenuhi salah satu dari ketiga kriteria kelayakan di atas langsung dieliminasi dari daftar pencarian. "
            "Dengan demikian, seluruh laptop yang masuk ke tahap penilaian dipastikan memenuhi kriteria dasar pengembangan yang telah ditetapkan dalam penelitian ini."
        )

    # D. Memberikan Skor Kesesuaian
    with st.container(border=True):
        st.markdown(
            "#### D. Memberikan Skor Kesesuaian\n"
            "Setiap laptop yang lolos penyaringan kelayakan dinilai kesesuaiannya terhadap target pengguna menggunakan metode **Weighted Scoring**.\n\n"
            "Tiga atribut laptop dinilai berdasarkan tingkat kepentingannya:\n"
            "- **RAM**: Bobot **45%** ($0.45$) — komponen paling berpengaruh terhadap kelancaran emulator dan multitasking.\n"
            "- **Performa Prosesor**: Bobot **35%** ($0.35$) — penentu kecepatan kompilasi kode dan kelancaran eksekusi aplikasi.\n"
            "- **Kapasitas Penyimpanan**: Bobot **20%** ($0.20$) — penampung file sistem, SDK, dan cache proyek.\n\n"
            "**Formula Skor Kesesuaian Akhir:**\n"
            "$$\\text{Skor Kesesuaian} = (\\text{Nilai RAM} \\times 0.45) + (\\text{Nilai Prosesor} \\times 0.35) + (\\text{Nilai Penyimpanan} \\times 0.20)$$\n\n"
            "**Penilaian untuk Setiap Atribut:**\n"
            "- **Jika spesifikasi laptop kurang dari target**: laptop menerima pengurangan skor secara proporsional sesuai selisihnya terhadap target.\n"
            "- **Jika spesifikasi laptop memenuhi atau melebihi target**: laptop menerima nilai dasar penuh (80%), ditambah bonus kesesuaian tambahan (hingga 100%) jika memiliki spesifikasi lebih tinggi dari target yang diminta pengguna.\n\n"
            "*Pembobotan tersebut merupakan preferensi desain model penelitian yang ditetapkan berdasarkan pertimbangan kebutuhan beban kerja pengembangan Android. Bobot tidak dimaksudkan sebagai hasil pengukuran empiris kontribusi masing-masing komponen terhadap performa aktual.*"
        )

    # E. Menentukan Urutan Rekomendasi
    with st.container(border=True):
        st.markdown(
            "#### E. Menentukan Urutan Rekomendasi\n"
            "Setelah seluruh laptop dinilai:\n\n"
            "1. Laptop diurutkan dari skor kesesuaian tertinggi ke terendah.\n"
            "2. **Aturan Penanganan Skor Seri**: Jika dua laptop memiliki skor kesesuaian yang sama persis (dalam batas toleransi $10^{-7}$), "
            "laptop dengan harga lebih murah akan ditempatkan di posisi lebih atas. Jika harganya juga sama, urutan katalog asli digunakan agar urutan selalu konsisten.\n"
            "3. Sistem menyajikan **5 laptop terbaik (Top-5)** kepada pengguna."
        )

    # F. Mengapa Menggunakan Metode Weighted Scoring?
    with st.container(border=True):
        st.markdown(
            "#### F. Mengapa Menggunakan Metode Weighted Scoring?\n"
            "Tiga metode rekomendasi berbasis konten telah dibandingkan dalam pengujian offline pada 72 skenario terkontrol:\n\n"
            "- **KNN Euclidean**: Rata-rata Kualitas Urutan (NDCG@5) = 0.7734\n"
            "- **Cosine Similarity**: Rata-rata Kualitas Urutan (NDCG@5) = 0.6988\n"
            "- **Weighted Scoring**: Rata-rata Kualitas Urutan (NDCG@5) = **0.9786**\n\n"
            "**Keunggulan Metode Weighted Scoring:**\n"
            "Pada pencarian laptop, pengguna menetapkan spesifikasi sebagai batas kebutuhan yang diinginkan. "
            "Laptop yang memiliki spesifikasi lebih tinggi dari target adalah sebuah keuntungan fungsional. "
            "Metode Weighted Scoring menghargai kelebihan spesifikasi ini sebagai bonus kesesuaian.\n\n"
            "Pada konfigurasi penelitian ini, metode KNN Euclidean mengukur kedekatan spesifikasi terhadap target. "
            "Akibatnya, laptop dengan spesifikasi yang jauh melampaui target dapat memperoleh posisi lebih rendah dibandingkan laptop yang spesifikasinya lebih dekat dengan target, "
            "meskipun spesifikasi yang lebih tinggi sebenarnya dapat memberikan kapasitas tambahan bagi pengguna.\n\n"
            "*(Catatan: Kemenangan metode ini didasarkan pada prosedur pengujian terkontrol pada data katalog yang diteliti, dan bukan merupakan jaminan kepuasan subjektif mutlak untuk setiap individu)*.\n\n"
            "*Penilaian relevansi dalam benchmark dibentuk berdasarkan aturan penelitian yang menggunakan atribut RAM, CPU, dan kapasitas penyimpanan. Oleh karena itu, hasil evaluasi mengukur kesesuaian pemeringkatan terhadap kriteria relevansi tersebut, bukan kepuasan pengguna nyata atau performa aktual laptop dalam menjalankan Android Studio.*"
        )


# ----------------------------------------------------------------------------
# main execution
# ----------------------------------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
DF, REF = load_catalog_cached()
init_state()

with st.sidebar:
    st.markdown("### Status Sistem")
    st.write("**Arsitektur**:  Zero-Database")
    st.write("**Metode Rekomendasi**: Weighted Scoring")
    st.write("**Sumber Data**: Katalog CSV")
    st.write(f"**Ukuran Katalog**: {len(DF)} laptop terverifikasi")
    if st.button("Muat ulang katalog"):
        st.cache_resource.clear()
        st.rerun()

st.title("Perbandingan Metode K-Nearest Neighbors, Cosine Similarity, dan Weighted Scoring")
st.caption(
    "Prototipe untuk evaluasi dan menguji metode rekomendasi laptop berdasarkan kebutuhan Android Developer."
)

tab_model_eval, tab_rec, tab_basis = st.tabs([
    "Evaluasi Model",
    "Uji Rekomendasi",
    "Lainnya",
])

# ----------------------------------------------------------------------------
# Tab 1: Evaluasi Model (Default Landing Page)
# ----------------------------------------------------------------------------
with tab_model_eval:
    render_model_evaluation_tab()

# ----------------------------------------------------------------------------
# Tab 2: Uji Rekomendasi
# ----------------------------------------------------------------------------
with tab_rec:
    st.subheader("1. Pilih Preset Kebutuhan")

    current_prof = st.session_state.get("profile", "programming")
    if current_prof not in S.PROFILE_ORDER:
        current_prof = "programming"
        st.session_state["profile"] = current_prof

    prof_idx = S.PROFILE_ORDER.index(current_prof)

    st.selectbox(
        "Preset profil beban kerja",
        S.PROFILE_ORDER,
        index=prof_idx,
        key="profile",
        format_func=lambda k: PROFILES[k]["label"],
        on_change=on_profile_change,
        label_visibility="collapsed"
    )

    prof_key = st.session_state["profile"]
    prof_desc = PROFILE_DESCRIPTIONS.get(prof_key, "")
    prof_label = PROFILES[prof_key]["label"]

    st.info(
        f"Preset **{prof_label}**: {prof_desc}\n\n"
        f"Target bawaan: RAM {PROFILES[prof_key]['target_ram']} GB, "
        f"Penyimpanan {PROFILES[prof_key]['target_storage']} GB, "
        f"Prosesor {S.CPU_LABELS[PROFILES[prof_key]['cpu']]}. "
        f"Anda dapat menyesuaikan kriteria di bawah secara fleksibel."
    )

    if st.session_state.get("profile_notice"):
        st.toast(st.session_state.pop("profile_notice"))

    cpu_opts = dict(S.cpu_options(REF))

    st.subheader("2. Tentukan Anggaran dan Target Spesifikasi")
    st.caption("Sistem secara otomatis menyaring laptop berdasarkan persyaratan minimum, yaitu RAM minimal 16 GB, SSD minimal 256 GB, dan harga di bawah atau sama dengan anggaran yang ditentukan.")

    with st.form("query_form"):
        # Baris 1: Budget & Target Performa Prosesor
        col1, col2 = st.columns(2)
        col1.number_input(
            "Budget Maksimum (Rp)",
            min_value=S.BUDGET_MIN, max_value=S.BUDGET_MAX,
            step=S.BUDGET_STEP, key="budget",
            help="Batas anggaran maksimal. Laptop dengan harga di atas anggaran akan otomatis dieliminasi."
        )

        cpu_val = st.session_state.get("cpu_level", PROFILES[prof_key]["cpu"])
        if cpu_val not in cpu_opts:
            cpu_val = PROFILES[prof_key]["cpu"]
        col2.selectbox(
            "Target Performa Prosesor",
            list(cpu_opts),
            index=list(cpu_opts).index(cpu_val),
            key="cpu_level",
            format_func=lambda k: cpu_opts[k],
            help="Target performa prosesor sebagai acuan penilaian kesesuaian."
        )

        # Baris 2: Target RAM & Target Storage
        r1, r2 = st.columns(2)
        tram_val = st.session_state.get("target_ram", PROFILES[prof_key]["target_ram"])
        if tram_val not in S.RAM_CHOICES:
            tram_val = PROFILES[prof_key]["target_ram"]
        r1.selectbox(
            "Target RAM",
            S.RAM_CHOICES,
            index=S.RAM_CHOICES.index(tram_val),
            key="target_ram",
            format_func=lambda v: f"{v} GB",
            help="Kapasitas RAM yang paling diinginkan untuk aktivitas pengembangan."
        )

        tsto_val = st.session_state.get("target_storage", PROFILES[prof_key]["target_storage"])
        if tsto_val not in S.STORAGE_CHOICES:
            tsto_val = PROFILES[prof_key]["target_storage"]
        r2.selectbox(
            "Target Kapasitas Penyimpanan",
            S.STORAGE_CHOICES,
            index=S.STORAGE_CHOICES.index(tsto_val),
            key="target_storage",
            format_func=lambda v: f"{v} GB" if v < 1024 else f"{v // 1024} TB",
            help="Kapasitas penyimpanan SSD yang paling diinginkan."
        )

        with st.expander("Filter layar (opsional)"):
            st.caption("Informasi layar dibaca dari deskripsi produk.")
            f1, f2 = st.columns(2)
            inch_val = st.session_state.get("screen_inch", 0)
            if inch_val not in S.SCREEN_INCH_CHOICES:
                inch_val = 0
            f1.selectbox(
                "Ukuran Layar Minimum",
                S.SCREEN_INCH_CHOICES,
                index=S.SCREEN_INCH_CHOICES.index(inch_val),
                key="screen_inch",
                format_func=lambda v: "Tanpa batas" if v == 0 else f"{v} inci"
            )

            res_choices = [""] + S.RES_CHOICES
            res_val = st.session_state.get("screen_res", "")
            if res_val not in res_choices:
                res_val = ""
            f2.selectbox(
                "Resolusi Layar Minimum",
                res_choices,
                index=res_choices.index(res_val),
                key="screen_res",
                format_func=lambda v: "Tanpa batas" if v == "" else v
            )
            st.checkbox("Hanya laptop dengan panel OLED", key="screen_oled")

        submitted = st.form_submit_button("Cari Rekomendasi", type="primary", use_container_width=True)

    if submitted:
        # Enforce fixed eligibility rules: Price <= budget, RAM >= 16 GB, Storage >= 256 GB
        params = {
            "profile": st.session_state["profile"],
            "budget": st.session_state["budget"],
            "target_ram": st.session_state["target_ram"],
            "min_ram": 16,
            "target_storage": st.session_state["target_storage"],
            "min_storage": 256,
            "cpu_level": st.session_state["cpu_level"],
            "min_screen_inch": st.session_state["screen_inch"],
            "min_resolution": st.session_state["screen_res"],
            "oled_only": st.session_state["screen_oled"]
        }
        try:
            data = S.run_query(DF, REF, params)
        except ValueError as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))
        else:
            st.session_state["result"] = {"data": data}

    if st.session_state.get("result"):
        render_result(st.session_state["result"])

# ----------------------------------------------------------------------------
# Tab 3: Lainnya
# ----------------------------------------------------------------------------
with tab_basis:
    render_basis_tab()

st.caption("Prototipe evaluasi perbandingan metode rekomendasi laptop menggunakan KNN, Cosine Similarity, dan Weighted Scoring.")
