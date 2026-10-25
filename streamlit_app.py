from __future__ import annotations

import pandas as pd
import streamlit as st

import db
import service as S

st.set_page_config(page_title="Rekomendasi Laptop", layout="centered")

PROFILES = S.profile_defaults()
USE_STARS = hasattr(st, "feedback")
FORM_KEYS = ("budget", "target_ram", "min_ram", "target_storage", "min_storage", "cpu_level", "gpu_level", "dedicated")

PROFILE_DESCRIPTIONS = {
    "office": (
        "Untuk mengetik, dokumen, spreadsheet, browsing, dan administrasi ringan. "
        "Mengutamakan efisiensi daya dengan RAM 8 GB dan performa prosesor hemat energi."
    ),
    "programming": (
        "Untuk coding, kompilasi software, multi-tab browsing, dan tools developer. "
        "Rekomendasi default profil: RAM 16 GB dan CPU level Sedang. "
        "Anda dapat mengubah target RAM ke 8 GB jika kebutuhan kerja Anda lebih ringan."
    ),
    "design": (
        "Untuk desain grafis, editing foto/video, dan rendering 2D/3D. "
        "Memerlukan prosesor level Tinggi dan akselerasi GPU (dedicated atau setara)."
    ),
    "gaming": (
        "Untuk bermain game 3D dengan lancar. "
        "Mewajibkan GPU dedicated dan CPU level Tinggi demi framerate optimal."
    ),
}

CUSTOM_CSS = """
<style>
/* Base spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 860px;
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
# configuration, data, DB helpers
# ----------------------------------------------------------------------------
def get_db_config():
    try:
        sec = st.secrets["mysql"]
        return dict(host=sec["host"], port=int(sec.get("port", 3306)), user=sec["user"],
                    password=sec.get("password", ""), database=sec["database"])
    except Exception:
        return db.read_config_from_env()


def db_call(fn, *args, **kwargs):
    cfg = get_db_config()
    if not cfg:
        raise RuntimeError("Database belum dikonfigurasi.")
    conn = db.connect(cfg)
    try:
        return fn(conn, *args, **kwargs)
    finally:
        conn.close()


@st.cache_resource(show_spinner="Memuat katalog...")
def load_catalog_cached():
    """Returns (DataFrame, reference targets, source, message). MySQL first, CSV as fallback."""
    message = ""
    if get_db_config():
        try:
            raw = db_call(db.load_catalog)
            if len(raw):
                df, ref = S.prepare_catalog(raw)
                return df, ref, "mysql", ""
            message = "Tabel laptops di MySQL kosong. Jalankan import_catalog.py."
        except Exception as exc:
            message = f"MySQL tidak dapat dipakai ({type(exc).__name__}: {exc})."
    else:
        message = "MySQL belum dikonfigurasi."
    raw = pd.read_csv(S.v8.DEFAULT_INPUT)
    df, ref = S.prepare_catalog(raw)
    return df, ref, "csv", message


# ----------------------------------------------------------------------------
# form state management
# ----------------------------------------------------------------------------
def on_profile_change():
    """Consistent state update: updates target specs when profile is changed, preserving budget."""
    new_prof = st.session_state.get("profile")
    if new_prof not in S.PROFILE_ORDER:
        new_prof = "programming"
        st.session_state["profile"] = new_prof

    d = PROFILES[new_prof]
    st.session_state["last_profile"] = new_prof
    st.session_state["target_ram"] = d["target_ram"]
    st.session_state["min_ram"] = d["min_ram"]
    st.session_state["target_storage"] = d["target_storage"]
    st.session_state["min_storage"] = d["min_storage"]
    st.session_state["cpu_level"] = d["cpu"]
    st.session_state["gpu_level"] = d["gpu"]
    st.session_state["dedicated"] = d["dedicated"]
    st.session_state["profile_notice"] = (
        f"Spesifikasi default diperbarui ke rekomendasi profil {d['label']}. "
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
    st.session_state.setdefault("min_ram", d["min_ram"])
    st.session_state.setdefault("target_storage", d["target_storage"])
    st.session_state.setdefault("min_storage", d["min_storage"])
    st.session_state.setdefault("cpu_level", d["cpu"])
    st.session_state.setdefault("gpu_level", d["gpu"])
    st.session_state.setdefault("dedicated", d["dedicated"])
    st.session_state.setdefault("consent", True)
    st.session_state.setdefault("screen_inch", 0)
    st.session_state.setdefault("screen_res", "")
    st.session_state.setdefault("screen_oled", False)


def md_escape(text: str) -> str:
    out = str(text)
    for ch in ("\\", "`", "*", "_", "[", "]", "$", "<", ">", "#", "|", "~"):
        out = out.replace(ch, "\\" + ch)
    return out


# ----------------------------------------------------------------------------
# feedback callbacks
# ----------------------------------------------------------------------------
def on_rate(session_id: str, laptop_id: int, key: str):
    value = st.session_state.get(key)
    if value is None:
        return
    rating = int(value) + 1 if USE_STARS else int(value)
    try:
        db_call(db.upsert_rating, session_id, int(laptop_id), rating)
        st.toast("Penilaian tersimpan. Terima kasih.")
    except Exception as exc:
        st.toast(f"Penilaian gagal disimpan: {exc}")


def on_comment(session_id: str, key: str):
    try:
        db_call(db.save_comment, session_id, st.session_state.get(key, ""))
        st.toast("Komentar tersimpan.")
    except Exception as exc:
        st.toast(f"Komentar gagal disimpan: {exc}")


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
def render_rating_widget(session_id: str, laptop_id: int):
    key = f"fb_{session_id}_{laptop_id}"
    st.caption("Kesesuaian laptop ini dengan kebutuhan Anda (1 = sangat kurang, 5 = sangat sesuai):")
    if USE_STARS:
        st.feedback("stars", key=key, on_change=on_rate, args=(session_id, laptop_id, key))
    else:
        st.radio("Penilaian", [1, 2, 3, 4, 5], index=None, horizontal=True, key=key,
                 label_visibility="collapsed", on_change=on_rate, args=(session_id, laptop_id, key))


def open_similar(source_row: int):
    st.session_state["similar_for"] = source_row


def close_similar():
    st.session_state["similar_for"] = None


def render_similar(params: dict, source_row: int):
    try:
        sim = S.similar_query(DF, REF, params, source_row)
    except ValueError as exc:
        st.error(str(exc))
        return
    with st.container(border=True):
        st.markdown(f"**Laptop yang mirip dengan {md_escape(sim['source_name'])}**")
        st.caption("Dicari berdasarkan kedekatan spesifikasi teknis dengan laptop acuan ini. "
                   "Batas anggaran dilonggarkan; RAM minimum, storage minimum, dan filter layar tetap berlaku.")
        if sim["message"]:
            st.warning(sim["message"])
        for it in sim["items"]:
            with st.container(border=True):
                header_html = (
                    f'<div style="margin-bottom:0.35rem; line-height:1.4;">'
                    f'<span style="font-weight:700; color:#64748b; font-size:1.05rem; margin-right:0.4rem;">#{it["rank"]}</span>'
                    f'<strong style="font-size:1.05rem;">{md_escape(it["name"])}</strong>'
                    f'</div>'
                )
                st.markdown(header_html, unsafe_allow_html=True)

                price_line = S.fmt_idr(it["price"])
                if it["over_budget"] > 0:
                    price_line += f"  (melampaui budget sebesar {S.fmt_idr(it['over_budget'])})"
                st.markdown(f'<div class="price-tag">{price_line}</div>', unsafe_allow_html=True)

                pills = (
                    f'<div class="specs-pills">'
                    f'<span class="spec-pill">RAM: {it["ram"]} GB</span>'
                    f'<span class="spec-pill">SSD: {it["storage"]} GB</span>'
                    f'<span class="spec-pill">CPU: {md_escape(it["processor"])}</span>'
                    f'<span class="spec-pill">GPU: {md_escape(it["graphics"])} ({it["gpu_class"]})</span>'
                    f'<span class="spec-pill">{md_escape(it["screen"])}</span>'
                    f'</div>'
                )
                st.markdown(pills, unsafe_allow_html=True)
                st.caption(f"Perbandingan dengan laptop acuan: {it['diff']}.")
                if it["variants"] > 0:
                    st.caption(f"+{it['variants']} varian lain dengan spesifikasi setara "
                               f"({S.fmt_idr(it['vmin'])} - {S.fmt_idr(it['vmax'])}).")
                if it["url"]:
                    st.link_button("Lihat di Plaza IT", it["url"])
        if sim["identical_n"] > 0:
            lo, hi = sim["identical_range"]
            st.caption(f"Ada {sim['identical_n']} laptop lain dengan spesifikasi identik pada fitur yang dipakai "
                       f"({S.fmt_idr(lo)} - {S.fmt_idr(hi)}); tidak ditampilkan sebagai mirip.")
        st.button("Tutup", key=f"simclose_{source_row}", on_click=close_similar)


def render_result(stored: dict):
    data, session_id = stored["data"], stored["session_id"]
    sm = data["summary"]
    st.subheader("Hasil Rekomendasi")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Katalog Total", S.fmt_num(sm["n_initial"]))
    c2.metric("Lolos Syarat Wajib", S.fmt_num(sm["n_after_basic_filter"]))
    c3.metric("Dapat Diranking", S.fmt_num(sm["n_rankable"]))
    c4.metric("Belum Diranking", S.fmt_num(data["unrankable_total"]))

    # Summary box
    target_parts = [f"**{t['label']}**: {t['value']}" for t in data["targets"]]
    if not data["gpu_in_ranking"]:
        target_parts.append("**GPU**: Tidak dihitung dalam ranking (profil non-GPU)")

    st.info(
        f"**Target Preferensi Aktif (Acuan Ranking KNN)**: {' · '.join(target_parts)}\n\n"
        f"**Batas Syarat Minimum**: Budget $\\le$ {S.fmt_idr(data['params']['budget'])} · "
        f"RAM $\\ge$ {data['params']['min_ram']} GB · "
        f"Storage $\\ge$ {data['params']['min_storage']} GB"
    )

    if data["screen_note"]:
        st.info(data["screen_note"])

    if not data["recs"]:
        st.warning(data["empty_reason"])
    else:
        if data["target_far"]:
            st.info("Hampir tidak ada laptop yang mencapai semua target pada budget ini. Hasil di bawah adalah "
                    "yang terdekat; perhatikan keterangan status pada tiap laptop.")

        for r in data["recs"]:
            with st.container(border=True):
                # 1. Nomor urutan dan nama laptop (menyatu, rapi, tanpa badge rounded)
                header_html = (
                    f'<div style="margin-bottom:0.35rem; line-height:1.4;">'
                    f'<span style="font-weight:700; color:#64748b; font-size:1.05rem; margin-right:0.4rem;">#{r["rank"]}</span>'
                    f'<strong style="font-size:1.05rem;">{md_escape(r["name"])}</strong>'
                    f'</div>'
                )
                st.markdown(header_html, unsafe_allow_html=True)

                # 2. Harga
                st.markdown(f'<div class="price-tag">{S.fmt_idr(r["price"])}</div>', unsafe_allow_html=True)

                # 3. Spesifikasi utama dalam format pills bersih tanpa emoji
                pills_html = (
                    f'<div class="specs-pills">'
                    f'<span class="spec-pill">RAM: {r["ram"]} GB</span>'
                    f'<span class="spec-pill">SSD: {r["storage"]} GB</span>'
                    f'<span class="spec-pill">CPU: {md_escape(r["processor"])}</span>'
                    f'<span class="spec-pill">GPU: {md_escape(r["graphics"])} ({r["gpu_class"]})</span>'
                    f'<span class="spec-pill">{md_escape(r["screen"])}</span>'
                    f'</div>'
                )
                st.markdown(pills_html, unsafe_allow_html=True)

                # 4. Status kesesuaian target dalam bentuk teks sederhana
                chips_html = '<div class="chips-container">' + "".join(
                    f'<span class="{"chip-ok" if c["ok"] else "chip-warn"}">{md_escape(c["text"])}</span>'
                    for c in r["chips"]
                ) + '</div>'
                st.markdown(chips_html, unsafe_allow_html=True)

                # 5. Bagian alasan rekomendasi yang dapat dibuka
                with st.expander("Mengapa laptop ini direkomendasikan?"):
                    st.markdown(f"**{md_escape(r['explain']['headline'])}**")
                    for p in r["explain"]["points"]:
                        st.markdown(f"- {md_escape(p)}")
                    st.caption("Penjelasan ini menunjukkan kedekatan spesifikasi terhadap target dan posisi harga, "
                               "bukan jaminan mutu komersial.")

                if r["variants"] > 0:
                    st.caption(f"+{r['variants']} varian lain dengan spesifikasi setara "
                               f"({S.fmt_idr(r['variant_price_min'])} - {S.fmt_idr(r['variant_price_max'])}).")

                # 6. Jarak KNN dengan penjelasan singkat
                st.caption(f"Jarak KNN: {r['distance']:.4f} (ukuran kedekatan multivariat terhadap target preferensi)")

                # 7. Tombol ke Plaza IT dan mencari laptop serupa
                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    if r["url"]:
                        st.link_button("Lihat di Plaza IT", r["url"], use_container_width=True)
                with btn_col2:
                    st.button("Cari yang mirip", key=f"sim_{session_id}_{r['source_row']}",
                              on_click=open_similar, args=(r["source_row"],), use_container_width=True)

                # 8. Evaluasi pengguna sederhana
                if session_id and r["laptop_id"] is not None:
                    render_rating_widget(session_id, r["laptop_id"])

            if st.session_state.get("similar_for") == r["source_row"]:
                render_similar(data["params"], r["source_row"])

        st.caption("Laptop dengan spesifikasi sama pada fitur yang dipakai dikelompokkan; yang ditampilkan adalah "
                   "varian termurah. Layar, baterai, garansi, dan merek tidak dinilai oleh model.")
        if session_id:
            ck = f"cm_{session_id}"
            st.text_area("Komentar untuk hasil ini (opsional)", key=ck, max_chars=1000)
            st.button("Simpan komentar", key=f"btn_{ck}", on_click=on_comment, args=(session_id, ck))

    if data["unrankable_total"] > 0:
        with st.expander(f"{data['unrankable_total']} laptop lolos syarat awal tetapi belum dapat diranking"):
            st.caption("Skor benchmark yang dibutuhkan tidak tersedia di basis data, atau jenis GPU belum dapat dipastikan "
                       "sebagai dedicated. Skor tidak diperkirakan sembarangan demi validitas ilmiah.")
            st.dataframe(data["unrankable"], hide_index=True, use_container_width=True)
            if data["unrankable_extra"] > 0:
                st.caption(f"dan {data['unrankable_extra']} laptop lainnya.")

    with st.expander("Cara membaca hasil dan algoritma"):
        st.markdown(
            "- **Batas Anggaran, RAM minimum, dan storage minimum** adalah syarat wajib (*hard filter*); "
            "laptop yang tidak memenuhinya tidak akan ditampilkan.\n"
            "- **Target RAM, Target Storage, CPU, dan GPU** adalah acuan ranking KNN. Laptop dengan spesifikasi "
            "paling mendekati target preferensi akan mendapatkan peringkat teratas.\n"
            "- Algoritma menggunakan **jarak Euclidean asimetris**: surplus spesifikasi di atas target diberi "
            "penalti lebih ringan daripada kekurangan (*shortfall*).\n"
            "- Harga bukan variabel penentu jarak ranking; harga murni batas kemampuan beli dan pemutus seri "
            "jika dua laptop memiliki jarak fitur identik."
        )


def render_eval_tab(source: str):
    st.subheader("Evaluasi Pengguna")
    if source != "mysql":
        st.info("Evaluasi pengguna memerlukan koneksi MySQL. Konfigurasikan database dan jalankan import_catalog.py.")
        return
    st.caption("Ringkasan umpan balik dari pengguna. Penilai memilih sendiri apakah menilai sehingga bersifat indikatif.")
    if not st.button("Muat ringkasan"):
        return
    try:
        ov = db_call(db.feedback_overview)
        export = db_call(db.export_feedback)
    except Exception as exc:
        st.error(f"Gagal membaca database: {exc}")
        return
    t = ov["totals"]
    a, b, c = st.columns(3)
    a.metric("Sesi pencarian", t["sessions"])
    b.metric("Sesi dinilai", t["sessions_with_rating"])
    c.metric("Jumlah rating", t["ratings"])
    if t["ratings"] == 0:
        st.info("Belum ada penilaian.")
        return
    st.markdown("**Rata-rata rating per profil**")
    st.dataframe(ov["by_profile"], hide_index=True)
    st.markdown("**Rata-rata rating per peringkat**")
    st.dataframe(ov["by_rank"], hide_index=True)
    if len(ov["by_rank"]):
        st.bar_chart(ov["by_rank"].set_index("peringkat")["rata2_rating"])
    st.markdown("**Sebaran rating**")
    st.dataframe(ov["distribution"], hide_index=True)
    st.download_button("Unduh data penilaian (CSV)", data=export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="feedback_export.csv", mime="text/csv")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
DF, REF, SOURCE, SOURCE_MSG = load_catalog_cached()
init_state()

with st.sidebar:
    st.markdown("### Sumber Katalog")
    st.write(f"**{'MySQL' if SOURCE == 'mysql' else 'CSV (cadangan)'}**: {len(DF)} laptop")
    if SOURCE_MSG:
        st.caption(SOURCE_MSG)
    if st.button("Muat ulang katalog"):
        st.cache_resource.clear()
        st.rerun()

st.title("Rekomendasi Laptop")
st.caption("Penerapan K-Nearest Neighbors (KNN) pada Content-Based Filtering Berbasis Web")

with st.expander("Informasi penelitian dan metodologi"):
    st.markdown(
        "Website ini membantu pengguna menemukan laptop yang sesuai dengan preferensi kebutuhan dan anggaran. "
        "Sistem menerapkan pemisahan tegas antara **Batas Minimum (Hard Filter)** untuk eliminasi laptop yang tidak memenuhi syarat, "
        "dan **Target Preferensi** sebagai magnet acuan perhitungan jarak KNN."
    )

tab_rec, tab_eval = st.tabs(["Rekomendasi", "Evaluasi Pengguna"])

with tab_rec:
    st.subheader("1. Pilih Profil Penggunaan")

    # Safe index retrieval
    current_prof = st.session_state.get("profile", "programming")
    if current_prof not in S.PROFILE_ORDER:
        current_prof = "programming"
        st.session_state["profile"] = current_prof

    prof_idx = S.PROFILE_ORDER.index(current_prof)

    st.selectbox(
        "Profil kebutuhan",
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
        f"Profil **{prof_label}**: {prof_desc}\n\n"
        f"Nilai bawaan: RAM target {PROFILES[prof_key]['target_ram']} GB, "
        f"Storage target {PROFILES[prof_key]['target_storage']} GB, "
        f"CPU {S.CPU_LABELS[PROFILES[prof_key]['cpu']]}. "
        f"Anda dapat menyesuaikan nilai di bawah sesuai kebutuhan."
    )

    if st.session_state.get("profile_notice"):
        st.toast(st.session_state.pop("profile_notice"))

    locked = PROFILES[st.session_state["profile"]]["dedicated_locked"]
    cpu_opts = dict(S.cpu_options(REF))
    gpu_opts = dict(S.gpu_options(REF))

    st.subheader("2. Tentukan Anggaran dan Preferensi Spesifikasi")

    with st.form("query_form"):
        # Baris 1: Budget & CPU
        col1, col2 = st.columns(2)
        col1.number_input(
            "Budget Maksimum (Rp)",
            min_value=S.BUDGET_MIN, max_value=S.BUDGET_MAX,
            step=S.BUDGET_STEP, key="budget",
            help="Batas anggaran maksimal. Laptop dengan harga di atas budget akan langsung dieliminasi."
        )

        cpu_val = st.session_state.get("cpu_level", PROFILES[prof_key]["cpu"])
        if cpu_val not in cpu_opts:
            cpu_val = PROFILES[prof_key]["cpu"]
        col2.selectbox(
            "Preferensi Performa CPU",
            list(cpu_opts),
            index=list(cpu_opts).index(cpu_val),
            key="cpu_level",
            format_func=lambda k: cpu_opts[k],
            help="Acuan benchmark PassMark CPU untuk peringkat rekomendasi KNN."
        )

        st.markdown("---")
        st.markdown("**Preferensi RAM dan Kapasitas Penyimpanan**")
        st.caption("Pilih Target sebagai acuan utama rekomendasi ranking KNN, dan Minimum sebagai batas spesifikasi terendah yang dapat diterima.")

        # Baris 2: RAM
        r1, r2 = st.columns(2)
        tram_val = st.session_state.get("target_ram", PROFILES[prof_key]["target_ram"])
        if tram_val not in S.RAM_CHOICES:
            tram_val = PROFILES[prof_key]["target_ram"]
        r1.selectbox(
            "Target RAM (Acuan Ranking)",
            S.RAM_CHOICES,
            index=S.RAM_CHOICES.index(tram_val),
            key="target_ram",
            format_func=lambda v: f"{v} GB",
            help="Spesifikasi RAM yang paling diinginkan. Laptop dengan RAM ini akan diprioritaskan."
        )

        mram_val = st.session_state.get("min_ram", PROFILES[prof_key]["min_ram"])
        if mram_val not in S.RAM_CHOICES:
            mram_val = PROFILES[prof_key]["min_ram"]
        r2.selectbox(
            "Batas Minimum RAM (Filter Wajib)",
            S.RAM_CHOICES,
            index=S.RAM_CHOICES.index(mram_val),
            key="min_ram",
            format_func=lambda v: f"{v} GB",
            help="Batas spesifikasi RAM terendah. Laptop di bawah nilai ini akan dieliminasi."
        )

        # Baris 3: Storage
        s1, s2 = st.columns(2)
        tsto_val = st.session_state.get("target_storage", PROFILES[prof_key]["target_storage"])
        if tsto_val not in S.STORAGE_CHOICES:
            tsto_val = PROFILES[prof_key]["target_storage"]
        s1.selectbox(
            "Target Storage (Acuan Ranking)",
            S.STORAGE_CHOICES,
            index=S.STORAGE_CHOICES.index(tsto_val),
            key="target_storage",
            format_func=lambda v: f"{v} GB" if v < 1024 else f"{v // 1024} TB",
            help="Spesifikasi kapasitas penyimpanan yang paling diinginkan."
        )

        msto_val = st.session_state.get("min_storage", PROFILES[prof_key]["min_storage"])
        if msto_val not in S.STORAGE_CHOICES:
            msto_val = PROFILES[prof_key]["min_storage"]
        s2.selectbox(
            "Batas Minimum Storage (Filter Wajib)",
            S.STORAGE_CHOICES,
            index=S.STORAGE_CHOICES.index(msto_val),
            key="min_storage",
            format_func=lambda v: f"{v} GB" if v < 1024 else f"{v // 1024} TB",
            help="Batas kapasitas penyimpanan terendah yang masih dapat Anda tolerir."
        )

        st.markdown("---")
        st.markdown("**Preferensi Grafis (GPU)**")
        gpu_val = st.session_state.get("gpu_level", PROFILES[prof_key]["gpu"])
        if gpu_val not in gpu_opts:
            gpu_val = PROFILES[prof_key]["gpu"]
        st.selectbox(
            "Preferensi Performa GPU",
            list(gpu_opts),
            index=list(gpu_opts).index(gpu_val),
            key="gpu_level",
            format_func=lambda k: gpu_opts[k],
            label_visibility="collapsed"
        )
        st.checkbox(
            "Wajib GPU dedicated (Gaming selalu mewajibkan; laptop dengan jenis GPU belum dapat dipastikan tidak diikutkan)",
            key="dedicated", disabled=locked
        )

        with st.expander("Filter layar (opsional)"):
            st.caption("Informasi layar dibaca dari nama produk. Laptop yang informasinya tidak tercantum akan disisihkan ke daftar belum dapat diranking bila filter ini diaktifkan.")
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

        st.checkbox("Izinkan penyimpanan anonim kriteria pencarian dan penilaian saya untuk penelitian (tanpa data pribadi)",
                    key="consent", disabled=(SOURCE != "mysql"))

        submitted = st.form_submit_button("Cari Rekomendasi", type="primary", use_container_width=True)

    if submitted:
        # Client validation
        validation_errors = []
        if st.session_state["min_ram"] > st.session_state["target_ram"]:
            validation_errors.append("Batas minimum RAM tidak boleh lebih besar dari Target RAM.")
        if st.session_state["min_storage"] > st.session_state["target_storage"]:
            validation_errors.append("Batas minimum storage tidak boleh lebih besar dari Target storage.")

        if validation_errors:
            st.error(" ".join(validation_errors))
        else:
            params = {
                "profile": st.session_state["profile"],
                "budget": st.session_state["budget"],
                "target_ram": st.session_state["target_ram"],
                "min_ram": st.session_state["min_ram"],
                "target_storage": st.session_state["target_storage"],
                "min_storage": st.session_state["min_storage"],
                "cpu_level": st.session_state["cpu_level"],
                "gpu_level": st.session_state["gpu_level"],
                "require_dedicated": st.session_state["dedicated"],
                "min_screen_inch": st.session_state["screen_inch"],
                "min_resolution": st.session_state["screen_res"],
                "oled_only": st.session_state["screen_oled"]
            }
            st.session_state["similar_for"] = None
            try:
                data = S.run_query(DF, REF, params)
            except ValueError as exc:
                st.session_state.pop("result", None)
                st.error(str(exc))
            else:
                session_id = None
                if SOURCE == "mysql" and st.session_state["consent"] and data["recs"]:
                    try:
                        session_id = db_call(db.insert_session, S.session_record(data, SOURCE), S.session_items(data))
                    except Exception as exc:
                        st.warning(f"Kriteria pencarian tidak dapat disimpan ({type(exc).__name__}); "
                                   "penilaian dinonaktifkan untuk hasil ini.")
                st.session_state["result"] = {"data": data, "session_id": session_id}

    if st.session_state.get("result"):
        render_result(st.session_state["result"])

with tab_eval:
    render_eval_tab(SOURCE)

st.caption("Prototipe penelitian. Kualitas rekomendasi dievaluasi secara objektif berdasarkan target kebutuhan spesifikasi.")
