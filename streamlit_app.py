from __future__ import annotations

import pandas as pd
import streamlit as st

import db
import service as S

st.set_page_config(page_title="Rekomendasi Laptop", layout="centered")

PROFILES = S.profile_defaults()
USE_STARS = hasattr(st, "feedback")
FORM_KEYS = ("budget", "min_ram", "min_storage", "cpu_level", "gpu_level", "dedicated")


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
# form state
# ----------------------------------------------------------------------------
def apply_profile_defaults():
    d = PROFILES[st.session_state["profile"]]
    st.session_state["min_ram"] = d["ram"]
    st.session_state["min_storage"] = d["storage"]
    st.session_state["cpu_level"] = d["cpu"]
    st.session_state["gpu_level"] = d["gpu"]
    st.session_state["dedicated"] = d["dedicated"]


def init_state():
    if "profile" not in st.session_state:
        st.session_state["profile"] = "programming"
        st.session_state["budget"] = 15_000_000
        apply_profile_defaults()
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
        st.toast("Penilaian tersimpan. Terima kasih!")
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
    st.caption("Seberapa sesuai laptop ini dengan kebutuhan Anda? (1 = sangat kurang, 5 = sangat sesuai)")
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
        st.caption("Dicari dari spesifikasi laptop ini (bukan target profil). Budget tidak membatasi hasil; "
                   "RAM minimum, storage minimum, dan filter layar tetap berlaku.")
        if sim["message"]:
            st.warning(sim["message"])
        for it in sim["items"]:
            st.markdown(f"**{it['rank']}. {md_escape(it['name'])}**")
            line = S.fmt_idr(it["price"])
            if it["over_budget"] > 0:
                line += f"  :orange[melampaui budget {S.fmt_idr(it['over_budget'])}]"
            st.markdown(line)
            st.caption(f"RAM {it['ram']} GB, Storage {it['storage']} GB, {it['processor']}, {it['graphics']} "
                       f"({it['gpu_class']}). {it['screen']}")
            st.caption("Dibanding laptop acuan: " + it["diff"] + ".")
            if it["variants"] > 0:
                st.caption(f"+{it['variants']} varian lain dengan spesifikasi setara "
                           f"({S.fmt_idr(it['vmin'])} - {S.fmt_idr(it['vmax'])}).")
            if it["url"]:
                st.markdown(f"[Lihat di Plaza IT]({it['url'].replace(')', '%29')})")
        if sim["identical_n"] > 0:
            lo, hi = sim["identical_range"]
            st.caption(f"Ada {sim['identical_n']} laptop lain dengan spesifikasi identik pada fitur yang dipakai "
                       f"({S.fmt_idr(lo)} - {S.fmt_idr(hi)}); tidak ditampilkan sebagai mirip.")
        st.button("Tutup", key=f"simclose_{source_row}", on_click=close_similar)


def render_result(stored: dict):
    data, session_id = stored["data"], stored["session_id"]
    sm = data["summary"]
    st.subheader("Hasil")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Laptop di katalog", S.fmt_num(sm["n_initial"]))
    c2.metric("Lolos budget, RAM, storage", S.fmt_num(sm["n_after_basic_filter"]))
    c3.metric("Dapat diranking", S.fmt_num(sm["n_rankable"]))
    c4.metric("Belum dapat diranking", S.fmt_num(data["unrankable_total"]))

    parts = [f"{t['label']}: {t['value']}" for t in data["targets"]]
    if not data["gpu_in_ranking"]:
        parts.append("GPU: tidak dipakai dalam ranking")
    st.caption("Target yang dipakai (RAM/storage = yang lebih tinggi antara minimum Anda dan rekomendasi profil): "
               + " | ".join(parts))

    if data["screen_note"]:
        st.info(data["screen_note"])

    if not data["recs"]:
        st.warning(data["empty_reason"])
    else:
        if data["target_far"]:
            st.info("Hampir tidak ada laptop yang mencapai semua target pada budget ini. Hasil di bawah adalah "
                    "yang terdekat; lihat keterangan 'di bawah target' pada tiap laptop.")
        for r in data["recs"]:
            with st.container(border=True):
                st.markdown(f"**{r['rank']}. {md_escape(r['name'])}**")
                st.markdown(f"**{S.fmt_idr(r['price'])}**")
                st.caption(r["screen"])
                st.caption(f"RAM {r['ram']} GB, Storage {r['storage']} GB, {r['processor']}, "
                           f"{r['graphics']} ({r['gpu_class']})")
                st.markdown("  \n".join(
                    (":green[" if c["ok"] else ":orange[") + c["text"] + "]" for c in r["chips"]))
                with st.expander("Mengapa direkomendasikan?"):
                    st.markdown(f"**{md_escape(r['explain']['headline'])}**")
                    st.markdown("\n".join(f"- {md_escape(p)}" for p in r["explain"]["points"]))
                    st.caption("Penjelasan ini menunjukkan kedekatan spesifikasi terhadap target dan posisi harga, "
                               "bukan jaminan mutu laptop.")
                if r["variants"] > 0:
                    st.caption(f"+{r['variants']} varian lain dengan spesifikasi setara "
                               f"({S.fmt_idr(r['variant_price_min'])} - {S.fmt_idr(r['variant_price_max'])}).")
                st.caption(f"Jarak KNN {r['distance']:.3f} (ukuran kedekatan terhadap target, bukan skor mutu)")
                if r["url"]:
                    st.link_button("Lihat di Plaza IT", r["url"])
                if session_id and r["laptop_id"] is not None:
                    render_rating_widget(session_id, r["laptop_id"])
                st.button("Cari yang mirip", key=f"sim_{session_id}_{r['source_row']}",
                          on_click=open_similar, args=(r["source_row"],))
            if st.session_state.get("similar_for") == r["source_row"]:
                render_similar(data["params"], r["source_row"])
        st.caption("Laptop dengan spesifikasi sama pada fitur yang dipakai dikelompokkan; yang ditampilkan adalah "
                   "varian termurah. Layar, baterai, garansi, dan merek tidak dinilai oleh model.")
        if session_id:
            ck = f"cm_{session_id}"
            st.text_area("Komentar untuk hasil ini (opsional)", key=ck, max_chars=1000)
            st.button("Simpan komentar", key=f"btn_{ck}", on_click=on_comment, args=(session_id, ck))

    if data["unrankable_total"] > 0:
        with st.expander(f"{data['unrankable_total']} laptop lolos syarat awal tetapi tidak dapat diranking"):
            st.caption("Skor benchmark yang dibutuhkan tidak tersedia, atau jenis GPU belum dapat dipastikan "
                       "sebagai dedicated. Skor tidak diperkirakan dan tidak dianggap nol.")
            st.dataframe(data["unrankable"], hide_index=True)
            if data["unrankable_extra"] > 0:
                st.caption(f"dan {data['unrankable_extra']} lainnya.")

    with st.expander("Cara membaca hasil"):
        st.markdown(
            "- Budget, RAM minimum, dan storage minimum adalah syarat wajib; laptop yang tidak memenuhinya "
            "tidak ditampilkan.\n"
            "- Ranking mengukur kedekatan spesifikasi dengan target profil. Kelebihan di atas target dihukum "
            "lebih ringan daripada kekurangan.\n"
            "- Harga bukan bagian dari ranking; harga hanya batas wajib dan pemutus seri.\n"
            "- Pada Office dan Programming, GPU tidak dipakai dalam ranking.\n"
            "- Skor CPU/GPU adalah PassMark. Skor GPU terintegrasi bersifat kasar (banyak laptop berbagi satu nilai).")


def render_eval_tab(source: str):
    st.subheader("Evaluasi pengguna")
    if source != "mysql":
        st.info("Evaluasi pengguna memerlukan MySQL. Konfigurasikan database dan jalankan import_catalog.py.")
        return
    st.caption("Ringkasan penilaian dari pengguna yang mencoba sistem. Penilai memilih sendiri apakah menilai dan "
               "jumlahnya kecil, sehingga hasil bersifat indikatif dan bukan ukuran akurasi.")
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
    b.metric("Sesi yang dinilai", t["sessions_with_rating"])
    c.metric("Jumlah penilaian", t["ratings"])
    if t["ratings"] == 0:
        st.info("Belum ada penilaian.")
        return
    st.markdown("**Rata-rata rating per profil**")
    st.dataframe(ov["by_profile"], hide_index=True)
    st.markdown("**Rata-rata rating per peringkat** (jika peringkat 1 dinilai lebih tinggi daripada peringkat 5, "
                "urutan ranking sejalan dengan penilaian pengguna)")
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
DF, REF, SOURCE, SOURCE_MSG = load_catalog_cached()
init_state()

with st.sidebar:
    st.markdown("**Sumber katalog**")
    st.write(f"{'MySQL' if SOURCE == 'mysql' else 'CSV (cadangan)'}: {len(DF)} laptop")
    if SOURCE_MSG:
        st.caption(SOURCE_MSG)
    if st.button("Muat ulang katalog"):
        st.cache_resource.clear()
        st.rerun()

st.title("Rekomendasi Laptop Berdasarkan Kebutuhan")
st.caption("KNN pada content-based filtering. Katalog Plaza IT.")
st.warning("Hasil menunjukkan kesesuaian spesifikasi terhadap target operasional untuk profil yang dipilih, "
           "bukan jaminan laptop terbaik. Target performa berasal dari persentil katalog (asumsi, bukan standar "
           "industri) dan belum divalidasi oleh ahli atau pengguna.")
if SOURCE == "csv":
    st.info("Katalog dimuat dari CSV cadangan; penilaian pengguna tidak dapat disimpan.")

tab_rec, tab_eval = st.tabs(["Rekomendasi", "Evaluasi pengguna"])

with tab_rec:
    st.selectbox("Profil penggunaan", S.PROFILE_ORDER, key="profile",
                 format_func=lambda k: PROFILES[k]["label"], on_change=apply_profile_defaults)
    locked = PROFILES[st.session_state["profile"]]["dedicated_locked"]
    cpu_opts = dict(S.cpu_options(REF))
    gpu_opts = dict(S.gpu_options(REF))

    with st.form("query_form"):
        col1, col2 = st.columns(2)
        col1.number_input("Budget maksimum (Rp)", min_value=S.BUDGET_MIN, max_value=S.BUDGET_MAX,
                          step=S.BUDGET_STEP, key="budget")
        col2.selectbox("RAM minimum (awal sesuai profil, bisa diubah)", S.RAM_CHOICES, key="min_ram",
                       format_func=lambda v: f"{v} GB")
        col1.selectbox("Storage minimum", S.STORAGE_CHOICES, key="min_storage",
                       format_func=lambda v: f"{v} GB" if v < 1024 else f"{v // 1024} TB")
        col2.selectbox("Preferensi performa CPU", list(cpu_opts), key="cpu_level",
                       format_func=lambda k: cpu_opts[k])
        st.selectbox("Preferensi performa GPU", list(gpu_opts), key="gpu_level",
                     format_func=lambda k: gpu_opts[k])
        st.checkbox("Wajib GPU dedicated (Gaming selalu mewajibkan; laptop yang jenis GPU-nya belum dapat "
                    "dipastikan tidak ikut)", key="dedicated", disabled=locked)
        with st.expander("Filter layar (opsional)"):
            st.caption("Informasi layar dibaca dari nama produk. Laptop yang resolusinya tidak tertulis tidak ikut "
                       "bila filter resolusi dipakai, dan dicantumkan pada daftar belum dapat diranking.")
            f1, f2 = st.columns(2)
            f1.selectbox("Ukuran layar minimum", S.SCREEN_INCH_CHOICES, key="screen_inch",
                         format_func=lambda v: "Tanpa batas" if v == 0 else f"{v} inci")
            f2.selectbox("Resolusi minimum", [""] + S.RES_CHOICES, key="screen_res",
                         format_func=lambda v: "Tanpa batas" if v == "" else v)
            st.checkbox("Hanya laptop yang tertulis OLED pada nama produk", key="screen_oled")
        st.checkbox("Izinkan penyimpanan anonim kriteria pencarian dan penilaian saya untuk penelitian "
                    "(tanpa data pribadi)", key="consent", disabled=(SOURCE != "mysql"))
        submitted = st.form_submit_button("Cari rekomendasi", type="primary")

    if submitted:
        params = {"profile": st.session_state["profile"], "budget": st.session_state["budget"],
                  "min_ram": st.session_state["min_ram"], "min_storage": st.session_state["min_storage"],
                  "cpu_level": st.session_state["cpu_level"], "gpu_level": st.session_state["gpu_level"],
                  "require_dedicated": st.session_state["dedicated"],
                  "min_screen_inch": st.session_state["screen_inch"],
                  "min_resolution": st.session_state["screen_res"],
                  "oled_only": st.session_state["screen_oled"]}
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

st.caption("Prototipe penelitian. Kualitas rekomendasi belum divalidasi oleh penilai ahli.")
