# Sistem Rekomendasi Laptop untuk Pengembangan Android

**Perbandingan Metode K-Nearest Neighbors, Cosine Similarity, dan Weighted Scoring pada Sistem Rekomendasi Laptop Berdasarkan Kebutuhan Hardware untuk Android Developer**

Proyek penelitian ini membandingkan tiga metode _content-based filtering_ untuk merekomendasikan laptop yang sesuai untuk beban kerja pengembangan aplikasi Android. Evaluasi offline dilakukan pada 636 laptop dari katalog satu retailer menggunakan 72 skenario sintetis yang terstruktur. Metode terbaik (Weighted Scoring) digunakan dalam aplikasi Streamlit interaktif yang dapat diuji langsung.

---

## Cara Menjalankan Aplikasi

```bash
# Install dependensi
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run streamlit_app.py
```

Buka `http://localhost:8501` di browser. Tab **Evaluasi Model** adalah halaman utama yang menampilkan perbandingan hasil benchmark ketiga metode.

---

## Cara Menjalankan Unit Test

```bash
.venv\Scripts\python.exe -m unittest test_preprocessor test_recommenders test_requirement_mapper test_evaluator test_service -v
```

Hasil yang diharapkan: **57 test, semua PASS**.

---

## Dokumentasi Teknis

Dokumentasi lengkap tersimpan di folder `docs/`. Urutan membaca yang disarankan:

| #   | Dokumen                                                                   | Topik                                                           |
| --- | ------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1   | [01_PROJECT_OVERVIEW.md](docs/01_PROJECT_OVERVIEW.md)                     | Latar belakang, pertanyaan penelitian, alur kerja keseluruhan   |
| 2   | [02_DATASET_AND_PREPROCESSING.md](docs/02_DATASET_AND_PREPROCESSING.md)   | Sumber data, kamus data, filter eligibility, normalisasi        |
| 3   | [03_RECOMMENDATION_METHODS.md](docs/03_RECOMMENDATION_METHODS.md)         | Detail algoritma KNN, Cosine, Weighted Scoring + contoh numerik |
| 4   | [04_METHODOLOGY_AND_EVALUATION.md](docs/04_METHODOLOGY_AND_EVALUATION.md) | 72 skenario, metrik, hasil benchmark, pemilihan pemenang        |
| 5   | [05_APPLICATION_AND_CODEBASE.md](docs/05_APPLICATION_AND_CODEBASE.md)     | Arsitektur kode, modul, cara menjalankan dan menguji            |

**Spesifikasi evaluasi terkunci (v2.1.0):** [`docs/EVALUATION_SPEC.md`](docs/EVALUATION_SPEC.md)

**Dokumen historis:** [`docs/archive/`](docs/archive/) — tidak aktif, disimpan sebagai referensi.

---

## Hasil Benchmark (Ringkasan)

| Metrik              | KNN Euclidean | Cosine Similarity | Weighted Scoring |
| ------------------- | ------------: | ----------------: | ---------------: |
| Mean NDCG@5         |      0,773432 |          0,698774 |     **0,978585** |
| Mean Precision@5    |      0,480556 |          0,494444 |     **0,738889** |
| Catalog Coverage    |      0,144654 |      **0,198113** |         0,050314 |
| Latency Median (ms) |        7,0474 |            6,8170 |       **6,3049** |

**Pemenang: Weighted Scoring** — menang di Tahap 1 seleksi hierarkis berdasarkan NDCG@5.

---

## Teknologi

Python · Streamlit · pandas · NumPy · scikit-learn · CSV-only (tanpa database aktif)
