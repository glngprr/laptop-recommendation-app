# 02 — Dataset dan Preprocessing

> Dokumen ini menjelaskan sumber data, struktur dataset, proses seleksi dan pembersihan data, serta tahap preprocessing. Untuk konteks penelitian, lihat [01_PROJECT_OVERVIEW.md](./01_PROJECT_OVERVIEW.md). Untuk cara atribut ini digunakan oleh algoritma, lihat [03_RECOMMENDATION_METHODS.md](./03_RECOMMENDATION_METHODS.md).

---

## 2.1 Sumber Data

| Aspek | Keterangan |
|---|---|
| Sumber | Katalog Plaza IT (satu retailer) |
| Tanggal pengambilan | 5 Oktober 2026 |
| Format | CSV (`dataset/laptops_benchmark_enriched_resolved.csv`) |
| Jumlah baris mentah | 648 |
| Jumlah baris valid (katalog referensi) | **636** |
| Jumlah baris dikecualikan | 12 |

> [!IMPORTANT]
> Dataset berasal dari satu retailer tunggal pada satu waktu tertentu. Ini adalah keterbatasan **representasi katalog** yang harus dipertimbangkan saat menginterpretasikan hasil evaluasi.

---

## 2.2 Alasan Pengecualian 12 Baris

Dua belas (12) baris dikecualikan karena tidak memiliki nilai `cpu_passmark_mark` yang valid (bernilai `NaN` atau tidak terhingga). Kasus ini seluruhnya terjadi pada model **Apple MacBook Pro dengan chip M4 dan M5 (Pro/Max)**, yang skor CPU PassMark-nya tidak tersedia pada basis data benchmark yang digunakan.

Aturan pengecualian diterapkan di fungsi `_build_reference_catalog()` dalam `core/preprocessor.py`:

```python
mask_valid = df["cpu_passmark_mark"].notna() & np.isfinite(df["cpu_passmark_mark"])
```

---

## 2.3 Kamus Data — Kolom Utama

| Kolom | Tipe | Satuan | Deskripsi |
|---|---|---|---|
| `price_numeric` | float | IDR (Rupiah) | Harga jual laptop. Rentang: Rp4.999.000 – Rp89.100.000 |
| `ram_gb` | int | GB | Kapasitas RAM. Rentang: 4 – 64 GB |
| `storage_gb` | int | GB | Kapasitas penyimpanan utama. Rentang: 256 – 4.096 GB |
| `cpu_passmark_mark` | float | Skor | Skor CPU PassMark (benchmark CPU-only). Rentang: 1.763 – 62.164 |
| `gpu_passmark_g3d_mark` | float | Skor | Skor GPU PassMark G3D Mark. 141 dari 648 baris mentah (21,76%) tidak memiliki data benchmark GPU yang valid (lihat §2.4). |
| `gpu_mapping_status` | string | — | Status pemetaan benchmark GPU: `MATCHED`, `AMBIGUOUS`, `NO_GPU_DATA` |
| `brand` | string | — | Merek laptop |
| `model_name` | string | — | Nama model |

**Statistik katalog referensi (636 laptop valid):**

| Kolom | Rata-rata (μ) | Standar Deviasi (σ) | Minimum | Maksimum |
|---|---:|---:|---:|---:|
| `ram_gb` | 17,33 GB | 8,95 GB | 4 GB | 64 GB |
| `storage_gb` | 683,87 GB | 368,30 GB | 256 GB | 4.096 GB |
| `cpu_passmark_mark` | 21.509,30 | 11.076,82 | 1.763 | 62.164 |
| `price_numeric` | — | — | Rp4.999.000 | Rp89.100.000 |

---

## 2.4 Penanganan Data GPU

Dari 648 baris mentah:

| Status | Jumlah Baris |
|---|---:|
| `MATCHED` — skor GPU ditemukan dengan keyakinan tinggi | 507 |
| `AMBIGUOUS` — beberapa kemungkinan model GPU cocok, tidak dapat dipilih satu | 94 |
| `NO_GPU_DATA` — tidak ada data benchmark GPU | 47 |

Total baris dengan data GPU tidak andal: **141 baris** (94 AMBIGUOUS + 47 NO_GPU_DATA).

**Keputusan desain:** GPU **tidak digunakan** sebagai fitur ranking karena 141 dari 648 baris mentah (21,76%) tidak memiliki data PassMark G3D yang valid dan andal. Menyertakan kolom dengan tingkat ketidaklengkapan setinggi itu akan merusak konsistensi peringkat. GPU juga bukan faktor penentu utama untuk beban kerja kompilasi Android Studio. Lihat §2.5 untuk fitur yang akhirnya dipilih.

---

## 2.5 Tiga Fitur Ranking yang Dipilih

Ketiga metode rekomendasi menggunakan fitur yang sama:

| Fitur | Kolom | Alasan Pemilihan |
|---|---|---|
| RAM | `ram_gb` | Faktor utama multitasking; AVD membutuhkan RAM besar |
| CPU | `cpu_passmark_mark` | Proksi kemampuan komputasi untuk Gradle build |
| Storage | `storage_gb` | Ruang untuk proyek, build cache, dan sistem |

**Atribut yang sengaja dikecualikan dari fitur ranking:**

| Atribut | Alasan Pengecualian |
|---|---|
| `price_numeric` | Harga digunakan sebagai filter batas atas (*budget ceiling*), bukan faktor kualitas. Memasukkan harga ke dalam ranking menimbulkan risiko *price leakage* (laptop murah otomatis skor lebih rendah, bukan karena spesifikasi) |
| `gpu_passmark_g3d_mark` | 141 dari 648 baris mentah (21,76%) tidak andal (lihat §2.4) |
| Layar, bobot, baterai | Tidak terukur secara andal dan relevansinya terhadap beban kerja Android Studio lebih subjektif |

---

## 2.6 Filter Kelayakan (Hard Eligibility Rules)

Sebelum metode apapun dijalankan, laptop difilter dari katalog berdasarkan dua aturan keras berikut, ditambah filter anggaran:

| Kriteria | Nilai Minimum | Kolom |
|---|---|---|
| Harga | ≤ anggaran pengguna | `price_numeric` |
| RAM | ≥ 16 GB | `ram_gb` |
| Storage | ≥ 256 GB | `storage_gb` |

Aturan ini diterapkan oleh `get_candidate_pool(budget)` di `core/preprocessor.py`. Laptop yang tidak memenuhi salah satu kriteria ini **tidak akan pernah direkomendasikan**, tidak peduli seberapa tinggi skor CPU-nya.

**Jumlah kandidat pool berdasarkan anggaran:**

| Anggaran | Jumlah Laptop Layak |
|---|---:|
| Rp12.000.000 | 56 |
| Rp18.000.000 | 192 |
| Rp25.000.000 | 329 |
| Rp40.000.000 | 428 |

---

## 2.7 Preprocessing: Normalisasi untuk Algoritma

Dua transformasi berbeda digunakan oleh dua metode yang berbeda. Keduanya di-*fit* satu kali pada seluruh 636 katalog referensi dan kemudian digunakan bersama.

### 2.7.1 StandardScaler (untuk KNN Euclidean)

*StandardScaler* mengubah setiap fitur sehingga memiliki rata-rata 0 dan standar deviasi 1:

$$z = \frac{x - \mu}{\sigma}$$

di mana μ dan σ dihitung dari **636 laptop** dalam katalog referensi (bukan hanya kandidat pool). Ini memastikan skala konsisten antar skenario.

Diimplementasikan di `_fit_preprocessing_parameters()` → `params.transform_knn(df)`.

### 2.7.2 MinMax Bounds (untuk Cosine Similarity)

*MinMax normalization* mengubah setiap fitur ke rentang [0, 1]:

$$\tilde{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

di mana x\_min dan x\_max juga diambil dari **636 laptop** katalog referensi.

Diimplementasikan di `_fit_preprocessing_parameters()` → `params.transform_cosine(df)`.

> [!NOTE]
> Weighted Scoring tidak menggunakan normalisasi pre-fitting. Metode ini menghitung utilitas secara langsung berdasarkan perbandingan nilai asli dengan target dan nilai maksimum dalam pool kandidat.

---

## 2.8 CPU PassMark sebagai Proxy

Skor CPU PassMark digunakan sebagai **proksi** kemampuan komputasi prosesor, bukan pengukuran langsung waktu kompilasi Android. Ini adalah keterbatasan metodologis yang harus diakui:

- Skor PassMark mencerminkan performa CPU pada beban kerja benchmark sintetis.
- Waktu build Gradle aktual dipengaruhi juga oleh kecepatan storage, RAM (JVM heap), dan konfigurasi build.

**Titik referensi CPU PassMark yang digunakan dalam penelitian ini:**

| Level | Nilai Skor | Konteks Penggunaan |
|---|---:|---|
| Ambang Batas Grade 1 (P10) | 10.510 | Minimum untuk relevansi Grade 1 dalam evaluasi |
| rendah | 10.510 | Target CPU di `service.py` (Baseline minimum P10) |
| menengah_bawah | 14.823 | Target CPU di `service.py`; P25 / Q1 dalam skenario evaluasi |
| sedang | 18.871 | Target CPU di `service.py`; P50 / Median dalam skenario evaluasi |
| tinggi | 24.969 | Target CPU di `service.py`; P75 / Q3 dalam skenario evaluasi |
| sangat_tinggi | 34.222 | Target CPU di `service.py`; nilai referensi sangat tinggi (Workstation / P90) |

Nilai-nilai ini berasal dari distribusi empiris dataset dan ditetapkan secara eksplisit di `service.py` (konstanta `CPU_LEVEL_TARGETS`).

---

## 2.9 Referensi Silang

- Cara fitur digunakan oleh masing-masing algoritma → [03_RECOMMENDATION_METHODS.md](./03_RECOMMENDATION_METHODS.md)
- Cara kandidat pool digunakan dalam evaluasi → [04_METHODOLOGY_AND_EVALUATION.md](./04_METHODOLOGY_AND_EVALUATION.md)
- Implementasi kode → `core/preprocessor.py`

