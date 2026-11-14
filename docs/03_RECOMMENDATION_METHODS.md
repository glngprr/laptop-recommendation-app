# 03 — Metode Rekomendasi

> Dokumen ini menjelaskan tiga metode rekomendasi yang dibandingkan dalam penelitian ini. Untuk data yang digunakan, lihat [02_DATASET_AND_PREPROCESSING.md](./02_DATASET_AND_PREPROCESSING.md). Untuk cara metode ini dievaluasi, lihat [04_METHODOLOGY_AND_EVALUATION.md](./04_METHODOLOGY_AND_EVALUATION.md).

---

## 3.0 Kerangka Bersama

Ketiga metode berbagi komponen yang sama:

- **Kandidat pool** — diperoleh dari `get_candidate_pool(budget)`: laptop ≤ budget, RAM ≥ 16 GB, Storage ≥ 256 GB.
- **Fitur ranking** — tiga atribut numerik: `ram_gb`, `storage_gb`, `cpu_passmark_mark`.
- **Parameter preprocessing** — `StandardScaler` (untuk KNN) dan batas MinMax (untuk Cosine) di-*fit* satu kali pada 636 katalog referensi.
- **Tie-breaking** — toleransi 1×10⁻⁷: laptop dengan selisih skor ≤ toleransi dianggap seri, kemudian diurutkan berdasarkan harga naik, lalu `original_index` naik.
- **Output** — Top-5 laptop berdasarkan metode masing-masing.

Implementasi seluruh metode ada di `core/recommenders.py`.

---

## 3.1 KNN Euclidean

### 3.1.1 Konsep

Dalam penelitian ini, KNN (*K-Nearest Neighbors*) diterapkan sebagai **metode pemeringkatan berbasis ketetanggaan terdekat** (*nearest-neighbor ranking*), bukan sebagai pipeline klasifikasi terbimbing dengan label kelas. Kandidat laptop diurutkan berdasarkan jarak Euclidean mereka dari profil hardware target dalam ruang fitur yang telah distandarisasi. Laptop dengan jarak terkecil menempati peringkat teratas.

**Karakteristik kunci:** Metode ini mengukur *kedekatan absolut* dalam ruang standar. Sebuah laptop yang jauh melebihi target (mis. RAM 64 GB ketika target hanya 16 GB) akan memiliki jarak Euclidean yang lebih besar dibandingkan laptop yang tepat memenuhi target — ini dikenal sebagai *overspecification penalty*.

### 3.1.2 Formula

**Langkah 1 — Standardisasi** setiap fitur menggunakan parameter dari 636 katalog referensi:

$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

di mana:
- $x_i$ = nilai asli fitur ke-$i$ (RAM, Storage, CPU)
- $\mu_i$ = rata-rata fitur ke-$i$ pada 636 katalog referensi
- $\sigma_i$ = standar deviasi fitur ke-$i$ pada 636 katalog referensi

**Langkah 2 — Jarak Euclidean** antara laptop kandidat $\mathbf{z}$ dan vektor target $\mathbf{z}_t$:

$$D(\mathbf{z}, \mathbf{z}_t) = \sqrt{\sum_{i=1}^{3} (z_i - z_{t,i})^2}$$

**Langkah 3 — Ranking:** Urutkan kandidat berdasarkan $D$ secara **menaik** (jarak terkecil = peringkat 1).

### 3.1.3 Contoh Numerik

Asumsikan parameter dari katalog referensi:

| Fitur | μ | σ |
|---|---:|---:|
| RAM (GB) | 17,33 | 8,95 |
| Storage (GB) | 683,87 | 368,30 |
| CPU PassMark | 21.509 | 11.077 |

Target pengguna: RAM=24 GB, Storage=512 GB, CPU=18.871.

Vektor target yang telah distandardisasi:

$$z_{t,\text{RAM}} = \frac{24 - 17{,}33}{8{,}95} = 0{,}745$$

$$z_{t,\text{Storage}} = \frac{512 - 683{,}87}{368{,}30} = -0{,}467$$

$$z_{t,\text{CPU}} = \frac{18{.}871 - 21{.}509}{11{.}077} = -0{,}238$$

Tiga kandidat laptop (nilai yang sudah distandardisasi):

| Laptop | z_RAM | z_Storage | z_CPU | D ke target |
|---|---:|---:|---:|---:|
| A | 0,745 | -0,467 | -0,238 | **0,000** |
| B | 0,969 | 0,083 | 0,131 | 0,699 |
| C | -0,037 | -0,467 | -0,238 | 0,782 |

Laptop A dipilih pertama (D = 0).

### 3.1.4 Implementasi

- Kelas: `KNNRecommender` di `core/recommenders.py`
- Fungsi utama: `recommend(targets, params, candidate_pool, top_k=5)`
- Transformasi: `params.transform_knn(df)` — mengembalikan DataFrame yang dinormalisasi
- Target juga ditransformasi dengan StandardScaler yang sama sebelum jarak dihitung

### 3.1.5 Keunggulan dan Keterbatasan

| Keunggulan | Keterbatasan |
|---|---|
| Intuitif: laptop yang "paling mendekati" spesifikasi target | *Overspecification penalty* — laptop jauh di atas target dihukum meski tidak ada pengguna yang menolaknya |
| Konsisten: jarak Euclidean adalah metrik yang terdefinisi dengan baik | Sensitif terhadap skala fitur (diatasi dengan StandardScaler) |
| Tidak ada parameter tuning yang diperlukan | Tidak membedakan shortfall vs. surplus |

---

## 3.2 Cosine Similarity

### 3.2.1 Konsep

Cosine Similarity mengukur **sudut antara dua vektor fitur** yang telah dinormalisasi ke rentang [0, 1]. Dua laptop dengan proporsi fitur yang sama akan memiliki kemiripan cosine yang tinggi, bahkan jika nilai absolutnya berbeda.

**Karakteristik kunci:** Metode ini mengukur *keselarasan arah*, bukan jarak. Laptop dengan proporsi RAM:Storage:CPU yang sama dengan target mendapat skor tinggi, terlepas dari apakah nilainya lebih besar atau lebih kecil dari target.

### 3.2.2 Formula

**Langkah 1 — Normalisasi MinMax** ke rentang [0, 1] menggunakan batas dari 636 katalog referensi:

$$\tilde{x}_i = \frac{x_i - x_{\min,i}}{x_{\max,i} - x_{\min,i}}$$

**Langkah 2 — Cosine Similarity** antara vektor laptop $\tilde{\mathbf{x}}$ dan vektor target $\tilde{\mathbf{t}}$:

$$\text{Sim}(\tilde{\mathbf{x}}, \tilde{\mathbf{t}}) = \frac{\tilde{\mathbf{x}} \cdot \tilde{\mathbf{t}}}{\|\tilde{\mathbf{x}}\| \cdot \|\tilde{\mathbf{t}}\|}$$

**Penanganan zero-norm:** Jika $\|\tilde{\mathbf{t}}\| = 0$ (semua nilai target sama dengan batas minimum), maka similarity semua laptop = 0,0.

**Langkah 3 — Ranking:** Urutkan berdasarkan $\text{Sim}$ secara **menurun** (skor tertinggi = peringkat 1).

### 3.2.3 Contoh Numerik

Batas MinMax dari katalog referensi:

| Fitur | min | max |
|---|---:|---:|
| RAM (GB) | 4 | 64 |
| Storage (GB) | 256 | 4.096 |
| CPU PassMark | 1.763 | 62.164 |

Target: RAM=24, Storage=512, CPU=18.871.

Target ternormalisasi:

$$\tilde{t}_{\text{RAM}} = \frac{24 - 4}{64 - 4} = 0{,}333$$

$$\tilde{t}_{\text{Storage}} = \frac{512 - 256}{4096 - 256} = 0{,}067$$

$$\tilde{t}_{\text{CPU}} = \frac{18871 - 1763}{62164 - 1763} = 0{,}283$$

Tiga kandidat laptop (nilai ternormalisasi):

| Laptop | x̃_RAM | x̃_Storage | x̃_CPU | Sim ke target |
|---|---:|---:|---:|---:|
| A | 0,333 | 0,067 | 0,283 | **1,000** |
| B | 0,667 | 0,067 | 0,283 | 0,969 |
| C | 0,167 | 0,133 | 0,141 | 0,972 |

Laptop A dipilih pertama (Sim = 1,0 — vektor identik dengan target).

### 3.2.4 Implementasi

- Kelas: `CosineRecommender` di `core/recommenders.py`
- Fungsi utama: `recommend(targets, params, candidate_pool, top_k=5)`
- Transformasi: `params.transform_cosine(df)` — normalisasi MinMax
- Target juga ditransformasi dengan batas MinMax yang sama

### 3.2.5 Keunggulan dan Keterbatasan

| Keunggulan | Keterbatasan |
|---|---|
| Sensitif terhadap proporsi/komposisi fitur | Tidak membedakan nilai absolut — laptop kecil dengan proporsi yang sama dengan laptop besar mendapat skor identik |
| Tidak sensitif terhadap besaran vektor secara keseluruhan | Jika target atau laptop memiliki semua nilai di minimum → similarity tidak terdefinisi (zero-norm) |
| Cocok untuk perbandingan dokumen/profil | Surplus besar tidak dihargai lebih dari surplus kecil selama proporsinya sama |

---

## 3.3 Weighted Scoring (Metode Aktif di Aplikasi)

### 3.3.1 Konsep

Weighted Scoring adalah implementasi dari *Multi-Attribute Utility Theory* (MAUT). Setiap fitur dikonversikan ke **skor utilitas** (0–1), kemudian dijumlahkan dengan bobot. Metode ini menggunakan fungsi utilitas sepotong-sepotong (*piecewise*) yang membedakan antara **kekurangan** (shortfall) dan **kelebihan** (surplus) dari target.

**Karakteristik kunci:** Ini adalah satu-satunya metode yang secara eksplisit mendefinisikan apa artinya "baik" vs "cukup" melalui bobot dan utilitas. Ini juga satu-satunya metode yang digunakan dalam aplikasi live.

### 3.3.2 Bobot Fitur

| Fitur | Bobot | Alasan |
|---|---:|---|
| RAM (`ram_gb`) | **0,45** | Faktor paling kritis untuk multitasking dan AVD |
| CPU (`cpu_passmark_mark`) | **0,35** | Penentu utama kecepatan Gradle build |
| Storage (`storage_gb`) | **0,20** | Penting namun lebih mudah diperluas |
| **Total** | **1,00** | — |

> [!NOTE]
> Bobot ini adalah **keputusan desain berbasis domain knowledge**, bukan hasil pengukuran kausal atau empiris. Bobot mencerminkan prioritas relatif yang diasumsikan untuk beban kerja Android development.

### 3.3.3 Fungsi Utilitas (Piecewise)

Untuk setiap fitur, utilitas $u(x, t)$ dihitung sebagai:

$$u(x, t) = \begin{cases} 0{,}8 \cdot \dfrac{x}{t} & \text{jika } x < t \text{ (shortfall)} \\[8pt] 0{,}8 + 0{,}2 \cdot g(x, t) & \text{jika } x \geq t \text{ (memenuhi atau surplus)} \end{cases}$$

di mana:

$$g(x, t) = \begin{cases} 0{,}0 & \text{jika } t \geq \max(X) \\[4pt] \min\!\left(1{,}0,\; \dfrac{x - t}{\max(X) - t}\right) & \text{sebaliknya} \end{cases}$$

**Variabel:**
- $x$ = nilai fitur laptop kandidat (nilai asli, tidak dinormalisasi)
- $t$ = nilai target fitur dari preset beban kerja
- $\max(X)$ = nilai maksimum fitur dalam *seluruh kandidat pool* (bukan 636 katalog)
- Rentang utilitas: [0, 1]; tepat memenuhi target → 0,8; melebihi maksimum pool → 1,0

> [!NOTE]
> Karena $\max(X)$ dihitung dari kandidat pool yang aktif (bukan dari seluruh katalog referensi), skor Weighted Scoring sebuah laptop dapat berbeda antar skenario tergantung pada komposisi pool — misalnya ketika anggaran berubah. Skor dari skenario yang berbeda tidak dapat dibandingkan secara langsung sebagai nilai absolut; skor hanya valid untuk perbandingan relatif *di dalam* skenario yang sama.

**Skor akhir:**

$$S = 0{,}45 \cdot u_{\text{RAM}} + 0{,}35 \cdot u_{\text{CPU}} + 0{,}20 \cdot u_{\text{Storage}}$$

### 3.3.4 Contoh Numerik

Target: RAM=24 GB, Storage=512 GB, CPU=18.871.

Asumsikan dalam kandidat pool: max\_RAM=64, max\_Storage=4.096, max\_CPU=55.000.

Tiga laptop kandidat:

**Laptop A** (RAM=16, Storage=512, CPU=24.969):
- $u_{\text{RAM}} = 0{,}8 \times \frac{16}{24} = 0{,}533$ (shortfall)
- $u_{\text{Storage}} = 0{,}8 + 0{,}2 \times \min(1,\, \frac{512-512}{4096-512}) = 0{,}8 + 0 = 0{,}800$ (tepat target)
- $u_{\text{CPU}} = 0{,}8 + 0{,}2 \times \min(1,\, \frac{24969-18871}{55000-18871}) = 0{,}8 + 0{,}2 \times 0{,}169 = 0{,}834$
- $S_A = 0{,}45 \times 0{,}533 + 0{,}35 \times 0{,}834 + 0{,}20 \times 0{,}800 = 0{,}240 + 0{,}292 + 0{,}160 = \mathbf{0{,}692}$

**Laptop B** (RAM=32, Storage=1.024, CPU=18.871):
- $u_{\text{RAM}} = 0{,}8 + 0{,}2 \times \min(1,\, \frac{32-24}{64-24}) = 0{,}8 + 0{,}040 = 0{,}840$
- $u_{\text{Storage}} = 0{,}8 + 0{,}2 \times \min(1,\, \frac{1024-512}{4096-512}) = 0{,}8 + 0{,}029 = 0{,}829$
- $u_{\text{CPU}} = 0{,}8 + 0{,}2 \times \min(1,\, \frac{18871-18871}{55000-18871}) = 0{,}800$
- $S_B = 0{,}45 \times 0{,}840 + 0{,}35 \times 0{,}800 + 0{,}20 \times 0{,}829 = 0{,}378 + 0{,}280 + 0{,}166 = \mathbf{0{,}824}$

**Laptop C** (RAM=24, Storage=256, CPU=14.823):
- $u_{\text{RAM}} = 0{,}8$ (tepat target)
- $u_{\text{Storage}} = 0{,}8 \times \frac{256}{512} = 0{,}400$ (shortfall)
- $u_{\text{CPU}} = 0{,}8 \times \frac{14823}{18871} = 0{,}628$ (shortfall)
- $S_C = 0{,}45 \times 0{,}800 + 0{,}35 \times 0{,}628 + 0{,}20 \times 0{,}400 = 0{,}360 + 0{,}220 + 0{,}080 = \mathbf{0{,}660}$

Urutan: **B (0,824) > A (0,692) > C (0,660)**

### 3.3.5 Implementasi

- Kelas: `WeightedScoringRecommender` di `core/recommenders.py`
- Fungsi utama: `recommend(targets, params, candidate_pool, top_k=5)`
- Fungsi utilitas: `_calculate_feature_utility(x, target, max_val)` — mengembalikan nilai [0, 1]
- Tie-breaking: `_apply_tie_breaking(df, score_col)` — toleransi 1e-7, kemudian harga naik, kemudian index naik
- **Status:** Ini adalah satu-satunya metode yang digunakan dalam aplikasi live (dipanggil dari `service.py`)

### 3.3.6 Keunggulan dan Keterbatasan

| Keunggulan | Keterbatasan |
|---|---|
| Transparan: skor dapat dijelaskan secara intuitif | Bobot adalah keputusan desain, bukan bukti empiris |
| Membedakan shortfall (penalti) vs. surplus (bonus terbatas) | Fungsi utilitas piecewise adalah aproksimasi sederhana |
| Performa evaluasi terbaik (NDCG@5 = 0,978) | *Construct-alignment risk* — lihat §3.4 |
| Digunakan langsung dalam live recommendation | Coverage rendah (5,03%) — cenderung menghasilkan rekomendasi yang terkonsentrasi pada sejumlah laptop yang sama di berbagai skenario |

---

## 3.4 Catatan Metodologis: Construct-Alignment Risk

> [!WARNING]
> Terdapat **risiko keselarasan konstruk** (*construct-alignment risk*) yang harus dipertimbangkan saat menginterpretasikan hasil evaluasi:
>
> - **Kriteria relevansi penelitian** (lihat [04_METHODOLOGY_AND_EVALUATION.md](./04_METHODOLOGY_AND_EVALUATION.md)) dinilai berdasarkan seberapa baik laptop memenuhi target RAM, CPU, dan Storage.
> - **Weighted Scoring** juga menghitung skor berdasarkan seberapa baik laptop memenuhi target RAM, CPU, dan Storage.
>
> Ini berarti Weighted Scoring secara struktural *diselaraskan* dengan kriteria relevansi penelitian yang digunakan. KNN dan Cosine menggunakan dimensi yang sama, tetapi dengan transformasi berbeda (standardisasi dan normalisasi) yang dapat menghasilkan urutan berbeda.
>
> Keunggulan performa Weighted Scoring mungkin sebagian mencerminkan keselarasan konstruk ini, bukan hanya superioritas metodologis yang murni.

---

## 3.5 Referensi Silang

- Data dan preprocessing yang digunakan → [02_DATASET_AND_PREPROCESSING.md](./02_DATASET_AND_PREPROCESSING.md)
- Bagaimana ketiga metode dievaluasi → [04_METHODOLOGY_AND_EVALUATION.md](./04_METHODOLOGY_AND_EVALUATION.md)
- Implementasi dan arsitektur kode → [05_APPLICATION_AND_CODEBASE.md](./05_APPLICATION_AND_CODEBASE.md)

