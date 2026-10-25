# Sistem Rekomendasi Laptop Berbasis KNN

**Penerapan K-Nearest Neighbors (KNN) pada Content-Based Filtering untuk Sistem Rekomendasi Laptop Berbasis Web**

Aplikasi web berbasis Streamlit yang merekomendasikan laptop dari katalog Plaza IT berdasarkan **kebutuhan dan preferensi pengguna**: profil penggunaan, batasan anggaran, spesifikasi minimum, serta target preferensi RAM, storage, CPU, dan GPU. Sistem mencocokkan spesifikasi laptop dengan target preferensi menggunakan algoritma K-Nearest Neighbors (KNN) dengan jarak Euclidean asimetris.

> **Status: Prototipe Penelitian.** Hasil rekomendasi menunjukkan *kesesuaian spesifikasi numerik laptop terhadap target kebutuhan pengguna*, bukan jaminan mutu laptop terbaik secara komersial. Target performa benchmark diturunkan dari persentil katalog (sebagai asumsi operasional, bukan standar industri).
> 
> Dokumentasi rinci mengenai metode, data, pengujian, dan batasan penelitian tersedia di [docs/METODOLOGI.md](docs/METODOLOGI.md).

---

## Daftar Isi

- [Deskripsi Proyek](#deskripsi-proyek)
- [Fitur Utama](#fitur-utama)
- [Cara Kerja Rekomendasi](#cara-kerja-rekomendasi)
- [Minimum Requirement vs Preferred Target](#minimum-requirement-vs-preferred-target)
- [Metodologi dan Keterbatasan](#metodologi-dan-keterbatasan)
- [Cara Memakai Aplikasi](#cara-memakai-aplikasi)
- [Panduan Instalasi dan Menjalankan Aplikasi](#panduan-instalasi-dan-menjalankan-aplikasi)
  - [Tanpa MySQL (Cepat via CSV)](#tanpa-mysql-cepat-via-csv)
  - [Dengan MySQL (Penyimpanan Sesi & Evaluasi)](#dengan-mysql-opsional)
- [Menjalankan Eksperimen dan Pengujian](#menjalankan-eksperimen-dan-pengujian)
- [Struktur Repositori](#struktur-repositori)
- [Pemecahan Masalah](#pemecahan-masalah)
- [Data Katalog](#data-katalog)

---

## Deskripsi Proyek

Sistem ini dikembangkan untuk membantu calon pembeli menemukan laptop yang paling sesuai dengan kebutuhan fungsional dan batas anggaran mereka. Pendekatan yang digunakan adalah **Content-Based Filtering** dengan algoritma **K-Nearest Neighbors (KNN)**.

Aplikasi tidak menggunakan rating dari pengguna lain (*collaborative filtering*), melainkan secara objektif mengukur kedekatan fitur teknis laptop di katalog terhadap vektor spesifikasi yang diharapkan pengguna (*preferred target*), setelah memangkas laptop yang tidak memenuhi batas wajib (*hard filter*).

### Teknologi Utama
- **Bahasa & Komputasi Numerik:** Python 3.11+ (NumPy, Pandas, Scikit-learn).
- **Antarmuka Web:** Streamlit (UI interaktif, responsif desktop dan mobile, bebas dekorasi berlebihan).
- **Database (Opsional):** MySQL 8 dengan PyMySQL (menyimpan data katalog, log sesi pencarian, dan feedback pengguna).

---

## Fitur Utama

- **Empat Profil Kebutuhan:**
  - *Office dan Kuliah*: hemat daya, target RAM 8 GB, prosesor level Rendah, GPU tidak dihitung dalam ranking.
  - *Programming dan Web Development*: target RAM 16 GB, CPU level Sedang, GPU tidak dihitung dalam ranking.
  - *Desain Grafis dan Editing Video*: target RAM 16 GB, CPU level Tinggi, akselerasi GPU (dedicated atau setara).
  - *Gaming*: target RAM 16 GB, CPU level Tinggi, wajib GPU dedicated.
- **Pemisahan Hard Filter dan Ranking Target:**
  - *Batas Wajib (Hard Filter)*: Laptop yang melebihi batas anggaran atau berada di bawah RAM/storage minimum langsung dieliminasi.
  - *Acuan Ranking (Preferred Target)*: Target spesifikasi RAM dan storage dapat disesuaikan pengguna secara mandiri tanpa ditimpa paksa oleh nilai bawaan profil.
- **Ranking KNN dengan Jarak Asimetris:**
  - Standardisasi fitur (`StandardScaler`) dilatih pada seluruh katalog tetap agar bobot dimensi stabil.
  - Surplus spesifikasi di atas target dikenai penalti lebih ringan dibanding kekurangan (*shortfall*).
  - Harga **bukan** variabel fitur ranking; harga murni batas wajib dan pemutus seri (*tie-breaker*).
- **Kebijakan Tanpa Imputasi (Zero-Imputation):**
  - Skor benchmark yang tidak tersedia di basis data tidak pernah ditebak atau diisi nol.
  - Laptop yang lolos batas wajib namun kehilangan benchmark yang disyaratkan dilaporkan secara transparan pada daftar *belum dapat diranking* beserta alasannya.
- **Transparansi Rekomendasi ("Mengapa direkomendasikan?"):**
  - Uraian rasional berbasis data aktual yang menjelaskan persentase pemenuhan target, posisi harga relatif terhadap kandidat lain, dan peringkat kedekatan.
- **Pencarian Laptop Serupa (Item-to-Item Similarity):**
  - Menemukan laptop dengan konfigurasi teknis paling mirip dengan satu laptop pilihan tanpa dibatasi anggaran.
- **Filter Layar Opsional:**
  - Filter ukuran layar minimum, kelas resolusi minimum, dan panel OLED (diekstraksi dari teks nama produk).
- **Evaluasi Pengguna (Opsional, Perlu MySQL):**
  - Penilaian bintang 1–5 secara anonim per laptop rekomendasi untuk analisis kepuasan.

---

## Cara Kerja Rekomendasi

Sistem memproses rekomendasi melalui lima tahapan berurutan:

```
[Input Pengguna] 
       │
       ▼
1. Penentuan Preferensi & Profil
       │
       ▼
2. Penyaringan Awal (Hard Filter: Anggaran, RAM Min, Storage Min, Dedicated GPU)
       │
       ▼
3. Pembentukan Vektor Target Preferensi (Target RAM, Target Storage, CPU Mark, G3D Mark)
       │
       ▼
4. Perhitungan Jarak KNN Asimetris (Terstandarisasi z-score seluruh katalog)
       │
       ▼
5. Pengelompokan Varian Setara & Penyajian Rekomendasi Terurut
```

1. **Pengguna Memilih Profil dan Menentukan Preferensi:** Pengguna memilih salah satu dari 4 profil kebutuhan, menentukan batas anggaran maksimum, batas minimum spesifikasi, serta target spesifikasi RAM, storage, performa CPU, dan GPU.
2. **Penyaringan Kandidat (Hard Filter):** Sistem memangkas katalog. Laptop yang harganya melebihi budget, memiliki RAM di bawah RAM minimum, atau storage di bawah storage minimum langsung dieliminasi. Pada profil Gaming (atau jika opsi dicentang), laptop tanpa GPU dedicated terkonfirmasi juga dieliminasi.
3. **Penyusunan Vektor Target:** Sistem menyusun titik koordinat target $t$ berdasarkan preferensi pengguna. Nilai bawaan profil hanya berfungsi mengisi nilai awal; jika pengguna memilih target yang berbeda, preferensi pengguna yang menjadi acuan target.
4. **Perhitungan Jarak Kedekatan KNN:** Fitur-fitur kandidat yang lolos ditransformasikan ke ruang terstandarisasi $z$-score (menggunakan mean dan varians dari seluruh populasi katalog acuan). Algoritma menghitung jarak Euclidean asimetris: kekurangan spesifikasi berbobot penuh, sedangkan surplus spesifikasi diberi bobot terdiskon ($\sqrt{\lambda}$ dengan $\lambda = 0{,}25$).
5. **Pengurutan dan Penyajian Rekomendasi:** Laptop diurutkan dari jarak terkecil. Laptop dengan spesifikasi identik pada fitur yang diranking dikelompokkan (hanya varian termurah yang ditampilkan sebagai representasi), kemudian disajikan lengkap dengan penjelasan kedekatan dan perbandingan harga.

---

## Minimum Requirement vs Preferred Target

Perbedaan antara batas minimum dan target preferensi merupakan prinsip inti sistem:

| Aspek | Minimum Requirement (Batas Wajib) | Preferred Target (Acuan Ranking) |
|---|---|---|
| **Peran** | *Hard Filter* (pemotong pool kandidat) | Titik acuan perhitungan jarak KNN |
| **Mekanisme** | `ram_gb >= min_ram` dan `storage_gb >= min_storage` | Dimensi target pada vektor KNN $t$ |
| **Efek Pelanggaran** | Laptop langsung didiskualifikasi dan tidak tampil | Laptop tetap dapat tampil, namun jaraknya bertambah (penalti shortfall) |
| **Pengaruh Nilai Profil** | Nilai rekomendasi profil hanya mengisi form awal | Profil tidak menimpa nilai yang diubah pengguna |

### Contoh Kasus Konkret

Misalkan pengguna memilih profil **Programming dan Web Development**:
- **Nilai Bawaan Profil:** Menyarankan target RAM 16 GB dan batas minimum 16 GB (atau 8 GB).
- **Kasus A: Minimum RAM 8 GB, Target RAM 16 GB**
  - Laptop dengan RAM 8 GB dan 16 GB sama-sama lolos ke pool kandidat.
  - Vektor target KNN tetap menargetkan RAM 16 GB. Laptop 16 GB memiliki jarak 0 pada dimensi RAM, sedangkan laptop 8 GB mengalami kekurangan (*shortfall*) sehingga peringkatnya cenderung di bawah laptop 16 GB.
- **Kasus B: Minimum RAM 8 GB, Target RAM Diubah Menjadi 8 GB**
  - Target ranking KNN pada dimensi RAM berubah menjadi **8 GB**.
  - Laptop berkapasitas 8 GB memiliki jarak 0 (selisih nol) terhadap target RAM. Sistem tidak memaksakan kenaikan target ke 16 GB. Laptop 8 GB dapat meraih peringkat teratas jika fitur CPU dan lainnya mendukung.

> **Catatan Multivariat:** Target RAM 8 GB **tidak menjamin** semua laptop 8 GB otomatis menempati peringkat teratas. Ranking KNN mengukur kedekatan gabungan multivariat (RAM, Storage, CPU PassMark, dan GPU bila aktif). Laptop dengan prosesor yang jauh dari target atau harga yang melampaui seri tetap memengaruhi urutan akhir.
> 
> Prinsip pemisahan yang sama berlaku penuh untuk **Storage** (misalnya minimum storage 256 GB dengan target 512 GB vs target 256 GB).

---

## Metodologi dan Keterbatasan

### Fitur yang Digunakan per Profil

| Profil | Fitur Ranking KNN | Syarat Hard Filter GPU |
|---|---|---|
| **Office dan kuliah** | RAM, Storage, CPU PassMark | GPU tidak dipakai dalam ranking |
| **Programming dan web** | RAM, Storage, CPU PassMark | GPU tidak dipakai dalam ranking |
| **Desain dan video** | RAM, Storage, CPU PassMark, GPU G3D Mark | Opsional / Tidak wajib dedicated |
| **Gaming** | RAM, Storage, CPU PassMark, GPU G3D Mark | **Wajib GPU dedicated** |

### Standardisasi dan Jarak KNN
- **StandardScaler Konsisten:** Scaler dilatih pada seluruh katalog yang memiliki kelengkapan data (populasi acuan tetap), bukan dihitung ulang pada pool kandidat yang lolos filter. Hal ini mencegah pergeseran skala bobot antar-query.
- **Jarak Asimetris ($\lambda = 0{,}25$):**
  $$\Delta_j = z_j(\text{laptop}) - z_j(\text{target})$$
  $$\Delta'_j = \begin{cases} \sqrt{\lambda}\,\Delta_j, & \text{jika } \Delta_j > 0 \text{ (surplus)} \\ \Delta_j, & \text{jika } \Delta_j \le 0 \text{ (kekurangan)} \end{cases}$$
  Laptop dengan spesifikasi di atas target tetap disukai (jarak kecil), namun kekurangan di bawah target dihukum secara penuh.

### Keterbatasan Sistem
1. **Bukan Bukti Kualitas Rekomendasi Mutlak:** Jarak KNN yang kecil hanya membuktikan kedekatan angka spesifikasi terhadap target numerik operasional, bukan bukti empiris bahwa laptop tersebut disukai pengguna atau paling awet di pasar.
2. **Ketiadaan Ground Truth:** Dataset katalog tidak memiliki label kebenaran mutlak (*ground truth*), sehingga evaluasi akurasi klasik (*precision/recall*) tidak dapat diterapkan tanpa studi pengguna langsung.
3. **Data Hilang pada Laptop Apple & iGPU:** Sekitar 141 laptop tidak memiliki skor benchmark GPU G3D (termasuk 45 laptop Apple yang grafisnya menyatu pada chip SoC dan GPU generik seperti Intel Arc/Radeon). Laptop-laptop ini tidak dapat diranking pada profil Desain dan Gaming, namun tetap dapat diranking pada Office dan Programming karena GPU tidak dijadikan fitur ranking.
4. **Informasi Layar Bersifat Parsial:** Fitur layar (ukuran, resolusi, panel OLED) diekstrak secara heuristik dari nama produk dan hanya difungsikan sebagai filter opsional, bukan fitur jarak KNN.

---

## Cara Memakai Aplikasi

Buka antarmuka web dan tentukan parameter pencarian:

| Isian / Kontrol | Keterangan |
|---|---|
| **Profil penggunaan** | Memilih salah satu dari 4 profil. Memperbarui nilai target bawaan tanpa mengubah budget. |
| **Budget maksimum (Rp)** | Batas kemampuan bayar tertinggi (*hard filter*). Laptop di atas budget dieliminasi. |
| **Target RAM (Acuan Ranking)** | Kapasitas RAM yang paling diinginkan pengguna (magnet acuan ranking KNN). |
| **Batas Minimum RAM (Filter Wajib)** | Batas RAM terendah yang dapat diterima. Laptop di bawah nilai ini dieliminasi. |
| **Target Storage (Acuan Ranking)** | Kapasitas penyimpanan SSD yang paling diinginkan pengguna. |
| **Batas Minimum Storage (Filter Wajib)**| Batas penyimpanan terendah yang masih dapat diterima. |
| **Preferensi performa CPU** | Target level benchmark PassMark CPU (Rendah: P25, Sedang: P50, Tinggi: P75, Sangat Tinggi: P90). |
| **Preferensi performa GPU** | Tingkat target benchmark grafis G3D Mark (wajib pada Desain & Gaming). |
| **Wajib GPU dedicated** | Membatasi hasil hanya pada laptop dengan kartu grafis terpisah (otomatis terkunci pada Gaming). |
| **Filter layar (opsional)** | Filter ukuran layar minimum, resolusi minimum, atau panel khusus OLED. |

---

## Panduan Instalasi dan Menjalankan Aplikasi

### Prasyarat
- Python 3.11 atau lebih baru (`python --version`).
- Git.

### Tanpa MySQL (Cepat via CSV)

Jalur ini membaca katalog langsung dari berkas CSV di repositori (`dataset/laptops_benchmark_enriched_resolved.csv`). Fitur penilaian pengguna tidak tersimpan di database.

**Windows (PowerShell):**
```powershell
git clone https://github.com/glngprr/laptop-recommendation-app.git
cd laptop-recommendation-app

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run streamlit_app.py
```

**macOS / Linux:**
```bash
git clone https://github.com/glngprr/laptop-recommendation-app.git
cd laptop-recommendation-app

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Buka peramban di `http://localhost:8501`.

---

### Dengan MySQL (Opsional)

Jalur ini memungkinkan penyimpanan log sesi pencarian dan rating evaluasi bintang 1–5 dari pengguna.

1. **Buat Database dan Pengguna di MySQL:**
   ```sql
   CREATE DATABASE laptop_recommender CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   CREATE USER 'laptop_app'@'localhost' IDENTIFIED BY 'password-rahasia-anda';
   GRANT ALL PRIVILEGES ON laptop_recommender.* TO 'laptop_app'@'localhost';
   ```

2. **Konfigurasi `secrets.toml`:**
   Salin berkas template:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Isi dengan kredensial MySQL Anda:
   ```toml
   [mysql]
   host = "127.0.0.1"
   port = 3306
   user = "laptop_app"
   password = "password-rahasia-anda"
   database = "laptop_recommender"
   ```

3. **Impor Katalog ke MySQL:**
   ```bash
   python import_catalog.py --dry-run
   python import_catalog.py
   ```
   Impor yang berhasil akan memverifikasi 648 baris katalog terunggah dengan status benchmark `NULL` (tanpa imputasi nol).

4. **Jalankan Aplikasi:**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Menjalankan Eksperimen dan Pengujian

Logika rekomendasi dapat dieksekusi langsung melalui terminal atau script pengujian tanpa membuka antarmuka web.

### 1. Menjalankan Suite Evaluasi 8 Skenario Baseline
```bash
python knn_v8_profile_recommender.py
```
Perintah ini menghasilkan berkas laporan dan audit di direktori `inspection/`:
- `knn_v8_evaluation.md`: laporan komprehensif evaluasi profil.
- `knn_v8_scenario_summary.csv`: ringkasan metrik kandidat per skenario.
- `knn_v8_recommendations.csv`: daftar Top-5 laptop rekomendasi.
- `knn_v8_unrankable_candidates.csv`: daftar kandidat yang kehilangan benchmark beserta alasannya.
- `knn_v8_gpu_classification_audit.csv`: audit aturan klasifikasi jenis GPU.

### 2. Menjalankan Kueri Kustom via CLI
```bash
python knn_v8_profile_recommender.py --profile programming --budget 15000000 --min-ram 8 --target-ram 8 --min-storage 512
```

### 3. Menjalankan Pengujian Logika Rekomendasi (Test Suite)
Untuk memverifikasi kepatuhan hard filter, target preferensi RAM 8 GB vs 16 GB, perubahan storage, dan zero-imputation policy:
```bash
python test_phase2_scenarios.py
```

---

## Struktur Repositori

```
laptop-recommendation-app/
├── streamlit_app.py              Antarmuka web Streamlit (bersih, responsif, fixed dropdowns)
├── service.py                    Lapisan logika bisnis, validasi input, penyusunan penjelasan
├── knn_v8_profile_recommender.py Inti KNN, hard filter, target profil, klasifikasi GPU, parsing layar
├── test_phase2_scenarios.py      Suite pengujian otomatis untuk 6 skenario logika rekomendasi
├── knn_v8_expert_eval.py         Alat evaluasi buta untuk penilai ahli (opsional)
├── db.py                         Modul koneksi dan query berparameter MySQL
├── import_catalog.py             Skrip impor katalog CSV ke database MySQL
├── schema.sql                    Skema tabel database (laptops, sessions, session_items, feedback)
├── requirements.txt              Daftar dependensi Python
├── dataset/
│   └── laptops_benchmark_enriched_resolved.csv   Katalog laptop Plaza IT berskor PassMark (648 baris)
├── inspection/                   Output laporan eksperimen dan evaluasi (dihasilkan otomatis)
├── docs/
│   └── METODOLOGI.md             Dokumentasi ilmiah metodologi, data, dan batasan penelitian
└── .streamlit/
    └── secrets.toml.example      Contoh template konfigurasi MySQL
```

---

## Pemecahan Masalah

| Gejala | Penyebab & Solusi |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` pada PowerShell, lalu ulangi aktivasi virtual environment. |
| `ModuleNotFoundError: service` atau modul lokal | Pastikan menjalankan perintah dari root repositori (`cd laptop-recommendation-app`). |
| Sidebar menampilkan "CSV (cadangan)" padahal MySQL sudah diisi | Periksa apakah service MySQL sedang berjalan dan kredensial pada `.streamlit/secrets.toml` sudah sesuai. Klik tombol **Muat ulang katalog**. |
| `Batas minimum RAM tidak boleh lebih besar dari Target RAM` | Nilai batas minimum (*hard filter*) tidak logis jika lebih tinggi dari target yang diharapkan. Sesuaikan batas minimum lebih kecil atau sama dengan Target RAM. |
| Dataset tidak ditemukan | Pastikan berkas `dataset/laptops_benchmark_enriched_resolved.csv` tersedia di folder repositori. |

---

## Data Katalog

Data katalog bersumber dari katalog daring **Plaza IT** dan skor benchmark dari **PassMark Software** (CPU Mark dan G3D Mark) per tanggal potret 5 Oktober 2026. Data digunakan murni untuk keperluan penelitian dan tugas akademik sistem basis data lanjut. Seluruh hak cipta merek dagang dan data tetap berada pada masing-masing pemiliknya.
