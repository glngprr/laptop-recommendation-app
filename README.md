# Sistem Rekomendasi Laptop Berbasis KNN

**Penerapan K-Nearest Neighbors (KNN) pada Content-Based Filtering untuk Sistem Rekomendasi Laptop Berbasis Web**

Aplikasi web (Streamlit) yang merekomendasikan laptop dari katalog Plaza IT berdasarkan **kebutuhan pengguna**: profil penggunaan, budget, serta batas RAM dan storage. Spesifikasi laptop dicocokkan dengan target numerik tiap profil menggunakan KNN.

> **Status: prototipe penelitian.** Hasil menunjukkan _kesesuaian spesifikasi terhadap target operasional_, bukan jaminan laptop terbaik. Target performa berasal dari persentil katalog (asumsi, bukan standar industri), dan kualitas rekomendasi **belum divalidasi** oleh ahli atau pengguna.

Penjelasan metode, data, hasil, dan keterbatasan ada di [docs/METODOLOGI.md](docs/METODOLOGI.md).

## Daftar isi

- [Fitur](#fitur)
- [Cara memakai aplikasi](#cara-memakai-aplikasi)
- [Menjalankan eksperimen dari terminal](#menjalankan-eksperimen-dari-terminal)
- [Struktur repositori](#struktur-repositori)
- [Pemecahan masalah](#pemecahan-masalah)
- [Data](#data)

## Fitur

- **Empat profil kebutuhan:** Office dan kuliah, Programming dan web development, Desain grafis dan editing video, Gaming. Tiap profil punya target RAM, storage, CPU, dan (bila relevan) GPU yang dapat diubah.
- **Filter wajib:** laptop yang melampaui budget atau di bawah RAM/storage minimum tidak pernah ditampilkan. Gaming mewajibkan GPU dedicated.
- **Ranking KNN** terhadap target profil. Harga bukan fitur ranking; harga hanya batas wajib dan pemutus seri.
- **Tanpa imputasi:** laptop yang skor benchmark-nya tidak tersedia tidak diperkirakan atau dianggap nol, melainkan dicantumkan di daftar _belum dapat diranking_ beserta alasannya.
- **Mengapa direkomendasikan?** Penjelasan per laptop, dirangkai hanya dari angka yang dipakai ranking.
- **Cari yang mirip:** laptop dengan spesifikasi terdekat dari satu laptop pilihan (budget boleh dilampaui).
- **Filter layar opsional:** ukuran minimum, resolusi minimum, dan hanya OLED (dibaca dari nama produk).
- **Penilaian pengguna (opsional, perlu MySQL):** bintang 1 sampai 5 per rekomendasi, disimpan anonim, dengan tab ringkasan evaluasi.
- **Dua sumber katalog:** MySQL bila dikonfigurasi, otomatis jatuh ke CSV bila tidak.

## tanpa MySQL

Jalur ini cukup untuk mencoba aplikasi. Tanpa MySQL, katalog dibaca dari CSV di repositori dan penilaian pengguna tidak tersimpan.

### Prasyarat

- **Python 3.11 atau lebih baru** (diuji pada 3.12 dan 3.13). Cek dengan `python --version`.
- **Git**.

### Langkah (Windows, PowerShell)

```powershell
git clone https://github.com/glngprr/laptop-recommendation-app.git
cd laptop-recommendation-app

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Jika `Activate.ps1` ditolak dengan pesan _running scripts is disabled_, jalankan dulu `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, lalu ulangi aktivasi.

### Langkah (macOS / Linux)

```bash
git clone https://github.com/glngprr/laptop-recommendation-app.git
cd laptop-recommendation-app

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Browser terbuka di `http://localhost:8501`. Hentikan aplikasi dengan `Ctrl+C`.

### Memastikan berjalan benar

- Sidebar menampilkan **"CSV (cadangan): 648 laptop"** dan keterangan bahwa MySQL belum dikonfigurasi. Itu normal pada jalur ini.
- Coba satu pencarian dan bandingkan dengan angka acuan berikut (dihitung dari data di repositori):

| Pengaturan                                                                                  | Hasil yang diharapkan                                                                       |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Programming, budget 15.000.000, RAM 16 GB, storage 512 GB, CPU Sedang, GPU Tidak diperlukan | 127 lolos filter, 127 dapat diranking, 5 rekomendasi; peringkat 1: Asus Vivobook 14 M1407KA |
| Office, budget 10.000.000, RAM 8 GB, storage 512 GB, CPU Rendah, GPU Tidak diperlukan       | 81 lolos filter, 81 dapat diranking; peringkat 1: HP 14-ep1733TU                            |
| Gaming, budget 20.000.000, RAM 16 GB, storage 512 GB, CPU Tinggi, GPU dedicated sedang      | 247 lolos filter dasar, 40 laptop ber-GPU dedicated, 40 dapat diranking                     |
| Desain, budget 25.000.000, RAM 16 GB, storage 512 GB, CPU Tinggi, GPU dedicated sedang      | 325 lolos filter, 252 dapat diranking, 73 belum dapat diranking                             |

Angka yang berbeda biasanya berarti ada isian formulir yang tidak sama (misalnya level CPU) atau dataset di repositori telah berubah.

## dengan MySQL (opsional)

Gunakan jalur ini bila ingin menyimpan penilaian pengguna dan melihat tab **Evaluasi pengguna**. Aplikasi membaca katalog dari MySQL dan otomatis kembali ke CSV bila koneksi gagal.

**1. Pasang MySQL Server 8 dan (opsional) MySQL Workbench**, lalu pastikan server berjalan.

**2. Buat database dan pengguna** (di Workbench atau klien `mysql`):

```sql
CREATE DATABASE laptop_recommender CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'laptop_app'@'localhost' IDENTIFIED BY 'password-yang-kuat';
GRANT ALL PRIVILEGES ON laptop_recommender.* TO 'laptop_app'@'localhost';
```

**3. Isi konfigurasi koneksi.** Salin berkas contoh lalu isi `password` dengan yang Anda buat di langkah 2:

```powershell
# Windows
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

```bash
# macOS / Linux
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Isi berkas:

```toml
[mysql]
host = "127.0.0.1"
port = 3306
user = "laptop_app"
password = "GANTI_DENGAN_PASSWORD_ANDA"
database = "laptop_recommender"
```

`.streamlit/secrets.toml` sudah masuk `.gitignore`. **Jangan pernah meng-commit file ini.**

**4. Periksa CSV, lalu impor ke MySQL:**

```bash
python import_catalog.py --dry-run
python import_catalog.py
```

`--dry-run` hanya memeriksa CSV dan tidak menyentuh database. Impor yang benar diakhiri dengan `VERIFIKASI: OK`, dengan jumlah baris 648, NULL skor CPU 12, NULL skor GPU 141, dan tidak ada nilai nol. Skor benchmark yang hilang disimpan sebagai `NULL`, tidak pernah 0. Menjalankan impor ulang aman: baris dicocokkan lewat `url`, dan keluarannya melaporkan berapa baris yang benar-benar berubah.

**5. (Opsional) Periksa di Workbench:**

```sql
SELECT COUNT(*) FROM laptops;   -- 648

SELECT processor_normalized, cpu_passmark_mark, COUNT(*) AS jumlah
FROM laptops WHERE processor_normalized IN ('M1', 'M3')
GROUP BY processor_normalized, cpu_passmark_mark;
-- M1 = 14120 (1 baris), M3 = 19051 (7 baris)
```

**6. Jalankan aplikasi:**

```bash
streamlit run streamlit_app.py
```

Sidebar harus menampilkan **"MySQL: 648 laptop"**. Jika masih tertulis "CSV (cadangan)", keterangan di bawahnya menyebut penyebabnya; lihat [Pemecahan masalah](#pemecahan-masalah), lalu klik **Muat ulang katalog**.

**Python di bawah 3.11:** `import_catalog.py` tidak dapat membaca `secrets.toml`. Gunakan variabel lingkungan sebagai gantinya (PowerShell):

```powershell
$env:MYSQL_HOST="127.0.0.1"; $env:MYSQL_PORT="3306"
$env:MYSQL_USER="laptop_app"; $env:MYSQL_PASSWORD="password-yang-kuat"
$env:MYSQL_DATABASE="laptop_recommender"
python import_catalog.py
```

### Data penilaian

Setiap pencarian dan penilaian disimpan **anonim** (tanpa nama, email, atau alamat IP) hanya bila MySQL aktif dan kotak persetujuan di formulir dicentang. Kolom komentar bersifat bebas dan opsional: jangan menuliskan data pribadi di dalamnya. Untuk mengosongkan data uji sebelum pengumpulan data sungguhan, jalankan `DELETE FROM sessions;` di Workbench (tabel `laptops` tidak tersentuh).

## Cara memakai aplikasi

| Isian                                | Arti                                                                                                                                                                                                  |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Profil penggunaan**                | Menentukan fitur ranking dan target awal. Mengganti profil mengisi ulang RAM, storage, level CPU, dan level GPU dengan nilai bawaan profil.                                                           |
| **Budget maksimum (Rp)**             | Batas wajib harga.                                                                                                                                                                                    |
| **RAM minimum**, **Storage minimum** | Batas wajib. Target ranking adalah yang lebih tinggi antara minimum Anda dan rekomendasi profil.                                                                                                      |
| **Preferensi performa CPU**          | Level target CPU (Rendah, Sedang, Tinggi, Sangat tinggi), berupa persentil skor PassMark seluruh katalog.                                                                                             |
| **Preferensi performa GPU**          | "Tidak diperlukan" berarti GPU tidak dipakai dalam ranking (Office dan Programming). Memilih level GPU memasukkan GPU ke ranking dan mewajibkan laptop punya skor GPU.                                |
| **Wajib GPU dedicated**              | Hanya laptop yang terklasifikasi GPU dedicated. Gaming selalu mewajibkannya.                                                                                                                          |
| **Filter layar**                     | Ukuran minimum, resolusi minimum, dan "hanya OLED". Laptop yang resolusinya tidak tertulis pada nama produk tidak ikut bila filter resolusi dipakai, dan dicantumkan di daftar belum dapat diranking. |
| **Izinkan penyimpanan anonim**       | Persetujuan menyimpan kriteria pencarian dan penilaian (hanya aktif bila MySQL tersedia).                                                                                                             |

Pada hasil:

- Ringkasan angka: laptop di katalog, lolos filter, dapat diranking, dan belum dapat diranking.
- **Kartu rekomendasi:** harga, spesifikasi, layar, serta keterangan tiap fitur ("memenuhi target" atau "N% di bawah target"). Laptop dengan spesifikasi identik pada fitur yang dipakai dikelompokkan; yang tampil adalah varian termurah.
- **Mengapa direkomendasikan?** Uraian posisi laptop terhadap target, posisi harga, dan peringkatnya.
- **Cari yang mirip:** lima laptop terdekat dengan spesifikasi laptop itu. Budget tidak membatasi; laptop yang melampaui budget diberi tanda.
- **Daftar belum dapat diranking:** laptop yang lolos syarat awal tetapi tidak punya skor benchmark yang dibutuhkan, lengkap dengan alasannya.
- **Jarak KNN** hanya ukuran kedekatan terhadap target, bukan skor mutu.

## Menjalankan eksperimen dari terminal

Logika rekomendasi (`knn_v8_profile_recommender.py`) dapat dijalankan tanpa antarmuka. Jalankan dari root repositori; semua keluaran masuk ke folder `inspection/` (dibuat otomatis).

```bash
# 8 skenario evaluasi (2 per profil)
python knn_v8_profile_recommender.py

# satu query kustom
python knn_v8_profile_recommender.py --profile programming --budget 15000000 --min-ram 16 --min-storage 512
```

Opsi lain: `--cpu-level`, `--gpu-level`, `--require-dedicated` / `--no-require-dedicated`, `--lambda-surplus`, `--top-n`, `--input`. Gunakan `--help` untuk daftar lengkap. Filter layar hanya tersedia di aplikasi web.

| Berkas di `inspection/`               | Isi                                                                          |
| ------------------------------------- | ---------------------------------------------------------------------------- |
| `knn_v8_evaluation.md`                | Laporan: jumlah kandidat, pemeriksaan constraint, sensitivitas, contoh audit |
| `knn_v8_scenario_summary.csv`         | Satu baris per skenario dengan semua hitungan dan diagnostik                 |
| `knn_v8_recommendations.csv`          | Top-5 per skenario                                                           |
| `knn_v8_full_ranking.csv`             | Seluruh kandidat yang dapat diranking, termasuk varian setara                |
| `knn_v8_unrankable_candidates.csv`    | Kandidat yang tidak dapat diranking beserta alasannya                        |
| `knn_v8_gpu_classification_audit.csv` | Setiap nama GPU, kelasnya, dasar aturan, dan status validasi                 |
| `knn_v8_reference_targets.csv`        | Nilai level CPU dan GPU yang dipakai sebagai target                          |

Query kustom menulis berkas dengan awalan `knn_v8_custom_`. Folder `inspection/` pada repositori berisi contoh keluaran dari satu eksekusi.

`knn_v8_expert_eval.py` adalah alat opsional untuk evaluasi oleh penilai ahli (lembar penilaian buta, lalu penghitungan skor):

```bash
python knn_v8_expert_eval.py make-sheet
python knn_v8_expert_eval.py score --ratings inspection/expert_eval/filled_R1.csv inspection/expert_eval/filled_R2.csv
```

## Struktur repositori

```
laptop-recommendation-app/
├── streamlit_app.py                 antarmuka web (Streamlit)
├── service.py                       validasi input, penyusunan hasil, penjelasan, laptop serupa
├── knn_v8_profile_recommender.py    inti: filter wajib, target profil, KNN, klasifikasi GPU, parsing layar
├── knn_v8_expert_eval.py            alat evaluasi ahli (opsional)
├── db.py                            akses MySQL (PyMySQL, kueri berparameter)
├── import_catalog.py                impor CSV ke MySQL + verifikasi
├── schema.sql                       skema tabel: laptops, sessions, session_items, feedback
├── requirements.txt
├── dataset/
│   └── laptops_benchmark_enriched_resolved.csv    katalog (648 laptop)
├── inspection/                      keluaran eksperimen v8 (dibuat oleh skrip)
├── docs/
│   └── METODOLOGI.md                metode, data, hasil, keterbatasan
└── .streamlit/
    └── secrets.toml.example         contoh konfigurasi MySQL (salin menjadi secrets.toml)
```

## Pemecahan masalah

| Gejala                                                            | Penyebab dan solusi                                                                                                                                                                                  |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Activate.ps1 ... running scripts is disabled`                    | Jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, lalu aktifkan ulang `.venv`.                                                                                                  |
| `streamlit` tidak dikenali                                        | `.venv` belum aktif atau paket belum terpasang. Aktifkan `.venv`, lalu `pip install -r requirements.txt`.                                                                                            |
| `ModuleNotFoundError: service` (atau modul lain milik repositori) | Perintah dijalankan bukan dari root repositori. Pindah ke folder yang berisi `streamlit_app.py`.                                                                                                     |
| Port 8501 dipakai                                                 | `streamlit run streamlit_app.py --server.port 8502`                                                                                                                                                  |
| Sidebar "CSV (cadangan)" padahal MySQL sudah diisi                | Baca keterangan di bawah label itu. Periksa server MySQL berjalan, `secrets.toml` berada di `.streamlit/` dan nilainya benar, dan `import_catalog.py` sudah dijalankan. Klik **Muat ulang katalog**. |
| `Access denied for user`                                          | Nama pengguna atau password pada `secrets.toml` tidak cocok dengan yang dibuat di MySQL.                                                                                                             |
| Galat autentikasi `caching_sha2_password`                         | Paket `cryptography` belum terpasang: `pip install cryptography` (sudah ada di `requirements.txt`).                                                                                                  |
| `Dataset tidak ditemukan`                                         | Jalankan dari root repositori; berkas `dataset/laptops_benchmark_enriched_resolved.csv` harus ada.                                                                                                   |
| `import_catalog.py` gagal membaca `secrets.toml`                  | Python di bawah 3.11. Gunakan variabel lingkungan (lihat bagian MySQL).                                                                                                                              |
| Penilaian bintang tidak muncul atau tidak tersimpan               | Memerlukan sumber katalog MySQL dan kotak persetujuan tercentang.                                                                                                                                    |
| Peringatan `LF will be replaced by CRLF` saat `git add`           | Hanya soal akhir baris di Windows; tidak berbahaya.                                                                                                                                                  |

## Data

- **Data katalog** bersumber dari **Plaza IT** dan **skor benchmark** dari **PassMark** (CPU Mark dan G3D Mark); hak atas data tetap pada pemiliknya. Data dipakai di sini untuk tujuan penelitian dan akademik, dan merupakan potret katalog pada tanggal 5 Oktober 2026. Harga dan ketersediaan dapat telah berubah. Jika Anda pemilik data dan keberatan, hubungi pemilik repositori.
- Selengkapnya dijelaskan di [docs/METODOLOGI.md](docs/METODOLOGI.md#2-data).
