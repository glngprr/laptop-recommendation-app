# KNN v8 - Evaluasi Profil Kebutuhan (skenario ilustrasi)

Laporan dibuat otomatis oleh `knn_v8_profile_recommender.py`. Seluruh angka berasal dari eksekusi script terhadap file input di bawah ini.

- Input: `laptops_benchmark_enriched_resolved.csv` (648 baris)
- Klasifikasi GPU: dedicated=168, integrated=477, unknown=3 (aturan terdokumentasi; belum seluruhnya divalidasi manual, lihat `knn_v8_gpu_classification_audit.csv`)

## 1. Metodologi singkat

1. **Hard filter**: harga <= budget, RAM >= minimum, storage >= minimum; Gaming juga mewajibkan `gpu_class = dedicated`.
2. **Target**: tabel profil eksplisit. RAM/storage target = max(minimum pengguna, rekomendasi profil). Target CPU/GPU = persentil referensi **tetap** (bukan median pool terfilter).
3. **KNN**: StandardScaler dipasang pada seluruh katalog (populasi referensi), jarak Euclidean ke target; kelebihan di atas target dikalikan sqrt(lambda), kekurangan berbobot penuh. Harga bukan fitur; harga hanya batas, pemutus seri, dan kolom tampilan.
4. **Seri/duplikasi**: jarak sama (1e-9) -> harga terendah. Laptop dengan vektor fitur identik dikelompokkan; satu perwakilan (termurah) ditampilkan, varian lain tetap ada di `knn_v8_full_ranking.csv`.
5. **Data hilang**: tidak ada imputasi. Kandidat yang lolos hard filter tetapi tidak punya skor yang dibutuhkan dicatat di `knn_v8_unrankable_candidates.csv`.

## 2. Target referensi (asumsi operasional, bukan standar industri)

| jenis | level | persentil | nilai | n | populasi |
|---|---|---|---|---|---|
| cpu | (populasi) |  |  | 636 | semua laptop dengan skor CPU |
| cpu | rendah | 0.25 | 14.823 | 636 | semua laptop dengan skor CPU |
| cpu | sedang | 0.5 | 18.871 | 636 | semua laptop dengan skor CPU |
| cpu | tinggi | 0.75 | 24.969 | 636 | semua laptop dengan skor CPU |
| cpu | sangat_tinggi | 0.9 | 34.222 | 636 | semua laptop dengan skor CPU |
| gpu | igpu_memadai | 0.75 | 6.186 | 19 | model GPU unik, kelas integrated, berskor G3D |
| gpu | dgpu_rendah | 0.25 | 12.352 | 16 | model GPU unik, kelas dedicated, berskor G3D |
| gpu | dgpu_sedang | 0.5 | 17.042 | 16 | model GPU unik, kelas dedicated, berskor G3D |
| gpu | dgpu_tinggi | 0.75 | 23.039 | 16 | model GPU unik, kelas dedicated, berskor G3D |
| gpu | dgpu_sangat_tinggi | 0.9 | 26.656 | 16 | model GPU unik, kelas dedicated, berskor G3D |

Catatan: persentil bersifat relatif terhadap katalog Plaza IT pada tanggal data ini dan akan berubah bila katalog berubah. Level GPU dihitung atas model GPU unik agar tidak ditentukan stok toko.

## 3. Jumlah kandidat per skenario

Skenario bersifat ilustrasi eksplorasi; bukan hasil rekomendasi final.

| skenario | budget | RAM_min | storage_min | awal | lolos_filter_dasar | lolos_syarat_GPU | dapat_diranking | keluar_CPU_hilang | keluar_GPU_hilang | tidak_terkonfirmasi_dedicated | rekomendasi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| office_A | Rp10.000.000 | 8 | 256 | 648 | 105 | 105 | 105 | 0 | 0 | 0 | 5 |
| office_B | Rp15.000.000 | 8 | 512 | 648 | 242 | 242 | 242 | 0 | 0 | 0 | 5 |
| programming_A | Rp14.999.994 | 16 | 512 | 648 | 127 | 127 | 127 | 0 | 0 | 0 | 5 |
| programming_B | Rp20.000.000 | 16 | 512 | 648 | 247 | 247 | 247 | 0 | 0 | 0 | 5 |
| design_A | Rp25.000.000 | 16 | 512 | 648 | 325 | 325 | 252 | 0 | 73 | 0 | 5 |
| design_B | Rp35.000.000 | 32 | 1024 | 648 | 52 | 52 | 45 | 0 | 7 | 0 | 5 |
| gaming_A | Rp20.000.000 | 16 | 512 | 648 | 247 | 40 | 40 | 0 | 0 | 0 | 5 |
| gaming_B | Rp30.000.000 | 16 | 1024 | 648 | 94 | 24 | 24 | 0 | 0 | 1 | 5 |

Klasifikasi GPU pada kandidat yang lolos filter dasar:

| skenario | dedicated | integrated | unknown | fitur |
|---|---|---|---|---|
| office_A | 3 | 101 | 1 | ram_gb+storage_gb+cpu_passmark_mark |
| office_B | 23 | 218 | 1 | ram_gb+storage_gb+cpu_passmark_mark |
| programming_A | 10 | 117 | 0 | ram_gb+storage_gb+cpu_passmark_mark |
| programming_B | 40 | 207 | 0 | ram_gb+storage_gb+cpu_passmark_mark |
| design_A | 63 | 261 | 1 | ram_gb+storage_gb+cpu_passmark_mark+gpu_passmark_g3d_mark |
| design_B | 11 | 41 | 0 | ram_gb+storage_gb+cpu_passmark_mark+gpu_passmark_g3d_mark |
| gaming_A | 40 | 207 | 0 | ram_gb+storage_gb+cpu_passmark_mark+gpu_passmark_g3d_mark |
| gaming_B | 24 | 69 | 1 | ram_gb+storage_gb+cpu_passmark_mark+gpu_passmark_g3d_mark |

## 4. Pemeriksaan constraint dan keragaman

| skenario | rekomendasi | budget_ok | RAM_ok | storage_ok | dedicated_ok | vektor_unik_pool | vektor_duplikat_di_rekomendasi | seri_pada_jarak_terbaik | rekomendasi_dgn_fitur_di_bawah_target | pool_mencapai_semua_target | pool_GPU_mencapai_target |
|---|---|---|---|---|---|---|---|---|---|---|---|
| office_A | 5 | True | True | True | True | 40 | 0 | 2 baris / 1 vektor | 0 | 15.2% | n/a |
| office_B | 5 | True | True | True | True | 76 | 0 | 13 baris / 1 vektor | 0 | 54.5% | n/a |
| programming_A | 5 | True | True | True | True | 41 | 0 | 4 baris / 1 vektor | 1 | 28.3% | n/a |
| programming_B | 5 | True | True | True | True | 78 | 0 | 1 baris / 1 vektor | 1 | 40.9% | n/a |
| design_A | 5 | True | True | True | True | 109 | 0 | 1 baris / 1 vektor | 5 | 1.6% | 3.2% |
| design_B | 5 | True | True | True | True | 24 | 0 | 1 baris / 1 vektor | 3 | 6.7% | 8.9% |
| gaming_A | 5 | True | True | True | True | 27 | 0 | 1 baris / 1 vektor | 4 | 2.5% | 10.0% |
| gaming_B | 5 | True | True | True | True | 21 | 0 | 2 baris / 1 vektor | 3 | 41.7% | 45.8% |

Catatan membaca jarak: jarak adalah ukuran kedekatan terhadap target, bukan mutu rekomendasi. Bila `pool_mencapai_semua_target` rendah atau 0%, target berada di luar jangkauan pool dan jarak yang besar tidak berarti rekomendasinya buruk; ranking menjadi 'terdekat dari bawah'. Gunakan kolom `kekurangan_terhadap_target` (persen di bawah target per fitur) untuk antarmuka pengguna.

## 5. Sensitivitas (tumpang tindih Jaccard himpunan vektor top-N terhadap baseline)

Nilai 1,00 = himpunan identik; nilai rendah = ranking sensitif terhadap asumsi tersebut.

| skenario | lambda_1 | lambda_0 | CPU_level_turun | CPU_level_naik | GPU_level_turun | GPU_level_naik |
|---|---|---|---|---|---|---|
| office_A | 0.43 | 0.43 | n/a | 0.25 | n/a | n/a |
| office_B | 0.43 | 0.11 | n/a | 0.00 | n/a | n/a |
| programming_A | 0.67 | 0.25 | 0.00 | 0.00 | n/a | n/a |
| programming_B | 0.67 | 0.11 | 0.00 | 0.00 | n/a | n/a |
| design_A | 1.00 | 0.00 | 0.67 | 0.00 | 0.00 | 0.43 |
| design_B | 0.67 | 0.67 | 0.67 | 0.43 | 0.43 | 0.67 |
| gaming_A | 0.67 | 0.67 | 0.43 | 0.25 | 0.11 | 0.67 |
| gaming_B | 0.67 | 0.11 | 0.67 | 0.25 | 0.43 | 0.67 |

## 6. Contoh audit ranking per skenario

### office_A - Office dan kuliah

- Fitur ranking: ram_gb+storage_gb+cpu_passmark_mark; lambda=0.25
- Target: RAM 8 GB, storage 512 GB, CPU 14.823 (level rendah), GPU tidak dipakai dalam ranking

| rank | produk | harga | RAM | storage | CPU | GPU_G3D | jarak | fitur_di_bawah_target | kekurangan_terhadap_target | varian_setara |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HP 14-ep1733TU - C45SLPA /Intel Core 5-120U/8GB/512GB SSD/In | Rp8.700.000 | 8 | 512 | 14.877 | 3.264 (integrated) | 0.0024 | 0 | - | 1 |
| 2 | HP 14-em0019AU - 7L9D0PA /AMD Ryzen 5-7530U/8GB/512GB SSD/AM | Rp7.099.000 | 8 | 512 | 14.882 | n/a (integrated) | 0.0027 | 0 | - | 0 |
| 3 | Acer Aspire Lite AL15-42P R50P /AMD Ryzen 5-7430U/8GB/512GB  | Rp8.999.000 | 8 | 512 | 15.082 | n/a (integrated) | 0.0117 | 0 | - | 1 |
| 4 | Infinix XBOOK B14 /AMD Ryzen 5-7535HS/8GB/512GB SSD/AMD Rade | Rp8.399.000 | 8 | 512 | 17.581 | 3.126 (integrated) | 0.1246 | 0 | - | 0 |
| 5 | Infinix XBOOK B14 /AMD Ryzen 7-5825U/8GB/512GB SSD/AMD Radeo | Rp7.899.000 | 8 | 512 | 17.777 | 4.016 (integrated) | 0.1334 | 0 | - | 1 |

### office_B - Office dan kuliah

- Fitur ranking: ram_gb+storage_gb+cpu_passmark_mark; lambda=0.25
- Target: RAM 8 GB, storage 512 GB, CPU 14.823 (level rendah), GPU tidak dipakai dalam ranking

| rank | produk | harga | RAM | storage | CPU | GPU_G3D | jarak | fitur_di_bawah_target | kekurangan_terhadap_target | varian_setara |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HP 14-ep1733TU - C45SLPA /Intel Core 5-120U/8GB/512GB SSD/In | Rp8.700.000 | 8 | 512 | 14.877 | 3.264 (integrated) | 0.0024 | 0 | - | 12 |
| 2 | HP 14-em0019AU - 7L9D0PA /AMD Ryzen 5-7530U/8GB/512GB SSD/AM | Rp7.099.000 | 8 | 512 | 14.882 | n/a (integrated) | 0.0027 | 0 | - | 2 |
| 3 | Acer Aspire Lite AL15-42P R50P /AMD Ryzen 5-7430U/8GB/512GB  | Rp8.999.000 | 8 | 512 | 15.082 | n/a (integrated) | 0.0117 | 0 | - | 1 |
| 4 | Lenovo IdeaPad Slim 3 14IRH10 - 83K000CVID /Intel Core i5-13 | Rp10.999.000 | 8 | 512 | 16.754 | 1.541 (integrated) | 0.0872 | 0 | - | 4 |
| 5 | Acer Aspire Lite 14 AL14-44P R1XH /AMD Ryzen 7-7730U/8GB/512 | Rp10.899.000 | 8 | 512 | 16.894 | n/a (integrated) | 0.0936 | 0 | - | 0 |

### programming_A - Programming dan web development

- Fitur ranking: ram_gb+storage_gb+cpu_passmark_mark; lambda=0.25
- Target: RAM 16 GB, storage 512 GB, CPU 18.871 (level sedang), GPU tidak dipakai dalam ranking

| rank | produk | harga | RAM | storage | CPU | GPU_G3D | jarak | fitur_di_bawah_target | kekurangan_terhadap_target | varian_setara |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Asus Vivobook 14 M1407KA-VIPS5153M /AMD Ryzen AI 5-340/16GB/ | Rp11.399.000 | 16 | 512 | 19.527 | 3.667 (integrated) | 0.0296 | 0 | - | 3 |
| 2 | HP Pavilion Aero 13-bg0222AU - A0DH3PA /AMD Ryzen 5-8640U/16 | Rp12.900.000 | 16 | 512 | 19.859 | n/a (integrated) | 0.0446 | 0 | - | 1 |
| 3 | HP 15-fd1222TU - 9Y3N9PA /Intel Core Ultra 5-125H/16GB/512GB | Rp9.000.000 | 16 | 512 | 19.971 | n/a (integrated) | 0.0497 | 0 | - | 5 |
| 4 | Asus ExpertBook P1 P1403CVA-S65152WS /Intel Core 5-210H/16GB | Rp12.999.000 | 16 | 512 | 18.103 | 3.264 (integrated) | 0.0694 | 1 | CPU -4% | 5 |
| 5 | Asus ExpertBook PM1 PM1403CDA-S67151WS /AMD Ryzen 7-170/16GB | Rp13.199.000 | 16 | 512 | 20.630 | n/a (integrated) | 0.0795 | 0 | - | 0 |

### programming_B - Programming dan web development

- Fitur ranking: ram_gb+storage_gb+cpu_passmark_mark; lambda=0.25
- Target: RAM 16 GB, storage 512 GB, CPU 18.871 (level sedang), GPU tidak dipakai dalam ranking

| rank | produk | harga | RAM | storage | CPU | GPU_G3D | jarak | fitur_di_bawah_target | kekurangan_terhadap_target | varian_setara |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Apple MacBook Air M3 (13-inch) 8-core CPU and 10-core GPU -  | Rp19.990.000 | 16 | 512 | 19.051 | n/a (integrated) | 0.0081 | 0 | - | 0 |
| 2 | Lenovo Yoga Slim 7 14ILL11 Aura Edition - 83R0001NID /Intel  | Rp19.799.000 | 16 | 512 | 19.491 | 5.142 (integrated) | 0.0280 | 0 | - | 0 |
| 3 | Asus Vivobook 14 M1407KA-VIPS5153M /AMD Ryzen AI 5-340/16GB/ | Rp11.399.000 | 16 | 512 | 19.527 | 3.667 (integrated) | 0.0296 | 0 | - | 7 |
| 4 | Acer Nitro V 15 ANV15-42 R60U /AMD Ryzen 7-7445HS/16GB/512GB | Rp17.799.000 | 16 | 512 | 18.499 | 14.201 (dedicated) | 0.0336 | 1 | CPU -2% | 1 |
| 5 | HP Pavilion Aero 13-bg0222AU - A0DH3PA /AMD Ryzen 5-8640U/16 | Rp12.900.000 | 16 | 512 | 19.859 | n/a (integrated) | 0.0446 | 0 | - | 1 |

### design_A - Desain grafis dan editing video

- Fitur ranking: ram_gb+storage_gb+cpu_passmark_mark+gpu_passmark_g3d_mark; lambda=0.25
- Target: RAM 16 GB, storage 512 GB, CPU 24.969 (level tinggi), GPU G3D 17.042 (level dgpu_sedang)

| rank | produk | harga | RAM | storage | CPU | GPU_G3D | jarak | fitur_di_bawah_target | kekurangan_terhadap_target | varian_setara |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Lenovo Gaming LOQ 15IRX11 Essential - 83SC003DID /Intel Core | Rp20.499.000 | 16 | 512 | 24.392 | 16.758 (dedicated) | 0.0628 | 2 | CPU -2%, GPU -2% | 0 |
| 2 | Asus Gaming V16 V3607VM-I7N56B6T-HM /Intel Core 7 240H/16GB/ | Rp21.399.000 | 16 | 512 | 23.970 | 16.758 (dedicated) | 0.0932 | 2 | CPU -4%, GPU -2% | 0 |
| 3 | Asus TUF F15 FX507VV-I745K6M-OM /Intel Core i7-13620H/16GB/5 | Rp21.999.000 | 16 | 512 | 23.226 | 17.326 (dedicated) | 0.1487 | 1 | CPU -7% | 0 |
| 4 | Asus TUF FA507NVR-R746K6M-OM /AMD Ryzen 7-7435HS/16GB/512GB  | Rp18.399.000 | 16 | 512 | 22.573 | 17.326 (dedicated) | 0.2035 | 1 | CPU -10% | 0 |
| 5 | Asus TUF Gaming A15 FA507NV-R746KLM-O /AMD Ryzen 7-7735HS/16 | Rp18.499.000 | 16 | 512 | 22.136 | 17.326 (dedicated) | 0.2403 | 1 | CPU -11% | 0 |

### design_B - Desain grafis dan editing video

- Fitur ranking: ram_gb+storage_gb+cpu_passmark_mark+gpu_passmark_g3d_mark; lambda=0.25
- Target: RAM 32 GB, storage 1.024 GB, CPU 24.969 (level tinggi), GPU G3D 17.042 (level dgpu_sedang)

| rank | produk | harga | RAM | storage | CPU | GPU_G3D | jarak | fitur_di_bawah_target | kekurangan_terhadap_target | varian_setara |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Asus TUF A14 FA401KM-R7N56G6G-HM /AMD Ryzen AI 7-350/32GB/1T | Rp28.699.000 | 32 | 1.024 | 24.969 | 16.758 (dedicated) | 0.0395 | 1 | GPU -2% | 0 |
| 2 | Acer Swift X 14 AI SFX14-72G 70K8 /Intel Core Ultra 7-155H/3 | Rp25.499.000 | 32 | 1.024 | 24.437 | 17.326 (dedicated) | 0.0491 | 1 | CPU -2% | 0 |
| 3 | Lenovo Yoga Pro 7 14IMH9 - 83E200DYID /Intel Core Ultra 9-18 | Rp29.799.000 | 32 | 1.024 | 28.963 | 17.326 (dedicated) | 0.1700 | 0 | - | 0 |
| 4 | HP OMEN 16-ap0196AX - C0DP3PA /AMD Ryzen AI 9-365/32GB/1TB S | Rp28.720.000 | 32 | 1.024 | 30.338 | 16.758 (dedicated) | 0.2304 | 1 | GPU -2% | 0 |
| 5 | Asus TUF F16 FX608JPR-I757G9G-HM /Intel Core i7-14650HX/32GB | Rp28.710.000 | 32 | 1.024 | 33.365 | 19.118 (dedicated) | 0.3831 | 0 | - | 0 |

### gaming_A - Gaming

- Fitur ranking: ram_gb+storage_gb+cpu_passmark_mark+gpu_passmark_g3d_mark; lambda=0.25
- Target: RAM 16 GB, storage 512 GB, CPU 24.969 (level tinggi), GPU G3D 17.042 (level dgpu_sedang)

| rank | produk | harga | RAM | storage | CPU | GPU_G3D | jarak | fitur_di_bawah_target | kekurangan_terhadap_target | varian_setara |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Asus TUF FA507NVR-R746K6M-OM /AMD Ryzen 7-7435HS/16GB/512GB  | Rp18.399.000 | 16 | 512 | 22.573 | 17.326 (dedicated) | 0.2035 | 1 | CPU -10% | 0 |
| 2 | Asus TUF Gaming A15 FA507NV-R746KLM-O /AMD Ryzen 7-7735HS/16 | Rp18.499.000 | 16 | 512 | 22.136 | 17.326 (dedicated) | 0.2403 | 1 | CPU -11% | 0 |
| 3 | HP Victus 15-fb2999AX - A4TZ2PA /AMD Ryzen 5-8645HS/16GB/512 | Rp15.599.000 | 16 | 512 | 21.979 | 17.326 (dedicated) | 0.2535 | 1 | CPU -12% | 0 |
| 4 | Acer Nitro V 16 ANV16-71 73GB /Intel Core i7-14650HX/16GB/51 | Rp18.299.000 | 16 | 512 | 33.365 | 17.326 (dedicated) | 0.3554 | 0 | - | 0 |
| 5 | Acer Nitro Lite 16 NL16-71G 78GF /Intel Core 7-240H/16GB/512 | Rp17.799.000 | 16 | 512 | 23.970 | 14.201 (dedicated) | 0.4041 | 2 | CPU -4%, GPU -17% | 2 |

### gaming_B - Gaming

- Fitur ranking: ram_gb+storage_gb+cpu_passmark_mark+gpu_passmark_g3d_mark; lambda=0.25
- Target: RAM 16 GB, storage 1.024 GB, CPU 24.969 (level tinggi), GPU G3D 17.042 (level dgpu_sedang)

| rank | produk | harga | RAM | storage | CPU | GPU_G3D | jarak | fitur_di_bawah_target | kekurangan_terhadap_target | varian_setara |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HP Victus 15-fb3999AX - BK6E3PA /AMD Ryzen AI 7-350/16GB/1TB | Rp21.420.000 | 16 | 1.024 | 24.969 | 16.758 (dedicated) | 0.0395 | 1 | GPU -2% | 1 |
| 2 | HP HyperX OMEN 15-gb0777AX - D36Z1PA /AMD Ryzen 7-260/16GB/1 | Rp24.520.000 | 16 | 1.024 | 27.983 | 16.758 (dedicated) | 0.1334 | 1 | GPU -2% | 0 |
| 3 | Asus TUF A16 FA608UP-R757J6G-HM /AMD Ryzen 7-260/16GB/1TB SS | Rp29.490.000 | 16 | 1.024 | 27.983 | 19.118 (dedicated) | 0.1925 | 0 | - | 0 |
| 4 | HP Victus 16-s1777AX - A4UE6PA /AMD Ryzen 7-8845HS/16GB/1TB  | Rp25.120.000 | 16 | 1.024 | 28.260 | 19.462 (dedicated) | 0.2183 | 0 | - | 0 |
| 5 | Asus TUF F16 FX608JMR-I7N56J6G-HM /Intel Core i7-14650HX/16G | Rp25.245.000 | 16 | 1.024 | 33.365 | 16.758 (dedicated) | 0.3571 | 1 | GPU -2% | 1 |

## 7. Keterbatasan (wajib dibaca)

- Tidak ada ground truth, penilaian ahli, atau evaluasi pengguna: laporan ini **tidak** membuktikan akurasi rekomendasi. Jarak kecil hanya berarti dekat dengan target numerik.
- Default Gaming diubah dari dgpu_tinggi ke dgpu_sedang setelah audit menunjukkan target tinggi tidak tercapai oleh kandidat mana pun pada budget uji; pilihan ini tetap asumsi.
- Persentil CPU/GPU, level default per profil, rekomendasi RAM/storage, dan lambda adalah asumsi operasional yang perlu diuji.
- Office dan Programming memakai fitur yang sama dan hanya berbeda pada target; perbedaan itu belum tervalidasi.
- Skor G3D bersifat per nama GPU dan sering generik (mis. satu skor untuk semua 'Intel Graphics'), sehingga tidak membedakan iGPU dengan baik; karena itu GPU tidak dipakai pada Office/Programming secara default.
- Jenis GPU (dedicated/integrated) dan tingkat performa (G3D) adalah dua hal berbeda (contoh: Radeon 8060S adalah iGPU dengan G3D tinggi). Kelas `unknown` tidak ditebak.
- Laptop Apple tanpa skor CPU/GPU tidak dapat diranking dan dicatat di `knn_v8_unrankable_candidates.csv`. Kesebandingan PassMark CPU Apple dan x86 adalah asumsi. Skor CPU M1 dan M3 identik pada dataset; perlu dicek manual.
- Pengelompokan vektor identik menyembunyikan perbedaan yang tidak ada di fitur (layar, baterai, garansi, merek).
- Dataset adalah snapshot katalog satu toko; hasil tidak berlaku umum.
