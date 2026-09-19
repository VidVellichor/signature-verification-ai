# MATERI PPT — VERSI FULL PAPER (Bab 1–5)
**Judul paper:** Offline Signature Forgery Detection with EfficientNet-B0 and Batch-Hard Triplet Loss Fine-Tuning
**Saran jumlah slide:** 16 · Durasi ±15–18 menit

---

## SLIDE 1 — Judul
- Offline Signature Forgery Detection with EfficientNet-B0 and Batch-Hard Triplet Loss Fine-Tuning
- David Nehemia Sunoto · Febrianto · Frolentika · Jessica Davalynda · Windy Sulistiawati
- (logo kampus + nama jurusan)

---

## SLIDE 2 — Latar Belakang
- Tanda tangan = autentikasi standar untuk transaksi bank, kontrak, dokumen resmi
- Verifikasi manual: **subjektif, tidak scalable, pemeriksa bisa saling tidak setuju**
- Titik terlemah: *skilled forgery* — pemalsu **berlatih** meniru bentuk, kemiringan, ritme goresan
- Kebutuhan: **Offline Signature Verification (OSV)** otomatis dari citra scan statis

---

## SLIDE 3 — Research Gap
1. Benchmark publik **kecil & mudah jenuh** — EER < 0,06% di CEDAR (55 penulis) tidak menjamin performa lintas dataset
2. Mayoritas paper melaporkan **satu akurasi gabungan** — menyembunyikan lemahnya sistem terhadap skilled forgery
3. Fine-tuning end-to-end **mahal** dan kontribusi tiap komponen tidak terukur

**→ Kontribusi kami: fine-tuning ringan (head-only) + pelaporan terpisah random vs skilled + protokol writer-disjoint yang reproducible**

---

## SLIDE 4 — Tujuan Penelitian
1. Model embedding **EfficientNet-B0 writer-independent** lintas CEDAR + GPDS + dataset lokal
2. Strategi fine-tuning **backbone-frozen** dengan **batch-hard triplet loss** → kontribusi head terukur terisolasi
3. Kuantifikasi terpisah: **random-impostor vs skilled-forgery** pada 33 penulis yang tak pernah dilihat model

---

## SLIDE 5 — Landasan: EfficientNet & Metric Learning
**EfficientNet-B0** (Tan & Le, 2019)
- Compound scaling (depth+width+resolusi bersamaan); akurasi setara ResNet-50 dengan parameter ~10× lebih sedikit
- Ideal saat data berlabel sedikit → risiko overfitting rendah

**Triplet Loss** (Schroff dkk., FaceNet)
- Belajar ruang embedding: jarak = identitas; anchor–positif–negatif + margin
- **Batch-hard mining** (Hermans dkk.): per anchor ambil positif & negatif TERSULIT dalam batch → gradien tetap informatif

---

## SLIDE 6 — Related Works
| Paper | Metode | Hasil | Keterbatasan |
|---|---|---|---|
| Al-Azzani & Musleh (2024) | PSO tuning CNN, 4 dataset | Akurasi 98,3% | Angka gabungan; tidak dipisah per jenis serangan |
| Chokshi dkk. (2023) SigScatNet | Siamese + Scattering wavelet | EER 0,0578% CEDAR; 3,689% SigComp | Gap ~60× antar dataset → CEDAR mudah jenuh |
| Ozyurt dkk. (2024) | MobileNetV2 + NCA feature selection | 91,3% → 97,7% | Data genuine saja → bukan deteksi forgery |

---

## SLIDE 7 — Dataset (Tabel I)
| Dataset | Penulis | Genuine | Forged | Peran |
|---|---|---|---|---|
| CEDAR | 55 | 1.320 | 1.320 | Fine-tuning + tes forgery |
| GPDS (subset) | 150 | 1.200 | 2.100 | Fine-tuning + tes forgery |
| Lokal | 16 | 257 | 0 | Fine-tuning saja |
| **Gabungan** | **221** | **2.052†** | 3.420* | **188 train / 33 held-out** |

- † cap maks. 12 genuine/penulis (660+1.200+192) · \* forged hanya untuk pengujian
- Split **level penulis**, seed tetap → tidak ada penulis yang bocor ke dua partisi
- Held-out: 18 GPDS + 12 CEDAR + 3 lokal

---

## SLIDE 8 — Metode: Pipeline (Fig. 1)
**[Tempel gambar flowchart.png di slide ini]**
- Preprocessing → fitur backbone (beku, cache) → fine-tuning head (batch-hard triplet) → validasi per 150 step → evaluasi held-out 2 skenario

---

## SLIDE 9 — Preprocessing & Arsitektur
**Preprocessing:** flatten alpha hanya jika benar ada transparansi (mencegah artefak frame hitam) → auto-crop tinta → 224×224 → normalisasi ImageNet

**Arsitektur:**
- Backbone EfficientNet-B0 (pre-trained triplet GPDS) — **dibekukan**
- Head: Linear 1280→512 → BN → ReLU → Dropout 0,3 → Linear 512→512 → BN → **L2-norm** (embedding 512-d)
- Skor pasangan = **cosine similarity**

---

## SLIDE 10 — Setup Training
- Batch-hard triplet loss, margin 0,3 (cosine) · batch 12 penulis × 4 gambar
- Adam lr 1×10⁻³ · weight decay 1×10⁻⁴ · 1.500 step · validasi AUC tiap 150 step
- Fitur backbone di-cache → **training tuntas dalam hitungan menit di CPU** (tanpa GPU!)
- Head < 1 juta parameter yang dilatih

---

## SLIDE 11 — Setup Evaluasi
- **495 pasangan genuine** (33 penulis held-out); forgery tersedia untuk 30/33 penulis
- Dua skenario negatif: **Random** (genuine penulis lain) vs **Skilled** (forgery penulis yang sama)
- AUC & EER = bebas threshold · Acc/Prec/Rec/F1 = pada **SATU threshold operasional per model** (titik EER gabungan) → meniru deployment nyata: sistem tak tahu jenis serangannya
- Pool negatif diseimbangkan ke jumlah pasangan positif → metrik tak terdistorsi rasio kelas

---

## SLIDE 12 — HASIL UTAMA (Tabel II) ⭐
| Model | Skenario | AUC | EER | Acc | Prec | Rec | F1 |
|---|---|---|---|---|---|---|---|
| Baseline | Random | 0,953 | 10,71 | 87,07 | 92,38 | 80,81 | 86,21 |
| Baseline | Skilled | 0,817 | 25,25 | 74,55 | 71,81 | 80,81 | 76,05 |
| **Proposed** | **Random** | **0,979** | **7,07** | **90,20** | 95,02 | 84,85 | 89,65 |
| **Proposed** | **Skilled** | **0,871** | **20,20** | **79,49** | 76,64 | 84,85 | 80,54 |

- Fine-tuning head memperbaiki **SEMUA metrik di KEDUA skenario**
- Biaya: head <1 jt parameter, **hitungan menit di CPU** — tanpa retraining end-to-end

> **Catatan presenter:** Ini slide terpenting. Highlight dua baris Proposed. "AUC random 0,953→0,979, skilled 0,817→0,871."

---

## SLIDE 13 — Error Analysis (Tabel III) ⭐
| Skenario | Genuine diterima (TP) | Genuine ditolak (FN) | Forgery LOLOS (FP) | Forgery ditolak (TN) |
|---|---|---|---|---|
| Random | 420 | 75 | **22** | 473 |
| Skilled | 420 | 75 | **128** | 367 |

- Sisi genuine identik (threshold sama) → **seluruh perbedaan ada di sisi impostor**
- Forgery skilled lolos **128 vs 22** (≈ **6× lebih sering**) → precision turun 95,02% → 76,64%
- EER skilled (20,20%) ≈ **3×** EER random (7,07%)

> **Catatan presenter:** Ini bukti kuantitatif kenapa laporan harus dipisah — kalau digabung, angka 6× ini tak terlihat.

---

## SLIDE 14 — Diskusi: Kenapa Skilled Forgery Sulit?
- **Random impostor** beda di bentuk global, kemiringan, layout goresan → mudah ditangkap embedding konvolusional
- **Skilled forger** justru berlatih menyamakan ciri-ciri global itu
- Pembeda yang tersisa = **dinamika goresan halus** (pen dynamics) — tidak sepenuhnya ter-encode oleh backbone yang dioptimalkan untuk klasifikasi citra alami
- → Skilled forgery tetap jadi **error mode dominan** setelah fine-tuning

---

## SLIDE 15 — Kesimpulan
- Fine-tuning **head-only** (backbone beku) + batch-hard triplet loss di korpus 221 penulis **terbukti efektif**:
  - Random: AUC 0,953 → **0,979** (EER 10,71% → **7,07%**) — akurasi **90,20%**
  - Skilled: AUC 0,817 → **0,871** (EER 25,25% → **20,20%**) — akurasi **79,49%**
- Kontribusi non-angka: **protokol pelaporan** — random vs skilled dipisah, partisi writer-disjoint, semua angka dari **satu skrip evaluasi reproducible**

---

## SLIDE 16 — Limitasi & Future Work
**Limitasi:**
1. Hanya head yang dilatih (kendala CPU-only); penulis evaluasi juga dipakai untuk pemilihan checkpoint
2. Penulis held-out masih dari keluarga dataset yang sama

**Future work:**
- Fine-tuning end-to-end di GPU + pool skilled forgery lebih besar/beragam
- Partisi tes yang sepenuhnya independen
- Menangkap dinamika goresan halus: attention, representasi graph, atau ViT self-supervised (**DINOv2**)

> **Catatan presenter (closing):** "Skilled forgery masih terbuka — dan justru itu arah riset paling menarik berikutnya. Terima kasih."

---

## TIPS UMUM
- Slide 12 & 13 = inti presentasi; alokasikan waktu terbanyak di sana
- Semua angka di materi ini **identik dengan paper** — aman kalau ditanya penguji silang-cek
- Antisipasi pertanyaan: (1) kenapa 2.052 ≠ jumlah kolom → cap 12/penulis; (2) kenapa CPU-only → fitur backbone di-cache; (3) kenapa threshold tunggal → deployment nyata tak tahu jenis serangan
