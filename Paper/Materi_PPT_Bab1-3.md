# MATERI PPT — VERSI BAB 1–3 (Proposal / Progress)
**Judul paper:** Offline Signature Forgery Detection with EfficientNet-B0 and Batch-Hard Triplet Loss Fine-Tuning
**Saran jumlah slide:** 12 · Durasi ±10–12 menit

---

## SLIDE 1 — Judul
- Offline Signature Forgery Detection with EfficientNet-B0 and Batch-Hard Triplet Loss Fine-Tuning
- David Nehemia Sunoto · Febrianto · Frolentika · Jessica Davalynda · Windy Sulistiawati
- (logo kampus + nama jurusan)

> **Catatan presenter:** Perkenalkan tim singkat saja, langsung masuk latar belakang.

---

## SLIDE 2 — Latar Belakang
- Tanda tangan masih jadi alat autentikasi utama: perbankan, kontrak, dokumen resmi
- Verifikasi manual oleh manusia: **subjektif, lambat, tidak konsisten antar pemeriksa**
- Paling rawan terhadap *skilled forgery* — pemalsu yang **berlatih** meniru bentuk, kemiringan, dan ritme goresan
- Solusi: **Offline Signature Verification (OSV)** otomatis — memutuskan asli/palsu hanya dari citra hasil scan

> **Catatan presenter:** Tekankan bedanya "offline" (hanya gambar statis) vs "online" (ada data tekanan/waktu dari pen tablet). Kita di kasus yang lebih sulit: offline.

---

## SLIDE 3 — Perkembangan Metode (State of the Art)
- **Era fitur manual:** SIFT / HOG + SVM → mudah dijelaskan, tapi tidak generalisasi antar penulis
- **Era CNN + transfer learning:** jaringan pre-trained ImageNet di-fine-tune untuk tanda tangan (mis. EfficientNet)
- **Era metric learning:** Siamese & triplet loss → belajar *ruang embedding*, bukan kelas tetap → cocok untuk penulis yang tak pernah dilihat saat training
- **Frontier:** self-supervised Vision Transformer (DINOv2) — belum banyak dipakai untuk tanda tangan

> **Catatan presenter:** Satu kalimat kunci: "sistem modern tidak menghafal penulis, tapi belajar mengukur kemiripan."

---

## SLIDE 4 — Research Gap
1. **Dataset publik kecil** — CEDAR hanya 55 penulis; hasil "hampir sempurna" di satu dataset (EER < 0,06% di CEDAR) belum tentu berlaku lintas dataset
2. **Skilled forgery jauh lebih sulit** ditolak daripada random impostor — tapi banyak paper hanya melaporkan **satu angka akurasi gabungan**
3. Fine-tuning **seluruh jaringan** itu mahal, dan tidak jelas bagian mana yang menyumbang kenaikan performa

> **Catatan presenter:** Gap #2 adalah jualan utama kita — kita LAPORKAN TERPISAH random vs skilled. Sistem bisa terlihat bagus "rata-rata" tapi tetap jebol oleh pemalsu yang niat.

---

## SLIDE 5 — Tujuan Penelitian (3 Objektif)
1. Membangun model embedding **EfficientNet-B0 writer-independent**, dievaluasi lintas **CEDAR + GPDS + dataset lokal**
2. Merancang & menguji strategi fine-tuning **ringan**: backbone dibekukan, **hanya embedding head** dilatih ulang dengan **batch-hard triplet loss**
3. Mengukur performa **secara terpisah**: random-impostor vs skilled-forgery, pada penulis yang **tidak pernah dilihat saat training**

> **Catatan presenter:** Kata kunci yang harus keucap: writer-independent, backbone-frozen, dilaporkan terpisah.

---

## SLIDE 6 — Studi Literatur: EfficientNet
- Tan & Le (2019): **compound scaling** — depth, width, resolusi dinaikkan bersama lewat satu koefisien
- **EfficientNet-B0**: hasil neural architecture search; akurasi setara ResNet-50 / Inception-v3 dengan parameter **~10× lebih sedikit**
- Relevan untuk biometrik: data berlabel sedikit → backbone kecil = risiko overfitting lebih rendah
- Umum dipakai via transfer learning: ganti layer klasifikasi dengan *head* khusus tugas → pola yang kami ikuti

---

## SLIDE 7 — Studi Literatur: Triplet Loss & Metric Learning
- Klasifikasi biasa: gambar → salah satu kelas tetap. **Metric learning:** gambar → vektor; **jarak antar vektor = kemiripan identitas**
- **Triplet loss** (FaceNet, Schroff dkk. 2015): anchor + positif + negatif; negatif harus lebih jauh dari positif minimal sebesar *margin*
- Masalah: triplet acak cepat jadi "mudah" → gradien nyaris nol
- Solusi: **batch-hard mining** (Hermans dkk.) — per anchor, ambil positif TERSULIT & negatif TERSULIT dalam batch → dipakai langsung di penelitian ini
- Di domain tanda tangan: **SigNet** (Dey dkk.) membuktikan embedding hasil belajar > fitur manual

---

## SLIDE 8 — Related Works (3 Paper Pembanding)
| Paper | Metode | Hasil | Keterbatasan |
|---|---|---|---|
| Al-Azzani & Musleh (2024) | PSO untuk tuning hyperparameter CNN; 4 dataset | Akurasi 98,3% | Satu angka gabungan; random vs skilled tidak dipisah |
| Chokshi dkk. (2023) — SigScatNet | Siamese + Scattering wavelet | EER 0,0578% (CEDAR); 3,689% (SigComp Dutch) | Gap ~60× antar dataset → CEDAR mudah jenuh |
| Ozyurt dkk. (2024) | MobileNetV2 + seleksi fitur NCA | 91,3% → 97,7% (300 fitur NCA) | Hanya data genuine → lebih ke identifikasi, bukan deteksi forgery |

> **Catatan presenter:** Ketiganya bagus, tapi tak satu pun memisahkan random vs skilled — itu celah yang kita isi.

---

## SLIDE 9 — Dataset (Tabel I)
| Dataset | Penulis | Genuine | Forged | Peran |
|---|---|---|---|---|
| CEDAR | 55 | 1.320 | 1.320 | Fine-tuning (genuine) + tes forgery |
| GPDS (subset) | 150 | 1.200 | 2.100 | Fine-tuning (genuine) + tes forgery |
| Lokal (dikumpulkan sendiri) | 16 | 257 | 0 | Fine-tuning (genuine) saja |
| **Gabungan** | **221** | **2.052†** | 3.420* | **188 train / 33 held-out** |

- † Setelah cap **maks. 12 gambar genuine per penulis** (660 CEDAR + 1.200 GPDS + 192 lokal)
- \* Gambar forged **tidak pernah** dipakai training — hanya untuk pengujian
- Split di **level penulis** (bukan level gambar), seed tetap → reproducible
- Held-out: 18 GPDS + 12 CEDAR + 3 lokal = 33 penulis

> **Catatan presenter:** Kalau ditanya kenapa 2.052 ≠ jumlah kolom: karena cap 12/penulis (sudah dijelaskan di paper, catatan kaki †).

---

## SLIDE 10 — Metode: Pipeline (Fig. 1)
**[Tempel gambar flowchart.png di slide ini]**
- Preprocessing → ekstraksi fitur backbone (beku, di-cache) → fine-tuning head (triplet loss) → validasi tiap 150 step → evaluasi held-out
- Dua skenario uji: **random-impostor** & **skilled-forgery**

> **Catatan presenter:** Jalankan penjelasan mengikuti panah dari atas ke bawah, ±30 detik.

---

## SLIDE 11 — Metode: Preprocessing & Arsitektur
**Preprocessing:**
- Flatten transparansi **hanya jika benar-benar ada piksel transparan** (mencegah artefak "frame hitam" — bug nyata yang kami temukan saat pengembangan)
- Auto-crop ke bounding box tinta → kanvas persegi → resize **224×224** → normalisasi ImageNet

**Arsitektur:**
- Backbone: **EfficientNet-B0** (pre-trained triplet loss di GPDS) — **DIBEKUKAN**
- Head (yang dilatih): Linear 1280→512 → BN → ReLU → Dropout 0,3 → Linear 512→512 → BN → **L2-normalize**
- Output: embedding 512-dimensi di unit hypersphere; skor = **cosine similarity**

---

## SLIDE 12 — Metode: Setup Training & Rencana Evaluasi
**Training (CPU-only!):**
- Batch-hard triplet loss, margin 0,3 (jarak cosine)
- Batch: 12 penulis × 4 gambar · Adam lr 1×10⁻³, weight decay 1×10⁻⁴ · 1.500 step
- Fitur backbone (1.280-d) dihitung sekali & di-cache → training selesai **hitungan menit tanpa GPU**
- Checkpoint terbaik dipilih dari validasi AUC (tiap 150 step, di 33 penulis held-out)

**Rencana evaluasi:**
- Metrik: AUC, EER + Accuracy, Precision, Recall, F1 pada satu threshold operasional
- Dilaporkan **terpisah**: random-impostor vs skilled-forgery

> **Catatan presenter (closing Bab 1–3):** "Dengan desain ini, kontribusi head bisa diukur terisolasi, dan hasilnya akan kami laporkan jujur per jenis serangan." → jembatan ke hasil (kalau ditanya, angka ada di full paper).
