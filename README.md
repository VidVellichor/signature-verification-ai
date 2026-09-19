# ✍️ Signature Verification AI — Deteksi Tanda Tangan Asli vs Palsu

> **Sistem verifikasi tanda tangan berbasis deep learning.** Ubah dua gambar tanda tangan menjadi *embedding* dengan EfficientNet-B0, lalu tentukan **COCOK (asli)** atau **BEDA (palsu)** dari skor kemiripan cosine.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B0-blueviolet)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi&logoColor=white)
![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)
![Val AUC](https://img.shields.io/badge/val_AUC-~0.975-success)
![Status](https://img.shields.io/badge/status-riset%20%2B%20demo%20web-brightgreen)

---

## 📖 Deskripsi

Project ini adalah **offline signature verification** — memverifikasi apakah sebuah tanda tangan **asli (genuine)** atau **palsu (forgery)** menggunakan deep metric learning. Alih-alih mengklasifikasi langsung, model mempelajari **ruang embedding**: tanda tangan dari orang yang sama dipetakan berdekatan, tanda tangan orang berbeda / palsu dipetakan berjauhan. Keputusan diambil dari **cosine similarity** antara embedding gambar uji dan tanda tangan acuan.

Sistem terdiri dari tiga bagian:
1. **Riset & training** — notebook untuk melatih dan mengevaluasi model (EfficientNet-B0 + varian GNN).
2. **Backend inferensi** — API FastAPI yang memuat checkpoint `.pth` dan melayani endpoint verifikasi.
3. **Front-end web** — halaman untuk upload & bandingkan tanda tangan, plus enrollment per-pengguna.

---

## 🧠 Model & Metode

- **Arsitektur:** `EfficientNetEmbedder` — backbone **EfficientNet-B0** (torchvision) + projection head (`Linear 1280→512 → BatchNorm → ReLU → Dropout → Linear 512→512 → BatchNorm`), output di-**L2-normalize** menjadi embedding 512 dimensi.
- **Training:** **batch-hard triplet loss**, dengan strategi **head fine-tuning** (backbone dibekukan, hanya projection head dilatih ulang) sebagai model **Proposed**.
- **Dataset:** gabungan **CEDAR + GPDS + dataset lokal** (± **221 penulis**).
- **Hasil validasi:** best **val AUC ≈ 0.975** (notebook head-ft); checkpoint head-ft dilaporkan AUC ~0.985 dengan separasi antar-orang jauh lebih tajam dibanding varian triplet-v4 lama.
- **Preprocessing:** load ke background putih → auto-crop area tinta + padding → resize 224×224 → normalisasi ImageNet (mean/std standar).
- **Threshold keputusan:** cosine similarity, default **0.50** (dikalibrasi dari hold-out GPDS: asli-orang-sama rata-rata ~0.85, beda-orang ~0.04). Bisa diatur lewat slider di UI.
- **Varian eksperimen:** notebook **GNN graph verification** sebagai pendekatan pembanding.

> Angka di atas diambil langsung dari kode (`Web/server.py`) dan output notebook training — bukan estimasi.

---

## ✨ Fitur Utama

- 🔍 **Bandingkan 2 tanda tangan (1-vs-1)** — upload dua gambar, dapatkan skor kemiripan cosine + verdict **COCOK / TIDAK COCOK** (endpoint `/api/compare`).
- 👤 **Enrollment per-pengguna** — daftarkan 1 tanda tangan asli per label (Nama/NIM/User ID). Sistem otomatis membuat **16 augmentasi ringan** (rotasi ±5°, skala ±5%, translasi ±3%, brightness/contrast ±8%, noise halus) dan menyimpan **embedding terkomputasi** sebagai BLOB di database.
- ✅ **Verifikasi terhadap acuan** — cocokkan tanda tangan uji dengan tanda tangan asli pengguna terdaftar (endpoint `/api/verify`), skor = mean cosine terhadap acuan.
- 🖊️ **Mode gambar kanvas (Voila)** — opsi menggambar tanda tangan langsung di kanvas untuk dibandingkan (layanan Voila terpisah).
- 🎚️ **Threshold interaktif** — slider di UI untuk menyetel titik keputusan sesuai kebutuhan.
- 🗄️ **Riwayat & manajemen** — capture history, daftar user terdaftar, thumbnail acuan, hapus enrollment (SQLite `detector.db`).
- 🔒 **Backend yang diamankan** — pembatasan ukuran body (anti-DoS), guard decompression-bomb, allow-list path & file front-end (mencegah kebocoran `detector.db` / source), validasi gambar (tolak kanvas kosong).

---

## 🧱 Tech Stack

| Layer | Teknologi |
|-------|-----------|
| **Deep learning** | PyTorch 2.x, torchvision (EfficientNet-B0) |
| **Metode** | Triplet loss (batch-hard), metric learning, cosine similarity |
| **Backend** | FastAPI + Uvicorn |
| **Image processing** | Pillow, NumPy |
| **Database** | SQLite (`users`, `refs`, `captures` + embedding BLOB) |
| **Frontend** | HTML/CSS/JS native, Voila (notebook interaktif) |
| **Deploy** | Render (`render.yaml`: service API + service Voila) |
| **Riset** | Jupyter Notebook, evaluasi model, dan visualisasi metrik |

---

## 🚀 Cara Install & Menjalankan

### Menjalankan Web App (backend + front-end)
```bash
# 1. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependensi (PyTorch versi CPU)
pip install -r Web/requirements.txt

# 3. Pastikan checkpoint model tersedia di root project atau folder Model/:
#    EfficientNet-B0-head-ft.pth  (diutamakan)  atau  EfficientNet-B0-triplet-v4.pth

# 4. Jalankan server
cd Web
uvicorn server:app --host 0.0.0.0 --port 3000
```
Buka `http://localhost:3000` untuk hub utama, atau `http://localhost:3000/compare.html` untuk bandingkan 2 foto.

### Deploy ke Render
`render.yaml` sudah mendefinisikan dua service: **ttd-app** (FastAPI) dan **ttd-voila** (Voila). Set env var `VOILA_URL` di dashboard Render setelah service Voila punya URL.

### Reproduksi Training
Buka `CodeTrain/signature_efficientnet_head_ft_TRAINING.ipynb` — melatih ulang projection head di atas backbone beku dengan batch-hard triplet loss (CEDAR + GPDS + data lokal) dan menyimpan checkpoint hasilnya.

---

## 📁 Struktur Folder

```
ProjectAI/
├── Web/                              # aplikasi web inferensi
│   ├── server.py                     # backend FastAPI (enroll / verify / compare)
│   ├── index.html, compare.html      # front-end hub & pembanding
│   ├── app.js, styles.css
│   ├── requirements.txt              # torch, torchvision, fastapi, uvicorn, pillow, numpy
│   └── data/                         # detector.db (SQLite), captures/, enrollments/
├── CodeTrain/                        # notebook training
│   ├── signature_efficientnet_head_ft_TRAINING.ipynb   # model Proposed (head fine-tune)
│   ├── signature_efficientnet_triplet_v4_TRAINING.ipynb
│   ├── signature_gnn_graph_verification.ipynb          # varian GNN
│   └── EfficientNet-B0-head-ft-reproduced.pth
├── DataSets/                         # CEDAR & dataset lain (genuine / forgery)
├── paper_figs/                       # visualisasi hasil evaluasi model
├── cek_model.ipynb                   # cek/uji model & threshold
├── compare_app.ipynb                 # app kanvas (dijalankan via Voila)
├── EfficientNet-B0-head-ft.pth       # checkpoint utama
├── EfficientNet-B0-triplet-v4.pth    # checkpoint fallback
└── render.yaml                       # konfigurasi deploy Render
```

---

## 📌 Status Project

🟢 **Riset selesai + demo web berjalan.** Model sudah terlatih dan tervalidasi (val AUC ~0.975), backend & front-end sudah berfungsi, siap di-deploy ke Render. Pengembangan lanjut: kalibrasi threshold lintas-dataset, evaluasi robustness terhadap skilled forgery, dan penyempurnaan varian GNN.

---

<sub>Dibuat oleh **David** — mahasiswa IT, penerima beasiswa **PPTI BCA**. Fokus di deep learning terapan & AI engineering. Terbuka untuk kolaborasi riset/produk. 🤝</sub>
