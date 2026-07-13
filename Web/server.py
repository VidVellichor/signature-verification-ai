import os
import io
import json
import base64
import time
import random
import sqlite3
import shutil
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
from PIL import Image

# Guard against decompression-bomb images: cap total pixels a single image may hold.
Image.MAX_IMAGE_PIXELS = 8_000_000  # ~8 MP; normal signature scans are far smaller

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pre-load Models Setup


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(DATA_DIR, "captures")
MODEL_DIR = os.path.abspath(os.path.join(ROOT, "..", "Model"))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

# Checkpoint aktif. Utamakan model hasil fine-tune head (AUC ~0.985, separasi jauh
# lebih tajam: beda-orang ~0.0); fallback ke triplet-v4 lama bila belum ada.
EFFNET_CKPT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "EfficientNet-B0-head-ft.pth"),
    os.path.join(MODEL_DIR, "EfficientNet-B0-head-ft.pth"),
    os.path.join(PROJECT_ROOT, "EfficientNet-B0-triplet-v4.pth"),
    os.path.join(MODEL_DIR, "EfficientNet-B0-triplet-v4.pth"),
]

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(IMG_DIR, "whiteboard"), exist_ok=True)
os.makedirs(os.path.join(IMG_DIR, "test"), exist_ok=True)

# Per-user enrollment references live here: data/enrollments/<label>/{original,aug_XX}.png
ENROLL_DIR = os.path.join(DATA_DIR, "enrollments")
os.makedirs(ENROLL_DIR, exist_ok=True)

# ------------------------------------------------------------------
#  Filesystem access confinement
#  Endpoints that accept a caller-supplied path may ONLY reach files
#  under these base directories — never arbitrary locations on disk.
# ------------------------------------------------------------------
DATASETS_DIR = os.path.abspath(os.path.join(ROOT, "..", "DataSets"))
ALLOWED_REF_BASES = [os.path.abspath(DATA_DIR), DATASETS_DIR]


def _is_within_allowed(p: str) -> bool:
    """True iff `p` resolves to a path inside one of the allowlisted base dirs."""
    ap = os.path.abspath(p)
    for base in ALLOWED_REF_BASES:
        try:
            if os.path.commonpath([base, ap]) == base:
                return True
        except ValueError:
            continue  # different drive on Windows -> not contained
    return False





# ============================================================
#  Database Setup
# ============================================================
db_path = os.path.join(DATA_DIR, "detector.db")
conn = sqlite3.connect(db_path, check_same_thread=False)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("""
  CREATE TABLE IF NOT EXISTS captures (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model        TEXT    NOT NULL,
    label        TEXT    NOT NULL DEFAULT 'test',
    verdict      TEXT    NOT NULL DEFAULT 'unscored',
    confidence   REAL    NOT NULL DEFAULT 0,
    models_json  TEXT    NOT NULL DEFAULT '{}',
    strokes_json TEXT    NOT NULL,
    image_path   TEXT    NOT NULL,
    stroke_count INTEGER NOT NULL,
    point_count  INTEGER NOT NULL,
    created_at   TEXT    NOT NULL
  );
""")

# Per-user enrollment: one row per enrolled label (Nama / NIM / User ID).
cursor.execute("""
  CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    label      TEXT    UNIQUE NOT NULL,
    created_at TEXT    NOT NULL,
    num_refs   INTEGER NOT NULL DEFAULT 0
  );
""")

# Reference signatures owned by a user: the original + its light augmentations.
# `embedding` is a raw float32 BLOB (256-dim, already L2-normalized) so verification
# never has to re-run the model over the reference images (precomputed embeddings).
cursor.execute("""
  CREATE TABLE IF NOT EXISTS refs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    image_path TEXT    NOT NULL,
    embedding  BLOB    NOT NULL,
    created_at TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
  );
""")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_refs_user ON refs(user_id);")
# Tag verification logs with which enrolled user they were checked against.
try:
    cursor.execute("ALTER TABLE captures ADD COLUMN user_label TEXT")
except sqlite3.OperationalError:
    pass  # column already exists
conn.commit()


# Model Architectures


class EfficientNetEmbedder(nn.Module):
    # Arsitektur ini HARUS sama persis dengan checkpoint triplet-v4 (embedding 512,
    # head: Linear->BN->ReLU->Dropout->Linear->BN) — identik dengan cek_model.ipynb.
    def __init__(self, embedding_dim=512, dropout=0.3):
        super().__init__()
        import torchvision.models as models
        effnet = models.efficientnet_b0(weights=None)
        self.backbone = nn.Sequential(effnet.features)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


# ============================================================
#  Image Preprocessing Helpers
# ============================================================
def _load_white_bg(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        alpha = img.getchannel("A")
        # Only flatten via the alpha channel when the image ACTUALLY has
        # transparent regions. A fully-opaque RGBA export (alpha=255 everywhere,
        # as HTML-canvas toDataURL produces) already holds ink-on-white in its
        # RGB channels — treating every pixel as "ink" would paint the whole
        # frame solid black and collapse every capture to the same image.
        if alpha.getextrema()[0] < 255:
            black_img = Image.new("RGBA", img.size, (0, 0, 0, 255))
            transparent_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
            img = Image.composite(black_img, transparent_img, alpha)
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def autocrop_and_pad(img, pad_frac=0.08, ink_thresh=245, size=224):
    gray = np.array(img.convert("L"))
    mask = gray < ink_thresh
    if mask.sum() == 0:
        cropped = img.convert("L")
    else:
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        h, w = gray.shape
        pad_y = max(2, int((y1 - y0 + 1) * pad_frac))
        pad_x = max(2, int((x1 - x0 + 1) * pad_frac))
        y0, y1 = max(0, y0 - pad_y), min(h - 1, y1 + pad_y)
        x0, x1 = max(0, x0 - pad_x), min(w - 1, x1 + pad_x)
        cropped = img.convert("L").crop((x0, y0, x1 + 1, y1 + 1))
    cw, ch = cropped.size
    side = max(cw, ch)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(cropped, ((side - cw) // 2, (side - ch) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


def preprocess_cnn_image(path, img_size=224):
    img = _load_white_bg(path)
    img = autocrop_and_pad(img, size=img_size)
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


# ============================================================
#  Pre-load Models on Startup
# ============================================================
device = torch.device("cpu")
models_cache = {}

# Cosine-similarity operating point used by BOTH /api/analyze and /api/verify
# (verdict genuine if mean similarity to the references >= threshold).
# Untuk model head-ft: verify vs TTD asli (1-vs-1). Kalibrasi GPDS hold-out:
# asli-orang-sama mean ~0.85, beda-orang mean ~0.04 -> 0.50 = titik terbaik
# (asli lolos ~98%, beda ditolak ~98%, error ~2%). Bisa diatur lewat slider UI.
COSINE_THRESHOLD = 0.50
EMB_DIM = 512   # dimensi embedding model aktif (diperbarui saat load_models)

def load_models():
    global EMB_DIM
    # EfficientNet Model Loading — cari checkpoint di kandidat path.
    effnet_path = next((p for p in EFFNET_CKPT_CANDIDATES if os.path.exists(p)), None)
    if effnet_path:
        try:
            ckpt = torch.load(effnet_path, map_location=device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            emb_dim = int(ckpt.get("embedding_dim", 512)) if isinstance(ckpt, dict) else 512
            EMB_DIM = emb_dim

            effnet = EfficientNetEmbedder(embedding_dim=emb_dim, dropout=0.3).to(device)
            effnet.load_state_dict(sd, strict=True)
            effnet.eval()
            models_cache["efficientnet-b0"] = effnet
            print(f"[Startup] Loaded EfficientNet (triplet-v4) from {effnet_path}. emb_dim={emb_dim}, cosine threshold default={COSINE_THRESHOLD}")
        except Exception as e:
            print(f"Error loading EfficientNet model: {e}")
    else:
        print(f"[Startup] EfficientNet model not loaded. Exist? {os.path.exists(effnet_path)}")

load_models()


# ============================================================
#  FastAPI Application Setup
# ============================================================
app = FastAPI(title="Fake Handwriting Detector Backend")

# Reject oversized request bodies early (base64 PNG DoS / disk-fill / RAM-bomb).
MAX_BODY_BYTES = 8 * 1024 * 1024  # 8 MB


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_BODY_BYTES:
                return JSONResponse(status_code=413,
                                    content={"detail": "Payload too large"})
        except ValueError:
            return JSONResponse(status_code=400,
                                content={"detail": "Invalid Content-Length"})
    return await call_next(request)


class AnalyzePayload(BaseModel):
    model: str
    label: str = "test"
    ref_dir: str = "custom"
    ref_filter: str = ""
    ref_threshold: Optional[float] = None
    image: str
    strokes: List[Any]
    meta: Optional[Dict[str, Any]] = None


class EnrollPayload(BaseModel):
    label: str
    image: str
    replace: bool = False
    strokes: Optional[List[Any]] = None
    meta: Optional[Dict[str, Any]] = None


class VerifyPayload(BaseModel):
    label: str
    image: str
    threshold: Optional[float] = None
    strokes: Optional[List[Any]] = None
    meta: Optional[Dict[str, Any]] = None


class ComparePayload(BaseModel):
    image1: str
    image2: str
    threshold: Optional[float] = None


# --- In-Memory Inference execution ---
@torch.no_grad()
def embed_cnn(model, path):
    tensor = preprocess_cnn_image(path).unsqueeze(0).to(device)
    return model(tensor)


# ============================================================
#  Enrollment: light augmentation + precomputed embeddings
# ============================================================
# Deterministic, GENTLE augmentations only. These preserve signature identity:
# small rotate / scale / translate / brightness / contrast + light gaussian noise.
# NO horizontal/vertical flip, NO perspective warp, NO extreme distortion.
AUG_COUNT = 16


def _augment_image(pil_img, seed):
    """Return one gently-augmented RGB copy of `pil_img` (on a white canvas)."""
    rs = np.random.RandomState(seed)
    img = pil_img.convert("RGB")
    w, h = img.size

    angle = float(rs.uniform(-5.0, 5.0))          # small rotation ±5°
    scale = float(rs.uniform(0.95, 1.05))         # small scale ±5%
    tx = float(rs.uniform(-0.03, 0.03)) * w       # small translation ±3%
    ty = float(rs.uniform(-0.03, 0.03)) * h

    # Rotate about center on a white background (expand=False keeps size).
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
    # Affine scale + translation (PIL affine matrix maps OUTPUT->INPUT coords).
    cx, cy = w / 2.0, h / 2.0
    a = 1.0 / scale
    e = 1.0 / scale
    c = cx - a * cx - tx
    f = cy - e * cy - ty
    img = img.transform((w, h), Image.AFFINE, (a, 0, c, 0, e, f),
                        resample=Image.BILINEAR, fillcolor=(255, 255, 255))

    # Brightness / contrast ±8%.
    from PIL import ImageEnhance
    img = ImageEnhance.Brightness(img).enhance(float(rs.uniform(0.92, 1.08)))
    img = ImageEnhance.Contrast(img).enhance(float(rs.uniform(0.92, 1.08)))

    # Light gaussian noise (std ~4/255).
    arr = np.array(img, dtype=np.float32)
    arr += rs.normal(0.0, 4.0, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def generate_augmentations(src_path, out_dir, count=AUG_COUNT):
    """Create `count` augmented PNGs from the original signature. Returns list of paths."""
    base = _load_white_bg(src_path)
    paths = []
    for i in range(count):
        aug = _augment_image(base, seed=1000 + i)
        p = os.path.join(out_dir, f"aug_{i + 1:02d}.png")
        aug.save(p)
        paths.append(p)
    return paths


@torch.no_grad()
def embed_paths(model, paths, batch_size=8):
    """Embed a list of image paths into L2-normalized vectors (batched). -> np.ndarray [N, D]."""
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        tensors = torch.stack([preprocess_cnn_image(p) for p in chunk]).to(device)
        emb = model(tensors)  # already F.normalize'd inside the model
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, EMB_DIM), dtype=np.float32)


def emb_to_blob(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_emb(blob):
    return np.frombuffer(blob, dtype=np.float32)


def run_inference_in_memory(model_id: str, test_path: str, ref_paths: List[str], custom_threshold: Optional[float] = None):
    model = models_cache.get(model_id)
    if not model:
        return {"ok": False, "error": "model_inactive"}

    # Decision metric is IDENTICAL to /api/verify: mean cosine similarity over all
    # references against COSINE_THRESHOLD. Mean-sim is far harder to fool than
    # min-distance (one close reference can't push a forgery over the line).
    threshold = custom_threshold if custom_threshold is not None else COSINE_THRESHOLD
    try:
        test_emb = embed_cnn(model, test_path).cpu().numpy()[0]

        sims = []
        for rp in ref_paths:
            try:
                ref_emb = embed_cnn(model, rp).cpu().numpy()[0]
                # Embeddings are L2-normalized -> cosine similarity == dot product.
                sims.append(float(np.dot(ref_emb, test_emb)))
            except Exception as e:
                print(f"Warning: failed to process ref {rp}: {e}")
                continue

        if not sims:
            return {"ok": False, "error": "embed_failed"}

        max_sim = float(max(sims))
        mean_sim = float(np.mean(sims))
        score = mean_sim
        verdict = "genuine" if score >= threshold else "forged"
        return {
            "ok": True,
            "verdict": verdict,
            "score": round(score, 4),
            "max_similarity": round(max_sim, 4),
            "mean_similarity": round(mean_sim, 4),
            "threshold": round(threshold, 4),
            "num_refs": len(sims)
        }
    except Exception as e:
        return {"ok": False, "error": "inference_failed", "detail": str(e)}


def list_genuine_refs(limit=5):
    genuine_dir = os.path.join(IMG_DIR, "whiteboard")
    try:
        files = [f for f in os.listdir(genuine_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(genuine_dir, x)), reverse=True)
        return [os.path.join(genuine_dir, f) for f in files[:limit]]
    except Exception:
        return []


# ============================================================
#  API Endpoints
# ============================================================

@app.post("/api/analyze/{model_id}")
async def analyze(model_id: str, payload: AnalyzePayload):
    if model_id != "efficientnet-b0":
        raise HTTPException(status_code=400, detail="unknown model")

    if not payload.strokes:
        raise HTTPException(status_code=400, detail="no stroke data")

    label = payload.label if payload.label in ("genuine", "test") else "test"
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    # Clear previous genuine references for whiteboard if enrolling a new genuine one
    if label == "genuine":
        whiteboard_dir = os.path.join(IMG_DIR, "whiteboard")
        if os.path.exists(whiteboard_dir):
            for f in os.listdir(whiteboard_dir):
                f_path = os.path.join(whiteboard_dir, f)
                try:
                    if os.path.isfile(f_path):
                        os.remove(f_path)
                except Exception as e:
                    print(f"Error clearing previous whiteboard file: {e}")
        try:
            cursor.execute("DELETE FROM captures WHERE label = 'genuine'")
            conn.commit()
        except Exception as e:
            print(f"Error clearing previous database entries: {e}")

    # Save base64 image to PNG
    image_path = ""
    abs_image_path = ""
    try:
        b64_data = payload.image
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
        
        if b64_data:
            file_name = f"sig_{int(time.time()*1000)}_{random.randint(100000, 999999)}.png"
            folder_name = "whiteboard" if label == "genuine" else "test"
            abs_image_path = os.path.join(IMG_DIR, folder_name, file_name)
            with open(abs_image_path, "wb") as fh:
                fh.write(base64.b64decode(b64_data))
            image_path = os.path.join("data", "captures", folder_name, file_name)
    except Exception as e:
        print(f"Could not save image: {e}")

    if not abs_image_path:
        raise HTTPException(status_code=400, detail="failed to save signature image")

    # Insert initial unscored record in DB
    cursor.execute("""
      INSERT INTO captures (model, label, verdict, confidence, models_json, strokes_json, image_path, stroke_count, point_count, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_id,
        label,
        "unscored",
        0.0,
        "{}",
        json.dumps(payload.strokes),
        image_path,
        payload.meta.get("strokeCount", 0) if payload.meta else 0,
        len(payload.strokes),
        created_at
    ))
    conn.commit()
    capture_id = cursor.lastrowid
    image_url = f"/api/captures/{capture_id}/image"

    base_response = {
        "id": capture_id,
        "created_at": created_at,
        "saved": True,
        "label": label,
        "image_url": image_url
    }

    has_effnet = "efficientnet-b0" in models_cache
    if not has_effnet:
        return {**base_response, "status": "model_inactive", "message": "Model EfficientNet-B0 belum aktif (bobot tidak ditemukan)."}

    # If enrolling, we just save and return
    if label == "genuine":
        return {**base_response, "status": "reference_saved", "message": "TTD asli disimpan sebagai acuan."}

    # Verify against reference files (either custom path or default captures/genuine)
    if payload.ref_dir and payload.ref_dir != "custom":
        ref_dir = os.path.abspath(payload.ref_dir.strip())
        if not _is_within_allowed(ref_dir):
            raise HTTPException(status_code=403, detail="Path not allowed")
        try:
            files = [f for f in os.listdir(ref_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            if payload.ref_filter:
                files = [f for f in files if f.startswith(payload.ref_filter)]
            ref_paths = [os.path.join(ref_dir, f) for f in files[:5]]
        except Exception:
            ref_paths = []
    else:
        ref_paths = list_genuine_refs(5)

    if not ref_paths:
        return {**base_response, "status": "no_reference", "message": "Belum ada TTD asli (acuan). Simpan TTD asli dulu."}

    # Run in-memory inference
    res = run_inference_in_memory("efficientnet-b0", abs_image_path, ref_paths, custom_threshold=payload.ref_threshold)
    if not res.get("ok"):
        return {
            **base_response,
            "status": "infer_error",
            "error": "infer_error",
            "message": f"Gagal menjalankan model analisis: {res.get('error')}"
        }

    score = res["score"]
    verdict = res["verdict"]
    confidence = score if verdict == "genuine" else 1.0 - score
    models_scores = {"efficientnet-b0": score}

    # Update database record with prediction scores
    cursor.execute("""
      UPDATE captures SET verdict = ?, confidence = ?, models_json = ?
      WHERE id = ?
    """, (verdict, confidence, json.dumps(models_scores), capture_id))
    conn.commit()

    return {
        **base_response,
        "status": "scored",
        "verdict": verdict,
        "confidence": confidence,
        "models": models_scores,
        "score": res["score"],
        "max_similarity": res["max_similarity"],
        "mean_similarity": res["mean_similarity"],
        "threshold": res["threshold"],
        "num_refs": res["num_refs"]
    }


def _safe_label(label: str) -> str:
    """Filesystem-safe folder name for a user label (keeps it human-readable)."""
    s = "".join(c if (c.isalnum() or c in "-_. ") else "_" for c in (label or "").strip())
    s = s.strip().replace(" ", "_")
    return s[:64]


def _save_b64_png(b64_data: str, dest_path: str):
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    if not b64_data:
        raise ValueError("empty image")
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        raise ValueError("invalid base64 image")
    if not raw:
        raise ValueError("empty image")

    # Verify the bytes are a genuine, decodable image AND that they contain
    # actual ink — a blank canvas or a single dot must not enroll/verify.
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()                       # structural integrity
        with Image.open(io.BytesIO(raw)) as img:
            gray = np.asarray(img.convert("L"))
    except Exception:
        raise ValueError("not a valid image")

    ink = int(np.count_nonzero(gray < 200))      # dark pixels = ink
    if ink < 40:
        raise ValueError("signature too small or blank")

    with open(dest_path, "wb") as fh:
        fh.write(raw)


# ============================================================
#  Enrollment  —  register 1 genuine signature per unique label
# ============================================================
@app.post("/api/enroll")
async def enroll(payload: EnrollPayload):
    if "efficientnet-b0" not in models_cache:
        raise HTTPException(status_code=503, detail="Model EfficientNet-B0 belum aktif.")
    model = models_cache["efficientnet-b0"]

    label = (payload.label or "").strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label (Nama/NIM/User ID) wajib diisi.")

    cursor.execute("SELECT id FROM users WHERE label = ?", (label,))
    existing = cursor.fetchone()
    if existing and not payload.replace:
        raise HTTPException(status_code=409, detail=f"Label '{label}' sudah terdaftar. Gunakan replace untuk menimpa.")

    folder = _safe_label(label)
    user_dir = os.path.join(ENROLL_DIR, folder)

    # If replacing, wipe the old references (files + rows) first.
    if existing:
        cursor.execute("DELETE FROM refs WHERE user_id = ?", (existing["id"],))
        cursor.execute("DELETE FROM users WHERE id = ?", (existing["id"],))
        conn.commit()
        if os.path.isdir(user_dir):
            shutil.rmtree(user_dir, ignore_errors=True)

    os.makedirs(user_dir, exist_ok=True)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    # 1) Save the original.
    original_path = os.path.join(user_dir, "original.png")
    try:
        _save_b64_png(payload.image, original_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal menyimpan gambar: {e}")

    # 2) Generate light augmentations (16) on disk.
    aug_paths = generate_augmentations(original_path, user_dir, count=AUG_COUNT)
    all_paths = [original_path] + aug_paths

    # 3) Precompute embeddings for original + all augmentations (batched).
    embs = embed_paths(model, all_paths)

    # 4) Persist user + refs (embeddings as BLOB).
    cursor.execute("INSERT INTO users (label, created_at, num_refs) VALUES (?, ?, ?)",
                   (label, created_at, len(all_paths)))
    user_id = cursor.lastrowid
    rel = lambda p: os.path.relpath(p, ROOT)
    rows = [(user_id, "original" if i == 0 else "augmented", rel(p),
             emb_to_blob(embs[i]), created_at) for i, p in enumerate(all_paths)]
    cursor.executemany(
        "INSERT INTO refs (user_id, kind, image_path, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
        rows)
    conn.commit()

    return {
        "ok": True,
        "status": "enrolled",
        "label": label,
        "user_id": user_id,
        "num_refs": len(all_paths),
        "message": f"'{label}' terdaftar dengan {len(all_paths)} referensi (1 asli + {AUG_COUNT} augmentasi).",
    }


# ============================================================
#  Verification  —  compare input ONLY against one user's refs
# ============================================================
@app.post("/api/verify")
async def verify(payload: VerifyPayload):
    if "efficientnet-b0" not in models_cache:
        raise HTTPException(status_code=503, detail="Model EfficientNet-B0 belum aktif.")
    model = models_cache["efficientnet-b0"]

    label = (payload.label or "").strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label yang ingin diverifikasi wajib diisi.")

    cursor.execute("SELECT id, num_refs FROM users WHERE label = ?", (label,))
    user = cursor.fetchone()
    if not user:
        return {"ok": False, "status": "no_reference",
                "message": f"Label '{label}' belum terdaftar. Lakukan enrollment dulu."}

    # Score ONLY against the enrolled ORIGINAL signature(s) — NOT the synthetic
    # augmentations. This triplet model gives augmentations low similarity even to
    # their own original, so averaging them in drags the score down enough that even
    # an IDENTICAL upload can fall below threshold. Comparing to the real enrolled
    # signature(s) makes "upload file yang sama" -> similarity 1.0 -> genuine, and
    # matches how cek_model.ipynb compares images directly.
    cursor.execute("SELECT embedding FROM refs WHERE user_id = ? AND kind = 'original'", (user["id"],))
    rows = cursor.fetchall()
    if not rows:  # fallback for older enrollments saved without a 'kind'
        cursor.execute("SELECT embedding FROM refs WHERE user_id = ?", (user["id"],))
        rows = cursor.fetchall()
    ref_embs = np.stack([blob_to_emb(r["embedding"]) for r in rows])

    # Save the incoming test image, then embed it (single forward pass).
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    file_name = f"sig_{int(time.time() * 1000)}_{random.randint(100000, 999999)}.png"
    abs_test = os.path.join(IMG_DIR, "test", file_name)
    try:
        _save_b64_png(payload.image, abs_test)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal menyimpan gambar uji: {e}")
    try:
        test_emb = embed_cnn(model, abs_test).cpu().numpy()[0]
    except Exception:
        raise HTTPException(status_code=400, detail="Gambar uji tidak dapat diproses.")

    # Cosine similarity (vectors are L2-normalized -> dot product).
    sims = ref_embs @ test_emb
    max_sim = float(np.max(sims))
    mean_sim = float(np.mean(sims))
    # Mean cosine vs the enrolled ORIGINAL signature(s). With one enrolled signature
    # this is a direct 1-vs-1 comparison (identical image -> 1.0).
    score = mean_sim

    threshold = float(payload.threshold) if payload.threshold is not None else COSINE_THRESHOLD
    verdict = "genuine" if score >= threshold else "forged"
    confidence = score if verdict == "genuine" else 1.0 - score

    image_path = os.path.join("data", "captures", "test", file_name)
    cursor.execute("""
      INSERT INTO captures (model, label, verdict, confidence, models_json, strokes_json,
                            image_path, stroke_count, point_count, created_at, user_label)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "efficientnet-b0", "test", verdict, confidence,
        json.dumps({"cosine_score": round(score, 4)}),
        json.dumps(payload.strokes or []), image_path,
        payload.meta.get("strokeCount", 0) if payload.meta else 0,
        len(payload.strokes or []), created_at, label,
    ))
    conn.commit()
    capture_id = cursor.lastrowid

    return {
        "ok": True,
        "status": "scored",
        "label": label,
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "score": round(score, 4),
        "max_similarity": round(max_sim, 4),
        "mean_similarity": round(mean_sim, 4),
        "threshold": round(threshold, 4),
        "num_refs": int(ref_embs.shape[0]),
        "id": capture_id,
        "image_url": f"/api/captures/{capture_id}/image",
    }


# ============================================================
#  Direct 2-image compare  —  logika sama dengan cek_model.ipynb
#  (embed dua gambar, hitung cosine, tanpa enroll/augmentasi)
# ============================================================
COMPARE_THRESHOLD = 0.50   # default untuk compare langsung 1-vs-1


@app.post("/api/compare")
async def compare_images(payload: ComparePayload):
    if "efficientnet-b0" not in models_cache:
        raise HTTPException(status_code=503, detail="Model EfficientNet-B0 belum aktif.")
    model = models_cache["efficientnet-b0"]

    tmp_dir = os.path.join(DATA_DIR, "compare_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    stamp = f"{int(time.time()*1000)}_{random.randint(100000, 999999)}"
    p1 = os.path.join(tmp_dir, f"a_{stamp}.png")
    p2 = os.path.join(tmp_dir, f"b_{stamp}.png")
    try:
        _save_b64_png(payload.image1, p1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gambar 1 tidak valid: {e}")
    try:
        _save_b64_png(payload.image2, p2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gambar 2 tidak valid: {e}")

    try:
        e1 = embed_cnn(model, p1).cpu().numpy()[0]
        e2 = embed_cnn(model, p2).cpu().numpy()[0]
    except Exception:
        raise HTTPException(status_code=400, detail="Gagal memproses gambar.")
    finally:
        for p in (p1, p2):
            try: os.remove(p)
            except Exception: pass

    sim = float(np.dot(e1, e2))   # cosine (vektor L2-normalized)
    threshold = float(payload.threshold) if payload.threshold is not None else COMPARE_THRESHOLD
    match = sim >= threshold
    return {
        "ok": True,
        "similarity": round(sim, 4),
        "threshold": round(threshold, 4),
        "match": match,
        "verdict": "COCOK — tanda tangan SAMA" if match else "TIDAK COCOK — tanda tangan BEDA",
    }


# ============================================================
#  Enrolled-user management
# ============================================================
@app.get("/api/users")
async def list_users():
    cursor.execute("SELECT id, label, created_at, num_refs FROM users ORDER BY label ASC")
    return [{"id": r["id"], "label": r["label"], "created_at": r["created_at"],
             "num_refs": r["num_refs"]} for r in cursor.fetchall()]


@app.get("/api/users/{label}")
async def get_user(label: str):
    cursor.execute("SELECT id, label, created_at, num_refs FROM users WHERE label = ?", (label,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    cursor.execute("SELECT id, kind, image_path FROM refs WHERE user_id = ? ORDER BY id ASC", (user["id"],))
    refs = [{"id": r["id"], "kind": r["kind"],
             "image_url": f"/api/refs/{r['id']}/image"} for r in cursor.fetchall()]
    return {"id": user["id"], "label": user["label"], "created_at": user["created_at"],
            "num_refs": user["num_refs"], "refs": refs}


@app.get("/api/refs/thumb")
async def get_ref_thumb(label: str):
    """Thumbnail = the 'original' reference image of an enrolled user, by label."""
    cursor.execute("""
      SELECT r.image_path FROM refs r JOIN users u ON u.id = r.user_id
      WHERE u.label = ? AND r.kind = 'original' LIMIT 1
    """, (label,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    full_path = os.path.join(ROOT, row["image_path"])
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(full_path)


@app.get("/api/refs/{ref_id}/image")
async def get_ref_image(ref_id: int):
    cursor.execute("SELECT image_path FROM refs WHERE id = ?", (ref_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Reference not found")
    full_path = os.path.join(ROOT, row["image_path"])
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(full_path)


@app.delete("/api/users/{label}")
async def delete_user(label: str):
    cursor.execute("SELECT id FROM users WHERE label = ?", (label,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    cursor.execute("DELETE FROM refs WHERE user_id = ?", (user["id"],))
    cursor.execute("DELETE FROM users WHERE id = ?", (user["id"],))
    conn.commit()
    user_dir = os.path.join(ENROLL_DIR, _safe_label(label))
    if os.path.isdir(user_dir):
        shutil.rmtree(user_dir, ignore_errors=True)
    return {"ok": True, "message": f"Enrollment '{label}' dihapus."}


@app.get("/api/captures")
async def get_captures(limit: int = 10, label: Optional[str] = None):
    limit = min(50, max(1, limit))
    if label in ("genuine", "test"):
        cursor.execute("""
          SELECT id, model, label, verdict, confidence, created_at FROM captures 
          WHERE label = ? ORDER BY id DESC LIMIT ?
        """, (label, limit))
    else:
        cursor.execute("""
          SELECT id, model, label, verdict, confidence, created_at FROM captures 
          ORDER BY id DESC LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    return [{
        "id": r["id"],
        "model": r["model"],
        "label": r["label"],
        "verdict": r["verdict"],
        "confidence": r["confidence"],
        "created_at": r["created_at"],
        "image_url": f"/api/captures/{r['id']}/image"
    } for r in rows]


@app.get("/api/captures/{capture_id}/image")
async def get_capture_image(capture_id: int):
    cursor.execute("SELECT image_path FROM captures WHERE id = ?", (capture_id,))
    row = cursor.fetchone()
    if not row or not row["image_path"]:
        raise HTTPException(status_code=404, detail="Capture image not found")
    
    full_path = os.path.join(ROOT, row["image_path"])
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    
    return FileResponse(full_path)


@app.get("/api/references")
async def get_references(path: Optional[str] = None, filter: Optional[str] = None):
    if path and path != "custom":
        genuine_dir = os.path.abspath(path.strip())
        if not _is_within_allowed(genuine_dir):
            raise HTTPException(status_code=403, detail="Path not allowed")
        if not os.path.exists(genuine_dir):
            raise HTTPException(status_code=404, detail="Directory not found")
        is_custom = False
    else:
        genuine_dir = os.path.join(IMG_DIR, "whiteboard")
        is_custom = True
        
    try:
        files = [f for f in os.listdir(genuine_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        if filter:
            files = [f for f in files if f.startswith(filter)]
        
        result = []
        for f in files:
            if is_custom:
                url = f"/api/references/custom/{f}"
            else:
                full_file_path = os.path.join(genuine_dir, f)
                url = f"/api/references/view?file={base64.urlsafe_b64encode(full_file_path.encode()).decode()}"
            result.append({"name": f, "image_url": url})
        return result
    except Exception as e:
        print(f"Error loading references: {e}")
        return []
@app.delete("/api/references/whiteboard")
async def clear_whiteboard():
    whiteboard_dir = os.path.join(IMG_DIR, "whiteboard")
    if os.path.exists(whiteboard_dir):
        for f in os.listdir(whiteboard_dir):
            f_path = os.path.join(whiteboard_dir, f)
            try:
                if os.path.isfile(f_path):
                    os.remove(f_path)
            except Exception as e:
                print(f"Error removing {f_path}: {e}")
    try:
        cursor.execute("DELETE FROM captures WHERE label = 'genuine'")
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    return {"ok": True, "message": "Whiteboard data reset successfully"}



@app.get("/api/references/view")
async def view_reference_image(file: str):
    try:
        decoded_path = base64.urlsafe_b64decode(file.encode()).decode()
        full_path = os.path.abspath(decoded_path.strip())
        if not full_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            raise HTTPException(status_code=400, detail="Invalid image file type")
        if not _is_within_allowed(full_path):
            raise HTTPException(status_code=403, detail="Path not allowed")
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(full_path)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path token")


@app.get("/api/references/custom/{name}")
async def get_custom_reference_image(name: str):
    safe_name = os.path.basename(name)
    if safe_name != name:
        raise HTTPException(status_code=400, detail="Invalid name")
        
    full_path = os.path.join(IMG_DIR, "whiteboard", safe_name)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Custom Reference image not found")
        
    return FileResponse(full_path)


# ------------------------------------------------------------------
#  Serve the front-end via an EXPLICIT allow-list.
#  A blanket StaticFiles(ROOT) mount would expose data/detector.db
#  (all biometric records), server.py source, and every capture PNG.
#  Only these top-level front-end assets are ever served.
# ------------------------------------------------------------------
FRONTEND_WHITELIST = {"index.html", "styles.css", "app.js", "compare.html"}


# Public URL of the separately-hosted Voila app (canvas-drawing compare feature).
# Defaults to localhost for local dev; set the VOILA_URL env var in production
# (e.g. https://ttd-voila.onrender.com) once that service is deployed.
VOILA_URL = os.environ.get("VOILA_URL", "http://127.0.0.1:8866")


@app.get("/")
async def get_index():
    index_path = os.path.join(ROOT, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index not found")
    html = open(index_path, "r", encoding="utf-8").read()
    html = html.replace("{{VOILA_URL}}", VOILA_URL)
    return HTMLResponse(html)


@app.get("/{fname}")
async def serve_frontend(fname: str):
    if fname not in FRONTEND_WHITELIST:
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.join(ROOT, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=3000, reload=False)
