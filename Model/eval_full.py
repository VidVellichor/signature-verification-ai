#!/usr/bin/env python
"""
Definitive evaluation for the paper. Reproduces the exact 188/33 writer split from
finetune_head.py, then evaluates BOTH the baseline (triplet-v4) and the head-fine-tuned
EfficientNet-B0 on the SAME 33 held-out writers, under two scenarios:
  - RANDOM  : genuine vs genuine-of-a-different-writer (random impostor)
  - SKILLED : genuine vs same-writer forgery (skilled forgery)
Reports AUC, EER, and at the EER-operating point: accuracy, precision, recall, F1,
plus the confusion-matrix counts (TP/FN/TN/FP). Backbone is frozen and shared, so
genuine backbone features are reused from the cache; forgery features are computed fresh.
"""
import os, glob, itertools, random, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import finetune_head as ft   # reuses gather(), Embedder, Head, preprocess, CACHE, SEED

random.seed(ft.SEED)   # replicate the split exactly (shuffle is the first random consumer)

# ---- 1. reproduce split ----
data = ft.gather()
z = np.load(ft.CACHE, allow_pickle=True)
feats = z["feats"]; writers = list(z["writers"]); paths = list(z["paths"])
uniq = sorted(set(writers)); random.shuffle(uniq)
n_val = max(20, int(0.15 * len(uniq)))
val_writers = set(uniq[:n_val])
print(f"[split] total {len(uniq)} writers -> {len(val_writers)} held-out for evaluation")

# ---- 2. load both models (shared frozen backbone) ----
base = ft.build_backbone()   # triplet-v4 full model, eval mode
ckft = torch.load(os.path.join(ft.PROJ, "EfficientNet-B0-head-ft.pth"), map_location="cpu", weights_only=False)
ftm = ft.Embedder(int(ckft.get("embedding_dim", 512))).to("cpu")
ftm.load_state_dict(ckft["model_state_dict"], strict=True); ftm.eval()

# ---- 3. gather val backbone features ----
gen_bf = {}
for i, w in enumerate(writers):
    if w in val_writers:
        gen_bf.setdefault(w, []).append(feats[i])
gen_bf = {w: np.asarray(v[:6], dtype=np.float32) for w, v in gen_bf.items() if len(v) >= 2}

@torch.no_grad()
def backbone_feat(path):
    x = ft.preprocess(path).unsqueeze(0)
    return F.adaptive_avg_pool2d(base.backbone(x), 1).flatten(1)[0].numpy().astype(np.float32)

def forg_paths(wid):
    kind, num = wid.split("_", 1)
    if kind == "gpds":
        return sorted(glob.glob(os.path.join(ft.PROJ, "DataSets", "GPDS", num, "forge", "*")))[:6]
    if kind == "cedar":
        return sorted(glob.glob(os.path.join(ft.PROJ, "DataSets", "Cedar", "full_forg", f"forgeries_{num}_*.png")))[:6]
    if kind == "own":
        base = os.path.join(ft.PROJ, "DataSets", "Dataset Sendiri", "TTD PALSU", "TTD PALSU")
        folders = glob.glob(os.path.join(base, num + " *"))   # "1 WINDY", etc.
        out = []
        for f in folders:
            out += sorted(glob.glob(os.path.join(f, "*")))
        return out[:6]
    return []

forg_bf = {}
for w in sorted(val_writers):
    fps = forg_paths(w)
    if fps and w in gen_bf:
        forg_bf[w] = np.asarray([backbone_feat(p) for p in fps], dtype=np.float32)
print(f"[data] genuine writers={len(gen_bf)}  writers-with-forgery={len(forg_bf)}")

# ---- 4. metrics ----
def auc_of(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-s); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    tpr = tp / tp[-1]; fpr = fp / fp[-1]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))

def eer_of(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    best = None
    for t in np.unique(np.concatenate([pos, neg])):
        far = float((neg >= t).mean()); frr = float((pos < t).mean())
        d = abs(far - frr)
        if best is None or d < best[0]:
            best = (d, float(t), (far + frr) / 2)
    return best[2], best[1]   # eer, threshold

def at_threshold(pos, neg, thr):
    pos = np.asarray(pos); neg = np.asarray(neg)
    TP = int((pos >= thr).sum()); FN = int((pos < thr).sum())
    TN = int((neg < thr).sum());  FP = int((neg >= thr).sum())
    tot = TP + FN + TN + FP
    acc = (TP + TN) / tot
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, TP=TP, FN=FN, TN=TN, FP=FP)

@torch.no_grad()
def embed_bf(head, bf):
    return F.normalize(head(torch.from_numpy(bf)), p=2, dim=1).numpy()

def evaluate(head):
    ge = {w: embed_bf(head, gen_bf[w]) for w in gen_bf}
    fe = {w: embed_bf(head, forg_bf[w]) for w in forg_bf}
    pos, neg_r, neg_s = [], [], []
    wl = list(ge)
    for w in wl:
        e = ge[w]
        for a, b in itertools.combinations(range(len(e)), 2):
            pos.append(float(e[a] @ e[b]))
    for x in range(len(wl)):
        for yy in range(x + 1, len(wl)):
            ea, eb = ge[wl[x]], ge[wl[yy]]
            for a in range(min(3, len(ea))):
                for b in range(min(3, len(eb))):
                    neg_r.append(float(ea[a] @ eb[b]))
    for w in fe:
        eg, ef = ge[w], fe[w]
        for a in range(min(3, len(eg))):
            for b in range(len(ef)):
                neg_s.append(float(eg[a] @ ef[b]))
    # Balance each negative pool to the number of genuine pairs (fixed seed).
    rng = np.random.RandomState(0)
    def bal(neg):
        neg = list(neg)
        if len(neg) > len(pos):
            idx = rng.choice(len(neg), len(pos), replace=False)
            return [neg[i] for i in idx]
        return neg
    neg_r, neg_s = bal(neg_r), bal(neg_s)
    # ONE operating threshold per model: EER point over genuine vs ALL impostors
    # (random + skilled combined) -- mimics deployment with a single threshold.
    _, thr = eer_of(pos, neg_r + neg_s)
    R = dict(auc=auc_of(pos, neg_r), eer=eer_of(pos, neg_r)[0], **at_threshold(pos, neg_r, thr))
    S = dict(auc=auc_of(pos, neg_s), eer=eer_of(pos, neg_s)[0], **at_threshold(pos, neg_s, thr))
    return R, S, thr, len(pos)

def show(name, m):
    print(f"  {name:8s} AUC={m['auc']:.3f}  EER={m['eer']*100:5.2f}%  Acc={m['acc']*100:5.2f}%  "
          f"P={m['prec']*100:5.2f}%  R={m['rec']*100:5.2f}%  F1={m['f1']*100:5.2f}%  "
          f"[TP={m['TP']} FN={m['FN']} TN={m['TN']} FP={m['FP']}]")

for tag, model in [("BASELINE (triplet-v4)", base), ("FINE-TUNED (head-ft)", ftm)]:
    R, S, thr, npos = evaluate(model.head)
    print(f"\n=== {tag} === (genuine pairs n={npos}, single threshold={thr:.3f})")
    show("RANDOM", R); show("SKILLED", S)
