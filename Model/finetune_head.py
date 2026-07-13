#!/usr/bin/env python
"""
Fine-tune HANYA projection head EfficientNet (backbone dibekukan) di CPU.
Gabungkan GPDS + CEDAR + Dataset Sendiri (penulis-disjoint train/val).
Precompute fitur backbone sekali (di-cache), lalu latih head dgn batch-hard triplet.
Ukur AUC genuine-vs-impostor di penulis validasi, bandingkan dgn model lama.
"""
import os, glob, re, time, json, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
CKPT = os.path.join(PROJ, "EfficientNet-B0-triplet-v4.pth")
CACHE = os.path.join(ROOT, "_feat_cache.npz")
OUT_CKPT = os.path.join(PROJ, "EfficientNet-B0-head-ft.pth")
IMG_SIZE = 224
MAX_PER_WRITER = 12
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cpu")

# ---------- preprocessing (sama dgn notebook, sudah fixed _load_white_bg) ----------
_MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
_STD  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def _load_white_bg(path):
    img = Image.open(path)
    if img.mode in ("RGBA","LA") or (img.mode=="P" and "transparency" in img.info):
        img = img.convert("RGBA"); a = img.getchannel("A")
        if a.getextrema()[0] < 255:
            bl=Image.new("RGBA",img.size,(0,0,0,255)); tr=Image.new("RGBA",img.size,(255,255,255,0))
            img=Image.composite(bl,tr,a); bg=Image.new("RGBA",img.size,(255,255,255,255)); img=Image.alpha_composite(bg,img)
    return img.convert("RGB")
def autocrop(img, size=IMG_SIZE, thr=245, pad=0.08):
    g=np.array(img.convert("L")); mask=g<thr
    if mask.sum()==0: c=img.convert("L")
    else:
        ys,xs=np.where(mask); y0,y1,x0,x1=int(ys.min()),int(ys.max()),int(xs.min()),int(xs.max()); h,w=g.shape
        py=max(2,int((y1-y0+1)*pad)); px=max(2,int((x1-x0+1)*pad))
        y0,y1,x0,x1=max(0,y0-py),min(h-1,y1+py),max(0,x0-px),min(w-1,x1+px)
        c=img.convert("L").crop((x0,y0,x1+1,y1+1))
    cw,ch=c.size; s=max(cw,ch); cv=Image.new("L",(s,s),255); cv.paste(c,((s-cw)//2,(s-ch)//2))
    return cv.resize((size,size),Image.BILINEAR)
def preprocess(path):
    im=autocrop(_load_white_bg(path)).convert("RGB")
    t=torch.from_numpy(np.array(im,dtype=np.float32)/255.0).permute(2,0,1)
    return (t-_MEAN)/_STD

# ---------- kumpulkan data: writer -> daftar path (genuine saja) ----------
def gather():
    data={}
    for w in glob.glob(os.path.join(PROJ,"DataSets","GPDS","*")+os.sep):
        wid="gpds_"+os.path.basename(w.rstrip(os.sep))
        imgs=sorted(glob.glob(os.path.join(w,"genuine","*")))
        if len(imgs)>=2: data[wid]=imgs[:MAX_PER_WRITER]
    ced={}
    for p in glob.glob(os.path.join(PROJ,"DataSets","Cedar","full_org","*.png")):
        mo=re.match(r"original_(\d+)_",os.path.basename(p))
        if mo: ced.setdefault("cedar_"+mo.group(1),[]).append(p)
    for k,v in ced.items():
        if len(v)>=2: data[k]=sorted(v)[:MAX_PER_WRITER]
    for w in glob.glob(os.path.join(PROJ,"DataSets","Dataset Sendiri","TTD ASLI","*")+os.sep):
        wid="own_"+os.path.basename(w.rstrip(os.sep))
        imgs=sorted(glob.glob(os.path.join(w,"*")))
        imgs=[p for p in imgs if p.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
        if len(imgs)>=2: data[wid]=imgs[:MAX_PER_WRITER]
    return data

# ---------- backbone beku ----------
class Embedder(nn.Module):
    def __init__(self, emb=512, dropout=0.3):
        super().__init__()
        self.backbone=nn.Sequential(tvm.efficientnet_b0(weights=None).features)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(1280,512),nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),nn.Dropout(dropout),
                                nn.Linear(512,emb),nn.BatchNorm1d(emb))
    def forward(self,x):
        x=self.backbone(x); x=F.adaptive_avg_pool2d(x,1); return F.normalize(self.head(x),p=2,dim=1)

def build_backbone():
    ck=torch.load(CKPT,map_location=device,weights_only=False)
    sd=ck.get("model_state_dict",ck); m=Embedder(int(ck.get("embedding_dim",512))).to(device)
    m.load_state_dict(sd,strict=True); m.eval()
    return m

# ---------- precompute fitur backbone (1280-d) ----------
@torch.no_grad()
def backbone_feat(backbone, path):
    t=preprocess(path).unsqueeze(0).to(device)
    x=backbone(t); x=F.adaptive_avg_pool2d(x,1).flatten(1)
    return x[0].cpu().numpy().astype(np.float32)

def precompute(data, model):
    if os.path.exists(CACHE):
        z=np.load(CACHE, allow_pickle=True)
        print("[cache] load", CACHE); return z["feats"], list(z["writers"]), list(z["paths"])
    feats=[]; writers=[]; paths=[]
    bb=model.backbone
    t0=time.time(); n=sum(len(v) for v in data.values()); i=0
    for wid,imgs in data.items():
        for p in imgs:
            try:
                x=preprocess(p).unsqueeze(0)
                with torch.no_grad():
                    f=F.adaptive_avg_pool2d(bb(x),1).flatten(1)[0].numpy().astype(np.float32)
                feats.append(f); writers.append(wid); paths.append(p)
            except Exception as e:
                pass
            i+=1
            if i%200==0: print(f"[feat] {i}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    feats=np.stack(feats)
    np.savez_compressed(CACHE, feats=feats, writers=np.array(writers), paths=np.array(paths))
    print(f"[feat] done {len(feats)} imgs in {time.time()-t0:.0f}s")
    return feats, writers, paths

# ---------- head baru (dilatih di atas fitur 1280-d) ----------
class Head(nn.Module):
    def __init__(self, emb=512, dropout=0.3):
        super().__init__()
        # Struktur PERSIS sama dgn Embedder.head (termasuk Flatten di index 0) supaya
        # state_dict-nya bisa langsung dimasukkan ke Embedder saat menyimpan.
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(1280,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),
                               nn.Dropout(dropout),nn.Linear(512,emb),nn.BatchNorm1d(emb))
    def forward(self,x): return F.normalize(self.net(x),p=2,dim=1)

def batch_hard_triplet(emb, labels, margin=0.3):
    # emb: [B,D] normalized; labels: [B]
    d = 1.0 - emb @ emb.t()                       # cosine distance
    B = emb.size(0)
    lab = labels.unsqueeze(0)==labels.unsqueeze(1)
    eye = torch.eye(B, dtype=torch.bool)
    pos = lab & ~eye
    neg = ~lab
    losses=[]
    for i in range(B):
        if pos[i].any() and neg[i].any():
            hp = d[i][pos[i]].max()               # positive terjauh
            hn = d[i][neg[i]].min()               # negative terdekat
            losses.append(F.relu(hp-hn+margin))
    if not losses: return torch.tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()

def sample_batch(feat_by_writer, writers, P=12, K=4):
    ws=random.sample(writers, min(P,len(writers)))
    xs=[]; ys=[]
    for j,w in enumerate(ws):
        idx=feat_by_writer[w]
        pick=random.sample(idx, min(K,len(idx)))
        for k in pick: xs.append(k); ys.append(j)
    return np.array(xs), np.array(ys)

def evaluate(head, feats, writers, val_writers):
    head.eval()
    with torch.no_grad():
        embs={}
        for w in val_writers:
            idx=[i for i in range(len(writers)) if writers[i]==w]
            if len(idx)<2: continue
            e=head(torch.from_numpy(feats[idx])).numpy()
            embs[w]=e
        same=[]; diff=[]
        wl=list(embs)
        for w in wl:
            e=embs[w]
            for a in range(len(e)):
                for b in range(a+1,len(e)): same.append(float(e[a]@e[b]))
        for x in range(len(wl)):
            for y in range(x+1,len(wl)):
                ea,eb=embs[wl[x]],embs[wl[y]]
                for a in range(min(2,len(ea))):
                    for b in range(min(2,len(eb))): diff.append(float(ea[a]@eb[b]))
        same=np.array(same); diff=np.array(diff)
        s=np.concatenate([same,diff]); y=np.concatenate([np.ones(len(same)),np.zeros(len(diff))])
        o=np.argsort(-s); y=y[o]; tp=np.cumsum(y); fp=np.cumsum(1-y)
        tpr=tp/tp[-1]; fpr=fp/fp[-1]
        auc=float(np.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1])/2))
    head.train()
    return auc, same.mean(), diff.mean()

def main():
    data=gather()
    print(f"[data] {len(data)} penulis, {sum(len(v) for v in data.values())} gambar")
    model=build_backbone()
    feats, writers, paths = precompute(data, model)
    uniq=sorted(set(writers)); random.shuffle(uniq)
    n_val=max(20, int(0.15*len(uniq)))
    val_writers=set(uniq[:n_val]); train_writers=[w for w in uniq if w not in val_writers]
    print(f"[split] train {len(train_writers)} penulis | val {len(val_writers)} penulis")

    feat_by_writer={}
    for i,w in enumerate(writers):
        if w in train_writers: feat_by_writer.setdefault(w,[]).append(i)
    train_ws=[w for w in feat_by_writer if len(feat_by_writer[w])>=2]

    head=Head(512).to(device)
    opt=torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    # baseline: head lama (dari checkpoint) di val
    base_auc,bs,bd=evaluate_original(model, feats, writers, val_writers)
    print(f"[BASELINE model lama] AUC={base_auc:.3f} (same {bs:.2f} / diff {bd:.2f})")

    best=0; STEPS=1500
    for step in range(1, STEPS+1):
        xs,ys=sample_batch(feat_by_writer, train_ws, P=12, K=4)
        x=torch.from_numpy(feats[xs]); y=torch.tensor(ys)
        emb=head(x); loss=batch_hard_triplet(emb,y,margin=0.3)
        opt.zero_grad(); loss.backward(); opt.step()
        if step%150==0:
            auc,sm,df=evaluate(head, feats, writers, val_writers)
            print(f"[step {step:4d}] loss={loss.item():.3f}  val AUC={auc:.3f} (same {sm:.2f}/diff {df:.2f})", flush=True)
            if auc>best:
                best=auc
                torch.save({"model_state_dict":_merge(model,head),"embedding_dim":512,
                            "threshold":0.5,"note":"head-finetuned","val_auc":auc}, OUT_CKPT)
    print(f"\n[SELESAI] best val AUC = {best:.3f}  (lama {base_auc:.3f}) -> {OUT_CKPT if best>base_auc else 'TIDAK disimpan (tak lebih baik)'}")

@torch.no_grad()
def evaluate_original(model, feats, writers, val_writers):
    # pakai head asli model lama di atas fitur backbone
    head=model.head
    def proj(F_):
        return torch.nn.functional.normalize(head(torch.from_numpy(F_)),p=2,dim=1).numpy()
    embs={}
    for w in val_writers:
        idx=[i for i in range(len(writers)) if writers[i]==w]
        if len(idx)<2: continue
        embs[w]=proj(feats[idx])
    same=[]; diff=[]; wl=list(embs)
    for w in wl:
        e=embs[w]
        for a in range(len(e)):
            for b in range(a+1,len(e)): same.append(float(e[a]@e[b]))
    for x in range(len(wl)):
        for y in range(x+1,len(wl)):
            ea,eb=embs[wl[x]],embs[wl[y]]
            for a in range(min(2,len(ea))):
                for b in range(min(2,len(eb))): diff.append(float(ea[a]@eb[b]))
    same=np.array(same); diff=np.array(diff)
    s=np.concatenate([same,diff]); y=np.concatenate([np.ones(len(same)),np.zeros(len(diff))])
    o=np.argsort(-s); y=y[o]; tp=np.cumsum(y); fp=np.cumsum(1-y); tpr=tp/tp[-1]; fpr=fp/fp[-1]
    auc=float(np.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1])/2))
    return auc, same.mean(), diff.mean()

def _merge(model, head):
    # gabung backbone lama + head baru -> state_dict kompatibel Embedder
    new=Embedder(512)
    new.backbone.load_state_dict(model.backbone.state_dict())
    new.head.load_state_dict(head.net.state_dict())
    return new.state_dict()

if __name__=="__main__":
    main()
