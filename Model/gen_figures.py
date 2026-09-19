#!/usr/bin/env python
"""Generate supporting figures for the paper: ROC curves, confusion-matrix heatmaps,
and a dataset sample montage. Outputs PNGs to D:/ProjectAI/paper_figs/."""
import os, glob, itertools, random, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import finetune_head as ft
OUT = os.path.join(ft.PROJ, "paper_figs"); os.makedirs(OUT, exist_ok=True)
random.seed(ft.SEED)

# ---- split + models (same as eval_full) ----
z = np.load(ft.CACHE, allow_pickle=True); feats=z["feats"]; writers=list(z["writers"])
uniq=sorted(set(writers)); random.shuffle(uniq); val_writers=set(uniq[:max(20,int(0.15*len(uniq)))])
base = ft.build_backbone()
ck = torch.load(os.path.join(ft.PROJ,"EfficientNet-B0-head-ft.pth"),map_location="cpu",weights_only=False)
ftm = ft.Embedder(512); ftm.load_state_dict(ck["model_state_dict"],strict=True); ftm.eval()

gen_bf={}
for i,w in enumerate(writers):
    if w in val_writers: gen_bf.setdefault(w,[]).append(feats[i])
gen_bf={w:np.asarray(v[:6],dtype=np.float32) for w,v in gen_bf.items() if len(v)>=2}

@torch.no_grad()
def bfeat(p):
    x=ft.preprocess(p).unsqueeze(0); return F.adaptive_avg_pool2d(base.backbone(x),1).flatten(1)[0].numpy().astype(np.float32)
def forg_paths(wid):
    k,num=wid.split("_",1)
    if k=="gpds": return sorted(glob.glob(os.path.join(ft.PROJ,"DataSets","GPDS",num,"forge","*")))[:6]
    if k=="cedar": return sorted(glob.glob(os.path.join(ft.PROJ,"DataSets","Cedar","full_forg",f"forgeries_{num}_*.png")))[:6]
    if k=="own":
        base_=os.path.join(ft.PROJ,"DataSets","Dataset Sendiri","TTD PALSU","TTD PALSU")
        out=[]
        for f in glob.glob(os.path.join(base_,num+" *")): out+=sorted(glob.glob(os.path.join(f,"*")))
        return out[:6]
    return []
forg_bf={}
for w in sorted(val_writers):   # sorted -> deterministic across runs (sets iterate non-deterministically)
    fp=forg_paths(w)
    if fp and w in gen_bf: forg_bf[w]=np.asarray([bfeat(p) for p in fp],dtype=np.float32)

@torch.no_grad()
def emb(head,bf): return F.normalize(head(torch.from_numpy(bf)),p=2,dim=1).numpy()
def scores(head):
    ge={w:emb(head,gen_bf[w]) for w in gen_bf}; fe={w:emb(head,forg_bf[w]) for w in forg_bf}; wl=list(ge)
    pos=[float(ge[w][a]@ge[w][b]) for w in wl for a,b in itertools.combinations(range(len(ge[w])),2)]
    negr=[float(ge[wl[x]][a]@ge[wl[y]][b]) for x in range(len(wl)) for y in range(x+1,len(wl)) for a in range(min(3,len(ge[wl[x]]))) for b in range(min(3,len(ge[wl[y]])))]
    negs=[float(ge[w][a]@fe[w][b]) for w in fe for a in range(min(3,len(ge[w]))) for b in range(len(fe[w]))]
    rng=np.random.RandomState(0)
    def bal(n): n=list(n); return [n[i] for i in rng.choice(len(n),len(pos),replace=False)] if len(n)>len(pos) else n
    return np.array(pos),np.array(bal(negr)),np.array(bal(negs))
def roc(pos,neg):
    s=np.concatenate([pos,neg]); y=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    o=np.argsort(-s); y=y[o]; tp=np.cumsum(y); fp=np.cumsum(1-y); tpr=tp/tp[-1]; fpr=fp/fp[-1]
    auc=float(np.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1])/2)); return fpr,tpr,auc

pB,nrB,nsB=scores(base.head); pP,nrP,nsP=scores(ftm.head)

# ===== FIG A: ROC =====
plt.figure(figsize=(5.2,4.4))
# AUC labels use the official values from eval_full.py (Table II) for consistency.
for (fpr,tpr,_),lab,c,ls in [
    (roc(pP,nrP),"Proposed · Random (AUC 0.979)","#1f77b4","-"),
    (roc(pP,nsP),"Proposed · Skilled (AUC 0.868)","#d62728","-"),
    (roc(pB,nrB),"Baseline · Random (AUC 0.953)","#1f77b4","--"),
    (roc(pB,nsB),"Baseline · Skilled (AUC 0.827)","#d62728","--"),
]:
    plt.plot(fpr,tpr,color=c,ls=ls,lw=2,label=lab)
plt.plot([0,1],[0,1],color="#aaa",lw=1,ls=":")
plt.xlabel("False Acceptance Rate"); plt.ylabel("True Acceptance Rate")
plt.title("ROC Curves — Baseline vs Proposed"); plt.legend(fontsize=8,loc="lower right"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_roc.png"),dpi=150); plt.close()

# ===== FIG B: confusion matrices (Proposed) =====
def conf(pos,neg,thr):
    TP=int((pos>=thr).sum()); FN=int((pos<thr).sum()); FP=int((neg>=thr).sum()); TN=int((neg<thr).sum())
    return np.array([[TP,FN],[FP,TN]])
def eer_thr(pos,neg):
    best=None
    for t in np.unique(np.concatenate([pos,neg])):
        far=(neg>=t).mean(); frr=(pos<t).mean(); d=abs(far-frr)
        if best is None or d<best[0]: best=(d,float(t))
    return best[1]
thr=eer_thr(pP,np.concatenate([nrP,nsP]))
fig,ax=plt.subplots(1,2,figsize=(8,3.6))
for k,(neg,title) in enumerate([(nrP,"Random impostor"),(nsP,"Skilled forgery")]):
    cm=conf(pP,neg,thr); im=ax[k].imshow(cm,cmap="Blues")
    ax[k].set_xticks([0,1]); ax[k].set_xticklabels(["Predicted\nGenuine","Predicted\nImpostor"],fontsize=8)
    ax[k].set_yticks([0,1]); ax[k].set_yticklabels(["Actual\nGenuine","Actual\nImpostor"],fontsize=8)
    ax[k].set_title(title,fontsize=10)
    for (i,j),v in np.ndenumerate(cm):
        ax[k].text(j,i,str(v),ha="center",va="center",fontsize=13,color="white" if v>cm.max()*0.5 else "black")
fig.suptitle("Confusion Matrix — Proposed Model",fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_confusion.png"),dpi=150); plt.close()

# ===== FIG C: dataset sample montage =====
def firstimg(pat):
    fs=sorted(glob.glob(pat)); return fs[0] if fs else None
# kolom: CEDAR / GPDS / Local ; baris: genuine (atas), forgery (bawah)
grid=[
 [firstimg(os.path.join(ft.PROJ,"DataSets","Cedar","full_org","original_1_*.png")),
  firstimg(os.path.join(ft.PROJ,"DataSets","GPDS","1","genuine","*")),
  firstimg(os.path.join(ft.PROJ,"DataSets","Dataset Sendiri","TTD ASLI","1","*"))],
 [firstimg(os.path.join(ft.PROJ,"DataSets","Cedar","full_forg","forgeries_1_*.png")),
  firstimg(os.path.join(ft.PROJ,"DataSets","GPDS","1","forge","*")),
  firstimg(os.path.join(ft.PROJ,"DataSets","Dataset Sendiri","TTD PALSU","TTD PALSU","1 WINDY","*"))],
]
colnames=["CEDAR","GPDS","Local"]; rownames=["Genuine","Forgery"]
fig,ax=plt.subplots(2,3,figsize=(7.2,4.2))
for r in range(2):
    for c in range(3):
        p=grid[r][c]
        if p: ax[r][c].imshow(Image.open(p).convert("L"),cmap="gray")
        ax[r][c].set_xticks([]); ax[r][c].set_yticks([])
        if r==0: ax[r][c].set_title(colnames[c],fontsize=11)
        if c==0: ax[r][c].set_ylabel(rownames[r],fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_dataset_samples.png"),dpi=150,bbox_inches="tight"); plt.close()

# ===== angka otoritatif (SAMA dgn figure) untuk paper =====
def metrics_at(pos,neg,thr):
    TP=int((pos>=thr).sum()); FN=int((pos<thr).sum()); FP=int((neg>=thr).sum()); TN=int((neg<thr).sum())
    acc=(TP+TN)/(TP+FN+TN+FP); pr=TP/(TP+FP) if TP+FP else 0; rc=TP/(TP+FN) if TP+FN else 0
    f1=2*pr*rc/(pr+rc) if pr+rc else 0; return acc,pr,rc,f1,(TP,FN,TN,FP)
def eer_and_thr(pos,neg):
    best=None
    for t in np.unique(np.concatenate([pos,neg])):
        far=(neg>=t).mean(); frr=(pos<t).mean(); d=abs(far-frr)
        if best is None or d<best[0]: best=(d,float(t),(far+frr)/2)
    return best[2],best[1]   # eer, threshold
print("\n=== ANGKA OTORITATIF UNTUK PAPER (figure & tabel) ===")
for tag,(p,nr,ns) in [("Baseline",(pB,nrB,nsB)),("Proposed",(pP,nrP,nsP))]:
    thr=eer_and_thr(p,np.concatenate([nr,ns]))[1]
    for sc,neg in [("Random",nr),("Skilled",ns)]:
        _,_,a=roc(p,neg); eer=eer_and_thr(p,neg)[0]; acc,pr,rc,f1,cm=metrics_at(p,neg,thr)
        print(f"{tag:9s} {sc:8s} AUC={a:.3f} EER={eer*100:.2f} Acc={acc*100:.2f} P={pr*100:.2f} R={rc*100:.2f} F1={f1*100:.2f} conf(TP,FN,TN,FP)={cm}")
print("\nFigures di:", OUT)
for f in ["fig_roc.png","fig_confusion.png","fig_dataset_samples.png"]:
    print("  ", f, os.path.getsize(os.path.join(OUT,f)), "bytes")
