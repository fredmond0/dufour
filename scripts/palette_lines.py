#!/usr/bin/env python3
"""Palette of THIN features (contours, roads, rails).

Flat k-means is dominated by area fills, so 1-2 px ink never forms a cluster.
Here we keep only pixels that deviate strongly from their local median -- i.e.
ink laid on top of a background -- and cluster those."""
import pathlib, sys
import numpy as np
from scipy.ndimage import median_filter
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.fetch import map_tile
from dufour.tiles import deg2tile

SITES={"zermatt":(46.020,7.749),"lauterbrunnen":(46.593,7.909),"grindelwald":(46.624,8.041),
 "andermatt":(46.637,8.594),"pontresina":(46.492,9.902),"saas":(46.122,7.930),
 "aletsch":(46.470,8.060),"matterhorn":(45.976,7.659)}
Z=15; keep=[]
for la,lo in SITES.values():
    cx,cy=deg2tile(la,lo,Z)
    for dx in range(-1,2):
        for dy in range(-1,2):
            t=map_tile(Z,cx+dx,cy+dy)
            if t is None: continue
            f=t.astype(np.float32)
            bg=np.stack([median_filter(f[...,c],9) for c in range(3)],-1)
            dev=np.abs(f-bg).sum(-1)
            keep.append(f[dev>60])
X=np.concatenate(keep); print(f"{len(X):,} ink pixels")
rng=np.random.default_rng(0)
S=X[rng.choice(len(X),min(150_000,len(X)),replace=False)]
K=14; C=S[rng.choice(len(S),K,replace=False)].copy()
for _ in range(40):
    lab=np.empty(len(S),np.int32)
    for i in range(0,len(S),20000):
        b=S[i:i+20000]; lab[i:i+20000]=((b[:,None,:]-C[None])**2).sum(2).argmin(1)
    for k in range(K):
        m=lab==k
        if m.any(): C[k]=S[m].mean(0)
cnt=np.bincount(lab,minlength=K)/len(lab)
for k in np.argsort(-cnt):
    r,g,b=C[k].round().astype(int); print(f"  #{r:02x}{g:02x}{b:02x}  {cnt[k]*100:5.2f}%")
