#!/usr/bin/env python3
"""Recover the LK25 palette empirically instead of guessing hex codes.

k-means over pixels drawn from valley + alpine tiles. The cluster centres are
the actual ink colours swisstopo prints with, which is what the deterministic
renderer must match."""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.fetch import map_tile
from dufour.tiles import deg2tile

SITES = {  # valleys with roads/rail/forest, plus high alpine
 "zermatt":(46.020,7.749),"lauterbrunnen":(46.593,7.909),"grindelwald":(46.624,8.041),
 "andermatt":(46.637,8.594),"saas":(46.122,7.930),"pontresina":(46.492,9.902),
 "matterhorn":(45.976,7.659),"aletsch":(46.470,8.060),"bernina":(46.382,9.909),
}
Z = 15
px = []
for name,(la,lo) in SITES.items():
    cx,cy = deg2tile(la,lo,Z)
    for dx in range(-1,2):
        for dy in range(-1,2):
            t = map_tile(Z,cx+dx,cy+dy)
            if t is not None: px.append(t.reshape(-1,3))
X = np.concatenate(px).astype(np.float32)
print(f"{len(X):,} pixels from {len(px)} tiles")

rng = np.random.default_rng(0)
S = X[rng.choice(len(X), 250_000, replace=False)]
K = 22
C = S[rng.choice(len(S), K, replace=False)].copy()
for it in range(40):
    d = ((S[:,None,:]-C[None])**2).sum(2) if False else None
    # chunked assignment to keep memory sane
    lab = np.empty(len(S), np.int32)
    for i in range(0, len(S), 20000):
        b = S[i:i+20000]
        lab[i:i+20000] = ((b[:,None,:]-C[None])**2).sum(2).argmin(1)
    for k in range(K):
        m = lab==k
        if m.any(): C[k] = S[m].mean(0)
cnt = np.bincount(lab, minlength=K)/len(lab)
order = np.argsort(-cnt)
print(f"\n{'hex':>9s} {'rgb':>16s} {'share':>7s}")
for k in order:
    r,g,b = C[k].round().astype(int)
    print(f"  #{r:02x}{g:02x}{b:02x} {str((r,g,b)):>16s} {cnt[k]*100:6.2f}%")
