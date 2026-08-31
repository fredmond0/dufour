#!/usr/bin/env python3
"""Attribute the supervision mask to its components, to find over-firing."""
import pathlib, sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.delabel import text_mask, word_mask
from dufour.fetch import map_tile
from dufour.separate import deterministic_mask, DETERMINISTIC, _dist, _neutral
from dufour.tiles import deg2tile

def main():
    z = 15
    sites = [(45.976, 7.659), (46.470, 8.060), (46.55, 8.05), (46.382, 9.909)]
    print(f"{'tile':6s}{'det':>7s}{'text':>7s}{'word':>7s}{'total':>7s}   per-class share of det")
    rows = []
    for i, (la, lo) in enumerate(sites):
        x, y = deg2tile(la, lo, z); m = map_tile(z, x, y)
        det, txt, wrd = deterministic_mask(m), text_mask(m), word_mask(m)
        tot = det | txt
        parts = {}
        for k, cols in DETERMINISTIC.items():
            parts[k] = float(((_dist(m, cols) < 52) & ~_neutral(m)).mean())
        print(f"{i:<6d}{det.mean():7.3f}{txt.mean():7.3f}{wrd.mean():7.3f}{tot.mean():7.3f}   "
              + " ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in parts.items()))
        def viz(mask): return np.stack([(~mask * 255).astype(np.uint8)] * 3, -1)
        rows.append(np.concatenate([m, viz(det), viz(txt), viz(tot)], 1))
    Image.fromarray(np.concatenate(rows, 0)).save("out/diag_mask.png")
    print("\nout/diag_mask.png  cols: swisstopo | deterministic | text | union")

if __name__ == "__main__":
    main()
