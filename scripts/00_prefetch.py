#!/usr/bin/env python3
"""Warm the Copernicus and Sentinel-2 caches for the manifest.

Without this the first epoch is network-bound at ~2 s/sample instead of
compute-bound at ~0.1 s/sample."""
import argparse, json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=8.5)
BUDGET = ap.parse_args().minutes * 60
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.copernicus import tile as cop_tile
from dufour.satellite import s2_tile

items = []
for f in ("data/train.json", "data/val.json"):
    items += [tuple(t) for t in json.loads(pathlib.Path(f).read_text())]
print(len(items), "tiles")

s2_need = set()
for z, x, y in items:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            s2_need.add((z, x + dx, y + dy))
print(len(s2_need), "sentinel-2 tiles")

done = [0]
def do_s2(t):
    if time.time() - T0 > BUDGET: return
    s2_tile(*t); done[0] += 1
    if done[0] % 1000 == 0: print("  s2", done[0], flush=True)
with ThreadPoolExecutor(16) as ex:
    list(ex.map(do_s2, sorted(s2_need)))

done[0] = 0
def do_cop(t):
    if time.time() - T0 > BUDGET: return
    cop_tile(*t, pad_px=96); done[0] += 1
    if done[0] % 500 == 0: print("  cop", done[0], flush=True)
with ThreadPoolExecutor(8) as ex:
    list(ex.map(do_cop, items))
print(f"round done in {time.time()-T0:.0f}s")
