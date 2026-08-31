#!/usr/bin/env python3
"""Harvest aligned (DEM, swisstopo PK25) tile pairs over the Swiss Alps.

The validation split is held out by REGION, not at random. Random splits leak:
neighbouring tiles share terrain, so a random val set mostly measures
memorisation. Holding out whole massifs asks the real question -- does this
transfer to mountains it has never seen?
"""
import argparse, json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.fetch import map_tile, dem_tile
from dufour.quality import accept
from dufour.tiles import tiles_in_bbox

# w, s, e, n
REGIONS = {
    "zermatt":    (7.50, 45.93, 7.95, 46.15),
    "mischabel":  (7.85, 46.00, 8.10, 46.20),
    "jungfrau":   (7.85, 46.42, 8.20, 46.62),
    "grimsel":    (8.15, 46.50, 8.50, 46.72),
    "bernina":    (9.75, 46.32, 10.10, 46.50),
    "silvretta":  (10.00, 46.75, 10.25, 46.95),
    "uri":        (8.40, 46.70, 8.80, 46.92),
    "gotthard":   (8.40, 46.50, 8.72, 46.68),
    "dentblanche":(7.35, 46.02, 7.62, 46.18),
    "adula":      (8.95, 46.40, 9.30, 46.58),
}
HOLDOUT = ("bernina", "uri")     # never seen during training


def probe(t):
    z, x, y = t
    m = map_tile(z, x, y)
    if m is None:
        return None
    d = dem_tile(z, x, y)
    if d is None:
        return None
    ok, info = accept(m, d)
    return (t, ok, info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=15)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-per-region", type=int, default=900)
    a = ap.parse_args()

    train, val, stats = [], [], {}
    for name, bbox in REGIONS.items():
        cand = list(tiles_in_bbox(*bbox, a.zoom))[: a.limit_per_region]
        with ThreadPoolExecutor(a.workers) as ex:
            res = [r for r in ex.map(probe, cand) if r]
        good = [t for t, ok, _ in res if ok]
        stats[name] = (len(good), len(res))
        (val if name in HOLDOUT else train).extend(good)
        print(f"  {name:12s} {len(good):5d}/{len(res):5d} accepted"
              f"{'   [HELD OUT]' if name in HOLDOUT else ''}", flush=True)

    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/train.json").write_text(json.dumps(train))
    pathlib.Path("data/val.json").write_text(json.dumps(val))
    print(f"\ntrain {len(train)}  val {len(val)}  (val = {'+'.join(HOLDOUT)})")


if __name__ == "__main__":
    main()
