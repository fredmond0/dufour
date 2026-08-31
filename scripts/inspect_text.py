#!/usr/bin/env python3
"""Inspect lettering detection on real training tiles.

Two ways to check it, because they catch different failures:

  visual  contact sheet -- original | mask in red | what the loss actually sees.
          False positives (mask eating rock hachures) are obvious here.

  audit   quantitative recall. We do not have ground-truth text boxes, but OSM
          knows where the named features are, and swisstopo prints a label next
          to each one. So: for every named peak/hut/village in a tile, look for
          mask coverage within a short radius of its position. Low coverage
          means we are missing real lettering -- the failure that actually
          poisons training.

Usage
  python3 scripts/inspect_text.py visual --n 18 --sort worst
  python3 scripts/inspect_text.py audit  --n 120
"""
import argparse, json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.delabel import text_mask, word_mask
from dufour.fetch import map_tile
from dufour.frame import Frame
from dufour.separate import deterministic_mask
from dufour.tiles import tile_bounds

OUT = pathlib.Path("out/inspect")


def load(split="train"):
    return [tuple(t) for t in json.loads(pathlib.Path(f"data/{split}.json").read_text())]


def overlay(rgb, mask, colour=(230, 30, 30), alpha=0.55):
    o = rgb.astype(np.float32).copy()
    for c in range(3):
        o[..., c] = np.where(mask, o[..., c] * (1 - alpha) + colour[c] * alpha,
                             o[..., c])
    return o.astype(np.uint8)


def loss_view(rgb, mask):
    """What the generator is actually supervised on. Masked pixels are shown
    as flat mid-grey because they contribute nothing to the loss."""
    o = rgb.copy()
    o[mask] = 150
    return o


def cmd_visual(a):
    items = load(a.split)
    rng = np.random.default_rng(a.seed)
    pick = [items[i] for i in rng.choice(len(items), min(a.pool, len(items)), replace=False)]
    scored = []
    for z, x, y in pick:
        m = map_tile(z, x, y)
        if m is None:
            continue
        tm = text_mask(m)
        scored.append((float(tm.mean()), (z, x, y), m, tm))
    scored.sort(key=lambda r: -r[0] if a.sort == "worst" else r[0])
    if a.sort == "random":
        rng.shuffle(scored)
    scored = scored[:a.n]

    rows = []
    for frac, (z, x, y), m, tm in scored:
        det = deterministic_mask(m)
        panel = np.concatenate([m, overlay(m, tm), loss_view(m, tm | det)], 1)
        im = Image.fromarray(panel)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 250, 13], fill=(255, 255, 255))
        d.text((3, 2), f"{z}/{x}/{y}   text {frac*100:.1f}%   det {det.mean()*100:.1f}%",
               fill=(180, 0, 0))
        rows.append(np.asarray(im))
    grid = np.concatenate(rows, 0)
    head = Image.new("RGB", (grid.shape[1], 20), (255, 255, 255))
    hd = ImageDraw.Draw(head)
    for i, t in enumerate(["ORIGINAL", "TEXT MASK (red)", "WHAT THE LOSS SEES (grey = ignored)"]):
        hd.text((i * 256 + 4, 5), t, fill=(0, 0, 170))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"text_{a.sort}_{a.n}.png"
    Image.fromarray(np.concatenate([np.asarray(head), grid], 0)).save(p)
    print(f"-> {p}")


def cmd_audit(a):
    from dufour.osm import fetch, nodes
    items = load(a.split)
    rng = np.random.default_rng(a.seed)
    pick = [items[i] for i in rng.choice(len(items), min(a.n, len(items)), replace=False)]

    hit = miss = 0
    missed_examples = []
    for z, x, y in pick:
        m = map_tile(z, x, y)
        if m is None:
            continue
        w, s, e, n = tile_bounds(x, y, z)
        try:
            osm = fetch((w, s, e, n), timeout=60)
        except Exception:
            continue
        named = [(t.get("name"), la, lo) for t, la, lo in nodes(osm) if t.get("name")]
        if not named:
            continue
        tm = text_mask(m)
        f = Frame(z, x, y, 1, 1)
        for name, la, lo in named:
            px, py = f.lonlat_to_px(lo, la)
            px, py = int(px), int(py)
            if not (0 <= px < 256 and 0 <= py < 256):
                continue
            r = a.radius
            win = tm[max(py - r, 0):py + r, max(px - r, 0):px + r]
            if win.size and win.any():
                hit += 1
            else:
                miss += 1
                if len(missed_examples) < 12:
                    missed_examples.append((f"{z}/{x}/{y}", name, px, py))
    tot = hit + miss
    print(f"\nnamed features found in sampled tiles: {tot}")
    if tot:
        print(f"  label detected within {a.radius}px : {hit:4d}  ({100*hit/tot:.1f}%)")
        print(f"  NO mask near the feature          : {miss:4d}  ({100*miss/tot:.1f}%)")
    if missed_examples:
        print("\n  examples with no detected lettering nearby:")
        for t, nm, px, py in missed_examples:
            print(f"    {t:20s} {nm[:28]:30s} at ({px},{py})")
    print("\nCaveat: swisstopo does not label every OSM node, so some 'misses'")
    print("are features the map simply does not name. Treat this as a floor on")
    print("recall, and read the visual sheet for false positives.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("visual"); v.set_defaults(fn=cmd_visual)
    v.add_argument("--n", type=int, default=14)
    v.add_argument("--pool", type=int, default=140)
    v.add_argument("--sort", choices=["worst", "best", "random"], default="worst")
    au = sub.add_parser("audit"); au.set_defaults(fn=cmd_audit)
    au.add_argument("--n", type=int, default=80)
    au.add_argument("--radius", type=int, default=26)
    for q in (v, au):
        q.add_argument("--split", default="train")
        q.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); a.fn(a)
