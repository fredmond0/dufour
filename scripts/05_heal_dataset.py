#!/usr/bin/env python3
"""Run LaMa AI inpainting over all tiles to produce clean, label-free RGB targets.

Saves 100% healed ground truth tiles to:
    data/tiles/healed/{z}/{x}/{y}.png

These pre-cleaned images feed directly into the PyTorch DataLoader with zero
runtime overhead during training.
"""
import argparse, json, pathlib, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.fetch import map_tile
from dufour.ocr import cached_mask

DEST_DIR = pathlib.Path("data/tiles/healed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    device = torch.device(a.device)
    print(f"Loading LaMa model onto {device}...")
    lama = torch.jit.load("data/models/big-lama.pt", map_location=device)
    lama.eval()

    items = []
    for f in ("data/train.json", "data/val.json"):
        p = pathlib.Path(f)
        if p.exists():
            items += [tuple(t) for t in json.loads(p.read_text())]

    # De-duplicate while preserving order
    seen = set()
    unique_items = []
    for t in items:
        if t not in seen:
            seen.add(t)
            unique_items.append(t)

    todo = [t for t in unique_items if not (DEST_DIR / f"{t[0]}/{t[1]}/{t[2]}.png").exists()]
    print(f"Total tiles: {len(unique_items)}, To heal/save: {len(todo)}")

    t0 = time.time()
    healed_count = 0
    clean_copy_count = 0

    batch_tiles = []
    batch_rgbs = []
    batch_masks = []

    def flush_batch():
        nonlocal healed_count, clean_copy_count
        if not batch_tiles:
            return

        # Separate into tiles with text (needs LaMa) vs clean tiles
        needs_lama = [i for i, m in enumerate(batch_masks) if m.any()]
        no_lama = [i for i, m in enumerate(batch_masks) if not m.any()]

        # 1. Clean tiles: save original directly
        for i in no_lama:
            z, x, y = batch_tiles[i]
            dest = DEST_DIR / f"{z}/{x}/{y}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(batch_rgbs[i]).save(dest, optimize=True)
            clean_copy_count += 1

        # 2. Tiles with text: run batched LaMa on GPU
        if needs_lama:
            imgs = np.stack([batch_rgbs[i] for i in needs_lama]).astype(np.float32) / 255.0
            masks = np.stack([batch_masks[i] for i in needs_lama]).astype(np.float32)[:, None, ...]

            t_imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
            t_masks = torch.from_numpy(masks).to(device)

            with torch.no_grad():
                t_outs = lama(t_imgs, t_masks)
                if device.type == "mps":
                    torch.mps.synchronize()

            out_rgbs = (t_outs.permute(0, 2, 3, 1).cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)

            for idx, i in enumerate(needs_lama):
                z, x, y = batch_tiles[i]
                dest = DEST_DIR / f"{z}/{x}/{y}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(out_rgbs[idx]).save(dest, optimize=True)
                healed_count += 1

        batch_tiles.clear()
        batch_rgbs.clear()
        batch_masks.clear()

    for idx, t in enumerate(todo, start=1):
        z, x, y = t
        rgb = map_tile(z, x, y)
        if rgb is None:
            continue

        mask = cached_mask(z, x, y, rgb=rgb)
        if mask is None:
            mask = np.zeros(rgb.shape[:2], dtype=bool)

        batch_tiles.append(t)
        batch_rgbs.append(rgb)
        batch_masks.append(mask)

        if len(batch_tiles) >= a.batch_size:
            flush_batch()

        if idx % 100 == 0:
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1e-6)
            eta_min = (len(todo) - idx) / max(rate, 1e-6) / 60
            print(f"  [{idx}/{len(todo)}]  {rate:.1f} tiles/s  (Healed: {healed_count}, Clean: {clean_copy_count})  ETA: {eta_min:.1f} min", flush=True)

    flush_batch()
    print(f"\nDone! Healed {healed_count} text-bearing tiles, saved {clean_copy_count} clean tiles in {(time.time()-t0)/60:.1f} min.")
    print(f"Output directory: {DEST_DIR}")


if __name__ == "__main__":
    main()
