#!/usr/bin/env python3
"""Precompute OCR lettering masks for the manifest (parallel, resumable)."""
import argparse, json, pathlib, sys, time
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def work(t):
    from dufour.ocr import cached_mask
    try:
        m = cached_mask(*t)
        return 0.0 if m is None else float(m.mean())
    except Exception:
        return -1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=8.5)
    a = ap.parse_args()
    items = []
    for f in ("data/train.json", "data/val.json"):
        items += [tuple(t) for t in json.loads(pathlib.Path(f).read_text())]
    todo = [t for t in items
            if not pathlib.Path(f"data/textmask/{t[0]}/{t[1]}/{t[2]}.png").exists()]
    print(f"{len(items)} tiles, {len(todo)} still to do", flush=True)
    t0 = time.time(); done = []
    with Pool(a.procs) as p:
        for i, r in enumerate(p.imap_unordered(work, todo, chunksize=4), 1):
            done.append(r)
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate:.1f} tiles/s  "
                      f"eta {(len(todo)-i)/max(rate,1e-6)/60:.1f} min", flush=True)
            if time.time() - t0 > a.minutes * 60:
                print("  time budget reached; rerun to continue", flush=True)
                p.terminate(); break
    ok = [d for d in done if d >= 0]
    if ok:
        import statistics
        print(f"done {len(ok)}  mean mask {100*statistics.mean(ok):.2f}% of pixels")

if __name__ == "__main__":
    main()
