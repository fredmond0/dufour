#!/usr/bin/env python3
"""Side-by-side: real swisstopo mosaic vs our render, same frame."""
import argparse, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.frame import Frame
from dufour.fetch import map_tile
from dufour.tiles import TILE_PX

def swisstopo_mosaic(frame):
    out = np.full((frame.height, frame.width, 3), 255, np.uint8)
    for i, j, x, y in frame.tiles:
        t = map_tile(frame.z, x, y)
        if t is not None:
            out[j*TILE_PX:(j+1)*TILE_PX, i*TILE_PX:(i+1)*TILE_PX] = t
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float); p.add_argument("--lon", type=float)
    p.add_argument("--km", type=float, default=6); p.add_argument("--zoom", type=int, default=15)
    p.add_argument("--ours", required=True); p.add_argument("--out", required=True)
    p.add_argument("--crop", type=int, default=0, help="1:1 crop size, 0 = full downscaled")
    a = p.parse_args()
    f = Frame.around(a.lat, a.lon, a.zoom, a.km)
    real = swisstopo_mosaic(f); ours = np.asarray(Image.open(a.ours).convert("RGB"))
    if a.crop:
        c = a.crop; cy, cx = f.height//2, f.width//2
        sl = (slice(cy-c//2, cy+c//2), slice(cx-c//2, cx+c//2))
        real, ours = real[sl], ours[sl]
    else:
        s = 1100/max(f.width,1)
        sz = (int(f.width*s), int(f.height*s))
        real = np.asarray(Image.fromarray(real).resize(sz, Image.LANCZOS))
        ours = np.asarray(Image.fromarray(ours).resize(sz, Image.LANCZOS))
    gap = np.full((real.shape[0], 8, 3), 40, np.uint8)
    im = Image.fromarray(np.concatenate([real, gap, ours], 1))
    d = ImageDraw.Draw(im)
    d.text((10,8), "swisstopo (real)", fill=(200,0,0))
    d.text((real.shape[1]+18,8), "dufour (generated)", fill=(200,0,0))
    im.save(a.out); print("->", a.out, im.size)
