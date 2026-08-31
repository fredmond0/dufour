#!/usr/bin/env python3
"""Compose a full map sheet for any bbox on Earth.

    terrain base  <- learned model (or analytic hillshade with --analytic)
    contours      <- DEM, exact
    landcover     <- OSM, exact
    roads/trails  <- OSM + LK25 legend, exact
    buildings     <- OSM, exact
    labels        <- OSM names + OSM/DEM elevations, exact
"""
import argparse, pathlib, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.frame import Frame
from dufour.osm import fetch
from dufour.render import render_vectors, draw_labels, glacier_mask
from dufour.copernicus import sample as cop_sample
from dufour.terrain import analytic_relief, contours, dem_mosaic, swiss_relief


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--km", type=float, default=8)
    p.add_argument("--zoom", type=int, default=15)
    p.add_argument("--out", default="out/map.png")
    p.add_argument("--analytic", action="store_true",
                   help="analytic hillshade base instead of the learned model")
    p.add_argument("--ckpt", default="out/ckpt/g_latest.pt")
    p.add_argument("--no-labels", action="store_true")
    a = p.parse_args()

    f = Frame.around(a.lat, a.lon, a.zoom, a.km)
    print(f, flush=True)

    # Copernicus GLO-30, the SAME source the model was conditioned on. The
    # terrain-tile mosaic used during harvesting is a patchwork of resolutions;
    # feeding it here would hand the network out-of-distribution input and undo
    # the whole point of training on a uniform global DEM.
    t = time.time(); dem, off = cop_sample(f, pad_px=256)
    print(f"  dem      {time.time()-t:5.1f}s  (Copernicus GLO-30)", flush=True)
    t = time.time(); osm = fetch(f.bbox());     print(f"  osm      {time.time()-t:5.1f}s  "
                                                      f"{len(osm['elements'])} elements", flush=True)
    ice = glacier_mask(f, osm)
    t = time.time(); cs = contours(f, dem, off, ice_mask=ice)
    print(f"  contours {time.time()-t:5.1f}s  {len(cs)} lines", flush=True)

    t = time.time()
    if a.analytic or not pathlib.Path(a.ckpt).exists():
        if not a.analytic:
            print("  (no checkpoint yet -> analytic relief)", flush=True)
        base = swiss_relief(f, dem, off)
    else:
        from dufour.infer import learned_relief
        base = learned_relief(f, dem, off, a.ckpt)
    print(f"  relief   {time.time()-t:5.1f}s", flush=True)

    t = time.time(); img = render_vectors(f, osm, base=base, contours=cs)
    print(f"  vectors  {time.time()-t:5.1f}s", flush=True)
    if not a.no_labels:
        t = time.time(); img = draw_labels(f, osm, dem, off, img)
        print(f"  labels   {time.time()-t:5.1f}s", flush=True)

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(a.out)
    print(f"-> {a.out}  {img.shape[1]}x{img.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
