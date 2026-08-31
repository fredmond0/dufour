"""Copernicus DEM GLO-30 as the elevation source.

Why not the terrain tiles used for the training harvest? Those are a patchwork
-- 10 m 3DEP in the USA, SRTM in Patagonia, EU-DEM in the Alps -- so ridge
sharpness varies by continent, which is exactly the wrong property for a model
meant to work anywhere. GLO-30 is TanDEM-X derived, uniformly 30 m worldwide,
free and unauthenticated, and markedly crisper in high mountains than SRTM.

Tiles are 1 deg x 1 deg COGs read over HTTP range requests via /vsicurl, so we
only pull the bytes the frame actually needs.
"""
import os

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")

import threading

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates

BASE = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
        "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
        "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
# GDAL dataset handles are NOT thread-safe. Sharing one /vsicurl handle across
# prefetch threads trips an assertion inside libtiff, so each thread keeps its
# own handle cache.
_local = threading.local()


def _ds(url):
    cache = getattr(_local, "cache", None)
    if cache is None:
        cache = _local.cache = {}
    if url not in cache:
        cache[url] = rasterio.open(url)
    return cache[url]


def _url(lat_deg, lon_deg):
    return BASE.format(ns="N" if lat_deg >= 0 else "S", lat=abs(lat_deg),
                       ew="E" if lon_deg >= 0 else "W", lon=abs(lon_deg))


def available(lat, lon):
    try:
        with rasterio.open(_url(int(np.floor(lat)), int(np.floor(lon)))):
            return True
    except Exception:
        return False


def sample(frame, pad_px=256):
    """Elevation on the frame's pixel grid, padded by pad_px on every side.

    Returns (array, offset) matching dufour.terrain.dem_mosaic's contract."""
    H = frame.height + 2 * pad_px
    W = frame.width + 2 * pad_px
    py, px = np.mgrid[0:H, 0:W].astype(np.float64)
    lon, lat = frame.px_to_lonlat(px - pad_px, py - pad_px)

    out = np.full((H, W), np.nan, np.float32)
    lat_lo, lat_hi = int(np.floor(lat.min())), int(np.floor(lat.max()))
    lon_lo, lon_hi = int(np.floor(lon.min())), int(np.floor(lon.max()))

    for la in range(lat_lo, lat_hi + 1):
        for lo in range(lon_lo, lon_hi + 1):
            u = _url(la, lo)
            try:
                ds = _ds(u)
            except Exception:
                continue
            m = (lat >= la) & (lat < la + 1) & (lon >= lo) & (lon < lo + 1)
            if not m.any():
                continue
            sub_lat, sub_lon = lat[m], lon[m]
            # Clamp to the dataset's real bounds. GLO-30 tiles start at
            # 46.000139 rather than exactly 46.0, so an unclamped window reads
            # outside the raster and leaves a one-tile-wide seam of fill value.
            b = ds.bounds
            win = from_bounds(max(sub_lon.min() - 0.01, b.left),
                              max(sub_lat.min() - 0.01, b.bottom),
                              min(sub_lon.max() + 0.01, b.right),
                              min(sub_lat.max() + 0.01, b.top),
                              ds.transform)
            arr = ds.read(1, window=win).astype(np.float32)
            if arr.size == 0:
                continue
            arr = np.nan_to_num(arr, nan=float(np.nanmedian(arr)))
            tr = ds.window_transform(win)
            # geographic -> array index, then bilinear sample
            col = (sub_lon - tr.c) / tr.a
            row = (sub_lat - tr.f) / tr.e
            out[m] = map_coordinates(arr, [row, col], order=1, mode="nearest")

    if np.isnan(out).any():
        out = np.nan_to_num(out, nan=float(np.nanmedian(out)))
    # Copernicus nodata is a large sentinel; clip before any float16 cast.
    return np.clip(out, -500.0, 9000.0), pad_px


# --- per-tile access for training -----------------------------------------
import pathlib

TILE_CACHE = pathlib.Path("data/cop")


def tile(z, x, y, pad_px=96):
    """Copernicus elevation for one XYZ tile plus `pad_px` context, cached.

    The training harvest used the AWS terrain tiles because they are trivially
    aligned, but the model must be conditioned on the same DEM it will see at
    inference anywhere on Earth -- so features come from GLO-30 everywhere."""
    from .frame import Frame
    dest = TILE_CACHE / f"{z}/{x}/{y}_{pad_px}.npy"
    if dest.exists():
        try:
            return np.load(dest).astype(np.float32)
        except Exception:
            pass
    arr, _ = sample(Frame(z, x, y, 1, 1), pad_px=pad_px)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, arr.astype(np.float16))   # ~0.4 MB/tile instead of 0.8
    return arr
