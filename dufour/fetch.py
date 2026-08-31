"""Tile fetching with an on-disk cache.

Two sources, both free and both on the standard XYZ grid:
  swisstopo  national map 1:25'000 raster  (target -- Swiss OpenData, CC-BY)
  terrarium  AWS elevation-tiles-prod DEM  (input  -- public, no key)
"""
import io, pathlib, time, urllib.error, urllib.request
import numpy as np
from PIL import Image

CACHE = pathlib.Path("data/tiles")
UA = {"User-Agent": "dufour/0.1 (cartography research)"}

SWISSTOPO = ("https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe-pk25"
             ".noscale/default/current/3857/{z}/{x}/{y}.jpeg")
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def _get(url, dest, tries=3):
    if dest.exists():
        return dest.read_bytes()
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            data = urllib.request.urlopen(req, timeout=30).read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (i + 1))
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def map_tile(z, x, y):
    """swisstopo PK25 RGB, uint8 (256,256,3), or None."""
    d = _get(SWISSTOPO.format(z=z, x=x, y=y), CACHE / f"map/{z}/{x}/{y}.jpeg")
    if d is None:
        return None
    return np.asarray(Image.open(io.BytesIO(d)).convert("RGB"))


def dem_tile(z, x, y):
    """Elevation in metres, float32 (256,256), or None.

    Terrarium encoding: h = R*256 + G + B/256 - 32768
    """
    d = _get(TERRARIUM.format(z=z, x=x, y=y), CACHE / f"dem/{z}/{x}/{y}.png")
    if d is None:
        return None
    a = np.asarray(Image.open(io.BytesIO(d)).convert("RGB")).astype(np.float32)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def dem_tile_padded(z, x, y, pad_tiles=1):
    """DEM for (z,x,y) plus a ring of neighbours, so gradient/curvature
    filters have real data at the edges instead of clamped nonsense.
    Returns (256*(2p+1))^2 array; the centre 256x256 is the tile itself."""
    n = 2 * pad_tiles + 1
    out = np.full((256 * n, 256 * n), np.nan, np.float32)
    for j, dy in enumerate(range(-pad_tiles, pad_tiles + 1)):
        for i, dx in enumerate(range(-pad_tiles, pad_tiles + 1)):
            t = dem_tile(z, x + dx, y + dy)
            if t is not None:
                out[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = t
    if np.isnan(out).any():
        m = np.nanmedian(out)
        out = np.nan_to_num(out, nan=0.0 if np.isnan(m) else m)
    return out
