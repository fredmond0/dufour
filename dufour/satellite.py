"""Sentinel-2 imagery as a landcover / high-frequency detail source.

A 30 m DEM cannot tell rock from grass from forest, and it cannot resolve
structure finer than ~60 m. Sentinel-2 is 10 m, global, free, and -- crucially
-- uniform in a way the DEM patchwork is not. It supplies exactly what the DEM
lacks: surface type, and fine structural detail on cliff faces.

EOX s2cloudless is an annual cloud-free Sentinel-2 mosaic served on the same
XYZ grid, so tiles are pixel-aligned with everything else with no warping.
Licence: CC BY-NC-SA 4.0 -- fine for research, NOT for commercial use. Swap in
a Copernicus Dataspace or AWS Sentinel-2 L2A reader if that matters.
"""
import io, pathlib, time, urllib.error, urllib.request
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

CACHE = pathlib.Path("data/tiles/s2")
URL = ("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/"
       "{z}/{y}/{x}.jpg")
UA = {"User-Agent": "dufour/0.1 (cartography research)"}


def s2_tile(z, x, y):
    dest = CACHE / f"{z}/{x}/{y}.jpg"
    if dest.exists():
        try:
            return np.asarray(Image.open(dest).convert("RGB"))
        except Exception:
            dest.unlink(missing_ok=True)
    for i in range(3):
        try:
            req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers=UA)
            d = urllib.request.urlopen(req, timeout=30).read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(d)
            return np.asarray(Image.open(io.BytesIO(d)).convert("RGB"))
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return None
            time.sleep(1.5 * (i + 1))
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def s2_padded(z, x, y, pad_px=96):
    """S2 RGB for a tile plus context, so the texture filter has real
    neighbours instead of clamped edges."""
    n = 256 + 2 * pad_px
    out = np.full((n, n, 3), 128, np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            t = s2_tile(z, x + dx, y + dy)
            if t is None:
                continue
            sy, sx = 256 + dy * 256 - (256 - pad_px), 256 + dx * 256 - (256 - pad_px)
            ty0, tx0 = max(sy, 0), max(sx, 0)
            ty1, tx1 = min(sy + 256, n), min(sx + 256, n)
            if ty1 <= ty0 or tx1 <= tx0:
                continue
            out[ty0:ty1, tx0:tx1] = t[ty0 - sy:ty1 - sy, tx0 - sx:tx1 - sx]
    return out


def channels(rgb):
    """Three conditioning channels in [0,1] from Sentinel-2 RGB.

      greenness  vegetation index proxy -- separates forest/pasture from rock
      brightness snow, ice and pale limestone vs dark shadowed rock
      texture    local high-frequency energy: the detail the DEM cannot resolve,
                 which is what makes a cliff face read as broken rock
    """
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    # s2cloudless is 8-bit JPEG, so the raw index occupies a narrow band --
    # stretch it or the channel arrives at the network almost constant.
    green = np.clip(((g - r) / (g + r + 1e-4)) / 0.28, -1, 1) * 0.5 + 0.5
    bright = f.mean(-1)
    lum = bright
    tex = np.sqrt(np.maximum(uniform_filter(lum * lum, 7) -
                             uniform_filter(lum, 7) ** 2, 0))
    # Lower gain than feels natural: at 0.10 the channel saturates on JPEG
    # block edges and stops carrying real structure.
    tex = np.clip(tex / 0.22, 0, 1)
    return np.stack([np.clip(green, 0, 1), np.clip(bright, 0, 1),
                     gaussian_filter(tex, 0.6)]).astype(np.float32)
