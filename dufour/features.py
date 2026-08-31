"""DEM -> conditioning channels for the generator.

Design notes
------------
* We give the network a *multi-azimuth* hillshade stack rather than one
  315-degree hillshade. Swiss relief shading rotates and blends illumination
  locally to keep ridges legible; handing over four lights lets the model
  learn that mixing instead of us hard-coding Imhof's rules.
* Absolute elevation is a channel because treeline / snowline / glacier
  colour all key off it. Without it the model cannot know green from white.
* Everything is low-passed to a fixed GROUND cutoff first, because the
  free global DEM is a patchwork of 10 m / 25 m / 30 m sources and we do
  not want the model keying off source resolution.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from .tiles import meters_per_pixel

CUTOFF_M = 30.0          # GLO-30 native posting
AZIMUTHS = (315, 45, 135, 225)
ALTITUDE = 45.0
ELEV_MAX = 9000.0
CHANNELS = ["hs315", "hs045", "hs135", "hs225", "hs_ridge", "hs_coarse",
            "slope", "elev_abs", "elev_loc", "curv", "aspect_s", "aspect_c",
            "rough", "s2_green", "s2_bright", "s2_tex"]


def normalize_resolution(dem, mpp):
    sigma = (CUTOFF_M / mpp) / 2.355
    return gaussian_filter(dem, sigma) if sigma > 0.5 else dem


def _grad(dem, mpp):
    gy, gx = np.gradient(dem, mpp)
    return gx, gy


def hillshade(dem, mpp, azimuth, altitude=ALTITUDE, zf=1.0):
    gx, gy = _grad(dem * zf, mpp)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    hs = (np.sin(alt) * np.cos(slope) +
          np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)


def stack(dem, lat, z, crop=None, s2=None):
    """dem: padded float32 DEM. Returns (C,H,W) float32 in [-1,1].
    If crop=(y0,y1,x0,x1) the stack is computed on the full padded array and
    then cropped, so edge pixels of the crop see real neighbours."""
    mpp = meters_per_pixel(lat, z)
    d = normalize_resolution(dem, mpp)

    gx, gy = _grad(d, mpp)
    slope_r = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)

    # curvature: laplacian of the smoothed surface, ridge(+)/valley(-)
    ds = gaussian_filter(d, 2.0)
    curv = (np.gradient(np.gradient(ds, mpp, axis=0), mpp, axis=0) +
            np.gradient(np.gradient(ds, mpp, axis=1), mpp, axis=1))

    rough = np.sqrt(np.maximum(uniform_filter(d * d, 15) -
                               uniform_filter(d, 15) ** 2, 0))

    ch = [hillshade(d, mpp, a) for a in AZIMUTHS]

    # Ridge-enhanced and coarse-scale lighting. Aretes are 50-100 m wide, right
    # at GLO-30's sampling limit, so a plain hillshade averages them away; the
    # unsharp-masked surface keeps them, and the coarse light carries the broad
    # massing that tells a summit from a shoulder.
    ridge = d + 0.85 * (d - gaussian_filter(d, (110.0 / mpp) / 2.355))
    ch.append(hillshade(ridge, mpp, 315, zf=1.5))
    ch.append(hillshade(gaussian_filter(d, max((220.0 / mpp) / 2.355, 0.5)),
                        mpp, 315, altitude=40))
    ch.append(slope_r / (np.pi / 2))
    ch.append(np.clip(d / ELEV_MAX, 0, 1))
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    ch.append(np.clip((d - lo) / max(hi - lo, 1e-3), 0, 1))
    ch.append(np.clip(curv * 200.0, -1, 1) * 0.5 + 0.5)
    ch.append(np.sin(aspect) * 0.5 + 0.5)
    ch.append(np.cos(aspect) * 0.5 + 0.5)
    ch.append(np.clip(rough / 80.0, 0, 1))

    if s2 is not None:
        from .satellite import channels as s2_channels
        ch.extend(list(s2_channels(s2)))
    else:
        ch.extend([np.full_like(ch[0], 0.5)] * 3)

    out = np.stack(ch).astype(np.float32)
    if crop is not None:
        y0, y1, x0, x1 = crop
        out = out[:, y0:y1, x0:x1]
    return out * 2.0 - 1.0
