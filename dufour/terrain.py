"""DEM mosaicking, contour extraction and the analytic relief base.

The analytic hillshade here is the FALLBACK / control condition. The point of
the project is to replace it with the learned Swiss-style relief -- keeping
both in one module makes the comparison a one-line change.
"""
import numpy as np
from contourpy import contour_generator
from scipy.ndimage import gaussian_filter, map_coordinates

from .features import hillshade, normalize_resolution
from .fetch import dem_tile_padded, dem_tile
from .tiles import TILE_PX


def dem_mosaic(frame, pad=1):
    """DEM covering the frame plus `pad` tiles of context on every side."""
    nx, ny = frame.nx + 2 * pad, frame.ny + 2 * pad
    out = np.full((ny * TILE_PX, nx * TILE_PX), np.nan, np.float32)
    for j in range(ny):
        for i in range(nx):
            t = dem_tile(frame.z, frame.x0 - pad + i, frame.y0 - pad + j)
            if t is not None:
                out[j * TILE_PX:(j + 1) * TILE_PX, i * TILE_PX:(i + 1) * TILE_PX] = t
    if np.isnan(out).any():
        out = np.nan_to_num(out, nan=float(np.nanmedian(out)))
    return out, pad * TILE_PX          # array, offset of frame origin


def _chaikin(pts, iters=2):
    """Corner-cutting smoothing -- turns marching-squares staircases into the
    flowing curves a contour is supposed to be."""
    p = pts
    for _ in range(iters):
        if len(p) < 3:
            break
        q = [p[0]]
        for a, b in zip(p, p[1:]):
            q.append(a + 0.25 * (b - a))
            q.append(a + 0.75 * (b - a))
        q.append(p[-1])
        p = np.asarray(q)
    return p


def contours(frame, dem=None, off=None, interval=20, index_every=5,
             smooth_m=55.0, ice_mask=None, min_pts=6):
    """Extract contour polylines in FRAME pixel coordinates.

    The DEM is low-passed before contouring: a 30 m source resampled to 3 m
    produces terraced, noisy isolines otherwise, and swisstopo's are smooth.
    """
    if dem is None:
        dem, off = dem_mosaic(frame)
    mpp = frame.mpp
    d = gaussian_filter(normalize_resolution(dem, mpp), (smooth_m / mpp) / 2.355)

    lo = int(np.floor(d.min() / interval) * interval)
    hi = int(np.ceil(d.max() / interval) * interval)
    cg = contour_generator(z=d, name="serial", line_type="SeparateCode")

    out = []
    for lvl in range(lo, hi + 1, interval):
        try:
            lines = cg.lines(float(lvl))[0]
        except Exception:
            continue
        if lines is None:
            continue
        segs = lines if isinstance(lines, list) else [lines]
        for s in segs:
            s = np.asarray(s, np.float64)
            if len(s) < min_pts:
                continue
            s = _chaikin(s)
            x = s[:, 0] - off
            y = s[:, 1] - off
            keep = ((x > -20) & (x < frame.width + 20) &
                    (y > -20) & (y < frame.height + 20))
            if keep.sum() < min_pts:
                continue
            ice = False
            if ice_mask is not None:
                xi = np.clip(y[keep].astype(int), 0, ice_mask.shape[0] - 1)
                yi = np.clip(x[keep].astype(int), 0, ice_mask.shape[1] - 1)
                ice = bool(ice_mask[xi, yi].mean() > 0.5)
            out.append((lvl, lvl % (interval * index_every) == 0, ice,
                        list(zip(x, y))))
    return out


def analytic_relief(frame, dem=None, off=None):
    """Imhof-flavoured analytic shading: NW key light, softened, with aerial
    perspective lightening the high ground. The baseline the GAN must beat."""
    if dem is None:
        dem, off = dem_mosaic(frame)
    mpp = frame.mpp
    d = normalize_resolution(dem, mpp)
    hs = (0.55 * hillshade(d, mpp, 315, 45, zf=1.6) +
          0.25 * hillshade(d, mpp, 355, 60, zf=1.6) +
          0.20 * hillshade(d, mpp, 275, 35, zf=1.6))
    hs = gaussian_filter(hs, 1.0)
    h = d[off:off + frame.height, off:off + frame.width]
    s = hs[off:off + frame.height, off:off + frame.width]
    lo, hi = np.percentile(h, 3), np.percentile(h, 97)
    alt = np.clip((h - lo) / max(hi - lo, 1e-3), 0, 1)
    tone = 0.62 + 0.38 * s                      # never fully black
    tone = tone * (0.90 + 0.14 * alt)           # aerial perspective
    warm = np.stack([tone * 1.00, tone * 0.995, tone * 0.965], -1)
    cool = np.stack([tone * 0.955, tone * 0.975, tone * 1.00], -1)
    m = alt[..., None]
    rgb = (1 - m) * warm + m * cool             # warm valleys, cool summits
    return np.clip(rgb * 252, 0, 255).astype(np.uint8)


def swiss_relief(frame, dem, off, ridge=0.85, ridge_m=110.0, zf=1.5,
                 local=0.55, snowline=None):
    """Imhof-flavoured relief with explicit ridge enhancement.

    A 30 m DEM shaded naively turns the Matterhorn into a cone: its aretes are
    50-100 m wide, right at the sampling limit, so a plain hillshade averages
    them away. Swiss cartographers solve this by hand -- they exaggerate the
    structural lines and give each arete a bright edge against a dark flank.
    The analytic equivalents:

      1. terrain unsharp mask -- d + k(d - blur(d)) pushes ridges up and
         gullies down before any lighting is computed, so aretes survive;
      2. multi-scale lighting -- fine/mid/coarse hillshades blended, so broad
         massing and fine structure both read;
      3. local contrast on the shading itself, which is what produces the
         characteristic bright-ridge/dark-flank edge.
    """
    mpp = frame.mpp
    d = gaussian_filter(dem, max((18.0 / mpp) / 2.355, 0.5))   # light denoise only

    # 1. ridge enhancement
    base = gaussian_filter(d, (ridge_m / mpp) / 2.355)
    d = d + ridge * (d - base)

    # 2. multi-scale illumination, NW key light
    def sh(src, az, alt, w):
        return w * hillshade(src, mpp, az, alt, zf=zf)
    fine = d
    mid = gaussian_filter(d, max((60.0 / mpp) / 2.355, 0.5))
    coarse = gaussian_filter(d, max((220.0 / mpp) / 2.355, 0.5))
    hs = (sh(fine, 315, 45, 0.34) + sh(fine, 300, 60, 0.14) +
          sh(mid, 315, 45, 0.30) + sh(coarse, 315, 40, 0.22))

    # 3. local contrast -> bright ridge, dark flank
    hs = hs + local * (hs - gaussian_filter(hs, max((260.0 / mpp) / 2.355, 1.0)))
    hs = np.clip(hs, 0, 1)

    s = hs[off:off + frame.height, off:off + frame.width]
    h = dem[off:off + frame.height, off:off + frame.width]

    # tone: swisstopo's plate is high-key -- near-white lit ground, shadows
    # that stop well short of black, never a muddy midtone everywhere
    tone = 0.55 + 0.48 * s
    lo, hi = np.percentile(h, 2), np.percentile(h, 98)
    alt = np.clip((h - lo) / max(hi - lo, 1e-3), 0, 1)
    tone = tone * (0.93 + 0.11 * alt)              # aerial perspective
    tone = np.clip(tone, 0.42, 1.0)

    if snowline is not None:
        snow = np.clip((h - snowline) / 350.0, 0, 1)
        tone = tone * (1 - snow) + np.clip(tone * 1.06 + 0.04, 0, 1) * snow

    warm = np.stack([tone * 1.000, tone * 0.988, tone * 0.958], -1)
    cool = np.stack([tone * 0.958, tone * 0.978, tone * 1.000], -1)
    m = alt[..., None]
    rgb = (1 - m) * warm + m * cool
    return np.clip(rgb * 254, 0, 255).astype(np.uint8)
