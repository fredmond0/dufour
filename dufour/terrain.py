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


def lighten_ice(rgb, ice_mask, amount=0.72):
    """Glaciers print as near-white with blue form lines, not grey shading."""
    if ice_mask is None or not ice_mask.any():
        return rgb
    out = rgb.astype(np.float32)
    tgt = np.array([250.0, 252.0, 254.0])
    m = ice_mask[..., None] * amount
    return np.clip(out * (1 - m) + tgt * m, 0, 255).astype(np.uint8)


def swiss_relief(frame, dem, off, ridge=1.35, ridge_m=110.0, zf=1.9,
                 local=0.95, snowline=None):
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
    tone = 0.66 + 0.36 * s
    lo, hi = np.percentile(h, 2), np.percentile(h, 98)
    alt = np.clip((h - lo) / max(hi - lo, 1e-3), 0, 1)
    tone = tone * (0.93 + 0.11 * alt)              # aerial perspective
    # swisstopo's plate is high-key: even deep shadow sits around 60-65% grey,
    # never the 40% this used to bottom out at. Side by side against the real
    # sheet the old floor read as muddy brown where swisstopo reads as light
    # cool grey, and that single number was the largest visual difference.
    tone = np.clip(tone, 0.60, 1.0)

    if snowline is not None:
        snow = np.clip((h - snowline) / 350.0, 0, 1)
        tone = tone * (1 - snow) + np.clip(tone * 1.06 + 0.04, 0, 1) * snow

    # swisstopo's sheet is COOL almost everywhere, with warmth only low down.
    # Weighting these evenly gave the whole plate a beige cast that read as
    # obviously wrong beside the real sheet.
    # Near-neutral throughout. Any appreciable warm bias reads as beige next to
    # swisstopo's cool grey plate, which was the last obvious tell.
    warm = np.stack([tone * 1.000, tone * 0.999, tone * 0.994], -1)
    cool = np.stack([tone * 0.972, tone * 0.986, tone * 1.000], -1)
    m = np.clip(alt[..., None] * 1.45, 0, 1)
    rgb = (1 - m) * warm + m * cool
    return np.clip(rgb * 254, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
def rock_hachures(frame, dem, off, s2=None, slope_min_deg=27.0,
                  spacing=4.9, max_len=13, step=1.1, seed=7,
                  strength=1.0, veg_cut=0.54, ice_mask=None):
    """Swiss-style rock drawing (Felszeichnung), derived rather than learned.

    Swisstopo's rock faces are drawn as fine strokes running down the fall
    line: they describe form through direction and shade through density and
    weight. That is a rule a cartographer follows, so it can be computed --
    which also makes it exact and stable, unlike a GAN that has to guess where
    every stroke goes from a 30 m DEM.

      where   slopes above ~27 degrees that Sentinel-2 says are not vegetated,
              minus mapped glaciers. The threshold is calibrated, not guessed:
              against swisstopo's own drawn rock the median terrain slope here
              is 30 degrees, and 27 recovers ~80% of the hatched area
      shape   each stroke traces steepest descent, so strokes follow aretes
              and gullies instead of lying in one direction
      weight  keyed to the NW key light: strokes darken and lengthen on shaded
              flanks and fade on lit ones, which is what makes the rock read
              as solid rather than as hatching pasted on top

    Returns (strokes, rock_mask): strokes are polylines in frame pixels with a
    per-stroke ink value, ready for dufour.render to draw.
    """
    mpp = frame.mpp
    d = gaussian_filter(normalize_resolution(dem, mpp), max((22.0 / mpp) / 2.355, 0.6))
    gy, gx = np.gradient(d, mpp)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    hs = hillshade(d, mpp, 315, 45, zf=1.6)

    H, W = frame.height, frame.width
    sl = slope[off:off + H, off:off + W]
    sh = hs[off:off + H, off:off + W]

    rock = sl >= slope_min_deg
    if s2 is not None:
        from .satellite import channels as s2ch
        ch = s2ch(s2)
        green, bright = ch[0], ch[1]
        if green.shape != (H, W):                   # s2 arrives padded
            o = (green.shape[0] - H) // 2
            green = green[o:o + H, o:o + W]; bright = bright[o:o + H, o:o + W]
        rock &= green < veg_cut
        # No brightness gate: measured against swisstopo's own hachured area,
        # S2 brightness does not separate rock from snow at all (median 0.51
        # inside the drawn rock versus 0.53 outside). Glaciers are excluded
        # properly below, from OSM geometry.
    if ice_mask is not None:
        rock &= ~ice_mask
    if not rock.any():
        return [], rock

    # Seed on a jittered grid: a regular lattice reads as wallpaper.
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[2:H - 2:spacing, 2:W - 2:spacing]
    ys = ys.ravel() + rng.uniform(-spacing / 2, spacing / 2, ys.size)
    xs = xs.ravel() + rng.uniform(-spacing / 2, spacing / 2, xs.size)
    yi = np.clip(ys.astype(int), 0, H - 1); xi = np.clip(xs.astype(int), 0, W - 1)
    keep = rock[yi, xi]
    # Denser on steep and on shaded ground, thinner on lit faces
    p = np.clip((sl[yi, xi] - slope_min_deg) / 20.0, 0.10, 1.0) * (1.20 - 0.85 * sh[yi, xi])
    keep &= rng.random(ys.size) < np.clip(p * strength, 0, 1)
    ys, xs = ys[keep], xs[keep]
    if ys.size == 0:
        return [], rock

    # March each seed down the fall line.
    gxf = gx[off:off + H, off:off + W]
    gyf = gy[off:off + H, off:off + W]
    n = ys.size
    L = np.clip((sl[np.clip(ys.astype(int), 0, H - 1),
                    np.clip(xs.astype(int), 0, W - 1)] - slope_min_deg) / 24.0, 0.3, 1.0)
    L = (4 + L * (max_len - 4)) * (0.75 + 0.5 * (1 - sh[np.clip(ys.astype(int), 0, H - 1),
                                                        np.clip(xs.astype(int), 0, W - 1)]))
    steps = int(max_len / step) + 1
    px = np.empty((steps, n)); py = np.empty((steps, n))
    cy, cx = ys.copy(), xs.copy()
    for s in range(steps):
        py[s] = cy; px[s] = cx
        u = map_coordinates(gxf, [cy, cx], order=1, mode="nearest")
        v = map_coordinates(gyf, [cy, cx], order=1, mode="nearest")
        m = np.hypot(u, v) + 1e-6
        cy = np.clip(cy + step * (v / m), 0, H - 1.01)   # descend
        cx = np.clip(cx + step * (u / m), 0, W - 1.01)

    tone = sh[np.clip(ys.astype(int), 0, H - 1), np.clip(xs.astype(int), 0, W - 1)]
    # Crisper: swisstopo's rock reads as distinct dark marks with white
    # between them, not an even grey wash, so ink runs darker and the seeds sit
    # slightly further apart to leave that white showing through.
    ink = np.clip(0.34 + 0.62 * (1 - tone), 0, 1)

    strokes = []
    nsteps = np.clip((L / step).astype(int), 3, steps)
    for i in range(n):
        k = nsteps[i]
        strokes.append((list(zip(px[:k, i], py[:k, i])), float(ink[i])))
    return strokes, rock
