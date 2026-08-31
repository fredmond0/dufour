"""Remove baked-in map lettering from the training targets.

A DEM cannot predict the string "Solvaybiwak SAC" or the number 2489, so any
lettering left in the target teaches the generator to emit text-shaped noise.
We detect glyph-like blobs and diffusion-fill them from surrounding map, so
the model is trained on an (almost) label-free national map.

Glyphs vs. map furniture:
  contour lines   huge, extremely elongated connected components
  rock hachures   thin strokes, but crucially they come in DENSE FIELDS
  glyphs / dots   compact, solid, and ISOLATED on light background

The isolation test does the heavy lifting: a component only counts as text if
the neighbourhood around it is mostly clear. That is what separates a "4" from
a hachure stroke of identical size and darkness.
"""
import numpy as np
from scipy.ndimage import (binary_dilation, gaussian_filter, label,
                           find_objects, uniform_filter)


def text_mask(rgb, dark=118, min_area=8, max_area=260, max_side=22,
              min_fill=0.34, max_density=0.13, grow=2):
    lum = rgb.astype(np.float32).mean(axis=2)
    local_bg = uniform_filter(lum, 31)
    dark_m = (lum < dark) & (lum < local_bg - 22)

    # local dark-fraction: high inside hachure fields, low around lettering
    density = uniform_filter(dark_m.astype(np.float32), 31)

    lab, n = label(dark_m)
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == i).sum())
        if not (min_area <= area <= max_area):
            continue
        if max(h, w) > max_side:
            continue
        if area / float(h * w) < min_fill:
            continue
        comp = (lab[sl] == i)
        if float(density[sl][comp].mean()) > max_density:
            continue                       # sitting in a hachure field
        keep[i] = True
    m = keep[lab]
    return binary_dilation(m, np.ones((grow * 2 + 1, grow * 2 + 1)))


def inpaint(rgb, mask, iters=48):
    """Cheap anisotropic-free diffusion fill: repeatedly blur and re-assert
    the known pixels. Good enough -- we only need the hole to stop looking
    like a letter, not to reconstruct the true map underneath."""
    out = rgb.astype(np.float32).copy()
    if not mask.any():
        return rgb
    known = ~mask
    for c in range(3):
        ch = out[..., c]
        fill = float(ch[known].mean())
        ch[mask] = fill
        for _ in range(iters):
            ch = gaussian_filter(ch, 1.2)
            ch[known] = out[..., c][known]
        out[..., c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


def clean(rgb, **kw):
    m = text_mask(rgb, **kw)
    return inpaint(rgb, m), m
