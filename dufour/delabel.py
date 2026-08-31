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


def word_mask(rgb, dark=112, grow=2):
    """Word-level detection: merge glyphs horizontally, then keep runs that
    look like set type.

    The per-glyph pass below misses exactly the labels that matter most --
    bold place names like MATTERHORN, where adjacent letters touch and the
    merged component blows past any single-glyph size limit. Closing the dark
    mask along x turns a word into one elongated blob with a characteristic
    height, which is easy to test for.
    """
    lum = rgb.astype(np.float32).mean(axis=2)
    dark_m = (lum < dark) & (lum < uniform_filter(lum, 31) - 22)
    density = uniform_filter(dark_m.astype(np.float32), 41)
    merged = binary_dilation(dark_m, np.ones((1, 9)))
    lab, n = label(merged)
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if not (5 <= h <= 30):            # cap height of set type at this scale
            continue
        if w < h * 1.1 or w > 240:
            continue
        comp = (lab[sl] == i)
        if comp.mean() < 0.30:
            continue
        if float(density[sl][comp].mean()) > 0.20:
            continue
        keep[i] = True
    m = keep[lab] & binary_dilation(dark_m, np.ones((3, 3)))
    return binary_dilation(m, np.ones((grow * 2 + 1, grow * 2 + 1)))


def text_mask(rgb, dark=112, min_area=8, max_area=330, max_side=26,
              min_fill=0.32, max_density=0.15, grow=2):
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
    m = binary_dilation(m, np.ones((grow * 2 + 1, grow * 2 + 1)))
    # Union with the word-level pass. Thresholds here are deliberately looser
    # than a precision-first detector would use: the training loss now MASKS
    # these pixels rather than inpainting them, so over-masking merely discards
    # a little supervision while under-masking teaches the model to draw text.
    return m | word_mask(rgb, dark=dark, grow=grow)


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
