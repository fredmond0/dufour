"""Remove baked-in map lettering from training targets using CRAFT neural text detection.

A DEM cannot predict place names or spot heights, so lettering left in the target
forces the generator to emit text-shaped noise smudges.

We use a CRAFT-based scene text detector (dufour.ocr) to accurately box lettering
and spot heights without chewing into alpine rock faces or hachure fields.
"""
import numpy as np
from .ocr import detect as ocr_detect


def text_mask(rgb, **kw):
    """Detect lettering via neural CRAFT detector."""
    m, _ = ocr_detect(rgb)
    return m


def inpaint(rgb, mask, iters=48):
    """Legacy diffusion inpainting fallback."""
    from scipy.ndimage import gaussian_filter
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
