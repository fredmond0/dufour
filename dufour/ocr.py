"""Lettering detection with a real scene-text detector.

The hand-tuned morphology this replaces did close to the opposite of its job:
it masked rock hachures (dark, compact, isolated -- indistinguishable from a
glyph by those rules) while leaving "Zuckerstock" and "MOOSSTOCK" untouched.
A CRAFT-based detector separates the two on shape and stroke statistics that no
threshold on darkness can express.

Confidence does the filtering. On map tiles the real labels come back at 0.5-1.0
and hachures misread as text come back at 0.06-0.20, so a modest threshold keeps
recall high without eating rock.

OCR costs ~0.7 s/tile, far too slow for the training loop, so masks are computed
once and cached as 1-bit PNGs.
"""
import pathlib
import numpy as np
from PIL import Image, ImageDraw

CACHE = pathlib.Path("data/textmask")
_reader = None

# Boxes larger than this fraction of the tile are detector noise, not type.
MAX_BOX_FRAC = 0.10


def reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["de", "fr", "en"], gpu=False, verbose=False)
    return _reader


def detect(rgb, scale=3, min_conf=0.25, strong_conf=0.50):
    """Return (mask, boxes). boxes = [(polygon, text, conf, kept)]."""
    h, w = rgb.shape[:2]
    big = np.asarray(Image.fromarray(rgb).resize((w * scale, h * scale), Image.LANCZOS))
    res = reader().readtext(big, low_text=0.3, text_threshold=0.5, link_threshold=0.3)

    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    boxes = []
    for poly, txt, conf in res:
        p = [(px / scale, py / scale) for px, py in poly]
        xs = [q[0] for q in p]; ys = [q[1] for q in p]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        keep = conf >= min_conf and 4 <= bh <= 46 and bw >= 3
        # A very tall, narrow box over a cliff is a misread hachure, unless the
        # detector is confident (map labels really are sometimes set vertically).
        if keep and bh > bw * 2.2 and conf < strong_conf:
            keep = False
        if keep and (bw * bh) > MAX_BOX_FRAC * w * h:
            keep = False
        if keep:
            d.polygon(p, fill=255)
        boxes.append((p, txt, float(conf), bool(keep)))
    m = np.asarray(im) > 127
    if m.any():
        from scipy.ndimage import binary_dilation
        m = binary_dilation(m, np.ones((5, 5)))
    return m, boxes


def cached_mask(z, x, y, rgb=None):
    dest = CACHE / f"{z}/{x}/{y}.png"
    if dest.exists():
        try:
            return np.asarray(Image.open(dest)) > 127
        except Exception:
            dest.unlink(missing_ok=True)
    if rgb is None:
        from .fetch import map_tile
        rgb = map_tile(z, x, y)
        if rgb is None:
            return None
    m, _ = detect(rgb)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((m * 255).astype(np.uint8)).save(dest, optimize=True)
    return m
