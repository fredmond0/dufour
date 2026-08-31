"""Training-tile selection.

The swisstopo raster has place names, spot heights and route lines baked in.
Nothing in a DEM predicts the string "Solvaybiwak SAC", so a model trained on
label-bearing tiles learns to emit text-shaped smudges. Two defences:
  1. reject tiles carrying much lettering        (here)
  2. inpaint the small glyphs that remain        (dufour.delabel)
We also skew selection towards high-alpine rock and ice, which is both where
the swisstopo look is most striking and where labels are naturally sparsest.
"""
import numpy as np

from .delabel import text_mask  # noqa: F401  (OCR; not used for harvesting)


def text_score(rgb):
    """Cheap lettering proxy for HARVEST-TIME filtering only.

    Deliberately not the CRAFT detector: harvesting probes ~8000 candidate
    tiles, and at ~7 s/tile that turns a minutes-long job into most of a day.
    Precision does not matter here -- we only need to skip obviously
    label-heavy tiles, and the real mask is computed later by dufour.ocr."""
    from scipy.ndimage import binary_erosion
    dark = rgb.max(axis=2) < 90
    return float(binary_erosion(dark, np.ones((3, 3))).mean())


def colour_flags(rgb):
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    return {
        "magenta": float(((r > 150) & (b > 120) & (g < r - 45)).mean()),
        "forest":  float(((g > r + 12) & (g > b + 12)).mean()),
        "white":   float(((r > 235) & (g > 235) & (b > 235)).mean()),
        "built":   float(((r < 70) & (g < 70) & (b < 70)).mean()),
    }


def accept(rgb, dem, min_elev=1500, max_text=0.045, max_magenta=0.03,
           max_white=0.95, max_built=0.10, min_relief=100):
    f = colour_flags(rgb)
    t = text_score(rgb)
    info = dict(text=t, elev=float(dem.mean()),
                relief=float(dem.max() - dem.min()), **f)
    ok = (t <= max_text and f["magenta"] <= max_magenta
          and f["white"] <= max_white and f["built"] <= max_built
          and info["elev"] >= min_elev and info["relief"] >= min_relief)
    return ok, info
