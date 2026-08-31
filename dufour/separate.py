"""Separate the swisstopo raster into a TERRAIN layer and everything else.

Under the hybrid design the network must learn only what cannot be derived from
data: relief shading and rock drawing (Felszeichnung). Contours, water, forest
tint and route lines are all rendered deterministically later, so if they stay
in the training target the network learns to draw a second, slightly wrong copy
of each -- and the finished sheet shows doubled contours.

So we classify each pixel against the measured LK25 ink palette and inpaint
away every class that the deterministic renderer owns.
"""
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, median_filter

from .delabel import inpaint, text_mask

# Measured ink colours (scripts/palette*.py), grouped by who renders them.
DETERMINISTIC = {
    "contour_brown": [(157, 140, 104), (192, 176, 146), (162, 166, 128)],
    "forest_green":  [(177, 199, 166), (201, 220, 175), (163, 180, 149),
                      (131, 141, 101)],
    # Only the darker blues are LINE ink. The pale (197,229,246) blue is the
    # glacier/snow area tint -- that is terrain shading and must survive, or
    # every ice field flattens to dead white.
    "water_blue":    [(77, 127, 153), (123, 166, 189)],
    "route_magenta": [(190, 90, 140), (215, 130, 170)],
    "road_warm":     [(233, 132, 58), (231, 148, 88), (245, 214, 120)],
}
# Kept: neutral greys and near-whites -- the shading and the rock hachures.


def _dist(rgb, colours):
    f = rgb.astype(np.float32)
    d = np.full(f.shape[:2], 1e9, np.float32)
    for c in colours:
        c = np.array(c, np.float32)
        d = np.minimum(d, np.sqrt(((f - c) ** 2).sum(-1)))
    return d


def _neutral(rgb, tol=16):
    """Grey-ish pixels: max channel spread small. These are shading/hachure."""
    f = rgb.astype(np.int16)
    return (f.max(-1) - f.min(-1)) <= tol


def deterministic_mask(rgb, thresh=52, grow=1):
    """Pixels owned by the deterministic renderer."""
    m = np.zeros(rgb.shape[:2], bool)
    for colours in DETERMINISTIC.values():
        m |= _dist(rgb, colours) < thresh
    m &= ~_neutral(rgb)          # never sacrifice grey rock drawing
    if grow:
        m = binary_dilation(m, np.ones((2 * grow + 1, 2 * grow + 1)))
    return m


def terrain_layer(rgb, drop_text=True):
    """Return (terrain_rgb, mask) -- swisstopo with the deterministic ink
    removed, i.e. the relief-and-rock layer the network should learn."""
    m = deterministic_mask(rgb)
    if drop_text:
        m = m | text_mask(rgb)
    out = inpaint(rgb, m, iters=40)
    # The stripped sheet should read as a neutral relief plate; pull the
    # remaining faint colour casts towards grey so the model targets tone.
    f = out.astype(np.float32)
    grey = f.mean(-1, keepdims=True)
    f = f * 0.78 + grey * 0.22
    return np.clip(f, 0, 255).astype(np.uint8), m
