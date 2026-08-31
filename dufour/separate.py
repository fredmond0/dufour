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


def chroma(rgb):
    """Channel spread. The single most useful discriminator in this whole file.

    swisstopo's relief plate is a DESATURATED blue-grey (#9ba6ad, spread 18);
    its coloured ink is saturated (contour #9d8c68 spread 53, water-line
    #4d7f99 spread 76). Colour distance alone confuses the two -- the shading
    grey sits within 36 units of the water blue -- and the result is that a
    4478 m rock face gets masked as "water" and "forest". Requiring real
    saturation separates them cleanly."""
    f = rgb.astype(np.int16)
    return (f.max(-1) - f.min(-1))


def _neutral(rgb, tol=26):
    return chroma(rgb) <= tol


def deterministic_mask(rgb, thresh=38, min_chroma=25, grow=1):
    """Pixels owned by the deterministic renderer."""
    m = np.zeros(rgb.shape[:2], bool)
    for colours in DETERMINISTIC.values():
        m |= _dist(rgb, colours) < thresh
    # Two independent guards: the pixel must be close to a known ink colour
    # AND actually saturated. Either alone lets the relief plate through.
    m &= chroma(rgb) >= min_chroma
    m &= ~_neutral(rgb)
    if grow:
        m = binary_dilation(m, np.ones((2 * grow + 1, 2 * grow + 1)))
    return m


def ignore_mask(rgb, z=None, x=None, y=None):
    """Pixels the generator must NOT be supervised on.

    Everything the deterministic renderer owns (contours, water, forest tint,
    route ink) plus all lettering. We do not inpaint these any more: diffusion
    fill leaves smooth grey discs, and an L1 loss over thousands of such discs
    teaches the network that rock faces contain textureless blobs. Masking the
    loss instead removes the pixels from supervision entirely, with no
    synthetic texture for the model to imitate.
    """
    if z is not None:
        # Use the precomputed OCR mask. Running CRAFT inside the DataLoader
        # costs ~7.4 s/tile versus 21 ms for a cache hit -- with 6 workers that
        # is 0.8 img/s against a GPU that wants 12, i.e. a 15x slowdown, and it
        # loads a copy of the detector into every worker process.
        from .ocr import cached_mask
        tm = cached_mask(z, x, y, rgb)
        if tm is None:
            tm = text_mask(rgb)
    else:
        tm = text_mask(rgb)
    return deterministic_mask(rgb) | tm


def terrain_layer(rgb, drop_text=True):
    """Legacy inpainting path, kept for visual inspection only."""
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
