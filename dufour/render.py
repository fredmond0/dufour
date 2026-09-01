"""Deterministic vector rendering of OSM + DEM into LK25 symbology.

Everything here is exact. No network sees this data; a trail is where OSM says
it is, drawn at the width the legend prescribes. Supersampled 3x and boxed down
so hairlines antialias the way printed line-work does.
"""
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .legend import (AERIAL, AREAS, BUILDING, CONTOUR, C, FONT_STACK, LINES,
                     POWER, RAIL, TEXT, WATER_LINE, mm_to_px, road_spec)

SS = 3  # supersample factor

_FONT_DIRS = ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental",
              "/Library/Fonts", "~/Library/Fonts"]
# Frutiger is swisstopo's face and is not free. Avenir Next is by the same
# designer (Adrian Frutiger) and is the closest thing shipped on macOS.
_PREFER = ["Avenir Next.ttc", "AvenirNext", "Inter", "HelveticaNeue.ttc",
           "Helvetica.ttc", "Arial.ttf"]
_cache = {}


def _font_path():
    import os
    for want in _PREFER:
        for d in _FONT_DIRS:
            for p in glob.glob(os.path.expanduser(f"{d}/{want}*")):
                return p
    return None


def font(pt, mpp, bold=False):
    """pt = printed points at 1:25'000; converted to device px via the frame."""
    px = int(round(mm_to_px(pt * 0.3527, mpp) * SS))
    key = (px, bold)
    if key in _cache:
        return _cache[key]
    p = _font_path()
    try:
        f = ImageFont.truetype(p, px, index=1 if bold else 0)
    except Exception:
        try:
            f = ImageFont.truetype(p, px)
        except Exception:
            f = ImageFont.load_default()
    _cache[key] = f
    return f


# --------------------------------------------------------------------------
def _px(frame, geom):
    lon = [g["lon"] for g in geom]; lat = [g["lat"] for g in geom]
    x, y = frame.lonlat_to_px(lon, lat)
    return list(zip(x * SS, y * SS))


def _dash(pts, on_px, off_px):
    """Split a polyline into dash segments of the given device lengths."""
    out, seg, dist, drawing = [], [], 0.0, True
    if len(pts) < 2:
        return out
    seg.append(pts[0])
    for a, b in zip(pts, pts[1:]):
        ax, ay = a; bx, by = b
        L = np.hypot(bx - ax, by - ay)
        t = 0.0
        while t < L:
            want = (on_px if drawing else off_px) - dist
            step = min(want, L - t)
            t += step; dist += step
            px_, py_ = ax + (bx - ax) * t / L, ay + (by - ay) * t / L
            if drawing:
                seg.append((px_, py_))
            if dist >= (on_px if drawing else off_px) - 1e-6:
                if drawing and len(seg) > 1:
                    out.append(seg)
                drawing = not drawing
                seg = [(px_, py_)]
                dist = 0.0
    if drawing and len(seg) > 1:
        out.append(seg)
    return out


def _line(d, pts, colour, width, dash=None, mpp=1.0):
    if len(pts) < 2:
        return
    w = max(int(round(width)), 1)
    if dash:
        on = mm_to_px(dash[0], mpp) * SS; off = mm_to_px(dash[1], mpp) * SS
        for s in _dash(pts, on, off):
            d.line(s, fill=colour, width=w, joint="curve")
    else:
        d.line(pts, fill=colour, width=w, joint="curve")


def _ticks(d, pts, colour, half_px, every_px):
    """Cross-ticks for railways, pylons for cableways/power lines."""
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        ax, ay = a; bx, by = b
        L = np.hypot(bx - ax, by - ay)
        if L < 1e-6:
            continue
        ux, uy = (bx - ax) / L, (by - ay) / L
        nx, ny = -uy, ux
        t = every_px - acc
        while t < L:
            cx, cy = ax + ux * t, ay + uy * t
            d.line([(cx - nx * half_px, cy - ny * half_px),
                    (cx + nx * half_px, cy + ny * half_px)],
                   fill=colour, width=max(int(round(half_px * 0.55)), 1))
            t += every_px
        acc = (acc + L) % every_px


# --------------------------------------------------------------------------
def render_vectors(frame, osm, base=None, contours=None):
    """Draw all deterministic layers over `base` (H,W,3 uint8) and return RGB."""
    mpp = frame.mpp
    W, H = frame.width * SS, frame.height * SS
    if base is None:
        img = Image.new("RGB", (W, H), (255, 255, 255))
    else:
        img = Image.fromarray(base).resize((W, H), Image.BICUBIC)
    d = ImageDraw.Draw(img, "RGBA")

    from .osm import ways
    items = list(ways(osm))

    # 1. area fills ---------------------------------------------------------
    areas = []
    for tags, geom in items:
        for (k, v), spec in AREAS.items():
            if tags.get(k) == v and len(geom) >= 3:
                areas.append((spec["order"], spec, geom))
    for _, spec, geom in sorted(areas, key=lambda a: a[0]):
        pts = _px(frame, geom)
        if len(pts) < 3:
            continue
        # semi-transparent so the learned relief shading reads through the fill,
        # exactly as swisstopo prints tints over the shaded base
        d.polygon(pts, fill=spec["fill"] + (168,))
        if spec.get("outline"):
            _line(d, pts + [pts[0]], spec["outline"], mm_to_px(spec["mm"], mpp) * SS)

    # 2. contours (under line-work, over fills) -----------------------------
    if contours is not None:
        draw_contours(d, frame, contours)

    # 3. linear features ----------------------------------------------------
    lines = []
    for tags, geom in items:
        if len(geom) < 2:
            continue
        s = road_spec(tags)
        if s:
            lines.append((s["order"], s, geom, "road"))
        elif tags.get("railway") in RAIL:
            lines.append((RAIL[tags["railway"]]["order"], RAIL[tags["railway"]], geom, "rail"))
        elif tags.get("waterway") in WATER_LINE:
            s = WATER_LINE[tags["waterway"]]
            lines.append((s["order"], s, geom, "water"))
        elif tags.get("aerialway"):
            lines.append((AERIAL["order"], AERIAL, geom, "aerial"))
        elif tags.get("power") == "line":
            lines.append((POWER["order"], POWER, geom, "power"))
    lines.sort(key=lambda a: a[0])

    # casings first for the whole road set, so junctions merge cleanly
    for _, s, geom, kind in lines:
        if kind == "road" and s.get("casing"):
            pts = _px(frame, geom)
            w = (mm_to_px(s["mm"], mpp) + 2 * mm_to_px(s["casing_mm"], mpp)) * SS
            _line(d, pts, s["casing"], w)
    for _, s, geom, kind in lines:
        pts = _px(frame, geom)
        w = mm_to_px(s["mm"], mpp) * SS
        _line(d, pts, s["fill"], w, dash=s.get("dash"), mpp=mpp)
        if kind == "rail" and s.get("ticks"):
            _ticks(d, pts, s["fill"], w * 1.7, mm_to_px(1.6, mpp) * SS)
        if kind in ("aerial", "power"):
            _ticks(d, pts, s["fill"], mm_to_px(s["pylon_mm"], mpp) * SS,
                   mm_to_px(s["pylon_every_mm"], mpp) * SS)

    # 4. buildings ----------------------------------------------------------
    for tags, geom in items:
        if tags.get("building") and len(geom) >= 3:
            pts = _px(frame, geom)
            if len(pts) >= 3:
                d.polygon(pts, fill=BUILDING["fill"])

    return np.asarray(img.resize((frame.width, frame.height), Image.LANCZOS))


def draw_contours(d, frame, contours):
    """contours: list of (level, is_index, is_ice, [(x,y)...]) in frame px."""
    mpp = frame.mpp
    for level, index, ice, pts in contours:
        col = CONTOUR["ice_colour"] if ice else CONTOUR["colour"]
        w = mm_to_px(CONTOUR["index_mm"] if index else CONTOUR["mm"], mpp) * SS
        p = [(x * SS, y * SS) for x, y in pts]
        if len(p) >= 2:
            d.line(p, fill=col + (215,), width=max(int(round(w)), 1), joint="curve")


# --------------------------------------------------------------------------
def draw_labels(frame, osm, dem=None, off=0, img=None):
    """Peak / col / hut names and spot heights, LK25 style.

    Elevation comes from the OSM `ele` tag when present and from the DEM
    otherwise, so summits get a height anywhere in the world -- OSM `ele`
    coverage outside the Alps is thin.
    """
    from .osm import nodes
    mpp = frame.mpp
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    W, H = im.size
    im = im.resize((W * SS, H * SS), Image.BICUBIC)
    d = ImageDraw.Draw(im, "RGBA")
    placed = []

    def free(box, pad=2 * SS):
        x0, y0, x1, y1 = box
        for a, b, c, e in placed:
            if not (x1 + pad < a or x0 - pad > c or y1 + pad < b or y0 - pad > e):
                return False
        return True

    def put(x, y, text, style, anchor="lt"):
        f = font(style["pt"], mpp, style.get("bold", False))
        box = d.textbbox((x, y), text, font=f, anchor=anchor)
        if not free(box):
            return False
        # halo: printed maps knock text out of the background for legibility
        for ox in (-SS, 0, SS):
            for oy in (-SS, 0, SS):
                if ox or oy:
                    d.text((x + ox, y + oy), text, font=f,
                           fill=(255, 255, 255, 205), anchor=anchor)
        d.text((x, y), text, font=f, fill=style["colour"], anchor=anchor)
        placed.append(box)
        return True

    items = []
    for tags, lat, lon in nodes(osm):
        kind = None
        if tags.get("natural") == "peak":
            kind = "peak"
        elif tags.get("natural") == "saddle":
            kind = "saddle"
        elif tags.get("tourism") in ("alpine_hut", "wilderness_hut"):
            kind = "hut"
        elif tags.get("place"):
            kind = "village"
        if kind is None:
            continue
        x, y = frame.lonlat_to_px(lon, lat)
        x, y = float(x), float(y)
        if not (-30 < x < frame.width + 30 and -30 < y < frame.height + 30):
            continue
        ele = tags.get("ele")
        try:
            ele = int(round(float(str(ele).split()[0])))
        except Exception:
            ele = None
        if ele is None and dem is not None:
            iy = int(np.clip(y + off, 0, dem.shape[0] - 1))
            ix = int(np.clip(x + off, 0, dem.shape[1] - 1))
            ele = int(round(float(dem[iy, ix])))
        items.append((kind, x, y, tags.get("name"), ele))

    # tallest first: important summits win the space they need
    items.sort(key=lambda t: -(t[4] or 0) - (5000 if t[0] == "village" else 0))

    for kind, x, y, name, ele in items:
        X, Y = x * SS, y * SS
        if kind in ("peak", "saddle"):
            r = mm_to_px(0.22, mpp) * SS
            d.line([(X - r, Y - r), (X + r, Y + r)], fill=C["black"],
                   width=max(int(SS * 0.7), 1))
            d.line([(X - r, Y + r), (X + r, Y - r)], fill=C["black"],
                   width=max(int(SS * 0.7), 1))
        dx = mm_to_px(0.5, mpp) * SS
        if name:
            put(X + dx, Y - dx, name, TEXT[kind if kind in TEXT else "peak"], "ls")
        if ele is not None and kind in ("peak", "saddle"):
            put(X + dx, Y + dx, str(ele), TEXT["peak_ele"], "lt")

    return np.asarray(im.resize((W, H), Image.LANCZOS))


def glacier_mask(frame, osm):
    """Raster mask of glaciated ground, used to colour contours blue."""
    from .osm import ways
    im = Image.new("L", (frame.width, frame.height), 0)
    d = ImageDraw.Draw(im)
    for tags, geom in ways(osm):
        if tags.get("natural") == "glacier" and len(geom) >= 3:
            lon = [g["lon"] for g in geom]; lat = [g["lat"] for g in geom]
            x, y = frame.lonlat_to_px(lon, lat)
            d.polygon(list(zip(x, y)), fill=255)
    return np.asarray(im) > 127


def draw_hachures(frame, base, strokes, width_mm=0.105):
    """Paint rock strokes onto the relief plate, under all other line work."""
    if not strokes:
        return base
    mpp = frame.mpp
    im = Image.fromarray(base).resize((frame.width * SS, frame.height * SS), Image.BICUBIC)
    d = ImageDraw.Draw(im, "RGBA")
    w = max(int(round(mm_to_px(width_mm, mpp) * SS)), 1)
    for pts, ink in strokes:
        if len(pts) < 2:
            continue
        p = [(x * SS, y * SS) for x, y in pts]
        # Warm-neutral rock ink, alpha carrying the shading weight
        a = int(np.clip(ink, 0, 1) * 255)
        d.line(p, fill=(38, 40, 40, a), width=w, joint="curve")
    return np.asarray(im.resize((frame.width, frame.height), Image.LANCZOS))
