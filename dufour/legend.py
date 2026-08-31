"""swisstopo LK25 symbology as data.

Widths are given in PAPER MILLIMETRES at 1:25'000, which is how the printed
legend actually specifies them. Converting via

    px = mm * 25000 / 1000 / metres_per_pixel

means the same spec renders correctly at any zoom, and it keeps the numbers
comparable with the published legend instead of being magic pixel constants.

Colours marked (measured) were recovered from the raster itself by
scripts/palette.py and scripts/palette_lines.py rather than guessed.
"""

SCALE_DENOM = 25000.0


def mm_to_px(mm, mpp):
    return max(mm * SCALE_DENOM / 1000.0 / mpp, 0.6)


C = {
    "black":        (8, 11, 8),        # (measured) building / path ink
    "rail":         (25, 27, 25),
    "contour":      (157, 140, 104),   # (measured) rock contour brown
    "contour_ice":  (123, 166, 189),   # (measured) glacier contour blue
    "water_line":   (77, 127, 153),    # (measured)
    "water_fill":   (197, 229, 246),   # (measured)
    "forest":       (201, 220, 175),   # (measured)
    "forest_dark":  (163, 180, 149),   # (measured)
    "meadow":       (244, 243, 226),   # (measured) open/cultivated buff
    "glacier":      (248, 250, 250),   # (measured)
    "motorway":     (233, 132, 58),
    "primary":      (231, 148, 88),
    "secondary":    (245, 214, 120),
    "minor":        (255, 255, 255),
    "built_area":   (232, 228, 220),
    "power":        (60, 62, 60),
}

# --- line features -------------------------------------------------------
# order = paint order (low first). casing is drawn under fill at fill+2*casing.
LINES = {
    "motorway":     dict(fill=C["motorway"],  mm=1.10, casing=C["black"], casing_mm=0.09, order=60),
    "trunk":        dict(fill=C["motorway"],  mm=0.95, casing=C["black"], casing_mm=0.09, order=59),
    "primary":      dict(fill=C["primary"],   mm=0.80, casing=C["black"], casing_mm=0.08, order=58),
    "secondary":    dict(fill=C["secondary"], mm=0.68, casing=C["black"], casing_mm=0.07, order=57),
    "tertiary":     dict(fill=C["minor"],     mm=0.58, casing=C["black"], casing_mm=0.07, order=56),
    "unclassified": dict(fill=C["minor"],     mm=0.50, casing=C["black"], casing_mm=0.06, order=55),
    "residential":  dict(fill=C["minor"],     mm=0.50, casing=C["black"], casing_mm=0.06, order=55),
    "service":      dict(fill=C["minor"],     mm=0.34, casing=C["black"], casing_mm=0.05, order=54),
    "track":        dict(fill=C["black"],     mm=0.26, dash=(2.4, 1.0), order=52),
    "path":         dict(fill=C["black"],     mm=0.20, dash=(1.5, 1.2), order=51),
    "footway":      dict(fill=C["black"],     mm=0.20, dash=(1.5, 1.2), order=51),
    "bridleway":    dict(fill=C["black"],     mm=0.20, dash=(1.5, 1.2), order=51),
    "steps":        dict(fill=C["black"],     mm=0.26, dash=(0.5, 0.5), order=51),
    "pedestrian":   dict(fill=C["minor"],     mm=0.34, casing=C["black"], casing_mm=0.05, order=53),
}
RAIL = {
    "rail":         dict(fill=C["rail"], mm=0.42, ticks=True,  order=70),
    "narrow_gauge": dict(fill=C["rail"], mm=0.32, ticks=True,  order=70),
    "light_rail":   dict(fill=C["rail"], mm=0.32, ticks=True,  order=70),
    "funicular":    dict(fill=C["rail"], mm=0.32, ticks=True,  order=70),
    "tram":         dict(fill=C["rail"], mm=0.26, ticks=False, order=69),
}
AERIAL = dict(fill=C["black"], mm=0.16, order=72, pylon_mm=0.55, pylon_every_mm=4.0)
POWER = dict(fill=C["power"], mm=0.14, order=71, pylon_mm=0.40, pylon_every_mm=6.0)
WATER_LINE = {
    "river":  dict(fill=C["water_line"], mm=0.40, order=30),
    "stream": dict(fill=C["water_line"], mm=0.22, order=29),
    "ditch":  dict(fill=C["water_line"], mm=0.16, order=28),
    "canal":  dict(fill=C["water_line"], mm=0.34, order=30),
}

# --- area features -------------------------------------------------------
AREAS = {
    ("natural", "water"):      dict(fill=C["water_fill"], outline=C["water_line"], mm=0.22, order=20),
    ("landuse", "reservoir"):  dict(fill=C["water_fill"], outline=C["water_line"], mm=0.22, order=20),
    ("natural", "glacier"):    dict(fill=C["glacier"],    order=12),
    ("natural", "wood"):       dict(fill=C["forest"],     order=14),
    ("landuse", "forest"):     dict(fill=C["forest"],     order=14),
    ("landuse", "meadow"):     dict(fill=C["meadow"],     order=11),
    ("landuse", "farmland"):   dict(fill=C["meadow"],     order=11),
    ("landuse", "residential"):dict(fill=C["built_area"], order=13),
    ("landuse", "vineyard"):   dict(fill=C["forest_dark"],order=13),
}
BUILDING = dict(fill=C["black"], order=80)

# --- contours ------------------------------------------------------------
CONTOUR = dict(interval_m=20, index_every=5,
               mm=0.11, index_mm=0.22,
               colour=C["contour"], ice_colour=C["contour_ice"])

# --- typography ----------------------------------------------------------
# swisstopo sets names in a humanist sans. Frutiger is the historical face and
# is not free; Inter / Source Sans / Helvetica are the closest metric-friendly
# substitutes that ship on most systems.
FONT_STACK = ["Frutiger", "Inter", "HelveticaNeue", "Helvetica", "Arial"]
TEXT = {
    "peak":     dict(pt=7.5, colour=C["black"],      italic=False, bold=False),
    "peak_ele": dict(pt=6.5, colour=C["black"],      italic=False, bold=False),
    "saddle":   dict(pt=6.5, colour=C["black"],      italic=True,  bold=False),
    "hut":      dict(pt=6.5, colour=C["black"],      italic=False, bold=False),
    "water":    dict(pt=6.5, colour=C["water_line"], italic=True,  bold=False),
    "village":  dict(pt=9.0, colour=C["black"],      italic=False, bold=True),
    "contour":  dict(pt=5.5, colour=C["contour"],    italic=False, bold=False),
}


def road_spec(tags):
    hw = tags.get("highway")
    if hw in LINES:
        return LINES[hw]
    if hw in ("living_street", "road"):
        return LINES["unclassified"]
    if hw and hw.endswith("_link"):
        return LINES.get(hw[:-5], LINES["unclassified"])
    return None
