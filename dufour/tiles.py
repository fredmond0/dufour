"""Slippy-map tile math (Web Mercator, EPSG:3857).

Everything in this project keys off (z, x, y) in the standard XYZ scheme.
swisstopo's WMTS and the AWS terrain tiles both serve that scheme, so a
map tile and its DEM tile are pixel-aligned by construction -- no warping.
"""
import math

TILE_PX = 256


def deg2tile(lat, lon, z):
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return int(x), int(y)


def tile2deg(x, y, z):
    """North-west corner of tile."""
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def tile_bounds(x, y, z):
    n_lat, w_lon = tile2deg(x, y, z)
    s_lat, e_lon = tile2deg(x + 1, y + 1, z)
    return w_lon, s_lat, e_lon, n_lat


def meters_per_pixel(lat, z):
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2.0 ** z)


def tiles_in_bbox(w, s, e, n, z):
    x0, y0 = deg2tile(n, w, z)
    x1, y1 = deg2tile(s, e, z)
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for y in range(min(y0, y1), max(y0, y1) + 1):
            yield z, x, y
