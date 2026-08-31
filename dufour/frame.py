"""A multi-tile mosaic frame -- the coordinate system every layer draws into.

All layers (learned terrain, contours, OSM symbology, labels) must land on the
exact same pixel grid, so they all take a Frame and use lonlat_to_px.
"""
import math
from dataclasses import dataclass

import numpy as np

from .tiles import TILE_PX, deg2tile, tile2deg, meters_per_pixel


@dataclass
class Frame:
    z: int
    x0: int
    y0: int
    nx: int
    ny: int

    @classmethod
    def from_bbox(cls, w, s, e, n, z):
        x0, y0 = deg2tile(n, w, z)
        x1, y1 = deg2tile(s, e, z)
        return cls(z, min(x0, x1), min(y0, y1),
                   abs(x1 - x0) + 1, abs(y1 - y0) + 1)

    @classmethod
    def around(cls, lat, lon, z, km):
        mpp = meters_per_pixel(lat, z)
        half = int((km * 1000 / mpp) / 2)
        cx, cy = deg2tile(lat, lon, z)
        # centre tile, expanded symmetrically to cover the requested extent
        t = max(1, int(math.ceil(half / TILE_PX)))
        return cls(z, cx - t, cy - t, 2 * t + 1, 2 * t + 1)

    @property
    def width(self):
        return self.nx * TILE_PX

    @property
    def height(self):
        return self.ny * TILE_PX

    @property
    def tiles(self):
        for j in range(self.ny):
            for i in range(self.nx):
                yield i, j, self.x0 + i, self.y0 + j

    def bbox(self):
        n_lat, w_lon = tile2deg(self.x0, self.y0, self.z)
        s_lat, e_lon = tile2deg(self.x0 + self.nx, self.y0 + self.ny, self.z)
        return w_lon, s_lat, e_lon, n_lat

    @property
    def centre_lat(self):
        w, s, e, n = self.bbox()
        return (s + n) / 2

    @property
    def mpp(self):
        return meters_per_pixel(self.centre_lat, self.z)

    def lonlat_to_px(self, lon, lat):
        """Vectorised web-mercator lon/lat -> frame pixel coords."""
        lon = np.asarray(lon, np.float64); lat = np.asarray(lat, np.float64)
        n = 2.0 ** self.z
        gx = (lon + 180.0) / 360.0 * n
        lat_r = np.radians(np.clip(lat, -85.05, 85.05))
        gy = (1.0 - np.arcsinh(np.tan(lat_r)) / np.pi) / 2.0 * n
        return (gx - self.x0) * TILE_PX, (gy - self.y0) * TILE_PX

    def px_to_lonlat(self, px, py):
        """Inverse of lonlat_to_px -- needed to sample geographic rasters
        (Copernicus DEM) onto this frame's grid."""
        n = 2.0 ** self.z
        gx = np.asarray(px, np.float64) / TILE_PX + self.x0
        gy = np.asarray(py, np.float64) / TILE_PX + self.y0
        lon = gx / n * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * gy / n))))
        return lon, lat

    def area_km2(self):
        return (self.width * self.mpp / 1000.0) * (self.height * self.mpp / 1000.0)

    def __str__(self):
        w, s, e, n = self.bbox()
        return (f"Frame z{self.z} {self.nx}x{self.ny} tiles = "
                f"{self.width}x{self.height}px, {self.mpp:.2f} m/px, "
                f"{self.area_km2():.1f} km2, bbox=({w:.4f},{s:.4f},{e:.4f},{n:.4f})")
