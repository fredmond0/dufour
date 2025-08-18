#!/usr/bin/env python3
"""
SwissALTI3D Elevation Service
Extracts elevation data from downloaded GeoTIFF files
"""

import os
import rasterio
from rasterio.warp import transform
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SwissALTI3DElevationService:
    """Service for extracting elevation data from SwissALTI3D GeoTIFF files"""
    
    def __init__(self, elevation_dir="data/elevation/swissalti3d"):
        self.elevation_dir = Path(elevation_dir)
        self.tile_cache = {}  # Cache opened tiles for performance
        
    def find_elevation_tile(self, lat, lng):
        """Find the appropriate elevation tile for given coordinates"""
        # Convert WGS84 to Swiss LV95 coordinates
        # This is a simplified approach - in production you'd use proper projection
        lv95_x, lv95_y = self._wgs84_to_lv95(lat, lng)
        
        # Find tile that contains these coordinates
        for tile_file in self.elevation_dir.glob("*.tif"):
            try:
                with rasterio.open(tile_file) as src:
                    # Check if coordinates are within tile bounds
                    if (src.bounds.left <= lv95_x <= src.bounds.right and 
                        src.bounds.bottom <= lv95_y <= src.bounds.top):
                        return tile_file
            except Exception as e:
                logger.warning(f"Could not read tile {tile_file}: {e}")
                continue
        
        return None
    
    def get_elevation(self, lat, lng):
        """Get elevation for given WGS84 coordinates"""
        tile_file = self.find_elevation_tile(lat, lng)
        
        if not tile_file:
            logger.warning(f"No elevation tile found for coordinates ({lat}, {lng})")
            return None
        
        try:
            # Use cached tile if available
            if tile_file not in self.tile_cache:
                self.tile_cache[tile_file] = rasterio.open(tile_file)
            
            src = self.tile_cache[tile_file]
            
            # Convert WGS84 to tile coordinates
            lv95_x, lv95_y = self._wgs84_to_lv95(lat, lng)
            
            # Get pixel coordinates
            row, col = src.index(lv95_x, lv95_y)
            
            # Read elevation value
            elevation = src.read(1, window=((row, row+1), (col, col+1)))
            
            if elevation.size > 0 and not np.isnan(elevation[0, 0]):
                return float(elevation[0, 0])
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error reading elevation from {tile_file}: {e}")
            return None
    
    def get_elevation_profile(self, coordinates):
        """Get elevation profile for a list of coordinates"""
        elevations = []
        
        for coord in coordinates:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                lat, lng = coord[0], coord[1]
            elif hasattr(coord, 'lat') and hasattr(coord, 'lng'):
                lat, lng = coord.lat, coord.lng
            else:
                logger.warning(f"Invalid coordinate format: {coord}")
                elevations.append(None)
                continue
            
            elevation = self.get_elevation(lat, lng)
            elevations.append(elevation)
        
        return elevations
    
    def _wgs84_to_lv95(self, lat, lng):
        """Convert WGS84 coordinates to Swiss LV95 (simplified)"""
        # This is a simplified conversion - in production you'd use pyproj
        # For now, we'll use a rough approximation
        
        # Switzerland bounds and LV95 ranges
        lat_min, lat_max = 45.8, 47.8
        lng_min, lng_max = 5.9, 10.5
        x_min, x_max = 2400000, 2900000
        y_min, y_max = 1000000, 1300000
        
        # Clamp coordinates to Switzerland bounds
        lat = max(lat_min, min(lat_max, lat))
        lng = max(lng_min, min(lng_max, lng))
        
        # Linear interpolation
        lat_ratio = (lat - lat_min) / (lat_max - lat_min)
        lng_ratio = (lng - lng_min) / (lng_max - lng_min)
        
        x = x_min + lng_ratio * (x_max - x_min)
        y = y_min + lat_ratio * (y_max - y_min)
        
        return x, y
    
    def close(self):
        """Close all open tile files"""
        for tile in self.tile_cache.values():
            try:
                tile.close()
            except:
                pass
        self.tile_cache.clear()

# Global instance for easy access
elevation_service = SwissALTI3DElevationService()

def get_elevation(lat, lng):
    """Get elevation for given coordinates"""
    return elevation_service.get_elevation(lat, lng)

def get_elevation_profile(coordinates):
    """Get elevation profile for coordinates"""
    return elevation_service.get_elevation_profile(coordinates)

if __name__ == "__main__":
    # Test the service
    test_coords = [
        (46.8182, 8.2275),  # Central Switzerland
        (47.3769, 8.5417),  # Zurich
        (46.2044, 6.1432)   # Geneva
    ]
    
    print("Testing SwissALTI3D Elevation Service")
    print("=" * 40)
    
    for lat, lng in test_coords:
        elevation = get_elevation(lat, lng)
        print(f"Coordinates: ({lat}, {lng}) -> Elevation: {elevation}m")
    
    elevation_service.close()
