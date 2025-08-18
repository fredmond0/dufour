# Swiss TLM3D Route Planner

A high-performance routing application for Switzerland using official Swiss topographic data (TLM3D) and elevation data (SwissALTI3D).

## 🗺️ Features

- **Point-to-Point Routing**: Route between any two points along roads
- **Multi-Point Routing**: Create complex routes with multiple waypoints
- **Real Swiss Topography**: Uses official TLM3D road network data
- **Real Elevation Data**: SwissALTI3D elevation profiles
- **Interactive Map**: Leaflet-based interface with Swisstopo maps
- **GPX Export**: Export routes for GPS devices
- **Performance Optimized**: Clipped network loading for fast routing

## 📁 Project Structure

```
dufour/
├── data/                          # Raw data (gitignored)
│   ├── elevation/
│   │   └── swissalti3d/          # SwissALTI3D elevation tiles
│   └── tlm3d/                    # TLM3D road network data
├── route_backend.py              # Core routing engine
├── route_interface.html          # Web interface
├── app.py                        # Flask web server
├── download_elevation.py         # Elevation data downloader
├── requirements.txt              # Python dependencies
├── requirements_download.txt     # Download script dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

**TLM3D Road Network:**
- Place your `SWISSTLM3D_2025.gpkg` file in the project root
- Update the path in `route_backend.py` if needed

**SwissALTI3D Elevation Data:**
```bash
# Install download dependencies
pip install -r requirements_download.txt

# Download sample tiles (recommended first)
python download_elevation.py --sample --num-sample 10

# Download all tiles (43,645 files - will take time!)
python download_elevation.py
```

### 3. Start the Application

```bash
# Start Flask server
python app.py

# Open browser to http://localhost:5000
```

## 🎯 Usage

### Point-to-Point Routing
1. Select **"Point to Point"** mode
2. Click **anywhere along a road** (not just at junctions!)
3. Set start and end points
4. Click **"Calculate Route"**

### Multi-Point Routing
1. Select **"Multi-Point Route"** mode
2. Click **anywhere along roads** to add waypoints
3. **Route draws automatically** after 2+ waypoints
4. **Drag waypoints** to update route in real-time
5. **Remove waypoints** with × button

### Interactive Features
- **Hover over elevation profile** to see route marker on map
- **Adjust buffer distance** to control routing area
- **Show/hide bounding box** for debugging
- **Export to GPX** for GPS devices

## 🔧 Technical Details

### Coordinate Systems
- **Input/Output**: WGS84 (Lat/Lng) for Leaflet display
- **Internal Processing**: Swiss LV95 (EPSG:2056) for routing
- **Automatic Conversion**: Handled transparently by the system

### Performance Optimizations
- **Network Clipping**: Only loads roads within buffer around route
- **Spatial Indexing**: Efficient nearest-point calculations
- **Geometry Interpolation**: Smooth routes with minimal straight lines

### Data Sources
- **Roads**: TLM3D (Swiss Federal Office of Topography)
- **Elevation**: SwissALTI3D (Swiss Federal Office of Topography)
- **Maps**: Leaflet.TileLayer.Swiss (official Swisstopo tiles)

## 📊 Data Requirements

### TLM3D Road Network
- **Format**: GeoPackage (.gpkg)
- **Layer**: `tlm_strassen_strasse`
- **Coverage**: Switzerland
- **Size**: ~100-500 MB

### SwissALTI3D Elevation
- **Format**: GeoTIFF (.tif)
- **Resolution**: 0.5m
- **Coverage**: Switzerland
- **Tiles**: 43,645 files
- **Total Size**: ~50-100 GB

## 🐛 Troubleshooting

### Common Issues

**"No module named 'geopandas'"**
```bash
pip install geopandas
```

**"No node found within X meters"**
- Increase buffer distance in the interface
- Check if TLM3D data is properly loaded

**"Projection failed"**
- Ensure map is using EPSG:2056 CRS
- Check coordinate conversion functions

**Slow routing**
- Reduce buffer distance
- Check if elevation data is in correct location

### Debug Mode
- Open browser console for detailed logging
- Check `elevation_download.log` for download issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project uses official Swiss government data. Please respect the data usage terms from Swisstopo.

## 🙏 Acknowledgments

- **Swisstopo**: For providing TLM3D and SwissALTI3D data
- **Leaflet.TileLayer.Swiss**: For official Swiss map tiles
- **GeoPandas & NetworkX**: For geospatial processing and routing

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the console logs
3. Open a GitHub issue with details
