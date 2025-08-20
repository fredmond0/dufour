# Swiss TLM3D Route Planner

A high-performance routing application for Switzerland using official Swiss topographic data (TLM3D) and elevation data (SwissALTI3D).

## 🗺️ Features

- **Point-to-Point Routing**: Route between any two points along roads
- **Multi-Point Routing**: Create complex routes with multiple waypoints
- **Real Swiss Topography**: Uses official TLM3D road network data
- **Ski Touring Routes**: Dedicated ski route network for winter sports
- **Real Elevation Data**: SwissALTI3D elevation profiles
- **Interactive Map**: Leaflet-based interface with Swisstopo maps
- **GPX Export**: Export routes for GPS devices
- **Performance Optimized**: Clipped network loading for fast routing
- **Robust Geometry Mapping**: Fixed momepy integration for reliable routing

## 📁 Project Structure

```
dufour/
├── data/                          # Raw data (gitignored)
│   ├── elevation/
│   │   └── swissalti3d/          # SwissALTI3D elevation tiles
│   ├── tlm3d/                    # TLM3D road network data
│   ├── skitouring/               # Ski touring route data
│   └── preprocessed/             # Pre-processed network files
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
- Place your `SWISSTLM3D_2025.gpkg` file in `data/tlm3d/`
- Update the path in `route_backend.py` if needed

**Ski Touring Routes:**
- Place your ski touring GPKG file in `data/skitouring/`
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

### 3. Pre-process Networks (Required for routing)

```bash
# Pre-process all networks (TLM3D + Ski routes)
python route_backend.py preprocess

# Or pre-process only ski routes (faster)
python route_backend.py preprocess-ski
```

### 4. Start the Application

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
- **Pre-processed Networks**: One-time conversion for faster loading

### Recent Improvements (Latest Update)
- **Fixed momepy Integration**: Robust geometry mapping between network and graph
- **Ski Touring Support**: Added dedicated ski route network
- **Improved Clipping**: Better edge weight handling for virtual connections
- **Debug Logging**: Enhanced troubleshooting and performance monitoring

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

**"Unknown network type: ski"**
- Use `ski_touring` instead of `ski` in the backend
- Ensure pre-processed networks are created first

## 📈 Current Status

### ✅ Completed
- **Core routing engine** with NetworkX and momepy
- **Coordinate system handling** (WGS84 ↔ LV95)
- **Network clipping** for performance optimization
- **Geometry mapping** between GeoDataFrame and NetworkX graph
- **Ski touring network support**
- **Pre-processing pipeline** for network optimization

### 🔄 In Progress
- **Route geometry building** from clipped network
- **Virtual edge weight optimization** for better routing
- **Performance tuning** of clipping algorithms

### 🎯 Next Steps
- **Complete route geometry extraction** from actual network edges
- **Optimize virtual edge handling** to prefer real routes
- **Add elevation profile support** for ski routes
- **Implement route caching** for repeated queries
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
