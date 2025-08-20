from flask import Flask, request, jsonify, send_from_directory
import json
import os
from route_backend import main_route_calculation, load_ski_touring_routes
from pyproj import Transformer

app = Flask(__name__)

# Initialize coordinate transformer for WGS84 to Swiss LV95 conversion
wgs84_to_lv95_transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

def convert_wgs84_to_lv95(lat, lng):
    """Convert WGS84 coordinates (latitude, longitude) to Swiss LV95 coordinates"""
    try:
        x, y = wgs84_to_lv95_transformer.transform(lng, lat)
        return [x, y]
    except Exception as e:
        print(f"Coordinate transformation failed: {e}")
        # Fallback to approximate conversion if pyproj fails
        return convert_wgs84_to_lv95_fallback(lat, lng)

def convert_wgs84_to_lv95_fallback(lat, lng):
    """Fallback coordinate conversion method (less accurate)"""
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
    
    return [x, y]

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return send_from_directory('.', 'route_interface.html')

@app.route('/calculate_route', methods=['POST'])
def calculate_route():
    """Handle route calculation requests from the frontend"""
    try:
        # Get data from frontend
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
        
        start_coords = data.get('start_coords')
        end_coords = data.get('end_coords')
        buffer_distance = data.get('buffer_distance', 5000)  # Default 5km
        network_type = data.get('network_type', 'tlm3d')  # Default to TLM3D
        
        if not start_coords or not end_coords:
            return jsonify({'success': False, 'error': 'Missing start or end coordinates'}), 400
        
        # Convert WGS84 coordinates to Swiss LV95
        # Frontend sends: [latitude, longitude]
        # Backend expects: [x, y] in LV95 projection
        start_lat, start_lng = start_coords[0], start_coords[1]
        end_lat, end_lng = end_coords[0], end_coords[1]
        
        print(f"Received WGS84 coordinates:")
        print(f"  Start: ({start_lat}, {start_lng})")
        print(f"  End: ({end_lat}, {end_lng})")
        
        # Convert to Swiss LV95
        start_lv95 = convert_wgs84_to_lv95(start_lat, start_lng)
        end_lv95 = convert_wgs84_to_lv95(end_lat, end_lng)
        
        print(f"Converted to LV95 coordinates:")
        print(f"  Start: ({start_lv95[0]:.0f}, {start_lv95[1]:.0f})")
        print(f"  End: ({end_lv95[0]:.0f}, {end_lv95[1]:.0f})")
        print(f"Calculating {network_type} route with buffer {buffer_distance}m")
        
        # Call your optimized routing function with LV95 coordinates
        shortest_path, path_length, segments_loaded = main_route_calculation(
            start_lv95, 
            end_lv95, 
            buffer_distance,
            network_type
        )
        
        if shortest_path and path_length:
            
            return jsonify({
                'success': True,
                'route': shortest_path,
                'path_length': path_length,
                'segments_loaded': segments_loaded,
                'buffer_distance': buffer_distance,
                'network_type': network_type
            })
        else:
            return jsonify({
                'success': False, 
                'error': 'No route found. Try increasing the buffer distance.'
            }), 404
            
    except Exception as e:
        print(f"Error in route calculation: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Route calculation failed: {str(e)}'
        }), 500

@app.route('/get_ski_routes', methods=['GET'])
def get_ski_routes():
    """Get ski touring routes for display on the map"""
    try:
        # Get optional bounding box parameters for performance
        bbox_params = request.args.get('bbox')
        bbox = None
        
        if bbox_params:
            try:
                # Parse bbox as "minx,miny,maxx,maxy"
                coords = [float(x) for x in bbox_params.split(',')]
                if len(coords) == 4:
                    bbox = coords
            except:
                pass
        
        # Load ski touring routes
        ski_routes = load_ski_touring_routes(bbox)
        
        if ski_routes is not None and len(ski_routes) > 0:
            # Convert to GeoJSON for frontend
            routes_geojson = ski_routes.to_json()
            return routes_geojson, 200, {'Content-Type': 'application/json'}
        else:
            return jsonify({'routes': []}), 200
            
    except Exception as e:
        print(f"Error loading ski touring routes: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Failed to load ski touring routes: {str(e)}'
        }), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    print("🚀 Starting Swiss TLM3D Route Planner Server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🗺️  The interface will load with a map where you can click to set start/end points")
    print("⚡ Routes are calculated using the optimized clipping approach!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
