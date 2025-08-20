# app.py
from flask import Flask, request, jsonify, send_from_directory
from route_backend import calculate_route_from_gpkg
from pyproj import Transformer

app = Flask(__name__)

# Initialize a single, reusable coordinate transformer
wgs84_to_lv95_transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

def convert_wgs84_to_lv95(lat, lng):
    """
    Converts WGS84 coordinates (latitude, longitude) to Swiss LV95 (x, y).
    Returns (None, None) if the conversion fails.
    """
    try:
        x, y = wgs84_to_lv95_transformer.transform(lng, lat)
        return x, y
    except Exception as e:
        print(f"Coordinate transformation failed for ({lat}, {lng}): {e}")
        return None, None

@app.route('/')
def index():
    """Serves the main HTML interface."""
    return send_from_directory('.', 'route_interface.html')

@app.route('/calculate_route', methods=['POST'])
def calculate_route():
    """Handles route calculation requests from the frontend."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400

        start_coords_wgs = data.get('start_coords')
        end_coords_wgs = data.get('end_coords')
        buffer_distance = data.get('buffer_distance', 5000)
        network_type = data.get('network_type', 'tlm3d')

        if not start_coords_wgs or not end_coords_wgs:
            return jsonify({'success': False, 'error': 'Missing start or end coordinates'}), 400

        # Convert coordinates from WGS84 to Swiss LV95
        start_x, start_y = convert_wgs84_to_lv95(start_coords_wgs[0], start_coords_wgs[1])
        end_x, end_y = convert_wgs84_to_lv95(end_coords_wgs[0], end_coords_wgs[1])

        if start_x is None or end_x is None:
            return jsonify({'success': False, 'error': 'Coordinate conversion failed'}), 500
        
        start_lv95 = (start_x, start_y)
        end_lv95 = (end_x, end_y)

        # Call the single, reliable "golden routing function" from the backend
        result = calculate_route_from_gpkg(
            start_lv95,
            end_lv95,
            buffer_distance,
            network_type
        )
        
        try:
            shortest_path, path_length, segments_loaded = result
        except Exception as e:
            return jsonify({'success': False, 'error': f'Internal unpacking error: {e}'}), 500

        if shortest_path:
            return jsonify({
                'success': True,
                'route': shortest_path,
                'path_length': path_length,
                'segments_loaded': segments_loaded,
                'buffer_distance': buffer_distance,
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No route found. Try increasing the buffer or selecting different points.'
            }), 404

    except Exception as e:
        print(f"An unexpected error occurred in route calculation: {e}")
        return jsonify({'success': False, 'error': 'An internal error occurred.'}), 500

if __name__ == '__main__':
    print("🚀 Starting Swiss Route Planner Server...")
    print("   Mode: Debugging Interface")
    print("   URL: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)