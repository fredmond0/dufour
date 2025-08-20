from flask import Flask, request, jsonify, send_from_directory
import json
import os
from route_backend import main_route_calculation, load_ski_touring_routes

app = Flask(__name__)

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
        
        print(f"Calculating {network_type} route from {start_coords} to {end_coords} with buffer {buffer_distance}m")
        
        # Call your optimized routing function
        shortest_path, path_length, segments_loaded = main_route_calculation(
            start_coords, 
            end_coords, 
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
