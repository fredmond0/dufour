from flask import Flask, request, jsonify, send_from_directory
import json
import os
from route_backend import main_route_calculation

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
        
        if not start_coords or not end_coords:
            return jsonify({'success': False, 'error': 'Missing start or end coordinates'}), 400
        
        print(f"Calculating route from {start_coords} to {end_coords} with buffer {buffer_distance}m")
        
        # Call your optimized routing function
        shortest_path, path_length, roads_loaded = main_route_calculation(
            start_coords, 
            end_coords, 
            buffer_distance
        )
        
        if shortest_path and path_length:
            
            return jsonify({
                'success': True,
                'route': shortest_path,
                'path_length': path_length,
                'roads_loaded': roads_loaded,
                'buffer_distance': buffer_distance
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
