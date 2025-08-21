#!/usr/bin/env python3
"""
Routing Validation Test Script

Tests the robust routing implementation with specific coordinate pairs
to ensure proper route calculation and geometry following.

Test coordinates (should each be ~2.4km):
1. (2706198.5, 1183207.5) to (2706725, 1183530) 
2. (2706198.5, 1183207.5) to (2707559.849, 1183207.530)
3. (2706725, 1183530) to (2707559.849, 1183207.530)
"""

import geopandas as gpd
import networkx as nx
import momepy
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from route_backend import calculate_route_from_gpkg

def test_coordinate_pair(start_coords, end_coords, pair_name):
    """Test routing between a specific coordinate pair."""
    print(f"\n{'='*60}")
    print(f"Testing {pair_name}")
    print(f"Start: {start_coords}")
    print(f"End: {end_coords}")
    print(f"{'='*60}")
    
    # Calculate route using the backend
    result = calculate_route_from_gpkg(
        start_coords, 
        end_coords, 
        buffer_distance=5000, 
        network_type='ski_touring'
    )
    
    route_geometry, path_length, segments_loaded, elevation_profile = result
    
    if route_geometry and path_length:
        print(f"✅ Route found!")
        print(f"   Length: {path_length:.1f}m ({path_length/1000:.2f}km)")
        print(f"   Segments loaded: {segments_loaded}")
        print(f"   Route points: {len(route_geometry)}")
        
        # Calculate straight-line distance for comparison
        start_point = Point(start_coords)
        end_point = Point(end_coords)
        straight_dist = start_point.distance(end_point)
        efficiency = (straight_dist / path_length) * 100 if path_length > 0 else 0
        
        print(f"   Straight-line: {straight_dist:.1f}m ({straight_dist/1000:.2f}km)")
        print(f"   Efficiency: {efficiency:.1f}%")
        
        return {
            'success': True,
            'route_geometry': route_geometry,
            'path_length': path_length,
            'straight_distance': straight_dist,
            'efficiency': efficiency,
            'segments_loaded': segments_loaded
        }
    else:
        print(f"❌ Route calculation failed")
        return {
            'success': False,
            'route_geometry': None,
            'path_length': None,
            'straight_distance': None,
            'efficiency': None,
            'segments_loaded': segments_loaded
        }

def visualize_all_routes(results):
    """Create a visualization showing all test routes."""
    print(f"\n{'='*60}")
    print("Creating route visualization...")
    print(f"{'='*60}")
    
    # Load the cleaned network for background
    try:
        gdf = gpd.read_file('./data/skitouring/cleaningtesting/cleaned_network.gpkg')
        print(f"Loaded cleaned network: {len(gdf)} segments")
    except Exception as e:
        print(f"Could not load cleaned network: {e}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    
    # Plot network background
    gdf.plot(ax=ax, color='lightgray', linewidth=0.8, alpha=0.6)
    
    # Define distinct colors and line styles for each route
    route_styles = [
        {'color': 'red', 'linewidth': 4, 'linestyle': '-', 'alpha': 0.9},
        {'color': 'blue', 'linewidth': 4, 'linestyle': '--', 'alpha': 0.9}, 
        {'color': 'green', 'linewidth': 4, 'linestyle': '-.', 'alpha': 0.9}
    ]
    
    # Plot each route with different styles to avoid overlap confusion
    for i, (pair_name, result) in enumerate(results.items()):
        if result['success'] and result['route_geometry']:
            route_coords = result['route_geometry']
            style = route_styles[i]
            
            # Plot route line with distinct style
            if len(route_coords) >= 2:
                x_coords = [coord[0] for coord in route_coords]
                y_coords = [coord[1] for coord in route_coords]
                
                ax.plot(x_coords, y_coords, 
                       color=style['color'], 
                       linewidth=style['linewidth'],
                       linestyle=style['linestyle'],
                       alpha=style['alpha'], 
                       label=f'{pair_name} ({result["path_length"]/1000:.2f}km, eff: {result["efficiency"]:.1f}%)',
                       zorder=10-i)  # Different z-order to control layering
                
                # Plot start and end points with larger, more visible markers
                ax.scatter([x_coords[0]], [y_coords[0]], color=style['color'], s=150, 
                          marker='o', zorder=15, edgecolors='white', linewidth=3,
                          label=f'{pair_name} Start' if i == 0 else "")
                ax.scatter([x_coords[-1]], [y_coords[-1]], color=style['color'], s=150, 
                          marker='s', zorder=15, edgecolors='white', linewidth=3,
                          label=f'End Points' if i == 0 else "")
                
                # Add route direction arrows at 1/3 and 2/3 points
                if len(x_coords) > 6:  # Only if route has enough points
                    third_idx = len(x_coords) // 3
                    two_third_idx = 2 * len(x_coords) // 3
                    
                    for arrow_idx in [third_idx, two_third_idx]:
                        if arrow_idx < len(x_coords) - 1:
                            dx = x_coords[arrow_idx + 1] - x_coords[arrow_idx]
                            dy = y_coords[arrow_idx + 1] - y_coords[arrow_idx]
                            ax.annotate('', xy=(x_coords[arrow_idx + 1], y_coords[arrow_idx + 1]),
                                      xytext=(x_coords[arrow_idx], y_coords[arrow_idx]),
                                      arrowprops=dict(arrowstyle='->', color=style['color'], 
                                                    lw=2, alpha=0.8), zorder=12)
    
    ax.set_title('Routing Validation Test Results\n(Circles=Start, Squares=End)')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('routing_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'routing_validation_results.png'")

def main():
    """Main test function."""
    print("🚀 Starting Routing Validation Tests")
    print("Testing robust routing with specific coordinate pairs...")
    
    # Define test coordinate pairs
    coord1 = (2706198.5, 1183207.5)  # Point 1
    coord2 = (2706725, 1183530)      # Point 2  
    coord3 = (2707977.395, 1180824.833)  # Point 3
    
    test_pairs = {
        "Route 1→2": (coord1, coord2),
        "Route 1→3": (coord1, coord3), 
        "Route 2→3": (coord2, coord3)
    }
    
    results = {}
    
    # Test each coordinate pair
    for pair_name, (start, end) in test_pairs.items():
        results[pair_name] = test_coordinate_pair(start, end, pair_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    
    successful_routes = 0
    for pair_name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        length_info = f"({result['path_length']/1000:.2f}km)" if result['success'] else ""
        print(f"{pair_name}: {status} {length_info}")
        if result['success']:
            successful_routes += 1
    
    print(f"\nSuccessful routes: {successful_routes}/3")
    
    # Create visualization if we have successful routes
    if successful_routes > 0:
        visualize_all_routes(results)
    
    # Check if routes are approximately 2.4km as expected
    print(f"\n{'='*60}")
    print("DISTANCE VALIDATION")
    print(f"{'='*60}")
    
    expected_distance = 3000  # ~3.0km in meters (based on straight-line distances)
    tolerance = 800  # 800m tolerance for mountain routing
    
    for pair_name, result in results.items():
        if result['success']:
            distance = result['path_length']
            diff = abs(distance - expected_distance)
            status = "✅ GOOD" if diff <= tolerance else "⚠️ CHECK"
            print(f"{pair_name}: {distance:.0f}m (diff: {diff:.0f}m) {status}")
    
    return results

if __name__ == "__main__":
    results = main()
