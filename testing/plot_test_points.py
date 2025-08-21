#!/usr/bin/env python3
"""
Plot the 3 test points on the cleaned network to visualize their positions
and understand the routing connectivity issues.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np

def plot_test_points():
    """Plot the 3 test coordinate points on the cleaned network."""
    
    # Load the cleaned network
    try:
        gdf = gpd.read_file('./data/skitouring/cleaningtesting/cleaned_network.gpkg')
        print(f"Loaded cleaned network: {len(gdf)} segments")
    except Exception as e:
        print(f"Could not load cleaned network: {e}")
        return
    
    # Define the 3 test points
    coord1 = (2706198.5, 1183207.5)   # Point 1
    coord2 = (2706725, 1183530)       # Point 2  
    coord3 = (2707977.395, 1180824.833)  # Point 3
    
    points = {
        'Point 1': coord1,
        'Point 2': coord2, 
        'Point 3': coord3
    }
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot the network
    gdf.plot(ax=ax, color='lightgray', linewidth=1, alpha=0.8)
    
    # Plot the 3 test points
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    
    for i, (label, coords) in enumerate(points.items()):
        ax.scatter([coords[0]], [coords[1]], 
                  color=colors[i], s=200, marker=markers[i], 
                  zorder=10, edgecolors='white', linewidth=3,
                  label=f'{label} ({coords[0]:.0f}, {coords[1]:.0f})')
    
    # Add point labels with offsets for readability
    offsets = [(100, 100), (-200, 100), (100, -150)]
    for i, (label, coords) in enumerate(points.items()):
        ax.annotate(label, 
                   xy=coords, 
                   xytext=(coords[0] + offsets[i][0], coords[1] + offsets[i][1]),
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                                 color='black', lw=2))
    
    # Draw straight lines between points to show expected routes
    line_pairs = [
        ('1→2', coord1, coord2, 'red'),
        ('1→3', coord1, coord3, 'blue'), 
        ('2→3', coord2, coord3, 'green')
    ]
    
    for label, start, end, color in line_pairs:
        ax.plot([start[0], end[0]], [start[1], end[1]], 
               color=color, linewidth=2, linestyle='--', alpha=0.6,
               label=f'Direct {label}')
    
    # Calculate and display straight-line distances
    distances = []
    for label, start, end, color in line_pairs:
        dist = Point(start).distance(Point(end))
        distances.append((label, dist))
        
        # Add distance label at midpoint
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, f'{dist:.0f}m', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Set title and labels
    ax.set_title('Test Points on Cleaned Ski Touring Network\n' + 
                'Dashed lines show direct distances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Print distance summary
    print(f"\n{'='*50}")
    print("STRAIGHT-LINE DISTANCES")
    print(f"{'='*50}")
    for label, dist in distances:
        print(f"Route {label}: {dist:.1f}m ({dist/1000:.2f}km)")
    
    # Show network bounds
    bounds = gdf.total_bounds
    print(f"\n{'='*50}")
    print("NETWORK BOUNDS")
    print(f"{'='*50}")
    print(f"X: {bounds[0]:.1f} to {bounds[2]:.1f} (width: {bounds[2]-bounds[0]:.1f}m)")
    print(f"Y: {bounds[1]:.1f} to {bounds[3]:.1f} (height: {bounds[3]-bounds[1]:.1f}m)")
    
    # Check if points are within network bounds
    print(f"\n{'='*50}")
    print("POINT COVERAGE CHECK")
    print(f"{'='*50}")
    for label, coords in points.items():
        x_in = bounds[0] <= coords[0] <= bounds[2]
        y_in = bounds[1] <= coords[1] <= bounds[3]
        status = "✅ INSIDE" if (x_in and y_in) else "❌ OUTSIDE"
        print(f"{label}: {status} network bounds")
    
    plt.tight_layout()
    plt.savefig('test_points_on_network.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'test_points_on_network.png'")

if __name__ == "__main__":
    plot_test_points()
