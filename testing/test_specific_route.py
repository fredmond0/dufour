import geopandas as gpd
import networkx as nx
import momepy
import numpy as np
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt

print("=== TESTING SPECIFIC ROUTE ===\n")

# Load the cleaned network
print("1. Loading cleaned network...")
cleaned_gdf = gpd.read_file('routing_network.gpkg')
print(f"   Loaded {len(cleaned_gdf)} segments")

# Convert to NetworkX graph
print("\n2. Converting to NetworkX graph...")
G = momepy.gdf_to_nx(cleaned_gdf, approach='primal')
print(f"   Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Find the largest connected component
components = list(nx.connected_components(G))
main_component = max(components, key=len)
print(f"\n3. Main component has {len(main_component)} nodes")

# Create subgraph of main component
main_G = G.subgraph(main_component)
print(f"   Main subgraph: {main_G.number_of_nodes()} nodes, {main_G.number_of_nodes()} edges")

# Test routing between the two specific points from your images
print("\n4. Testing route between specific points...")

# Point 1: fid 113, coordinates (2706198.5, 1183207.5)
point1_coords = (2706198.5, 1183207.5)
point1 = Point(point1_coords)

# Point 2: fid 115, coordinates (2706725, 1183530)  
point2_coords = (2706725, 1183530)
point2 = Point(point2_coords)

print(f"   Point 1: {point1_coords}")
print(f"   Point 2: {point2_coords}")

# Find nearest nodes to these points
def find_nearest_node(graph, point):
    """Find the nearest node in the graph to a given point"""
    min_dist = float('inf')
    nearest_node = None
    
    for node in graph.nodes():
        node_coords = (graph.nodes[node]['x'], graph.nodes[node]['y'])
        dist = point.distance(Point(node_coords))
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node, min_dist

print("\n5. Finding nearest nodes...")
node1, dist1 = find_nearest_node(main_G, point1)
node2, dist2 = find_nearest_node(main_G, point2)

print(f"   Nearest to Point 1: {node1} (distance: {dist1:.1f}m)")
print(f"   Nearest to Point 2: {node2} (distance: {dist2:.1f}m)")

# Test routing between these nodes
print("\n6. Testing routing...")
route_path = None
route_length = None

try:
    path = nx.shortest_path(main_G, node1, node2, weight='length')
    path_length = nx.shortest_path_length(main_G, node1, node2, weight='length')
    
    print(f"   ✅ Route found!")
    print(f"   Path length: {path_length:.1f}m")
    print(f"   Number of nodes: {len(path)}")
    
    # Show the path
    print(f"   Path: {node1} → ... → {node2}")
    if len(path) > 2:
        print(f"   Intermediate nodes: {len(path)-2}")
    
    # Calculate straight-line distance for comparison
    straight_dist = point1.distance(point2)
    print(f"   Straight-line distance: {straight_dist:.1f}m")
    print(f"   Route efficiency: {(straight_dist/path_length)*100:.1f}%")
    
    # Store for plotting
    route_path = path
    route_length = path_length
    
except nx.NetworkXNoPath:
    print(f"   ❌ No path found between {node1} and {node2}")
except Exception as e:
    print(f"   ❌ Error during routing: {e}")

# Visualize the route
if route_path:
    print("\n7. Creating route visualization...")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Network with route highlighted
    cleaned_gdf.plot(ax=ax1, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Plot the route path
    route_coords = []
    for i in range(len(route_path) - 1):
        u, v = route_path[i], route_path[i + 1]
        if main_G.has_edge(u, v):
            # Get the edge geometry
            edge_data = main_G[u][v]
            if 'geometry' in edge_data:
                route_coords.append(edge_data['geometry'])
    
    # Plot route segments
    for geom in route_coords:
        if geom.geom_type == 'LineString':
            x_coords, y_coords = geom.xy
            ax1.plot(x_coords, y_coords, color='red', linewidth=3, alpha=0.8)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                x_coords, y_coords = line.xy
                ax1.plot(x_coords, y_coords, color='red', linewidth=3, alpha=0.8)
    
    # Plot start and end points
    ax1.scatter([point1_coords[0]], [point1_coords[1]], color='green', s=100, zorder=5, label='Start Point')
    ax1.scatter([point2_coords[0]], [point2_coords[1]], color='blue', s=100, zorder=5, label='End Point')
    
    # Plot the actual route nodes
    route_x = [main_G.nodes[node]['x'] for node in route_path]
    route_y = [main_G.nodes[node]['y'] for node in route_path]
    ax1.plot(route_x, route_y, 'o-', color='orange', markersize=8, linewidth=2, alpha=0.8, label='Route Nodes')
    
    ax1.set_title(f'Route Visualization\nPath Length: {route_length:.1f}m vs Straight: {straight_dist:.1f}m')
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Plot 2: Zoomed in view around the route
    # Calculate bounding box around the route
    route_x_coords = [main_G.nodes[node]['x'] for node in route_path]
    route_y_coords = [main_G.nodes[node]['y'] for node in route_path]
    
    x_min, x_max = min(route_x_coords), max(route_x_coords)
    y_min, y_max = min(route_y_coords), min(route_y_coords)
    
    # Add some padding
    x_pad = (x_max - x_min) * 0.2
    y_pad = (y_max - y_min) * 0.2
    
    # Plot zoomed view
    cleaned_gdf.plot(ax=ax2, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Plot route in zoomed view
    for geom in route_coords:
        if geom.geom_type == 'LineString':
            x_coords, y_coords = geom.xy
            ax2.plot(x_coords, y_coords, color='red', linewidth=4, alpha=0.8)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                x_coords, y_coords = line.xy
                ax2.plot(x_coords, y_coords, color='red', linewidth=4, alpha=0.8)
    
    # Plot points and nodes
    ax2.scatter([point1_coords[0]], [point1_coords[1]], color='green', s=150, zorder=5, label='Start Point')
    ax2.scatter([point2_coords[0]], [point2_coords[1]], color='blue', s=150, zorder=5, label='End Point')
    ax2.plot(route_x, route_y, 'o-', color='orange', markersize=10, linewidth=3, alpha=0.8, label='Route Nodes')
    
    # Set zoomed extent
    ax2.set_xlim(x_min - x_pad, x_max + x_pad)
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)
    
    ax2.set_title('Zoomed Route View')
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    print("   Route visualization displayed!")
    print("   Red lines = actual route path")
    print("   Orange dots = route nodes")
    print("   Green dot = start point")
    print("   Blue dot = end point")

print("\n=== ROUTE TEST COMPLETE ===")
print("If routing works, you should see a reasonable path length and efficiency.")
print("If it still takes the 'long way around', we may need to adjust the vertex collapsing.")
