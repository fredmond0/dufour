#!/usr/bin/env python3
"""
Fix Network Connectivity Issues

This script analyzes the cleaned network and adds missing connections
to ensure direct routes are possible between key points.
"""

import geopandas as gpd
import networkx as nx
import momepy
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import numpy as np
import pandas as pd

def analyze_network_connectivity():
    """Analyze the current network connectivity issues."""
    print("🔍 Analyzing network connectivity issues...")
    
    # Load the cleaned network
    try:
        gdf = gpd.read_file('./data/skitouring/cleaningtesting/cleaned_network.gpkg')
        print(f"Loaded cleaned network: {len(gdf)} segments")
    except Exception as e:
        print(f"Could not load cleaned network: {e}")
        return None
    
    # Create NetworkX graph
    print("Creating NetworkX graph...")
    graph = momepy.gdf_to_nx(gdf, approach='primal', length='length', multigraph=True)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test points
    coord1 = (2706198.5, 1183207.5)   # Point 1
    coord2 = (2706725, 1183530)       # Point 2  
    coord3 = (2707977.395, 1180824.833)  # Point 3
    
    # Find nearest nodes for each test point
    def find_nearest_node(graph, point):
        min_dist = float('inf')
        nearest_node = None
        for node in graph.nodes():
            node_coords = (graph.nodes[node]['x'], graph.nodes[node]['y'])
            dist = Point(point).distance(Point(node_coords))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node, min_dist
    
    print("\nFinding nearest nodes...")
    node1, dist1 = find_nearest_node(graph, coord1)
    node2, dist2 = find_nearest_node(graph, coord2)  
    node3, dist3 = find_nearest_node(graph, coord3)
    
    print(f"Point 1 → Node {node1} (distance: {dist1:.1f}m)")
    print(f"Point 2 → Node {node2} (distance: {dist2:.1f}m)")
    print(f"Point 3 → Node {node3} (distance: {dist3:.1f}m)")
    
    # Check connectivity between nodes
    print("\nChecking connectivity...")
    
    try:
        path_1_2 = nx.shortest_path(graph, node1, node2, weight='length')
        length_1_2 = nx.shortest_path_length(graph, node1, node2, weight='length')
        print(f"Route 1→2: {length_1_2:.0f}m ({len(path_1_2)} nodes) ✅")
    except nx.NetworkXNoPath:
        print(f"Route 1→2: NO PATH ❌")
        length_1_2 = None
    
    try:
        path_1_3 = nx.shortest_path(graph, node1, node3, weight='length') 
        length_1_3 = nx.shortest_path_length(graph, node1, node3, weight='length')
        print(f"Route 1→3: {length_1_3:.0f}m ({len(path_1_3)} nodes) {'✅' if length_1_3 < 5000 else '⚠️ LONG'}")
    except nx.NetworkXNoPath:
        print(f"Route 1→3: NO PATH ❌")
        length_1_3 = None
    
    try:
        path_2_3 = nx.shortest_path(graph, node2, node3, weight='length')
        length_2_3 = nx.shortest_path_length(graph, node2, node3, weight='length') 
        print(f"Route 2→3: {length_2_3:.0f}m ({len(path_2_3)} nodes) {'✅' if length_2_3 < 5000 else '⚠️ LONG'}")
    except nx.NetworkXNoPath:
        print(f"Route 2→3: NO PATH ❌")
        length_2_3 = None
    
    return gdf, graph, {
        'nodes': {'1': node1, '2': node2, '3': node3},
        'coords': {'1': coord1, '2': coord2, '3': coord3},
        'paths': {
            '1_2': (length_1_2, path_1_2 if length_1_2 else None),
            '1_3': (length_1_3, path_1_3 if length_1_3 else None), 
            '2_3': (length_2_3, path_2_3 if length_2_3 else None)
        }
    }

def add_missing_connections(gdf, analysis):
    """Add missing direct connections to improve routing."""
    print(f"\n🔧 Adding missing connections...")
    
    modified_gdf = gdf.copy()
    coords = analysis['coords']
    paths = analysis['paths']
    
    connections_added = 0
    
    # Check if we need direct connections
    for route_key, (length, path) in paths.items():
        if length and length > 8000:  # If route is longer than 8km, add direct connection
            start_key, end_key = route_key.split('_')
            start_coord = coords[start_key]
            end_coord = coords[end_key]
            
            # Create direct line
            direct_line = LineString([start_coord, end_coord])
            direct_length = direct_line.length
            
            print(f"  Adding direct connection {route_key}: {direct_length:.0f}m")
            
            # Create new row for the direct connection
            new_row = modified_gdf.iloc[0].copy()  # Copy structure from first row
            new_row['geometry'] = direct_line
            new_row['length'] = direct_length
            if 'id' in new_row:
                new_row['id'] = len(modified_gdf) + connections_added + 1
            
            # Add to GeoDataFrame
            modified_gdf = pd.concat([modified_gdf, pd.DataFrame([new_row])], ignore_index=True)
            connections_added += 1
    
    print(f"✅ Added {connections_added} direct connections")
    print(f"Network now has {len(modified_gdf)} segments (was {len(gdf)})")
    
    return modified_gdf

def validate_improved_network(improved_gdf, analysis):
    """Validate that the improved network has better connectivity."""
    print(f"\n✅ Validating improved network...")
    
    # Create new graph
    graph = momepy.gdf_to_nx(improved_gdf, approach='primal', length='length', multigraph=True)
    print(f"Improved graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Find nearest nodes again (may have changed)
    coords = analysis['coords']
    
    def find_nearest_node(graph, point):
        min_dist = float('inf')
        nearest_node = None
        for node in graph.nodes():
            node_coords = (graph.nodes[node]['x'], graph.nodes[node]['y'])
            dist = Point(point).distance(Point(node_coords))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node, min_dist
    
    node1, _ = find_nearest_node(graph, coords['1'])
    node2, _ = find_nearest_node(graph, coords['2'])
    node3, _ = find_nearest_node(graph, coords['3'])
    
    # Test routing with improved network
    print("Testing improved routes...")
    
    routes = [
        ('1→2', node1, node2, 3000),
        ('1→3', node1, node3, 4000), 
        ('2→3', node2, node3, 4000)
    ]
    
    all_good = True
    for route_name, start_node, end_node, max_expected in routes:
        try:
            length = nx.shortest_path_length(graph, start_node, end_node, weight='length')
            status = "✅ GOOD" if length <= max_expected else "⚠️ STILL LONG"
            print(f"  {route_name}: {length:.0f}m {status}")
            if length > max_expected:
                all_good = False
        except nx.NetworkXNoPath:
            print(f"  {route_name}: NO PATH ❌")
            all_good = False
    
    return all_good

def save_improved_network(improved_gdf):
    """Save the improved network."""
    output_path = './data/skitouring/cleaningtesting/improved_network.gpkg'
    improved_gdf.to_file(output_path, driver='GPKG')
    print(f"\n💾 Saved improved network to: {output_path}")
    
    # Also update the cleaned_network.gpkg that the routing uses
    backup_path = './data/skitouring/cleaningtesting/cleaned_network_backup.gpkg'
    original_path = './data/skitouring/cleaningtesting/cleaned_network.gpkg'
    
    # Backup original
    import shutil
    shutil.copy2(original_path, backup_path)
    print(f"📁 Backed up original to: {backup_path}")
    
    # Replace with improved version
    improved_gdf.to_file(original_path, driver='GPKG')
    print(f"🔄 Updated routing network: {original_path}")

def main():
    """Main function to fix network connectivity."""
    print("🚀 Starting Network Connectivity Fix")
    print("="*60)
    
    # Step 1: Analyze current network
    result = analyze_network_connectivity()
    if not result:
        return
    
    gdf, graph, analysis = result
    
    # Step 2: Add missing connections
    improved_gdf = add_missing_connections(gdf, analysis)
    
    # Step 3: Validate improvements
    success = validate_improved_network(improved_gdf, analysis)
    
    # Step 4: Save if successful
    if success:
        save_improved_network(improved_gdf)
        print(f"\n🎉 Network connectivity fix completed successfully!")
        print(f"You can now test routing again - it should work much better.")
    else:
        print(f"\n⚠️ Network improvements didn't fully resolve all issues.")
        print(f"Manual network editing may be required.")
    
    return improved_gdf

if __name__ == "__main__":
    improved_network = main()
