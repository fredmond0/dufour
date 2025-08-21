#!/usr/bin/env python3
"""
Debug Robust Routing Issues

Compare the simple nearest-node approach vs robust approach
to understand why robust routing creates massive detours.
"""

import geopandas as gpd
import networkx as nx
import momepy
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import numpy as np

def test_simple_vs_robust_routing():
    """Compare simple nearest-node routing vs robust routing."""
    print("🔍 Debugging Robust Routing Issues")
    print("="*60)
    
    # Load network
    gdf = gpd.read_file('./data/skitouring/cleaningtesting/cleaned_network.gpkg')
    print(f"Loaded network: {len(gdf)} segments")
    
    # Create graph
    graph = momepy.gdf_to_nx(gdf, approach='primal', length='length', multigraph=True)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test coordinates
    coord1 = (2706198.5, 1183207.5)   # Point 1
    coord3 = (2707977.395, 1180824.833)  # Point 3 (problematic route)
    
    print(f"\nTesting Route 1→3:")
    print(f"Start: {coord1}")
    print(f"End: {coord3}")
    
    # Method 1: Simple nearest-node approach
    print(f"\n{'='*40}")
    print("METHOD 1: Simple Nearest-Node Routing")
    print(f"{'='*40}")
    
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
    
    start_node, start_dist = find_nearest_node(graph, coord1)
    end_node, end_dist = find_nearest_node(graph, coord3)
    
    print(f"Start node: {start_node} (distance: {start_dist:.1f}m)")
    print(f"End node: {end_node} (distance: {end_dist:.1f}m)")
    
    try:
        simple_path = nx.shortest_path(graph, start_node, end_node, weight='length')
        simple_length = nx.shortest_path_length(graph, start_node, end_node, weight='length')
        print(f"✅ Simple route: {simple_length:.0f}m ({len(simple_path)} nodes)")
        simple_success = True
    except nx.NetworkXNoPath:
        print(f"❌ Simple route: NO PATH")
        simple_success = False
        simple_length = None
        simple_path = None
    
    # Method 2: Robust approach analysis
    print(f"\n{'='*40}")
    print("METHOD 2: Robust Routing Analysis")
    print(f"{'='*40}")
    
    # Convert to edges GDF like robust routing does
    nodes_gdf, edges_gdf = momepy.nx_to_gdf(graph)
    print(f"Edges GDF: {len(edges_gdf)} segments")
    
    # Find nearest segments (like robust routing)
    def find_nearest_segment(gdf, point):
        min_dist = float('inf')
        nearest_segment = None
        nearest_idx = None
        for idx, row in gdf.iterrows():
            dist = point.distance(row.geometry)
            if dist < min_dist:
                min_dist = dist
                nearest_segment = row.geometry
                nearest_idx = idx
        return nearest_segment, nearest_idx, min_dist
    
    start_point = Point(coord1)
    end_point = Point(coord3)
    
    start_segment, start_idx, start_seg_dist = find_nearest_segment(edges_gdf, start_point)
    end_segment, end_idx, end_seg_dist = find_nearest_segment(edges_gdf, end_point)
    
    print(f"Start segment {start_idx}: distance {start_seg_dist:.1f}m")
    print(f"End segment {end_idx}: distance {end_seg_dist:.1f}m")
    
    # Project points onto segments
    start_snap_point = start_segment.interpolate(start_segment.project(start_point))
    end_snap_point = end_segment.interpolate(end_segment.project(end_point))
    
    print(f"Start snap point: ({start_snap_point.x:.1f}, {start_snap_point.y:.1f})")
    print(f"End snap point: ({end_snap_point.x:.1f}, {end_snap_point.y:.1f})")
    
    # Check if these are the same segments that simple routing would use
    start_node_coords = (graph.nodes[start_node]['x'], graph.nodes[start_node]['y'])
    end_node_coords = (graph.nodes[end_node]['x'], graph.nodes[end_node]['y'])
    
    print(f"\nComparison:")
    print(f"Simple start node: ({start_node_coords[0]:.1f}, {start_node_coords[1]:.1f})")
    print(f"Robust start snap: ({start_snap_point.x:.1f}, {start_snap_point.y:.1f})")
    print(f"Distance difference: {Point(start_node_coords).distance(start_snap_point):.1f}m")
    
    print(f"Simple end node: ({end_node_coords[0]:.1f}, {end_node_coords[1]:.1f})")
    print(f"Robust end snap: ({end_snap_point.x:.1f}, {end_snap_point.y:.1f})")
    print(f"Distance difference: {Point(end_node_coords).distance(end_snap_point):.1f}m")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Simple routing
    gdf.plot(ax=ax1, color='lightgray', linewidth=0.5, alpha=0.7)
    
    if simple_success:
        # Plot simple route
        route_x = [graph.nodes[node]['x'] for node in simple_path]
        route_y = [graph.nodes[node]['y'] for node in simple_path]
        ax1.plot(route_x, route_y, 'b-', linewidth=3, alpha=0.8, label=f'Simple Route ({simple_length:.0f}m)')
        
        # Plot nodes
        ax1.scatter(route_x, route_y, color='blue', s=50, zorder=5)
    
    # Plot test points
    ax1.scatter([coord1[0]], [coord1[1]], color='red', s=150, marker='o', zorder=10, label='Start')
    ax1.scatter([coord3[0]], [coord3[1]], color='green', s=150, marker='s', zorder=10, label='End')
    
    # Plot nearest nodes
    ax1.scatter([start_node_coords[0]], [start_node_coords[1]], color='orange', s=100, marker='^', zorder=8, label='Nearest Nodes')
    ax1.scatter([end_node_coords[0]], [end_node_coords[1]], color='orange', s=100, marker='^', zorder=8)
    
    ax1.set_title('Simple Nearest-Node Routing')
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Plot 2: Robust routing analysis
    gdf.plot(ax=ax2, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Plot test points
    ax2.scatter([coord1[0]], [coord1[1]], color='red', s=150, marker='o', zorder=10, label='Start')
    ax2.scatter([coord3[0]], [coord3[1]], color='green', s=150, marker='s', zorder=10, label='End')
    
    # Plot snap points
    ax2.scatter([start_snap_point.x], [start_snap_point.y], color='purple', s=100, marker='*', zorder=8, label='Snap Points')
    ax2.scatter([end_snap_point.x], [end_snap_point.y], color='purple', s=100, marker='*', zorder=8)
    
    # Plot nearest segments
    if start_segment:
        x_coords, y_coords = start_segment.xy
        ax2.plot(x_coords, y_coords, 'r-', linewidth=4, alpha=0.8, label='Nearest Segments')
    if end_segment:
        x_coords, y_coords = end_segment.xy
        ax2.plot(x_coords, y_coords, 'r-', linewidth=4, alpha=0.8)
    
    ax2.set_title('Robust Routing: Nearest Segments & Snap Points')
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('routing_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if simple_success:
        straight_dist = Point(coord1).distance(Point(coord3))
        simple_efficiency = (straight_dist / simple_length) * 100
        print(f"Simple routing: {simple_length:.0f}m (efficiency: {simple_efficiency:.1f}%)")
        print(f"Straight-line: {straight_dist:.0f}m")
        
        if simple_length < 5000:
            print(f"✅ Simple routing finds reasonable path")
            print(f"❌ Robust routing creates detours - likely due to segment splitting issues")
        else:
            print(f"⚠️ Even simple routing is long - network connectivity issue")
    
    print(f"\nVisualization saved as 'routing_comparison.png'")

if __name__ == "__main__":
    test_simple_vs_robust_routing()
