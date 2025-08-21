import geopandas as gpd
import networkx as nx
import os
import requests
import math
import pandas as pd
from shapely.geometry import Point, box, LineString
from shapely.ops import nearest_points
import momepy

# --- Configuration ---
PREPROCESSED_TLM3D_PATH = './data/preprocessed/tlm3d_network_processed.gpkg'
PREPROCESSED_SKI_PATH = './data/skitouring/cleaningtesting/cleaned_network.gpkg'

def create_clipped_network(start_coords, end_coords, buffer_distance, network_type):
    """Creates a clipped GeoDataFrame from a pre-processed file."""
    filepath = {
        'tlm3d': PREPROCESSED_TLM3D_PATH,
        'ski_touring': PREPROCESSED_SKI_PATH
    }.get(network_type)
    if not filepath:
        return None
    if not os.path.exists(filepath):
        return None
    bbox = box(
        min(start_coords[0], end_coords[0]) - buffer_distance,
        min(start_coords[1], end_coords[1]) - buffer_distance,
        max(start_coords[0], end_coords[0]) + buffer_distance,
        max(start_coords[1], end_coords[1]) + buffer_distance
    )
    try:
        if network_type == 'tlm3d':
            clipped_gdf = gpd.read_file(filepath, layer='tlm_strassen_strasse', bbox=bbox)
        else:
            clipped_gdf = gpd.read_file(filepath, bbox=bbox)
        if clipped_gdf.empty:
            return None
        return clipped_gdf
    except Exception as e:
        print(f"Error reading clipped network: {e}")
        return None

def load_network_to_graph(gdf):
    """Converts a GeoDataFrame into a NetworkX graph."""
    print("Creating network graph using momepy...")
    gdf = gdf.reset_index(drop=True)
    if 'length' not in gdf.columns:
        gdf['length'] = gdf.geometry.length
    
    # Create graph with geometry preservation
    graph = momepy.gdf_to_nx(gdf, approach='primal', length='length', multigraph=True)
    
    print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

def fetch_elevations_for_coordinates(coordinates_2d, chunk_size=100):
    """Fetches elevation data for a route from the Swisstopo Profile API."""
    if not coordinates_2d or len(coordinates_2d) < 2:
        return []
        
    if len(coordinates_2d) > chunk_size:
        step = len(coordinates_2d) // chunk_size
        coordinates_2d = coordinates_2d[::step]

    coordinates_array = [[coord[0], coord[1]] for coord in coordinates_2d if len(coord) >= 2]
    geojson = {"type": "LineString", "coordinates": coordinates_array}
    
    import json
    import urllib.parse
    geom_param = urllib.parse.quote(json.dumps(geojson))
    api_url = f"https://api3.geo.admin.ch/rest/services/profile.json?geom={geom_param}&sr=2056&nb_points=200"
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return [(p['easting'], p['northing'], p['alts']['COMB']) for p in data if 'alts' in p]
    except Exception as e:
        print(f"Warning: Failed to fetch elevation profile: {e}")
    return []

def calculate_elevation_profile(coordinates_3d):
    """Converts 3D route coordinates into an elevation profile."""
    if len(coordinates_3d) < 2:
        return []
    profile, cumulative_distance = [], 0.0
    for i in range(len(coordinates_3d)):
        x, y, elevation = coordinates_3d[i]
        if i > 0:
            prev_x, prev_y, _ = coordinates_3d[i-1]
            cumulative_distance += math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        profile.append([cumulative_distance, elevation])
    return profile

# Removed unused find_nearest_edge function - using simpler approach

def find_nearest_segment(gdf, point):
    """Finds the geometrically nearest line segment to a point."""
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

def project_point_to_line(point, line):
    """Projects a point onto a line and returns the projected point."""
    return line.interpolate(line.project(point))

def split_line_at_point(line, point, tolerance=0.1):
    """Splits a line at a point with robust handling."""
    from shapely.ops import split
    
    print(f"      Attempting to split line (length: {line.length:.1f}m)")
    print(f"      Point coords: ({point.x:.1f}, {point.y:.1f})")
    
    # First check if point is too close to endpoints
    coords = list(line.coords)
    start_point = Point(coords[0])
    end_point = Point(coords[-1])
    
    start_dist = point.distance(start_point)
    end_dist = point.distance(end_point)
    print(f"      Distance to start: {start_dist:.3f}m, to end: {end_dist:.3f}m")
    
    if start_dist < tolerance or end_dist < tolerance:
        print(f"      Point too close to line endpoints - skipping split")
        return [line]
    
    # Check if point is actually on the line
    line_dist = line.distance(point)
    print(f"      Distance from point to line: {line_dist:.6f}m")
    
    try:
        # Try direct split first
        print(f"      Trying direct split...")
        split_result = split(line, point)
        print(f"      Split result type: {split_result.geom_type}")
        
        if split_result.geom_type == 'GeometryCollection':
            lines = []
            print(f"      GeometryCollection with {len(split_result.geoms)} geometries")
            for i, geom in enumerate(split_result.geoms):
                print(f"        Geom {i}: {geom.geom_type}, length: {geom.length:.3f}m")
                if geom.geom_type == 'LineString' and geom.length > tolerance:
                    lines.append(geom)
                elif geom.geom_type == 'MultiLineString':
                    lines.extend([g for g in geom.geoms if g.length > tolerance])
            
            print(f"      Found {len(lines)} valid line parts")
            if len(lines) >= 2:
                return lines[:2]  # Return only first 2 parts
            else:
                print(f"      Not enough valid parts, trying buffered approach...")
                
        elif split_result.geom_type == 'MultiLineString':
            lines = [g for g in split_result.geoms if g.length > tolerance]
            print(f"      MultiLineString with {len(lines)} valid parts")
            return lines if len(lines) >= 2 else [line]
        else:
            print(f"      Unexpected split result type: {split_result.geom_type}")
        
        # If direct split didn't work, try with buffer
        print(f"      Trying buffered split...")
        buffered_point = point.buffer(0.01)  # 1cm buffer
        split_result = split(line, buffered_point)
        print(f"      Buffered split result type: {split_result.geom_type}")
        
        if split_result.geom_type == 'GeometryCollection':
            lines = []
            for geom in split_result.geoms:
                if geom.geom_type == 'LineString' and geom.length > tolerance:
                    lines.append(geom)
                elif geom.geom_type == 'MultiLineString':
                    lines.extend([g for g in geom.geoms if g.length > tolerance])
            
            print(f"      Buffered approach found {len(lines)} valid parts")
            if len(lines) >= 2:
                return lines[:2]
            
        print(f"      All split attempts failed")
        return [line]
            
    except Exception as e:
        print(f"      Exception during split: {e}")
        return [line]

def calculate_shortest_path(graph, start_coords, end_coords):
    """
    Robust shortest path calculation using the two-phase approach from robust_routing.py
    """
    print(f"  Using robust two-phase routing approach...")
    
    # Convert graph back to GeoDataFrame for the robust routing approach
    nodes_gdf, edges_gdf = momepy.nx_to_gdf(graph)
    gdf = edges_gdf
    
    # Create Point objects
    start_point = Point(start_coords)
    end_point = Point(end_coords)
    
    print("  Phase 1: Preparing network with start/end points...")
    
    # Create a copy to modify
    modified_gdf = gdf.copy()
    
    # Process start point
    print("  Processing start point...")
    start_segment, start_idx, start_dist = find_nearest_segment(modified_gdf, start_point)
    print(f"    Start segment {start_idx}: distance {start_dist:.1f}m")
    
    start_snap_point = project_point_to_line(start_point, start_segment)
    print(f"    Projected start point: ({start_snap_point.x:.1f}, {start_snap_point.y:.1f})")
    
    # Process end point
    print("  Processing end point...")
    end_segment, end_idx, end_dist = find_nearest_segment(modified_gdf, end_point)
    print(f"    End segment {end_idx}: distance {end_dist:.1f}m")
    
    end_snap_point = project_point_to_line(end_point, end_segment)
    print(f"    Projected end point: ({end_snap_point.x:.1f}, {end_snap_point.y:.1f})")
    
    # Check if both points are on the same segment
    if start_idx == end_idx:
        print("    ⚠️ Both points on same segment - creating direct route")
        # Create a direct line segment between the two snap points
        from shapely.geometry import LineString
        direct_line = LineString([start_snap_point.coords[0], end_snap_point.coords[0]])
        direct_length = direct_line.length
        
        # Return direct route without graph processing
        route_geometry = [start_coords, end_coords]
        return route_geometry, direct_length
    
    # Split segments (only if different segments)
    print("  Splitting segments...")
    
    # Split start segment
    start_parts = split_line_at_point(start_segment, start_snap_point)
    print(f"    Split start segment into {len(start_parts)} parts")
    if len(start_parts) == 2:
        original_row = modified_gdf.loc[start_idx].copy()
        modified_gdf = modified_gdf.drop(start_idx).reset_index(drop=True)
        
        # Adjust end_idx if it was after start_idx
        if end_idx > start_idx:
            end_idx -= 1
        
        # Add split parts
        first_part = original_row.copy()
        first_part['geometry'] = start_parts[0]
        modified_gdf = pd.concat([modified_gdf, pd.DataFrame([first_part])], ignore_index=True)
        
        second_part = original_row.copy()
        second_part['geometry'] = start_parts[1]
        modified_gdf = pd.concat([modified_gdf, pd.DataFrame([second_part])], ignore_index=True)
        
        print(f"    ✅ Successfully split start segment")
    
    # Split end segment
    end_parts = split_line_at_point(end_segment, end_snap_point)
    print(f"    Split end segment into {len(end_parts)} parts")
    if len(end_parts) == 2:
        original_row = modified_gdf.loc[end_idx].copy()
        modified_gdf = modified_gdf.drop(end_idx).reset_index(drop=True)
        
        # Add split parts
        first_part = original_row.copy()
        first_part['geometry'] = end_parts[0]
        modified_gdf = pd.concat([modified_gdf, pd.DataFrame([first_part])], ignore_index=True)
        
        second_part = original_row.copy()
        second_part['geometry'] = end_parts[1]
        modified_gdf = pd.concat([modified_gdf, pd.DataFrame([second_part])], ignore_index=True)
        
        print(f"    ✅ Successfully split end segment")
    
    # Recalculate lengths
    modified_gdf['length'] = modified_gdf.geometry.length
    
    print("  Phase 2: Finding route on modified network...")
    print(f"  Modified GDF: {len(modified_gdf)} segments (was {len(gdf)})")
    
    # Convert to NetworkX graph
    modified_graph = momepy.gdf_to_nx(modified_gdf, approach='primal', length='length', multigraph=True)
    print(f"  Modified graph: {modified_graph.number_of_nodes()} nodes, {modified_graph.number_of_edges()} edges")
    
    # Find exact snap point nodes
    start_node = None
    end_node = None
    tolerance = 5.0  # 5 meter tolerance - increased from 1.0
    
    for node, data in modified_graph.nodes(data=True):
        node_point = Point(data['x'], data['y'])
        
        if start_node is None:
            start_dist_to_snap = node_point.distance(start_snap_point)
            if start_dist_to_snap <= tolerance:
                start_node = node
                print(f"    ✅ Found start node: {node} (distance: {start_dist_to_snap:.3f}m)")
        
        if end_node is None:
            end_dist_to_snap = node_point.distance(end_snap_point)
            if end_dist_to_snap <= tolerance:
                end_node = node
                print(f"    ✅ Found end node: {node} (distance: {end_dist_to_snap:.3f}m)")
        
        if start_node and end_node:
            break
    
    if not start_node or not end_node:
        print("  ❌ Could not find snap point nodes in modified graph")
        return None, None
    
    # Find shortest path
    try:
        path_nodes = nx.shortest_path(modified_graph, source=start_node, target=end_node, weight='length')
        path_length = nx.shortest_path_length(modified_graph, source=start_node, target=end_node, weight='length')
        
        print(f"  ✅ Route found!")
        print(f"  Path length: {path_length:.1f}m")
        print(f"  Number of nodes: {len(path_nodes)}")
        
        # Reconstruct route geometry from path nodes
        route_geometry = [start_coords]
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            if modified_graph.has_edge(u, v):
                # Get edge geometry
                edge_data = modified_graph[u][v]
                if isinstance(edge_data, dict):
                    edge_geom = edge_data.get('geometry')
                else:
                    # Multiple edges - get first one with geometry
                    edge_geom = None
                    for key, data in edge_data.items():
                        if 'geometry' in data:
                            edge_geom = data['geometry']
                            break
                
                if edge_geom and hasattr(edge_geom, 'coords'):
                    coords = list(edge_geom.coords)
                    if Point(coords[0]).distance(Point(u)) > 1e-9:
                        coords.reverse()
                    route_geometry.extend(coords[1:])
        
        route_geometry.append(end_coords)
        
        return route_geometry, path_length
        
    except nx.NetworkXNoPath:
        print("  ❌ No path found between start and end nodes")
        return None, None

# Removed unused functions - using simpler nearest-node approach

def calculate_route_from_gpkg(start_coords, end_coords, buffer_distance=5000, network_type='tlm3d'):
    """The main routing function."""
    print(f"\n=== Starting Route Calculation ({network_type}) ===")
    clipped_gdf = create_clipped_network(start_coords, end_coords, buffer_distance, network_type)
    
    if clipped_gdf is None or clipped_gdf.empty:
        return None, None, 0, []
    
    segments_loaded = len(clipped_gdf)
    network_graph = load_network_to_graph(clipped_gdf)
    
    if network_graph is None:
        return None, None, segments_loaded, []
        
    route_geometry, path_length = calculate_shortest_path(network_graph, start_coords, end_coords)
    
    if route_geometry:
        print("=== Route Calculation Successful ===")
        coordinates_3d = fetch_elevations_for_coordinates(route_geometry)
        elevation_profile = calculate_elevation_profile(coordinates_3d)
        return route_geometry, path_length, segments_loaded, elevation_profile
    else:
        print("=== Route Calculation Failed ===")
        return None, None, segments_loaded, []

if __name__ == '__main__':
    # (No changes to the testing section)
    pass