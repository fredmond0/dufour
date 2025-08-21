import geopandas as gpd
import networkx as nx
import os
import requests
import math
from shapely.geometry import Point, box, LineString
from shapely.ops import nearest_points, split
import momepy

# --- Configuration ---
PREPROCESSED_TLM3D_PATH = './data/preprocessed/tlm3d_network_processed.gpkg'
PREPROCESSED_SKI_PATH = './data/preprocessed/ski_network_processed.gpkg'

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
    graph = momepy.gdf_to_nx(gdf, approach='primal', length='length', multigraph=True)
    
    # Momepy doesn't reliably transfer geometry to MultiGraphs, so we do it manually.
    for u, v, key, data in graph.edges(keys=True, data=True):
        # Find the original geometry for this edge pair
        original_edge = gdf[
            (gdf.geometry.apply(lambda g: tuple(g.coords[0]) in [(u), (v)])) &
            (gdf.geometry.apply(lambda g: tuple(g.coords[-1]) in [(u), (v)]))
        ]
        if not original_edge.empty:
            # Get the geometry that most closely matches the length
            # This helps differentiate parallel edges
            best_match_idx = (original_edge['length'] - data['length']).abs().idxmin()
            data['geometry'] = original_edge.loc[best_match_idx, 'geometry']
            
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

def find_nearest_edge(graph, point):
    """Finds the nearest edge in the graph to a given point."""
    min_dist = float('inf')
    nearest_edge = None
    
    try:
        for u, v, key, data in graph.edges(data=True, keys=True):
            if 'geometry' in data:
                dist = point.distance(data['geometry'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_edge = ((u, v, key), data['geometry'])
        
        if nearest_edge is None:
            print(f"Warning: No edges with geometry found in graph")
            return None
            
        return nearest_edge
        
    except Exception as e:
        print(f"Error in find_nearest_edge: {e}")
        return None

def calculate_shortest_path(graph, start_coords, end_coords):
    """
    Finds the shortest path by temporarily adding start and end points to the graph.
    This simplifies routing and correctly handles parallel trails.
    """
    # Create a copy of the graph to modify
    G = graph.copy()
    
    # Find the nearest network segments to the user's start and end points
    start_edge_data = find_nearest_edge(G, Point(start_coords))
    end_edge_data = find_nearest_edge(G, Point(end_coords))
    
    if not start_edge_data or not end_edge_data:
        print("Could not find nearest edge for start or end point.")
        return None, None
    
    # Safely unpack the edge data with flexible unpacking
    try:
        if len(start_edge_data) == 2:
            (start_u, start_v, start_key), start_geom = start_edge_data
        else:
            print(f"Unexpected start_edge_data structure: {start_edge_data}")
            return None, None
            
        if len(end_edge_data) == 2:
            (end_u, end_v, end_key), end_geom = end_edge_data
        else:
            print(f"Unexpected end_edge_data structure: {end_edge_data}")
            return None, None
            
    except (ValueError, TypeError) as e:
        print(f"Error unpacking edge data: {e}")
        print(f"start_edge_data: {start_edge_data}")
        print(f"end_edge_data: {end_edge_data}")
        return None, None

    # Find the exact points on the lines that are closest to the user's clicks
    start_proj_point = nearest_points(start_geom, Point(start_coords))[0]
    end_proj_point = nearest_points(end_geom, Point(end_coords))[0]
    
    # Define new node names for the projected start and end points
    start_node = tuple(start_proj_point.coords[0])
    end_node = tuple(end_proj_point.coords[0])
    
    # --- Add projected start point to the graph ---
    # Split the original edge at the projected point
    geom1, geom2 = split(start_geom, start_proj_point).geoms
    # Add the new node and two new edges
    G.add_node(start_node)
    G.add_edge(start_u, start_node, length=geom1.length, geometry=geom1)
    G.add_edge(start_node, start_v, length=geom2.length, geometry=geom2)
    G.remove_edge(start_u, start_v, key=start_key)

    # --- Add projected end point to the graph ---
    # Need to check if the end point is on the same edge as the start
    if (end_u, end_v, end_key) == (start_u, start_v, start_key):
        # If so, split one of the newly created segments
        if end_proj_point.within(geom1):
             _, _, _, new_start_data = find_nearest_edge(G, end_proj_point)
             (new_start_u, new_start_v, new_start_key) = new_start_data[0]
             geom3, geom4 = split(geom1, end_proj_point).geoms
             G.add_node(end_node)
             G.add_edge(new_start_u, end_node, length=geom3.length, geometry=geom3)
             G.add_edge(end_node, new_start_v, length=geom4.length, geometry=geom4)
             G.remove_edge(new_start_u, new_start_v, key=new_start_key)
        else:
             _, _, _, new_start_data = find_nearest_edge(G, end_proj_point)
             (new_start_u, new_start_v, new_start_key) = new_start_data[0]
             geom3, geom4 = split(geom2, end_proj_point).geoms
             G.add_node(end_node)
             G.add_edge(new_start_u, end_node, length=geom3.length, geometry=geom3)
             G.add_edge(end_node, new_start_v, length=geom4.length, geometry=geom4)
             G.remove_edge(new_start_u, new_start_v, key=new_start_key)
    else:
        # Otherwise, split the original end edge
        geom3, geom4 = split(end_geom, end_proj_point).geoms
        G.add_node(end_node)
        G.add_edge(end_u, end_node, length=geom3.length, geometry=geom3)
        G.add_edge(end_node, end_v, length=geom4.length, geometry=geom4)
        G.remove_edge(end_u, end_v, key=end_key)

    # Now, find the shortest path on the modified graph
    try:
        path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
        path_length = nx.shortest_path_length(G, source=start_node, target=end_node, weight='length')
    except nx.NetworkXNoPath:
        print("No path found between the start and end nodes.")
        return None, None
        
    # Reconstruct the full route geometry from the path nodes
    route_geometry = [start_coords]
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i+1]
        # Find the best edge to use between these nodes
        best_key = min(G[u][v], key=lambda k: G[u][v][k]['length'])
        edge_geom = G[u][v][best_key]['geometry']
        
        coords = list(edge_geom.coords)
        if Point(coords[0]).distance(Point(u)) > 1e-9:
            coords.reverse()
        route_geometry.extend(coords[1:])
    route_geometry.append(end_coords)
    
    return route_geometry, path_length

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