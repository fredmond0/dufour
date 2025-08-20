import geopandas as gpd
import networkx as nx
import os
import requests
import math
from shapely.geometry import Point, box, LineString, MultiPoint
from shapely.ops import nearest_points, split
import momepy
from pyproj import Transformer

# --- Configuration ---
PREPROCESSED_TLM3D_PATH = './data/preprocessed/tlm3d_network_processed.gpkg'
PREPROCESSED_SKI_PATH = './data/preprocessed/ski_network_processed.gpkg'

def create_clipped_network(start_coords, end_coords, buffer_distance, network_type):
    """Creates a clipped GeoDataFrame from a pre-processed file."""
    
    # Define file paths for each network type
    filepath = {
        'tlm3d': PREPROCESSED_TLM3D_PATH,
        'ski_touring': PREPROCESSED_SKI_PATH
    }.get(network_type)

    if not filepath:
        print(f"Error: Unknown network type '{network_type}'")
        return None
    
    # Check if file exists and provide helpful error message
    if not os.path.exists(filepath):
        if network_type == 'tlm3d':
            print(f"Error: TLM3D network file not found at {filepath}")
            print("This file needs to be preprocessed first. Ski routing is available.")
            return None
        else:
            print(f"Error: Network file not found at {filepath}")
            return None

    bbox = box(
        min(start_coords[0], end_coords[0]) - buffer_distance,
        min(start_coords[1], end_coords[1]) - buffer_distance,
        max(start_coords[0], end_coords[0]) + buffer_distance,
        max(start_coords[1], end_coords[1]) + buffer_distance
    )
    
    try:
        clipped_gdf = gpd.read_file(filepath, bbox=bbox)
        if clipped_gdf.empty:
            print("Warning: No segments found in the clipped area.")
            print("Try increasing the buffer distance or selecting different coordinates.")
            return None
        print(f"Loaded {len(clipped_gdf)} segments for {network_type} network.")
        return clipped_gdf
    except Exception as e:
        print(f"Error reading clipped network: {e}")
        return None

def load_network_to_graph(gdf):
    """
    Converts a GeoDataFrame into a NetworkX graph, ensuring geometry is
    stored on each edge.
    """
    print("Creating network graph using momepy...")
    print(f"Input GeoDataFrame has {len(gdf)} rows and columns: {gdf.columns.tolist()}")
    
    gdf = gdf.reset_index(drop=True)
    if 'length' not in gdf.columns:
        print("Adding length column to GeoDataFrame...")
        gdf['length'] = gdf.geometry.length
        
    print(f"GeoDataFrame CRS: {gdf.crs}")
    print(f"Sample geometry types: {gdf.geometry.geom_type.unique()}")
    
    # Create the graph using momepy
    graph = momepy.gdf_to_nx(gdf, approach='primal', length='length')
    print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    # --- Start of Corrected Section ---
    # Create a dictionary mapping node pairs to their LineString geometry
    # This is a more reliable way to map geometries than the previous method.
    edge_geometries = {}
    for index, row in gdf.iterrows():
        line = row.geometry
        start_node = tuple(line.coords[0])
        end_node = tuple(line.coords[-1])
        # Store geometry for both directions, as graph edges might not be ordered
        edge_geometries[(start_node, end_node)] = line
        edge_geometries[(end_node, start_node)] = line

    # Assign the geometry from the dictionary to each edge in the graph
    for u, v, data in graph.edges(data=True):
        if (u, v) in edge_geometries:
            data['geometry'] = edge_geometries[(u, v)]
    # --- End of Corrected Section ---
            
    # Verify that the geometries were assigned
    geometry_assigned = sum(1 for _, _, data in graph.edges(data=True) if 'geometry' in data)
    print(f"Geometry assigned to {geometry_assigned} edges out of {graph.number_of_edges()} total edges.")
    
    if geometry_assigned != graph.number_of_edges():
        print("Warning: Some edges were not assigned a geometry.")
        
    return graph

def fetch_elevations_for_coordinates(coordinates_2d, chunk_size=100):
    """
    Fetches elevation data for a route using Swisstopo Profile service.
    This is much more efficient than individual point requests.
    
    Args:
        coordinates_2d: List of (x, y) coordinate tuples in LV95
        chunk_size: Not used for profile service, kept for compatibility
    
    Returns:
        List of (x, y, elevation) coordinate tuples
    """
    if not coordinates_2d or len(coordinates_2d) < 2:
        return []
    
    # Create a GeoJSON LineString from the route coordinates
    # The API expects coordinates as [x, y] arrays, not tuples
    coordinates_array = [[x, y] for x, y in coordinates_2d]
    
    geojson = {
        "type": "LineString",
        "coordinates": coordinates_array
    }
    
    # Convert to proper JSON string and URL encode it
    import json
    import urllib.parse
    geom_json = json.dumps(geojson)
    geom_param = urllib.parse.quote(geom_json)
    
    # Build the Profile API URL
    api_url = f"https://api3.geo.admin.ch/rest/services/profile.json?geom={geom_param}&sr=2056&nb_points=200"
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # The API returns the data directly as a list, not wrapped in a 'profile' key
        if isinstance(data, list) and len(data) > 0:
            elevations = []
            
            # Extract elevation data from profile response
            for point in data:
                if 'alts' in point and 'easting' in point and 'northing' in point:
                    x = point['easting']
                    y = point['northing']
                    # Use the COMB elevation (combined model)
                    elevation = point['alts']['COMB']
                    elevations.append((x, y, elevation))
            
            print(f"Elevation data fetched for {len(elevations)} points")
            return elevations
        else:
            print(f"Warning: No profile data in API response")
            return []
            
    except Exception as e:
        print(f"Warning: Failed to fetch elevation profile: {e}")
        return []

def calculate_elevation_profile(coordinates_3d):
    """
    Converts 3D route coordinates into elevation profile data.
    
    Args:
        coordinates_3d: List of (x, y, elevation) coordinate tuples
    
    Returns:
        List of [distance, elevation] pairs for charting
    """
    if len(coordinates_3d) < 2:
        return []
    
    profile = []
    cumulative_distance = 0.0
    
    for i in range(len(coordinates_3d)):
        x, y, elevation = coordinates_3d[i]
        
        # Calculate distance from previous point
        if i > 0:
            prev_x, prev_y, _ = coordinates_3d[i-1]
            segment_distance = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            cumulative_distance += segment_distance
        
        profile.append([cumulative_distance, elevation])
    
    return profile

def find_nearest_edge(graph, point):
    """
    Finds the nearest edge in the graph to a given point by checking
    the geometry of each edge.
    """

    snapped_point = None
    nearest_edge = None
    min_dist = float('inf')

    for u, v, data in graph.edges(data=True):
        try:
            if not isinstance(data, dict):
                print(f"Warning: Edge data is not a dictionary for edge ({u}, {v}). Data: {data}. Skipping.")
                continue

            if data and 'geometry' in data:
                line = data['geometry']
                dist = point.distance(line)
                if dist < min_dist:
                    min_dist = dist
                    nearest_edge = (u, v)
                    snapped_point = nearest_points(point, line)[1]
            else:
                continue

        except Exception as e:
            continue  # Skip to the next edge
    return nearest_edge, snapped_point









def add_point_to_graph(graph, point, edge):
    """Adds a point to the graph by splitting an existing edge."""
    try:
        node_u, node_v = edge
    except Exception as e:
        raise

    try:
        edge_data = graph[node_u][node_v][0]
        edge_geom = edge_data['geometry']
    except Exception as e:
        raise
    
    # --- Start of New, Corrected Logic ---
    # Attempt to split the geometry at the user's snapped point.
    parts = split(edge_geom, point)

    # If the split fails (returns only 1 piece), it means the point is
    # mathematically at a vertex. In this specific case, and only this one,
    # we find and return the closest existing vertex.
    if len(parts.geoms) != 2:
        dist_u = point.distance(Point(node_u))
        dist_v = point.distance(Point(node_v))
        if dist_u < dist_v:
            return node_u
        else:
            return node_v
    
    # If the split succeeds, we proceed to create the new virtual node.
    geom1, geom2 = parts.geoms[0], parts.geoms[1]
    # --- End of New, Corrected Logic ---

    # Determine which new geometry piece connects to which original node.
    if geom1.distance(Point(node_u)) < 1e-9:
        geom_u, geom_v = geom1, geom2
    else:
        geom_u, geom_v = geom2, geom1

    # Remove the original edge.
    graph.remove_edge(node_u, node_v, key=0)
    
    # Add the new node and the two new edges with their correct, curved geometries.
    new_node = tuple(point.coords)[0]
    graph.add_node(new_node)
    graph.add_edge(node_u, new_node, weight=geom_u.length, geometry=geom_u)
    graph.add_edge(new_node, node_v, weight=geom_v.length, geometry=geom_v)
    
    return new_node

def calculate_shortest_path(graph, start_coords, end_coords):
    """Calculates the shortest path using a unified geometry-first approach."""
    
    def get_partial_segment(line, p1_coords, p2_coords):
        """
        Extracts a sub-string from a line between two points on that line,
        preserving all intermediate vertices.
        """
        dist1 = line.project(Point(p1_coords))
        dist2 = line.project(Point(p2_coords))

        if dist1 > dist2:
            dist1, dist2 = dist2, dist1
            
        p_start = line.interpolate(dist1)
        p_end = line.interpolate(dist2)

        new_coords = [tuple(p_start.coords)[0]]
        for coord in list(line.coords):
            p_dist = line.project(Point(coord))
            if p_dist > dist1 + 1e-9 and p_dist < dist2 - 1e-9:
                new_coords.append(coord)
        new_coords.append(tuple(p_end.coords)[0])
        
        sub_line = LineString(new_coords)
        return list(sub_line.coords), sub_line.length

    start_edge, _ = find_nearest_edge(graph, Point(start_coords))
    end_edge, _ = find_nearest_edge(graph, Point(end_coords))

    if start_edge is None or end_edge is None:
        return None

    if start_edge == end_edge:
        line = graph[start_edge[0]][start_edge[1]][0]['geometry']
        route_coords, route_len = get_partial_segment(line, start_coords, end_coords)
        return route_coords, route_len

    start_u, start_v = start_edge
    end_u, end_v = end_edge

    paths = {}
    for s_node in [start_u, start_v]:
        for e_node in [end_u, end_v]:
            try:
                paths[(s_node, e_node)] = nx.shortest_path_length(graph, s_node, e_node, weight='weight')
            except nx.NetworkXNoPath:
                continue
    
    if not paths: return None
    
    start_node_main, end_node_main = min(paths, key=paths.get)

    # --- Start of Corrected Route Assembly Logic ---
    
    # Part 1: Get the start partial segment
    start_line = graph[start_u][start_v][0]['geometry']
    start_coords_list, start_len = get_partial_segment(start_line, start_coords, start_node_main)
    # Ensure the start segment is oriented correctly (ending at the main path)
    if Point(start_coords_list[0]).distance(Point(start_node_main)) < 1e-9:
        start_coords_list.reverse()
    
    # Part 2: Get the end partial segment
    end_line = graph[end_u][end_v][0]['geometry']
    end_coords_list, end_len = get_partial_segment(end_line, end_node_main, end_coords)
    # Ensure the end segment is oriented correctly (starting at the main path)
    if Point(end_coords_list[0]).distance(Point(end_node_main)) > 1e-9:
        end_coords_list.reverse()
        
    # Part 3: Get the middle full segments
    if start_node_main != end_node_main:
        main_path_nodes = nx.shortest_path(graph, start_node_main, end_node_main, weight='weight')
        middle_path_len = paths[(start_node_main, end_node_main)]
    else: # This handles the two-segment case
        main_path_nodes = []
        middle_path_len = 0

    # Stitch all three parts together
    route_geometry = start_coords_list
    
    for i in range(len(main_path_nodes) - 1):
        u, v = main_path_nodes[i], main_path_nodes[i+1]
        edge_data = graph.get_edge_data(u, v)[0]
        coords = list(edge_data['geometry'].coords)
        if Point(coords[0]).distance(Point(u)) > Point(coords[-1]).distance(Point(u)):
            coords.reverse()
        # Append without the first point, which is a duplicate
        route_geometry.extend(coords[1:])

    # Append the end segment
    route_geometry.extend(end_coords_list[1:])
    
    total_length = start_len + middle_path_len + end_len
    
    return route_geometry, total_length
    # --- End of Corrected Route Assembly Logic ---
        
def calculate_route_from_gpkg(start_coords, end_coords, buffer_distance=5000, network_type='tlm3d'):
    """The Golden Routing Function."""
    print(f"\n=== Starting Route Calculation ({network_type}) ===")
    
    # Special handling for TLM3D when file is missing
    if network_type == 'tlm3d' and not os.path.exists(PREPROCESSED_TLM3D_PATH):
        print("❌ TLM3D routing is not available yet.")
        print("   The TLM3D network file needs to be preprocessed first.")
        print("   For now, please use 'ski_touring' network type.")
        return None, None, 0
    
    clipped_gdf = create_clipped_network(start_coords, end_coords, buffer_distance, network_type)
    
    if clipped_gdf is None or clipped_gdf.empty:
        return None, None, 0
    
    segments_loaded = len(clipped_gdf)
    network_graph = load_network_to_graph(clipped_gdf)
    
    if network_graph is None:
        return None, None, segments_loaded

    result = calculate_shortest_path(
        network_graph, start_coords, end_coords
    )
    
    if result is None:
        print("=== Route Calculation Failed ===")
        return None, None, segments_loaded, []
    
    try:
        route_geometry, path_length = result
    except Exception as e:
        return None, None, segments_loaded, []
    
    if route_geometry:
        print("=== Route Calculation Successful ===")
        
        # Enrich route with elevation data
        print("Fetching elevation data...")
        try:
            # Get 3D coordinates with elevation
            coordinates_3d = fetch_elevations_for_coordinates(route_geometry)
            
            # Calculate elevation profile for charting
            elevation_profile = calculate_elevation_profile(coordinates_3d)
            
            print(f"Elevation data fetched for {len(coordinates_3d)} points")
            print(f"Elevation profile created with {len(elevation_profile)} data points")
            
        except Exception as e:
            print(f"Warning: Failed to fetch elevation data: {e}")
            elevation_profile = []
    else:
        print("=== Route Calculation Failed ===")
        elevation_profile = []

    return route_geometry, path_length, segments_loaded, elevation_profile

if __name__ == '__main__':
    # (No changes to the testing section)
    pass