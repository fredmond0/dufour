import geopandas as gpd
import networkx as nx
import os
from shapely.geometry import Point, box, LineString, MultiPoint
from shapely.ops import nearest_points, split
import momepy

# --- Configuration ---
PREPROCESSED_TLM3D_PATH = './data/preprocessed/tlm3d_network_processed.gpkg'
PREPROCESSED_SKI_PATH = './data/preprocessed/ski_network_processed.gpkg'

def create_clipped_network(start_coords, end_coords, buffer_distance, network_type):
    """Creates a clipped GeoDataFrame from a pre-processed file."""
    print(f"Creating clipped {network_type} network with buffer: {buffer_distance}m")
    
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
    
    # The edge is ((x1, y1), (x2, y2)) - coordinate tuples
    # We need to access the graph with these exact coordinate tuples
    try:
        edge_data = graph[node_u][node_v] # <-- CORRECTED LINE
        edge_geom = edge_data[0]['geometry']  # MultiGraph: access first edge at index 0
        
    except Exception as e:
        raise
    
    if point.distance(Point(node_u)) < 1e-6: 
        return node_u
    if point.distance(Point(node_v)) < 1e-6: 
        return node_v

    graph.remove_edge(node_u, node_v)
    
    new_node = tuple(point.coords)[0]
    
    graph.add_node(new_node)
    
    dist_u = point.distance(Point(node_u))
    dist_v = point.distance(Point(node_v))
    graph.add_edge(node_u, new_node, weight=dist_u, geometry=LineString([Point(node_u), point]))
    graph.add_edge(new_node, node_v, weight=dist_v, geometry=LineString([point, Point(node_v)]))
    
    return new_node

def calculate_shortest_path(graph, start_coords, end_coords):
    """Calculates the shortest path using a point-to-segment routing model."""
    temp_graph = graph.copy()

    start_edge, start_snap = find_nearest_edge(temp_graph, Point(start_coords))
    
    end_edge, end_snap = find_nearest_edge(temp_graph, Point(end_coords))
    
    if start_edge is None or end_edge is None:
        print("Error: Could not snap points to network.")
        return None

    if start_edge == end_edge:
        print("--- Intra-edge route detected ---")
        # Fix: NetworkX expects graph.edges[u, v], not graph.edges[(u, v)]
        node_u, node_v = start_edge
        line = temp_graph[node_u][node_v][0]['geometry']  # MultiGraph: access first edge at index 0
        start_dist = line.project(start_snap)
        end_dist = line.project(end_snap)
        
        splitter = MultiPoint([line.interpolate(start_dist), line.interpolate(end_dist)])
        
        try:
            parts = split(line, splitter)
            
            for i, part in enumerate(parts.geoms):
                if part.distance(start_snap) < 1e-9 and part.distance(end_snap) < 1e-9:
                    result = (list(part.coords), part.length)
                    return result
        except Exception as e:
            return None
        return None

    start_node = add_point_to_graph(temp_graph, start_snap, start_edge)
    end_node = add_point_to_graph(temp_graph, end_snap, end_edge)

    try:
        path_nodes = nx.shortest_path(temp_graph, source=start_node, target=end_node, weight='weight')
        path_length = nx.shortest_path_length(temp_graph, source=start_node, target=end_node, weight='weight')

        route_geometry = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            edge_data = temp_graph.get_edge_data(u, v)[0]  # MultiGraph: access first edge at index 0
            print(f"DEBUG: Edge {i}: u={u}, v={v}, edge_data={edge_data}")
            if 'geometry' in edge_data:
                coords = list(edge_data['geometry'].coords)  # Now we can access geometry directly
                print(f"DEBUG: Got coords: {len(coords)} points")
                if Point(coords[0]).distance(Point(u)) > Point(coords[-1]).distance(Point(u)):
                    coords.reverse()
                if not route_geometry or coords[0] != route_geometry[-1]:
                    route_geometry.extend(coords)
                else:
                    route_geometry.extend(coords[1:])
            else:
                print(f"DEBUG: No geometry found for edge {i}")
        
        result = (route_geometry, path_length)
        return result
    except nx.NetworkXNoPath:
        print("Error: No path could be found. The network may be disconnected.")
        return None
        
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
        return None, None, segments_loaded
    
    try:
        route_geometry, path_length = result
    except Exception as e:
        return None, None, segments_loaded
    
    if route_geometry:
        print("=== Route Calculation Successful ===")
    else:
        print("=== Route Calculation Failed ===")

    return route_geometry, path_length, segments_loaded

if __name__ == '__main__':
    # (No changes to the testing section)
    pass