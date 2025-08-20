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
    
    graph = momepy.gdf_to_nx(gdf, approach='primal', length='length')
    print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    # After graph creation, iterate through the original GeoDataFrame and
    # assign the geometry to the corresponding edge in the graph.
    # This is more reliable than trying to map back from the graph.
    geometry_assigned = 0
    for index, row in gdf.iterrows():
        # momepy stores the original index in 'mm_len'
        for u, v, data in graph.edges(data=True):
            if data.get('mm_len') == index:
                data['geometry'] = row.geometry
                geometry_assigned += 1
                break
    
    print(f"Geometry assigned to {geometry_assigned} edges out of {graph.number_of_edges()} total edges.")
    return graph

def find_nearest_edge(graph, point):
    """
    Finds the nearest edge in the graph to a given point by checking
    the geometry of each edge.
    """

    snapped_point = None
    nearest_edge = None
    min_dist = float('inf')

    print(f"DEBUG: find_nearest_edge called with point: {point}")
    print(f"DEBUG: Graph has {graph.number_of_edges()} edges")

    for u, v, data in graph.edges(data=True):
        try:
            # print(f"DEBUG: Inspecting edge: ({u}, {v})")  # Commented out for clarity
            if not isinstance(data, dict):
                print(f"Warning: Edge data is not a dictionary for edge ({u}, {v}). Data: {data}. Skipping.")
                continue

            if data and 'geometry' in data:
                line = data['geometry']
                dist = point.distance(line)
                if dist < min_dist:
                    print(
                        f"DEBUG: New nearest edge found: ({u}, {v}), distance: {dist}"
                    )

                    min_dist = dist
                    nearest_edge = (u, v)
                    snapped_point = nearest_points(point, line)[1]
            else:
                print(f"DEBUG: Edge ({u}, {v}) missing 'geometry' data. Data: {data}")

        except Exception as e:
            print(f"DEBUG: Exception during edge inspection for edge ({u}, {v}): {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            continue  # Skip to the next edge

    print(f"DEBUG: find_nearest_edge returning: nearest_edge={nearest_edge}, snapped_point={snapped_point}")
    return nearest_edge, snapped_point









def add_point_to_graph(graph, point, edge):
    """Adds a point to the graph by splitting an existing edge."""
    print(f"DEBUG: add_point_to_graph called with point={point}, edge={edge}")
    print(f"DEBUG: edge type: {type(edge)}, edge contents: {edge}")
    
    try:
        node_u, node_v = edge
        print(f"DEBUG: Successfully unpacked edge: node_u={node_u}, node_v={node_v}")
    except Exception as e:
        print(f"DEBUG: Failed to unpack edge: {e}")
        print(f"DEBUG: edge was: {edge}")
        raise
    
    print(f"DEBUG: About to get edge geometry...")
    print(f"DEBUG: edge: {edge}")
    print(f"DEBUG: edge type: {type(edge)}")
    print(f"DEBUG: edge[0] type: {type(edge[0])}")
    print(f"DEBUG: edge[1] type: {type(edge[1])}")
    
    # The edge is ((x1, y1), (x2, y2)) - coordinate tuples
    # We need to access the graph with these exact coordinate tuples
    try:
        edge_data = graph.edges[edge]
        print(f"DEBUG: Successfully got edge data: {edge_data}")
        print(f"DEBUG: Edge data type: {type(edge_data)}")
        print(f"DEBUG: Edge data keys: {list(edge_data.keys()) if hasattr(edge_data, 'keys') else 'No keys method'}")
        
        if hasattr(edge_data, 'keys'):
            for key, value in edge_data.items():
                print(f"DEBUG: key: {key}, value: {value}, type: {type(value)}")
        
        edge_geom = edge_data['geometry']
        print(f"DEBUG: Got edge geometry: {type(edge_geom)}")
        
    except Exception as e:
        print(f"DEBUG: Error accessing edge data: {e}")
        print(f"DEBUG: Available edges in graph: {list(graph.edges())}")
        raise
    
    print(f"DEBUG: About to check distances...")
    if point.distance(Point(node_u)) < 1e-6: 
        print(f"DEBUG: Point too close to node_u, returning: {node_u}")
        return node_u
    if point.distance(Point(node_v)) < 1e-6: 
        print(f"DEBUG: Point too close to node_v, returning: {node_v}")
        return node_v

    print(f"DEBUG: About to remove edge...")
    graph.remove_edge(node_u, node_v)
    
    print(f"DEBUG: About to create new node...")
    new_node = tuple(point.coords)[0]
    print(f"DEBUG: New node created: {new_node}")
    
    print(f"DEBUG: About to add node to graph...")
    graph.add_node(new_node)
    
    print(f"DEBUG: About to calculate distances...")
    dist_u = point.distance(Point(node_u))
    dist_v = point.distance(Point(node_v))
    print(f"DEBUG: Distances calculated: dist_u={dist_u}, dist_v={dist_v}")
    
    print(f"DEBUG: About to add edges...")
    graph.add_edge(node_u, new_node, weight=dist_u, geometry=LineString([Point(node_u), point]))
    graph.add_edge(new_node, node_v, weight=dist_v, geometry=LineString([point, Point(node_v)]))
    
    print(f"DEBUG: add_point_to_graph returning: {new_node}")
    return new_node

def calculate_shortest_path(graph, start_coords, end_coords):
    """Calculates the shortest path using a point-to-segment routing model."""
    # print(f"DEBUG: calculate_shortest_path called with start_coords={start_coords}, end_coords={end_coords}")
    temp_graph = graph.copy()

    # print("DEBUG: Finding start edge...")
    start_edge, start_snap = find_nearest_edge(temp_graph, Point(start_coords))
    # print(f"DEBUG: Start edge found: {start_edge}, start_snap: {start_snap}")
    
    # print("DEBUG: Finding end edge...")
    end_edge, end_snap = find_nearest_edge(temp_graph, Point(end_coords))
    # print(f"DEBUG: End edge found: {end_edge}, end_snap: {end_snap}")
    
    if start_edge is None or end_edge is None:
        print("Error: Could not snap points to network.")
        return None

    if start_edge == end_edge:
        print("--- Intra-edge route detected ---")
        print(f"DEBUG: Processing intra-edge route...")
        # Fix: NetworkX expects graph.edges[u, v], not graph.edges[(u, v)]
        node_u, node_v = start_edge
        line = temp_graph.edges[node_u, node_v]['geometry']
        start_dist = line.project(start_snap)
        end_dist = line.project(end_snap)
        
        print(f"DEBUG: About to create splitter...")
        splitter = MultiPoint([line.interpolate(start_dist), line.interpolate(end_dist)])
        print(f"DEBUG: Splitter created: {splitter}")
        
        try:
            print(f"DEBUG: About to split line...")
            parts = split(line, splitter)
            print(f"DEBUG: Line split into {len(parts.geoms)} parts")
            
            for i, part in enumerate(parts.geoms):
                print(f"DEBUG: Checking part {i}: distance to start={part.distance(start_snap)}, distance to end={part.distance(end_snap)}")
                if part.distance(start_snap) < 1e-9 and part.distance(end_snap) < 1e-9:
                    result = (list(part.coords), part.length)
                    print(f"DEBUG: Intra-edge route returning: {result}")
                    print(f"DEBUG: Return type: {type(result)}, length: {len(result)}")
                    return result
        except Exception as e:
            print(f"DEBUG: Exception in intra-edge route processing: {e}")
            import traceback
            traceback.print_exc()
            return None

    print(f"DEBUG: About to add points to graph...")
    print(f"DEBUG: start_edge: {start_edge}, start_snap: {start_snap}")
    start_node = add_point_to_graph(temp_graph, start_snap, start_edge)
    end_node = add_point_to_graph(temp_graph, end_snap, end_edge)

    print(f"DEBUG: Routing from new temporary node {start_node} to {end_node}")

    try:
        print(f"DEBUG: About to find shortest path...")
        path_nodes = nx.shortest_path(temp_graph, source=start_node, target=end_node, weight='weight')
        path_length = nx.shortest_path_length(temp_graph, source=start_node, target=end_node, weight='weight')
        print(f"DEBUG: Path found with {len(path_nodes)} nodes and length {path_length:.2f}m")

        print(f"DEBUG: About to build route geometry...")
        route_geometry = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            edge_data = temp_graph.get_edge_data(u, v)
            if 'geometry' in edge_data:
                coords = list(edge_data['geometry'].coords)
                if Point(coords[0]).distance(Point(u)) > Point(coords[-1]).distance(Point(u)):
                    coords.reverse()
                if not route_geometry or coords[0] != route_geometry[-1]:
                    route_geometry.extend(coords)
                else:
                    route_geometry.extend(coords[1:])
        
        result = (route_geometry, path_length)
        print(f"DEBUG: Normal route returning: {type(result)}, length: {len(result)}")
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
    
    print(f"DEBUG: calculate_shortest_path returned: {result}")
    print(f"DEBUG: result type: {type(result)}")
    if result is not None:
        print(f"DEBUG: result length: {len(result)}")
        print(f"DEBUG: result contents: {result}")
    
    if result is None:
        print("=== Route Calculation Failed ===")
        return None, None, segments_loaded
    
    try:
        route_geometry, path_length = result
        print(f"DEBUG: Successfully unpacked: route_geometry={type(route_geometry)}, path_length={type(path_length)}")
    except Exception as e:
        print(f"DEBUG: Failed to unpack result: {e}")
        print(f"DEBUG: result was: {result}")
        return None, None, segments_loaded
    
    if route_geometry:
        print("=== Route Calculation Successful ===")
    else:
        print("=== Route Calculation Failed ===")

    return route_geometry, path_length, segments_loaded

if __name__ == '__main__':
    # (No changes to the testing section)
    pass