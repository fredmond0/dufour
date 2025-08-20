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
    
    # --- Start of Corrected Helper Function ---
    def get_partial_segment(line, p1_coords, p2_coords):
        """
        Extracts a sub-string from a line between two points on that line,
        preserving all intermediate vertices.
        """
        # Project the start and end coordinates onto the line to find their distance along it
        dist1 = line.project(Point(p1_coords))
        dist2 = line.project(Point(p2_coords))

        # Ensure the distances are in the correct order
        if dist1 > dist2:
            dist1, dist2 = dist2, dist1
            
        # Get the actual start and end points on the line
        p_start = line.interpolate(dist1)
        p_end = line.interpolate(dist2)

        # Build the new list of coordinates for the sub-path
        new_coords = [tuple(p_start.coords)[0]]
        
        # Add all of the original line's vertices that lie between the start and end points
        for coord in list(line.coords):
            p_dist = line.project(Point(coord))
            # Add a small tolerance (1e-9) to avoid floating point issues
            if p_dist > dist1 + 1e-9 and p_dist < dist2 - 1e-9:
                new_coords.append(coord)
                
        new_coords.append(tuple(p_end.coords)[0])
        
        # Create the new LineString and return its coordinates and length
        sub_line = LineString(new_coords)
        return list(sub_line.coords), sub_line.length
    # --- End of Corrected Helper Function ---

    # --- Main logic starts here ---
    start_edge, _ = find_nearest_edge(graph, Point(start_coords))
    end_edge, _ = find_nearest_edge(graph, Point(end_coords))

    if start_edge is None or end_edge is None:
        return None

    # Case 1: Start and end are on the same edge.
    if start_edge == end_edge:
        line = graph[start_edge[0]][start_edge[1]][0]['geometry']
        route_coords, route_len = get_partial_segment(line, start_coords, end_coords)
        return route_coords, route_len

    # Case 2: Start and end are on different edges.
    start_u, start_v = start_edge
    end_u, end_v = end_edge

    # Find the shortest path between the four possible endpoint combinations of the start/end edges.
    paths = {}
    for s_node in [start_u, start_v]:
        for e_node in [end_u, end_v]:
            try:
                paths[(s_node, e_node)] = nx.shortest_path_length(graph, s_node, e_node, weight='weight')
            except nx.NetworkXNoPath:
                continue
    
    if not paths: return None # No path exists.
    
    # Determine the best nodes on the start and end segments to form the main path.
    start_node_main, end_node_main = min(paths, key=paths.get)

    # Build the final route from three parts: start partial, middle full, end partial.
    
    # Part 1: Start partial segment
    start_line = graph[start_u][start_v][0]['geometry']
    start_coords_list, start_len = get_partial_segment(start_line, start_coords, start_node_main)

    # Part 2: Middle full segments
    # Handle the two-segment case where start and end are on adjacent trails
    if start_node_main == end_node_main:
        # Two-segment route: no middle path needed
        main_path_nodes = [start_node_main]
        route_geometry = start_coords_list
    else:
        # Multi-segment route: find the path between different nodes
        main_path_nodes = nx.shortest_path(graph, start_node_main, end_node_main, weight='weight')
        route_geometry = start_coords_list
    
    for i in range(len(main_path_nodes) - 1):
        u, v = main_path_nodes[i], main_path_nodes[i+1]
        edge_data = graph.get_edge_data(u, v)[0]
        coords = list(edge_data['geometry'].coords)
        if Point(coords[0]).distance(Point(u)) > Point(coords[-1]).distance(Point(u)):
            coords.reverse()
        if route_geometry[-1] == coords[0]:
            route_geometry.extend(coords[1:])
        else:
            route_geometry.extend(coords)
            
    # Part 3: End partial segment
    end_line = graph[end_u][end_v][0]['geometry']
    end_coords_list, end_len = get_partial_segment(end_line, end_node_main, end_coords)

    if route_geometry[-1] == end_coords_list[0]:
        route_geometry.extend(end_coords_list[1:])
    else:
        route_geometry.extend(end_coords_list)
        
    total_length = start_len + paths[(start_node_main, end_node_main)] + end_len
    
    # Debug: Print the route geometry to see what we're returning
    print(f"DEBUG: Final route has {len(route_geometry)} coordinate pairs")
    print(f"DEBUG: Start partial: {len(start_coords_list)} coords, length: {start_len:.2f}m")
    print(f"DEBUG: Middle path: {len(main_path_nodes)} nodes, length: {paths[(start_node_main, end_node_main)]:.2f}m")
    print(f"DEBUG: End partial: {len(end_coords_list)} coords, length: {end_len:.2f}m")
    print(f"DEBUG: Total route length: {total_length:.2f}m")
    
    # Debug: Show the first few and last few coordinates to see the structure
    if len(route_geometry) > 0:
        print(f"DEBUG: First 3 coords: {route_geometry[:3]}")
        print(f"DEBUG: Last 3 coords: {route_geometry[-3:]}")
        print(f"DEBUG: Start partial first/last: {start_coords_list[0]} -> {start_coords_list[-1]}")
        print(f"DEBUG: End partial first/last: {end_coords_list[0]} -> {end_coords_list[-1]}")
    
    return route_geometry, total_length
        
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