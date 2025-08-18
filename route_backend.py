import geopandas as gpd
import networkx as nx
import os
from shapely.geometry import Point, box
from shapely.ops import unary_union
import numpy as np

# --- Configuration ---
# Updated path to your actual GeoPackage file
GPKG_PATH = './data/tlm3d/swisstlm3d_2025-03_2056_5728.gpkg/SWISSTLM3D_2025.gpkg'
# The name of the road layer inside your GeoPackage
LAYER_NAME = 'tlm_strassen_strasse'

def create_clipped_network(start_coords, end_coords, buffer_distance=5000):
    """
    Creates a clipped network around the start and end points for better performance.
    This is the key optimization - instead of loading the entire network,
    we only load roads within a buffer around our area of interest.
    """
    print(f"Creating clipped network with buffer distance: {buffer_distance} meters")
    print(f"Start coords: {start_coords}")
    print(f"End coords: {end_coords}")
    
    # Create a bounding box around the start and end points
    min_x = min(start_coords[0], end_coords[0]) - buffer_distance
    max_x = max(start_coords[0], end_coords[0]) + buffer_distance
    min_y = min(start_coords[1], end_coords[1]) - buffer_distance
    max_y = max(start_coords[1], end_coords[1]) + buffer_distance
    
    bbox = box(min_x, min_y, max_x, max_y)
    print(f"Bounding box: ({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f})")
    
    # Check if bbox is within reasonable Swiss bounds
    if min_x < 2000000 or max_x > 3000000 or min_y < 500000 or max_y > 1500000:
        print("⚠️  Warning: Bounding box appears to be outside typical Swiss bounds")
        print("   Expected X: 2.0M - 3.0M, Y: 0.5M - 1.5M")
    
    try:
        # Read only the roads within the bounding box
        gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME, bbox=bbox)
        print(f"Loaded {len(gdf)} road segments in clipped area (vs entire network)")
        return gdf
    except Exception as e:
        print(f"Error reading clipped network: {e}")
        return None

def load_network_to_graph(gdf):
    """
    Loads a road network from a GeoDataFrame into a NetworkX graph.
    Now optimized to work with the clipped data.
    """
    if gdf is None or len(gdf) == 0:
        print("Error: No road data to process")
        return None

    print("Creating network graph from clipped data...")
    
    # Create an empty graph
    G = nx.Graph()

    # Add nodes and edges to the graph from the GeoDataFrame
    for index, row in gdf.iterrows():
        line = row.geometry
        
        # Get the start and end points of the line segment
        start_node = tuple(line.coords[0])  # Convert to tuple for hashability
        end_node = tuple(line.coords[-1])
        
        # Calculate length in meters (assuming coordinates are in meters)
        length = line.length
        
        # Add the edge to the graph with its length as the weight
        # Remove geometry from row data to avoid conflicts
        row_data = row.to_dict()
        if 'geometry' in row_data:
            del row_data['geometry']
        
        G.add_edge(start_node, end_node, weight=length, geometry=line, **row_data)

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def find_nearest_node(graph, point, max_search_distance=1000):
    """
    Finds the nearest point on the nearest road segment to the given coordinates.
    This allows starting/ending anywhere along a road, not just at junctions.
    """
    nearest_point = None
    min_dist = float('inf')
    nearest_edge = None
    
    # Convert to tuple for consistency
    query_point = tuple(point)
    
    # Check all edges (road segments) to find the closest point on any road
    for u, v, data in graph.edges(data=True):
        if 'geometry' in data:
            # Get the road geometry
            road_geometry = data['geometry']
            
            # Find the closest point on this road segment
            try:
                # Project the query point onto the road geometry
                closest_point = road_geometry.interpolate(road_geometry.project(Point(query_point)))
                dist = Point(query_point).distance(closest_point)
                
                if dist < min_dist and dist <= max_search_distance:
                    min_dist = dist
                    nearest_point = closest_point
                    nearest_edge = (u, v, data)
            except Exception as e:
                # Skip this edge if geometry operations fail
                continue
    
    if nearest_point is not None:
        print(f"Found nearest point on road at {list(nearest_point.coords)[0]} (distance: {min_dist:.2f}m)")
        return list(nearest_point.coords)[0]
    else:
        # Fallback: find nearest node if no road segments found
        print("No road segments found, falling back to nearest node")
        return find_nearest_node_fallback(graph, point, max_search_distance)

def find_nearest_node_fallback(graph, point, max_search_distance=1000):
    """
    Fallback method to find the nearest node if no road segments are available.
    """
    nearest_node = None
    min_dist = float('inf')
    
    # Convert to tuple for consistency
    query_point = tuple(point)
    
    for node in graph.nodes():
        # Calculate distance between coordinate tuples
        dist = np.sqrt((node[0] - query_point[0])**2 + (node[1] - query_point[1])**2)
        
        # Early exit if we're too far
        if dist > max_search_distance:
            continue
            
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
            
    if nearest_node:
        print(f"Nearest node found at a distance of {min_dist:.2f} meters.")
    else:
        print(f"No node found within {max_search_distance} meters. Try increasing buffer distance.")
    
    return nearest_node

def connect_point_to_network(graph, point, point_type):
    """
    Connects a point (start or end) to the nearest road network node.
    This allows routing from any point along a road, not just at junctions.
    """
    nearest_node = None
    min_dist = float('inf')
    
    # Find the nearest actual road network node
    for node in graph.nodes():
        if node != point:  # Don't connect to ourselves
            dist = np.sqrt((node[0] - point[0])**2 + (node[1] - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
    
    if nearest_node:
        # Create a virtual edge from our point to the nearest network node
        # The weight is the actual distance
        graph.add_edge(point, nearest_node, weight=min_dist, geometry=Point(point).buffer(0.1))
        print(f"Connected {point_type} point to network node at distance {min_dist:.2f}m")
        return True
    else:
        print(f"Could not find network node to connect {point_type} point")
        return False

def interpolateRoadSegment(start_node, end_node, num_points=10):
    """
    Interpolate between two nodes to create smooth routing.
    This ensures we don't have straight lines between vertices.
    """
    if num_points < 2:
        num_points = 2
    
    interpolated = []
    for i in range(num_points):
        # Linear interpolation between start and end nodes
        t = i / (num_points - 1)
        x = start_node[0] + t * (end_node[0] - start_node[0])
        y = start_node[1] + t * (end_node[1] - start_node[1])
        interpolated.append((x, y))
    
    return interpolated

def calculate_shortest_path(graph, start_coords, end_coords):
    """
    Calculates the shortest path between two coordinate pairs.
    Now returns the actual road geometry for smooth route display.
    """
    if graph is None:
        return None

    print("\nFinding nearest point on road for start point...")
    start_point = find_nearest_node(graph, start_coords)
    
    print("Finding nearest point on road for end point...")
    end_point = find_nearest_node(graph, end_coords)

    if start_point and end_point:
        print(f"Calculating shortest path between {start_point} and {end_point}...")
        
        # Create a temporary graph for routing that includes our start/end points
        temp_graph = graph.copy()
        
        # Add start and end points as temporary nodes if they're not already in the graph
        if start_point not in temp_graph:
            temp_graph.add_node(start_point)
        if end_point not in temp_graph:
            temp_graph.add_node(end_point)
        
        # Connect start point to nearest actual road node
        start_connected = connect_point_to_network(temp_graph, start_point, 'start')
        end_connected = connect_point_to_network(temp_graph, end_point, 'end')
        
        if not start_connected or not end_connected:
            print("Could not connect start/end points to road network")
            return None, None
        
        try:
            # Use Dijkstra's algorithm to find the shortest path based on edge 'weight'
            path_nodes = nx.shortest_path(temp_graph, source=start_point, target=end_point, weight='weight')
            path_length = nx.shortest_path_length(temp_graph, source=start_point, target=end_point, weight='weight')
            
            print(f"Path found with {len(path_nodes)} nodes and a total length of {path_length:.2f} meters.")
            
            # Now extract the actual road geometry for smooth route display
            route_geometry = []
            
            # Convert path nodes to coordinates
            for i in range(len(path_nodes) - 1):
                current_node = path_nodes[i]
                next_node = path_nodes[i + 1]
                
                # Get the edge between these nodes from the temp_graph
                edge_data = temp_graph.get_edge_data(current_node, next_node)
                if edge_data and 'geometry' in edge_data:
                    # Extract coordinates from the road geometry
                    road_coords = list(edge_data['geometry'].coords)
                    
                    # Ensure we have enough coordinates for smooth routing
                    if len(road_coords) < 2:
                        # If geometry has insufficient points, interpolate
                        road_coords = interpolateRoadSegment(current_node, next_node, 10)
                    
                    # Only add coordinates if we don't already have them (avoid duplicates)
                    if not route_geometry or road_coords[0] != route_geometry[-1]:
                        route_geometry.extend(road_coords)
                    else:
                        # Skip first coordinate if it's the same as the last one we added
                        route_geometry.extend(road_coords[1:])
                else:
                    # Fallback: interpolate between nodes for smooth routing
                    interpolated_coords = interpolateRoadSegment(current_node, next_node, 10)
                    if not route_geometry or interpolated_coords[0] != route_geometry[-1]:
                        route_geometry.extend(interpolated_coords)
                    else:
                        route_geometry.extend(interpolated_coords[1:])
            
            # Ensure start and end points are included in the route
            if route_geometry and len(route_geometry) > 0:
                # Add start point if it's not already the first point
                if route_geometry[0] != start_point:
                    route_geometry.insert(0, start_point)
                
                # Add end point if it's not already the last point
                if route_geometry[-1] != end_point:
                    route_geometry.append(end_point)
            
            print(f"Route geometry created with {len(route_geometry)} coordinate points")
            return route_geometry, path_length
        except nx.NetworkXNoPath:
            print("No path could be found between the start and end points.")
            return None, None
    else:
        print("Could not find nearest nodes for the given coordinates.")
        return None, None

def main_route_calculation(start_coords, end_coords, buffer_distance=5000):
    """
    Main function that orchestrates the entire routing process with clipping optimization.
    """
    print("=== SWISS TLM3D Route Optimization ===")
    print(f"Start point: {start_coords}")
    print(f"End point: {end_coords}")
    print(f"Buffer distance: {buffer_distance} meters")
    
    # Step 1: Create clipped network (the key optimization!)
    clipped_gdf = create_clipped_network(start_coords, end_coords, buffer_distance)
    
    if clipped_gdf is None:
        print("Failed to create clipped network")
        return None, None, 0
    
    roads_loaded = len(clipped_gdf)
    print(f"Loaded {roads_loaded} road segments for routing")
    
    # Step 2: Convert to NetworkX graph
    road_network_graph = load_network_to_graph(clipped_gdf)
    
    if road_network_graph is None:
        print("Failed to create network graph")
        return None, None, roads_loaded
    
    # Step 3: Calculate the path
    route_geometry, path_length = calculate_shortest_path(road_network_graph, start_coords, end_coords)
    
    return route_geometry, path_length, roads_loaded

# --- Main Execution ---
if __name__ == "__main__":
    # --- DEFINE YOUR START AND END POINTS HERE ---
    # These are example coordinates in the Swiss LV95 projection (EPSG:2056)
    # Replace these with coordinates from your area of interest
    start_point_coords = (2683000, 1249000)  # Example: Near Zurich HB
    end_point_coords = (2683500, 1247500)    # Example: Near Paradeplatz
    
    # You can adjust the buffer distance based on your needs
    # Larger buffer = more roads included but slower performance
    # Smaller buffer = faster but might miss some route options
    buffer_distance = 5000  # 5km buffer
    
    # Calculate the path using the optimized approach
    route_geometry, path_length = main_route_calculation(
        start_point_coords, 
        end_point_coords, 
        buffer_distance
    )

    if route_geometry:
        print("\n=== Route Summary ===")
        print(f"Total path length: {path_length:.2f} meters")
        print(f"Number of coordinate points: {len(route_geometry)}")
        
        print("\n--- Route Geometry (Coordinates) ---")
        # Print the first 5 and last 5 coordinates of the route for verification
        for i, coord in enumerate(route_geometry[:5]):
            print(f"  {i+1}: {coord}")
        if len(route_geometry) > 10:
            print("  ...")
            for i, coord in enumerate(route_geometry[-5:], len(route_geometry)-4):
                print(f"  {i+1}: {coord}")
        elif len(route_geometry) > 5:
            for i, coord in enumerate(route_geometry[5:], 6):
                print(f"  {i}: {coord}")
    else:
        print("Failed to calculate route. Try increasing the buffer distance.")

