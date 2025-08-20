import geopandas as gpd
import networkx as nx
import os
import pandas as pd
from shapely.geometry import Point, box, LineString
from shapely.ops import unary_union
import numpy as np
import momepy

# --- Configuration ---
# Updated path to your actual GeoPackage file
GPKG_PATH = './data/tlm3d/swisstlm3d_2025-03_2056_5728.gpkg/SWISSTLM3D_2025.gpkg'
# The name of the road layer inside your GeoPackage
LAYER_NAME = 'tlm_strassen_strasse'

# Ski touring data paths
SKI_ROUTES_PATH = './data/skitouring/skitouren_2056.gpkg/ski_routes_2056.gpkg'
SKI_NETWORK_PATH = './data/skitouring/skitouren_2056.gpkg/ski_network_2056.gpkg'

def find_network_intersections(road_gdf, ski_gdf, tolerance=10):
    """
    Finds intersection points between road and ski touring networks.
    tolerance: distance in meters to consider lines as intersecting
    """
    intersection_points = []
    
    print(f"Checking {len(road_gdf)} roads against {len(ski_gdf)} ski routes for intersections...")
    
    for road_idx, road_row in road_gdf.iterrows():
        road_geom = road_row.geometry
        
        for ski_idx, ski_row in ski_gdf.iterrows():
            ski_geom = ski_row.geometry
            
            try:
                # Check if geometries intersect
                if road_geom.intersects(ski_geom):
                    intersection = road_geom.intersection(ski_geom)
                    
                    try:
                        if intersection.geom_type == 'Point':
                            # Single intersection point
                            intersection_points.append({
                                'point': intersection,
                                'road_idx': road_idx,
                                'ski_idx': ski_idx,
                                'type': 'point'
                            })
                        elif intersection.geom_type == 'MultiPoint':
                            # Multiple intersection points
                            for point in intersection.geoms:
                                intersection_points.append({
                                    'point': point,
                                    'road_idx': road_idx,
                                    'ski_idx': ski_idx,
                                    'type': 'multipoint'
                                })
                        elif intersection.geom_type == 'LineString':
                            # Lines overlap - find start and end points
                            try:
                                coords = list(intersection.coords)
                                if len(coords) >= 2:
                                    intersection_points.append({
                                        'point': Point(coords[0]),
                                        'road_idx': road_idx,
                                        'ski_idx': ski_idx,
                                        'type': 'overlap_start'
                                    })
                                    intersection_points.append({
                                        'point': Point(coords[-1]),
                                        'road_idx': road_idx,
                                        'ski_idx': ski_idx,
                                        'type': 'overlap_end'
                                    })
                            except Exception as e:
                                print(f"Error processing LineString intersection: {e}")
                                continue
                        elif intersection.geom_type == 'MultiLineString':
                            # Handle MultiLineString intersections
                            for line in intersection.geoms:
                                try:
                                    coords = list(line.coords)
                                    if len(coords) >= 2:
                                        intersection_points.append({
                                            'point': Point(coords[0]),
                                            'road_idx': road_idx,
                                            'ski_idx': ski_idx,
                                            'type': 'multiline_start'
                                        })
                                        intersection_points.append({
                                            'point': Point(coords[-1]),
                                            'road_idx': road_idx,
                                            'ski_idx': ski_idx,
                                            'type': 'multiline_end'
                                        })
                                except Exception as e:
                                    print(f"Error processing MultiLineString part: {e}")
                                    continue
                        else:
                            print(f"Unhandled intersection type: {intersection.geom_type}")
                            
                    except Exception as e:
                        print(f"Error processing intersection result: {e}")
                        continue
                        
            except Exception as e:
                # Skip problematic geometries
                print(f"Error checking intersection: {e}")
                continue
    
    # Remove duplicate intersection points (within tolerance)
    unique_intersections = []
    for intersection in intersection_points:
        is_duplicate = False
        for existing in unique_intersections:
            if intersection['point'].distance(existing['point']) < tolerance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_intersections.append(intersection)
    
    print(f"Found {len(unique_intersections)} unique intersection points")
    return unique_intersections

def connect_networks_at_intersections(road_gdf, ski_gdf, intersection_points):
    """
    Connects road and ski touring networks at intersection points by:
    1. Adding intersection nodes to both networks
    2. Creating virtual edges between networks at intersections
    3. Ensuring seamless routing between networks
    """
    print("Connecting networks at intersection points...")
    
    # Create a copy of the original dataframes
    connected_roads = road_gdf.copy()
    connected_skis = ski_gdf.copy()
    
    # Add intersection nodes to both networks
    for i, intersection in enumerate(intersection_points):
        intersection_point = intersection['point']
        intersection_coords = (intersection_point.x, intersection_point.y)
        
        # Create a unique identifier for this intersection
        intersection_id = f"intersection_{i}"
        
        # Add intersection node to roads network
        road_intersection_row = {
            'route_type': 'intersection',
            'intersection_id': intersection_id,
            'connected_networks': ['road', 'ski_tour'],
            'geometry': intersection_point
        }
        
        # Add intersection node to ski network
        ski_intersection_row = {
            'route_type': 'intersection',
            'intersection_id': intersection_id,
            'connected_networks': ['road', 'ski_tour'],
            'geometry': intersection_point
        }
        
        # Add to respective networks
        connected_roads = pd.concat([
            connected_roads, 
            gpd.GeoDataFrame([road_intersection_row], crs=connected_roads.crs)
        ], ignore_index=True)
        
        connected_skis = pd.concat([
            connected_skis, 
            gpd.GeoDataFrame([ski_intersection_row], crs=connected_skis.crs)
        ], ignore_index=True)
    
    # Combine the connected networks
    combined_network = pd.concat([connected_roads, connected_skis], ignore_index=True)
    
    print(f"Connected network created with {len(combined_network)} total segments")
    print(f"Including {len(intersection_points)} intersection nodes")
    
    return combined_network

def load_ski_touring_routes(bbox=None):
    """
    Loads ski touring routes for display on the map.
    If bbox is provided, only loads routes within that bounding box.
    """
    try:
        if bbox:
            # Load only routes within the bounding box for performance
            gdf = gpd.read_file(SKI_ROUTES_PATH, bbox=bbox)
        else:
            # Load all routes (for initial map display)
            gdf = gpd.read_file(SKI_ROUTES_PATH)
        
        print(f"Loaded {len(gdf)} ski touring routes")
        return gdf
    except Exception as e:
        print(f"Error loading ski touring routes: {e}")
        return None

def create_clipped_network(start_coords, end_coords, buffer_distance=5000, network_type='tlm3d'):
    """
    Creates a clipped network around the start and end points for better performance.
    Now supports two network types: TLM3D (roads/trails) and ski touring.
    """
    print(f"Creating clipped {network_type} network with buffer distance: {buffer_distance} meters")
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
        if network_type == 'tlm3d':
            # Load only TLM3D roads and trails
            road_gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME, bbox=bbox)
            print(f"Loaded {len(road_gdf)} TLM3D road/trail segments in clipped area")
            road_gdf['route_type'] = 'tlm3d'
            return road_gdf
            
        elif network_type == 'ski_touring':
            # Load only ski touring routes (use routes file instead of network file)
            try:
                ski_gdf = gpd.read_file(SKI_ROUTES_PATH, layer='ski_routes_2056', bbox=bbox)
                print(f"Loaded {len(ski_gdf)} ski touring routes in clipped area")
                
                # Debug: Check what we actually loaded
                if len(ski_gdf) > 0:
                    print(f"Ski routes columns: {ski_gdf.columns.tolist()}")
                    print(f"Ski routes geometry types: {ski_gdf.geometry.geom_type.value_counts().to_dict()}")
                    print(f"First few geometries: {ski_gdf.geometry.head(3).tolist()}")
                    
                    # Check for any invalid geometries
                    invalid_count = (~ski_gdf.geometry.is_valid).sum()
                    if invalid_count > 0:
                        print(f"Warning: {invalid_count} invalid ski geometries found")
                    
                    # Check coordinate ranges
                    bounds = ski_gdf.total_bounds
                    print(f"Ski routes bounds: {bounds}")
                
                ski_gdf['route_type'] = 'ski_tour'
                return ski_gdf
            except Exception as e:
                print(f"Error loading ski routes: {e}")
                # Fallback to network file
                print("Falling back to ski network file...")
                ski_gdf = gpd.read_file(SKI_NETWORK_PATH, layer='ski_network_2056', bbox=bbox)
                print(f"Loaded {len(ski_gdf)} ski touring segments in clipped area")
                ski_gdf['route_type'] = 'ski_tour'
                return ski_gdf
            
        else:
            print(f"Unknown network type: {network_type}")
            return None
            
    except Exception as e:
        print(f"Error reading clipped network: {e}")
        return None

def load_network_to_graph(gdf):
    """
    Loads a road network from a GeoDataFrame into a NetworkX graph using momepy.
    Clean, robust implementation that handles all geometry types automatically.
    """
    if gdf is None or len(gdf) == 0:
        print("Error: No road data to process")
        return None

    print("Creating network graph using momepy...")
    
    try:
        # Ensure we have a length column for momepy
        if 'length' not in gdf.columns:
            gdf = gdf.copy()
            gdf['length'] = gdf.geometry.length
        
        # Use momepy to convert GeoDataFrame to NetworkX graph
        # momepy handles MultiLineString geometries automatically
        G = momepy.gdf_to_nx(gdf, approach='primal', length='length')
        
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        # Add geometry data to edges for routing
        for u, v, data in G.edges(data=True):
            # Find the corresponding geometry from the original GeoDataFrame
            edge_geom = gdf.iloc[data.get('momepy_idx', 0)].geometry
            data['geometry'] = edge_geom
            
            # Ensure we have proper weight for routing
            if 'length' in data and data['length'] is not None:
                data['weight'] = data['length']
            else:
                data['weight'] = edge_geom.length
        
        return G
        
    except Exception as e:
        print(f"Error creating network graph with momepy: {e}")
        return None

def create_intersection_connections(graph, intersection_point, network_gdf, intersection_data):
    """
    Creates virtual edges from intersection points to nearby network nodes.
    This ensures seamless routing between different network types.
    """
    if intersection_data.get('route_type') != 'intersection':
        return
    
    # Find nearby network nodes within a reasonable distance
    search_radius = 50  # meters - adjust as needed
    
    for idx, row in network_gdf.iterrows():
        if row.get('route_type') == 'intersection':
            continue  # Skip other intersection nodes
            
        line_geom = row.geometry
        
        if hasattr(line_geom, 'coords'):
            # Check distance to line geometry
            try:
                dist = line_geom.distance(Point(intersection_point))
                if dist <= search_radius:
                    # Create virtual edge to the nearest point on this line
                    nearest_point = line_geom.interpolate(line_geom.project(Point(intersection_point)))
                    nearest_coords = tuple(nearest_point.coords[0])
                    
                    # Add virtual edge with very low weight for seamless transitions
                    graph.add_edge(
                        intersection_point, 
                        nearest_coords, 
                        weight=dist, 
                        geometry=LineString([intersection_point, nearest_coords]),
                        route_type='virtual_connection',
                        source_network=intersection_data.get('connected_networks', [])
                    )
                    
            except Exception as e:
                continue

def find_nearest_node(graph, point, max_search_distance=1000):
    """
    Finds the nearest point on the nearest route segment to the given coordinates.
    Uses spatial indexing for efficient nearest edge finding.
    """
    from shapely.ops import nearest_points
    from shapely.geometry import Point as ShapelyPoint
    
    # Convert to tuple for consistency
    query_point = tuple(point)
    
    print(f"Searching for nearest point on route segments within {max_search_distance}m of {query_point}")
    print(f"Graph has {graph.number_of_edges()} edges to check")
    
    try:
        # Create a GeoSeries of all edges in the graph for spatial indexing
        edge_geometries = []
        edge_data_list = []
        
        for u, v, data in graph.edges(data=True):
            if 'geometry' in data:
                edge_geometries.append(data['geometry'])
                edge_data_list.append((u, v, data))
        
        if not edge_geometries:
            print("No edges with geometry data found")
            return None
        
        # Create GeoSeries for spatial operations
        edges_gseries = gpd.GeoSeries(edge_geometries)
        
        # Find the nearest edge using spatial indexing
        query_shapely_point = ShapelyPoint(query_point)
        
        # Use nearest_points to find the closest edge
        nearest_edge_geom = nearest_points(query_shapely_point, edges_gseries.unary_union)[1]
        
        # Find the closest point on that edge
        nearest_point_on_edge = nearest_edge_geom.interpolate(
            nearest_edge_geom.project(query_shapely_point)
        )
        
        # Calculate distance to verify it's within search range
        distance = query_shapely_point.distance(nearest_point_on_edge)
        
        if distance <= max_search_distance:
            # Extract coordinates
            if hasattr(nearest_point_on_edge, 'coords'):
                coords = list(nearest_point_on_edge.coords)[0]
            else:
                coords = (nearest_point_on_edge.x, nearest_point_on_edge.y)
            
            print(f"✅ Found nearest point on route at {coords} (distance: {distance:.2f}m)")
            return coords
        else:
            print(f"❌ Nearest edge found at {distance:.2f}m, but exceeds search limit of {max_search_distance}m")
            return None
            
    except Exception as e:
        print(f"Error in spatial search: {e}")
        return None

def connect_point_to_network(graph, point, point_type):
    """
    Connects a point (start or end) to the nearest route network node.
    This allows routing from any point along a route, not just at junctions.
    """
    nearest_node = None
    min_dist = float('inf')
    
    # Find the nearest actual route network node
    for node in graph.nodes():
        if node != point:  # Don't connect to ourselves
            dist = np.sqrt((node[0] - point[0])**2 + (node[1] - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
    
    if nearest_node:
        # Create a virtual edge from our point to the nearest network node
        # The weight is the actual distance
        # Create a simple LineString geometry for the virtual edge
        virtual_geometry = LineString([point, nearest_node])
        graph.add_edge(point, nearest_node, weight=min_dist, geometry=virtual_geometry)
        print(f"Connected {point_type} point to route network node at distance {min_dist:.2f}m")
        return True
    else:
        print(f"Could not find route network node to connect {point_type} point")
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
    Now returns full route geometry that follows the actual road/ski route paths.
    """
    if graph is None:
        return None

    print("\nFinding nearest point on route for start point...")
    start_point = find_nearest_node(graph, start_coords)
    
    print("Finding nearest point on route for end point...")
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
            print("Could not connect start/end points to route network")
            return None, None
        
        try:
            # Use Dijkstra's algorithm to find the shortest path based on edge 'weight'
            path_nodes = nx.shortest_path(temp_graph, source=start_point, target=end_point, weight='weight')
            path_length = nx.shortest_path_length(temp_graph, source=start_point, target=end_point, weight='weight')
            
            print(f"Path found with {len(path_nodes)} nodes and a total length of {path_length:.2f} meters.")
            
            # Build full route geometry by following the actual road/ski route paths
            route_geometry = []
            
            print(f"Building route geometry from {len(path_nodes)} path nodes...")
            
            for i in range(len(path_nodes) - 1):
                current_node = path_nodes[i]
                next_node = path_nodes[i + 1]
                
                print(f"Processing edge {i}: {current_node} -> {next_node}")
                
                # Get the edge data between these nodes
                edge_data = temp_graph.get_edge_data(current_node, next_node)
                
                if edge_data and 'geometry' in edge_data:
                    # This edge has actual geometry - use it
                    geom = edge_data['geometry']
                    print(f"  Edge has geometry: {type(geom)}")
                    
                    if hasattr(geom, 'coords'):
                        coords = list(geom.coords)
                        print(f"  Geometry has {len(coords)} coordinates")
                        
                        # Add coordinates (skip first if not the first segment to avoid duplication)
                        if i == 0:
                            route_geometry.extend(coords)
                            print(f"  Added {len(coords)} coordinates (first segment)")
                        else:
                            route_geometry.extend(coords[1:])  # Skip first coordinate to avoid duplication
                            print(f"  Added {len(coords)-1} coordinates (subsequent segment)")
                    else:
                        print(f"  Geometry has no coords attribute")
                        # Fallback: just add the nodes
                        route_geometry.append(current_node)
                        if i == len(path_nodes) - 2:  # Last segment
                            route_geometry.append(next_node)
                else:
                    print(f"  Edge has no geometry data")
                    # Virtual edge (like start/end connections) - just add the nodes
                    route_geometry.append(current_node)
                    if i == len(path_nodes) - 2:  # Last segment
                        route_geometry.append(next_node)
            
            print(f"Final route geometry has {len(route_geometry)} coordinate points")
            
            print(f"Route geometry created with {len(route_geometry)} coordinate points")
            return route_geometry, path_length
        except nx.NetworkXNoPath:
            print("No path could be found between the start and end points.")
            return None, None
    else:
        print("Could not find nearest nodes for the given coordinates.")
        return None, None

def main_route_calculation(start_coords, end_coords, buffer_distance=5000, network_type='tlm3d'):
    """
    Main function that orchestrates the entire routing process with clipping optimization.
    Now supports two network types: TLM3D (roads/trails) and ski touring.
    """
    print(f"=== SWISS {network_type.upper()} Route Optimization ===")
    print(f"Start point: {start_coords}")
    print(f"End point: {end_coords}")
    print(f"Buffer distance: {buffer_distance} meters")
    print(f"Network type: {network_type}")
    
    # Step 1: Create clipped network (the key optimization!)
    clipped_gdf = create_clipped_network(start_coords, end_coords, buffer_distance, network_type)
    
    if clipped_gdf is None:
        print("Failed to create clipped network")
        return None, None, 0
    
    segments_loaded = len(clipped_gdf)
    print(f"Loaded {segments_loaded} {network_type} segments for routing")
    
    # Step 2: Convert to NetworkX graph using momepy
    print("Creating network graph using momepy...")
    network_graph = load_network_to_graph(clipped_gdf)
    
    if network_graph is None:
        print("Failed to create network graph with momepy")
        return None, None, segments_loaded
    
    # Step 3: Calculate the path
    route_geometry, path_length = calculate_shortest_path(network_graph, start_coords, end_coords)
    
    return route_geometry, path_length, segments_loaded

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

