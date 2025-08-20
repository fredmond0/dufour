import geopandas as gpd
import networkx as nx
import os
import pandas as pd
from shapely.geometry import Point, box, LineString
from shapely.ops import unary_union
import numpy as np

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
    Loads a road network from a GeoDataFrame into a NetworkX graph.
    Simple, reliable approach that works for both roads and ski routes.
    """
    if gdf is None or len(gdf) == 0:
        print("Error: No road data to process")
        return None

    print("Creating network graph from clipped data...")
    
    # Create an empty graph
    G = nx.Graph()
    
    # Track processed geometries for debugging
    processed_count = 0
    error_count = 0

    # Pre-process MultiLineString geometries using WKT conversion
    print("Pre-processing MultiLineString geometries using WKT...")
    processed_gdf = gdf.copy()
    
    for index, row in processed_gdf.iterrows():
        try:
            line = row.geometry
            if line.geom_type == 'MultiLineString':
                # Convert MultiLineString to individual LineStrings using WKT
                from shapely import wkt
                from shapely.geometry import LineString
                
                try:
                    # Convert to WKT and back to bypass the Sub-geometries error
                    wkt_string = line.wkt
                    reconstructed_geom = wkt.loads(wkt_string)
                    
                    # Now try to extract individual parts
                    line_strings = []
                    if reconstructed_geom.geom_type == 'MultiLineString':
                        # Extract each part of the MultiLineString
                        for part in reconstructed_geom.geoms:
                            if part.is_valid and part.geom_type == 'LineString':
                                coords = list(part.coords)
                                if len(coords) >= 2:
                                    line_strings.append(LineString(coords))
                                    
                        # Use the first valid LineString
                        if line_strings:
                            processed_gdf.at[index, 'geometry'] = line_strings[0]
                            print(f"WKT: Converted MultiLineString to LineString at index {index}")
                        else:
                            # Fallback: convert entire MultiLineString to single LineString
                            all_coords = []
                            for part in reconstructed_geom.geoms:
                                if part.geom_type == 'LineString':
                                    all_coords.extend(list(part.coords))
                            if len(all_coords) >= 2:
                                processed_gdf.at[index, 'geometry'] = LineString(all_coords)
                                print(f"WKT: Converted MultiLineString to single LineString at index {index}")
                            else:
                                print(f"WKT: Could not extract valid coordinates from MultiLineString at index {index}")
                                continue
                    else:
                        # If it's not a MultiLineString after WKT conversion, use it directly
                        processed_gdf.at[index, 'geometry'] = reconstructed_geom
                        print(f"WKT: Geometry became {reconstructed_geom.geom_type} at index {index}")
                        
                except Exception as e:
                    print(f"WKT conversion failed for index {index}: {e}")
                    # Last resort: try to create a simple LineString from bounds
                    try:
                        bounds = line.bounds
                        if len(bounds) == 4:
                            # Create a simple LineString from bounds (not ideal but better than nothing)
                            simple_line = LineString([(bounds[0], bounds[1]), (bounds[2], bounds[3])])
                            processed_gdf.at[index, 'geometry'] = simple_line
                            print(f"Fallback: Created simple LineString from bounds at index {index}")
                        else:
                            print(f"Could not create fallback geometry for index {index}")
                            continue
                    except Exception as e2:
                        print(f"Fallback also failed for index {index}: {e2}")
                        continue
                        
        except Exception as e:
            print(f"Error pre-processing MultiLineString at index {index}: {e}")
            continue

    # Now process the pre-processed GeoDataFrame - simple approach
    print("Processing pre-processed geometries...")
    
    for index, row in processed_gdf.iterrows():
        try:
            line = row.geometry
            
            # Skip invalid geometries
            if line is None or not line.is_valid:
                print(f"Skipping invalid geometry at index {index}")
                continue
                
            # Handle line geometry
            if hasattr(line, 'coords') and line.geom_type == 'LineString':
                coords = list(line.coords)
                if len(coords) >= 2:
                    # Ensure coordinates are valid numbers
                    start_coords = coords[0]
                    end_coords = coords[-1]
                    
                    if (isinstance(start_coords[0], (int, float)) and 
                        isinstance(start_coords[1], (int, float)) and
                        isinstance(end_coords[0], (int, float)) and 
                        isinstance(end_coords[1], (int, float))):
                        
                        start_node = tuple(start_coords)
                        end_node = tuple(end_coords)
                        
                        # Calculate length in meters (assuming coordinates are in meters)
                        length = line.length
                        
                        # Add the edge to the graph with its length as the weight
                        # Remove geometry from row data to avoid conflicts
                        row_data = row.to_dict()
                        if 'geometry' in row_data:
                            del row_data['geometry']
                        
                        G.add_edge(start_node, end_node, weight=length, geometry=line, **row_data)
                        processed_count += 1
                    else:
                        print(f"Invalid coordinate types at index {index}: {start_coords}, {end_coords}")
                        error_count += 1
                else:
                    print(f"LineString at index {index} has insufficient coordinates: {len(coords)}")
                    error_count += 1
            else:
                print(f"Unexpected geometry type: {line.geom_type} at index {index}")
                error_count += 1
                    
        except Exception as e:
            print(f"Error processing geometry at index {index}: {e}")
            error_count += 1
            continue

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Processed {processed_count} geometries successfully, {error_count} errors")
    
    if G.number_of_edges() == 0:
        print("Warning: No edges were created. Check geometry data.")
        return None
        
    return G

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
    This allows starting/ending anywhere along a route, not just at junctions.
    Works for both roads and ski routes.
    """
    nearest_point = None
    min_dist = float('inf')
    nearest_edge = None
    
    # Convert to tuple for consistency
    query_point = tuple(point)
    
    print(f"Searching for nearest point on route segments within {max_search_distance}m of {query_point}")
    print(f"Graph has {graph.number_of_edges()} edges to check")
    
    # Check all edges (route segments) to find the closest point on any route
    for u, v, data in graph.edges(data=True):
        if 'geometry' in data:
            # Get the route geometry
            route_geometry = data['geometry']
            
            # Find the closest point on this route segment
            try:
                # Handle different geometry types
                if route_geometry.geom_type == 'LineString':
                    # Project the query point onto the route geometry
                    closest_point = route_geometry.interpolate(route_geometry.project(Point(query_point)))
                    dist = Point(query_point).distance(closest_point)
                    
                    if dist < min_dist and dist <= max_search_distance:
                        min_dist = dist
                        nearest_point = closest_point
                        nearest_edge = (u, v, data)
                        print(f"  New closest: {closest_point.coords[0]} at {dist:.2f}m")
                        
                elif route_geometry.geom_type == 'MultiLineString':
                    # For MultiLineString, check each part
                    for part in route_geometry.geoms:
                        try:
                            closest_point = part.interpolate(part.project(Point(query_point)))
                            dist = Point(query_point).distance(closest_point)
                            
                            if dist < min_dist and dist <= max_search_distance:
                                min_dist = dist
                                nearest_point = closest_point
                                nearest_edge = (u, v, data)
                                print(f"  New closest (MultiLineString): {closest_point.coords[0]} at {dist:.2f}m")
                        except Exception as e:
                            continue
                            
            except Exception as e:
                print(f"  Error processing edge {u}->{v}: {e}")
                continue
        else:
            print(f"  Edge {u}->{v} has no geometry data")
    
    if nearest_point is not None:
        try:
            if nearest_point.geom_type == 'Point':
                coords = list(nearest_point.coords)[0]
            elif nearest_point.geom_type == 'LineString':
                coords = list(nearest_point.coords)[0]
            elif nearest_point.geom_type == 'MultiLineString':
                # For MultiLineString, get the first coordinate of the first part
                coords = list(nearest_point.geoms[0].coords)[0]
            else:
                # Fallback for other geometry types
                coords = (nearest_point.x, nearest_point.y) if hasattr(nearest_point, 'x') else tuple(nearest_point.coords[0])
            
            print(f"✅ Found nearest point on route at {coords} (distance: {min_dist:.2f}m)")
            return coords
        except Exception as e:
            print(f"Error extracting coordinates from nearest point: {e}")
            # Fallback: return the point as a tuple if possible
            if hasattr(nearest_point, 'x') and hasattr(nearest_point, 'y'):
                return (nearest_point.x, nearest_point.y)
            else:
                return None
    else:
        # No route segments found - this should not happen if the graph is properly constructed
        print("❌ No route segments found with geometry data. Check graph construction.")
        return None

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
    
    # Step 2: Convert to NetworkX graph
    network_graph = load_network_to_graph(clipped_gdf)
    
    if network_graph is None:
        print("Failed to create network graph")
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

