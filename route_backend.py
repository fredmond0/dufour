import geopandas as gpd
import networkx as nx
import os
import requests
import math
import pandas as pd
from shapely.geometry import Point, box, LineString
from shapely.ops import nearest_points
import momepy
from pyproj import Transformer

# --- Configuration ---
PREPROCESSED_TLM3D_PATH = './data/preprocessed/tlm3d_network_processed.gpkg'
PREPROCESSED_SKI_PATH = './data/preprocessed/ski_network_processed.gpkg'
PREPROCESSED_SAC_PATH = './data/preprocessed/sac_network_processed.gpkg'

def create_clipped_network(start_coords, end_coords, buffer_distance, network_type):
    """Creates a clipped GeoDataFrame from a pre-processed file."""
    filepath = {
        'tlm3d': PREPROCESSED_TLM3D_PATH,
        'ski_touring': PREPROCESSED_SKI_PATH,
        'sac_alpine': PREPROCESSED_SAC_PATH
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
    """Converts a GeoDataFrame into a NetworkX graph with preserved geometry."""
    print("Creating network graph using momepy...")
    gdf = gdf.reset_index(drop=True)
    if 'length' not in gdf.columns:
        gdf['length'] = gdf.geometry.length
    
    # Create graph with geometry preservation
    graph = momepy.gdf_to_nx(gdf, approach='primal', length='length', multigraph=True)
    
    # CRITICAL: Manually preserve detailed geometry in edges
    print("Preserving detailed geometry in graph edges...")
    geometry_preserved = 0
    
    for u, v, key, data in graph.edges(keys=True, data=True):
        # Find the corresponding edge in the original GDF
        edge_found = False
        
        for idx, row in gdf.iterrows():
            edge_geom = row.geometry
            if edge_geom.geom_type == 'LineString':
                # Check if this edge connects nodes u and v
                coords = list(edge_geom.coords)
                start_point = Point(coords[0])
                end_point = Point(coords[-1])
                
                u_point = Point(graph.nodes[u]['x'], graph.nodes[u]['y'])
                v_point = Point(graph.nodes[v]['x'], graph.nodes[v]['y'])
                
                # Check if edge endpoints match graph nodes (within tolerance)
                tolerance = 1.0  # 1 meter tolerance
                
                if ((start_point.distance(u_point) < tolerance and end_point.distance(v_point) < tolerance) or
                    (start_point.distance(v_point) < tolerance and end_point.distance(u_point) < tolerance)):
                    
                    # Preserve the FULL detailed geometry
                    data['geometry'] = edge_geom
                    data['detailed_coords'] = coords  # Store coordinates for easy access
                    geometry_preserved += 1
                    edge_found = True
                    break
        
        if not edge_found:
            # Fallback: create simple line between nodes
            u_coords = (graph.nodes[u]['x'], graph.nodes[u]['y'])
            v_coords = (graph.nodes[v]['x'], graph.nodes[v]['y'])
            data['geometry'] = LineString([u_coords, v_coords])
            data['detailed_coords'] = [u_coords, v_coords]
    
    print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Preserved detailed geometry for {geometry_preserved} edges.")
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
        print(f"⚠️ Elevation profile unavailable")
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
    """Splits a line at a point with robust handling - PRESERVES detailed geometry."""
    from shapely.ops import split
    
    print(f"      Attempting to split line (length: {line.length:.1f}m)")
    print(f"      Point coords: ({point.x:.1f}, {point.y:.1f})")
    
    # Get line coordinates
    coords = list(line.coords)
    print(f"      Original line has {len(coords)} coordinates")
    
    start_point = Point(coords[0])
    end_point = Point(coords[-1])
    
    start_dist = point.distance(start_point)
    end_dist = point.distance(end_point)
    print(f"      Distance to start: {start_dist:.3f}m, to end: {end_dist:.3f}m")
    
    # If point is too close to endpoints, don't split
    if start_dist < tolerance or end_dist < tolerance:
        print(f"      Point too close to line endpoints - skipping split")
        return [line]
    
    try:
        # Project point onto line to get exact position
        projected_distance = line.project(point)
        total_length = line.length
        
        print(f"      Projected distance along line: {projected_distance:.3f}m / {total_length:.3f}m")
        
        # Find the segment of the line where the point should be inserted
        current_distance = 0.0
        split_index = -1
        
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            segment_length = segment.length
            
            if current_distance <= projected_distance <= current_distance + segment_length:
                split_index = i
                break
            current_distance += segment_length
        
        if split_index == -1:
            print(f"      ❌ Could not find split location - using simple split")
            segment1 = LineString([coords[0], point.coords[0]])
            segment2 = LineString([point.coords[0], coords[-1]])
        else:
            print(f"      Found split location at segment {split_index}")
            
            # Build first segment: from start to split point
            first_coords = coords[:split_index + 1] + [point.coords[0]]
            segment1 = LineString(first_coords)
            
            # Build second segment: from split point to end
            second_coords = [point.coords[0]] + coords[split_index + 1:]
            segment2 = LineString(second_coords)
            
            print(f"      Segment 1: {len(first_coords)} coords, length: {segment1.length:.1f}m")
            print(f"      Segment 2: {len(second_coords)} coords, length: {segment2.length:.1f}m")
        
        # Validate segments
        if segment1.length > tolerance and segment2.length > tolerance:
            print(f"      ✅ Detailed geometry split successful!")
            return [segment1, segment2]
        else:
            print(f"      ⚠️ Split created too-short segments")
            return [line]
            
    except Exception as e:
        print(f"      ❌ Detailed split failed: {e}")
        
        # Final fallback: try shapely split with small buffer
        try:
            buffered_point = point.buffer(0.001)  # 1mm buffer
            split_result = split(line, buffered_point)
            
            if split_result.geom_type == 'GeometryCollection':
                lines = [geom for geom in split_result.geoms 
                        if geom.geom_type == 'LineString' and geom.length > tolerance]
                if len(lines) >= 2:
                    print(f"      ✅ Fallback split successful: {len(lines)} parts")
                    return lines[:2]
            
            print(f"      ❌ All split methods failed - using original segment")
            return [line]
            
        except Exception as e2:
            print(f"      ❌ Fallback split also failed: {e2}")
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
        print("    ⚠️ Both points on same segment - extracting geometry between projected points")
        
        # Get the original segment geometry
        original_segment = start_segment  # Same as end_segment since start_idx == end_idx
        
        # Calculate the projected distances along the segment
        start_distance = original_segment.project(start_snap_point)
        end_distance = original_segment.project(end_snap_point)
        
        # Ensure start_distance < end_distance for proper extraction
        if start_distance > end_distance:
            start_distance, end_distance = end_distance, start_distance
            start_snap_point, end_snap_point = end_snap_point, start_snap_point
        
        print(f"    Extracting geometry from {start_distance:.1f}m to {end_distance:.1f}m along segment")
        
        # Extract the geometry between the two points using interpolate
        from shapely.geometry import LineString
        from shapely.ops import substring
        
        try:
            # Use substring to get the portion of the line between the two distances
            route_segment = substring(original_segment, start_distance, end_distance)
            
            if route_segment.geom_type == 'LineString' and route_segment.length > 0.1:
                # Extract coordinates in LV95 (same as regular routing flow)
                route_coords = list(route_segment.coords)
                route_length = route_segment.length
                print(f"    ✅ Extracted {len(route_coords)} coordinates, length: {route_length:.1f}m")
                return route_coords, route_length
            else:
                print("    ⚠️ Substring extraction failed, using direct line")
                # Fallback to direct line in LV95 coordinates
                route_geometry = [start_snap_point.coords[0], end_snap_point.coords[0]]
                return route_geometry, start_snap_point.distance(end_snap_point)
                
        except Exception as e:
            print(f"    ❌ Error extracting segment geometry: {e}")
            # Fallback to direct line in LV95 coordinates
            route_geometry = [start_snap_point.coords[0], end_snap_point.coords[0]]
            return route_geometry, start_snap_point.distance(end_snap_point)
    
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
    
    # Convert to NetworkX graph and preserve detailed geometry
    modified_graph = momepy.gdf_to_nx(modified_gdf, approach='primal', length='length', multigraph=True)
    
    # CRITICAL: Re-preserve detailed geometry in the modified graph
    print("  Re-preserving detailed geometry in modified graph...")
    geometry_preserved = 0
    
    for u, v, key, data in modified_graph.edges(keys=True, data=True):
        # Find the corresponding edge in the modified GDF
        edge_found = False
        
        for idx, row in modified_gdf.iterrows():
            edge_geom = row.geometry
            if edge_geom.geom_type == 'LineString':
                # Check if this edge connects nodes u and v
                coords = list(edge_geom.coords)
                if len(coords) < 2:
                    continue
                    
                start_point = Point(coords[0])
                end_point = Point(coords[-1])
                
                u_point = Point(modified_graph.nodes[u]['x'], modified_graph.nodes[u]['y'])
                v_point = Point(modified_graph.nodes[v]['x'], modified_graph.nodes[v]['y'])
                
                # Check if edge endpoints match graph nodes (within tolerance)
                tolerance = 1.0  # 1 meter tolerance
                
                if ((start_point.distance(u_point) < tolerance and end_point.distance(v_point) < tolerance) or
                    (start_point.distance(v_point) < tolerance and end_point.distance(u_point) < tolerance)):
                    
                    # Preserve the FULL detailed geometry
                    data['geometry'] = edge_geom
                    data['detailed_coords'] = coords  # Store coordinates for easy access
                    geometry_preserved += 1
                    edge_found = True
                    break
        
        if not edge_found:
            # Fallback: create simple line between nodes
            u_coords = (modified_graph.nodes[u]['x'], modified_graph.nodes[u]['y'])
            v_coords = (modified_graph.nodes[v]['x'], modified_graph.nodes[v]['y'])
            data['geometry'] = LineString([u_coords, v_coords])
            data['detailed_coords'] = [u_coords, v_coords]
    
    print(f"  Modified graph: {modified_graph.number_of_nodes()} nodes, {modified_graph.number_of_edges()} edges")
    print(f"  Re-preserved detailed geometry for {geometry_preserved} edges")
    
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
        
        # Reconstruct route geometry from path nodes using actual edge geometries
        route_geometry = []
        
        print(f"  Reconstructing route geometry from {len(path_nodes)} nodes...")
        
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            print(f"    Processing edge {u} → {v}")
            
            if modified_graph.has_edge(u, v):
                # Get edge geometry
                edge_data = modified_graph[u][v]
                edge_geom = None
                
                if isinstance(edge_data, dict):
                    edge_geom = edge_data.get('geometry')
                    print(f"      Single edge, has geometry: {edge_geom is not None}")
                else:
                    # Multiple edges - get first one with geometry
                    for key, data in edge_data.items():
                        if 'geometry' in data:
                            edge_geom = data['geometry']
                            print(f"      Multi-edge key {key}, found geometry")
                            break
                
                if edge_geom and hasattr(edge_geom, 'coords'):
                    coords = list(edge_geom.coords)
                    print(f"      Edge geometry has {len(coords)} coordinates")
                    
                    # Determine direction - check which end is closer to current node
                    u_point = Point(modified_graph.nodes[u]['x'], modified_graph.nodes[u]['y'])
                    start_dist = u_point.distance(Point(coords[0]))
                    end_dist = u_point.distance(Point(coords[-1]))
                    
                    if end_dist < start_dist:
                        coords.reverse()
                        print(f"      Reversed edge direction")
                    
                    # Add coordinates to route
                    if not route_geometry:
                        # First edge - add all coordinates
                        route_geometry.extend(coords)
                        print(f"      Added {len(coords)} coords (first edge)")
                    else:
                        # Subsequent edges - skip first coordinate to avoid duplication
                        route_geometry.extend(coords[1:])
                        print(f"      Added {len(coords)-1} coords (skip first)")
                else:
                    print(f"      ❌ No geometry found for edge {u} → {v}")
                    # Fallback to straight line between nodes
                    u_coords = (modified_graph.nodes[u]['x'], modified_graph.nodes[u]['y'])
                    v_coords = (modified_graph.nodes[v]['x'], modified_graph.nodes[v]['y'])
                    if not route_geometry:
                        route_geometry.append(u_coords)
                    route_geometry.append(v_coords)
                    print(f"      Added fallback straight line")
            else:
                print(f"      ❌ Edge {u} → {v} not found in graph!")
        
        print(f"  Final route geometry: {len(route_geometry)} points")
        
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