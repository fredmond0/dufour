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

# Pre-processed network file paths
PREPROCESSED_TLM3D_PATH = './data/preprocessed/tlm3d_network_processed.gpkg'
PREPROCESSED_SKI_PATH = './data/preprocessed/ski_network_processed.gpkg'

def create_preprocessed_ski_network():
    """
    Creates pre-processed ski touring network file that momepy can work with directly.
    This is faster than processing the full TLM3D network.
    """
    print("🔄 Creating pre-processed ski touring network file...")
    
    # Create output directory
    os.makedirs('./data/preprocessed', exist_ok=True)
    
    # Process ski touring network only
    print("Processing ski touring network...")
    try:
        ski_gdf = gpd.read_file(SKI_ROUTES_PATH, layer='ski_routes_2056')
        print(f"Loaded {len(ski_gdf)} ski touring routes")
        
        # Explode MultiLineStrings to LineStrings
        ski_processed = ski_gdf.explode(index_parts=True)
        print(f"Exploded to {len(ski_processed)} LineString segments")
        
        # Add route type
        ski_processed['route_type'] = 'ski_tour'
        
        # Save pre-processed file
        ski_processed.to_file(PREPROCESSED_SKI_PATH, driver='GPKG')
        print(f"✅ Saved pre-processed ski network to {PREPROCESSED_SKI_PATH}")
        
        # Show some stats
        print(f"📊 Final network stats:")
        print(f"   - Total segments: {len(ski_processed)}")
        print(f"   - Geometry types: {ski_processed.geometry.geom_type.value_counts().to_dict()}")
        print(f"   - CRS: {ski_processed.crs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing ski network: {e}")
        return False

def create_preprocessed_networks():
    """
    Creates pre-processed network files that momepy can work with directly.
    This function should be run once to prepare the data.
    """
    print("🔄 Creating pre-processed network files...")
    
    # Create output directory
    os.makedirs('./data/preprocessed', exist_ok=True)
    
    # Process TLM3D network
    print("Processing TLM3D network...")
    try:
        tlm3d_gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
        print(f"Loaded {len(tlm3d_gdf)} TLM3D segments")
        
        # Explode MultiLineStrings to LineStrings
        tlm3d_processed = tlm3d_gdf.explode(index_parts=True)
        print(f"Exploded to {len(tlm3d_processed)} LineString segments")
        
        # Add route type
        tlm3d_processed['route_type'] = 'tlm3d'
        
        # Save pre-processed file
        tlm3d_processed.to_file(PREPROCESSED_TLM3D_PATH, driver='GPKG')
        print(f"✅ Saved pre-processed TLM3D network to {PREPROCESSED_TLM3D_PATH}")
        
    except Exception as e:
        print(f"❌ Error processing TLM3D network: {e}")
    
    # Process ski touring network
    print("Processing ski touring network...")
    try:
        ski_gdf = gpd.read_file(SKI_ROUTES_PATH, layer='ski_routes_2056')
        print(f"Loaded {len(ski_gdf)} ski touring routes")
        
        # Explode MultiLineStrings to LineStrings
        ski_processed = ski_gdf.explode(index_parts=True)
        print(f"Exploded to {len(ski_processed)} LineString segments")
        
        # Add route type
        ski_processed['route_type'] = 'ski_tour'
        
        # Save pre-processed file
        ski_processed.to_file(PREPROCESSED_SKI_PATH, driver='GPKG')
        print(f"✅ Saved pre-processed ski network to {PREPROCESSED_SKI_PATH}")
        
    except Exception as e:
        print(f"❌ Error processing ski network: {e}")
    
    print("🎉 Pre-processing complete!")

def load_preprocessed_network(network_type='tlm3d'):
    """
    Loads a pre-processed network file that momepy can work with directly.
    """
    try:
        if network_type == 'tlm3d':
            if os.path.exists(PREPROCESSED_TLM3D_PATH):
                gdf = gpd.read_file(PREPROCESSED_TLM3D_PATH)
                print(f"✅ Loaded pre-processed TLM3D network: {len(gdf)} segments")
                return gdf
            else:
                print("⚠️ Pre-processed TLM3D network not found. Run create_preprocessed_networks() first.")
                return None
                
        elif network_type == 'ski_touring':
            if os.path.exists(PREPROCESSED_SKI_PATH):
                gdf = gpd.read_file(PREPROCESSED_SKI_PATH)
                print(f"✅ Loaded pre-processed ski network: {len(gdf)} segments")
                return gdf
            else:
                print("⚠️ Pre-processed ski network not found. Run create_preprocessed_networks() first.")
                return None
        else:
            print(f"Unknown network type: {network_type}")
            return None
            
    except Exception as e:
        print(f"Error loading pre-processed network: {e}")
        return None

def create_clipped_network_from_graph(graph, start_coords, end_coords, buffer_distance=5000):
    """
    Creates a clipped network by filtering the pre-processed graph.
    This is much more efficient than clipping geometries.
    """
    print(f"Creating clipped network with buffer distance: {buffer_distance} meters")
    
    # Create bounding box
    min_x = min(start_coords[0], end_coords[0]) - buffer_distance
    max_x = max(start_coords[0], end_coords[0]) + buffer_distance
    min_y = min(start_coords[1], end_coords[1]) - buffer_distance
    max_y = max(start_coords[1], end_coords[1]) + buffer_distance
    
    print(f"Bounding box: ({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f})")
    
    # Debug: Check a few edges to see their coordinate ranges
    print(f"Debug: Checking coordinate ranges of first few edges...")
    edge_count = 0
    for u, v, data in list(graph.edges(data=True))[:5]:
        if 'geometry' in data:
            bounds = data['geometry'].bounds
            print(f"  Edge {u}->{v}: bounds {bounds}")
            edge_count += 1
    
    # Filter edges within bounding box
    clipped_edges = []
    total_edges = graph.number_of_edges()
    edges_with_geometry = 0
    
    for u, v, data in graph.edges(data=True):
        if 'geometry' in data:
            edges_with_geometry += 1
            # Check if edge intersects with bounding box
            bbox_geom = box(min_x, min_y, max_x, max_y)
            if data['geometry'].intersects(bbox_geom):
                clipped_edges.append((u, v))  # Only store edge tuple, not data
    
    print(f"Debug: {edges_with_geometry}/{total_edges} edges have geometry data")
    print(f"Debug: {len(clipped_edges)} edges intersect with bounding box")
    
    if len(clipped_edges) == 0:
        print("⚠️ No edges found in bounding box. Trying alternative approach...")
        # Fallback: include edges that have nodes within the bounding box
        for u, v, data in graph.edges(data=True):
            if 'geometry' in data:
                # Check if either node is within the bounding box
                u_in_box = min_x <= u[0] <= max_x and min_y <= u[1] <= max_y
                v_in_box = min_x <= v[0] <= max_x and min_y <= v[1] <= max_y
                if u_in_box or v_in_box:
                    clipped_edges.append((u, v))  # Only store edge tuple
        
        print(f"Fallback approach found {len(clipped_edges)} edges")
    
    # Create subgraph with only the clipped edges
    if clipped_edges:
        try:
            # Use NetworkX's built-in edge_subgraph which preserves all data
            # Convert 2-tuples to 3-tuples with default key 0
            edges_with_keys = [(u, v, 0) for u, v in clipped_edges]
            clipped_graph = graph.edge_subgraph(edges_with_keys)
            
            # Verify geometry data is preserved
            edges_with_geom = sum(1 for _, _, data in clipped_graph.edges(data=True) if 'geometry' in data)
            print(f"✅ Clipped network created: {clipped_graph.number_of_nodes()} nodes, {clipped_graph.number_of_edges()} edges")
            print(f"✅ Edges with geometry data: {edges_with_geom}/{clipped_graph.number_of_edges()}")
            return clipped_graph
            
        except Exception as e:
            print(f"⚠️ edge_subgraph failed: {e}")
            print("🔄 Using manual subgraph creation...")
            
            # Fallback: Create subgraph manually
            clipped_graph = nx.Graph()
            
            # Add the clipped edges and their data
            for u, v in clipped_edges:
                edge_data = graph.get_edge_data(u, v)
                if edge_data:
                    clipped_graph.add_edge(u, v, **edge_data)
            
            # Verify geometry data is preserved
            edges_with_geom = sum(1 for _, _, data in clipped_graph.edges(data=True) if 'geometry' in data)
            print(f"✅ Manual clipped network created: {clipped_graph.number_of_nodes()} nodes, {clipped_graph.number_of_edges()} edges")
            print(f"✅ Edges with geometry data: {edges_with_geom}/{clipped_graph.number_of_edges()}")
            return clipped_graph
    else:
        print("❌ No edges found in clipping area")
        return None

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
    Ensures all geometries are LineStrings for momepy compatibility.
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
            
            # Ensure all geometries are LineStrings for momepy compatibility
            road_gdf = road_gdf.explode(index_parts=True)
            print(f"Exploded to {len(road_gdf)} LineString segments")
            
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
                
                # CRITICAL: Explode MultiLineStrings to LineStrings for momepy compatibility
                print("Exploding MultiLineString geometries to LineStrings for momepy...")
                ski_gdf = ski_gdf.explode(index_parts=True)
                print(f"Exploded to {len(ski_gdf)} LineString segments")
                
                # Verify all geometries are now LineStrings
                final_geom_types = ski_gdf.geometry.geom_type.value_counts().to_dict()
                print(f"Final geometry types: {final_geom_types}")
                
                return ski_gdf
                
            except Exception as e:
                print(f"Error loading ski routes: {e}")
                # Fallback to network file
                print("Falling back to ski network file...")
                ski_gdf = gpd.read_file(SKI_NETWORK_PATH, layer='ski_network_2056', bbox=bbox)
                print(f"Loaded {len(ski_gdf)} ski touring segments in clipped area")
                ski_gdf['route_type'] = 'ski_tour'
                
                # Also explode this to LineStrings
                ski_gdf = ski_gdf.explode(index_parts=True)
                print(f"Exploded network to {len(ski_gdf)} LineString segments")
                
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
        
        # Debug: Check what momepy actually provides
        print(f"Debug: Checking momepy edge data structure...")
        sample_edges = list(G.edges(data=True))
        if sample_edges:
            sample_edge = sample_edges[0]
            print(f"Sample edge data keys: {list(sample_edge[2].keys())}")
        else:
            print("No edges found in graph")
        
        # Create a more robust geometry mapping
        print(f"Creating robust geometry mapping for {G.number_of_edges()} edges...")
        
        # First, try to understand what momepy provides
        sample_edges = list(G.edges(data=True))[:5]
        if sample_edges:
            sample_edge = sample_edges[0]
            print(f"Sample edge data keys: {list(sample_edge[2].keys())}")
            
            # Check if momepy provides any useful indexing
            if 'momepy_idx' in sample_edge[2]:
                print(f"momepy_idx found: {sample_edge[2]['momepy_idx']}")
            else:
                print("No momepy_idx found")
        
        # Create a spatial index for efficient geometry lookup
        from shapely.geometry import Point
        import numpy as np
        
        # Build a spatial index mapping coordinates to route indices
        coord_to_route = {}
        for idx, row in gdf.iterrows():
            route_geom = row.geometry
            # Sample points along the route for indexing
            if hasattr(route_geom, 'interpolate'):
                for t in np.linspace(0, 1, 5):  # Sample 5 points along route
                    try:
                        point = route_geom.interpolate(t, normalized=True)
                        coord_key = (round(point.x, 0), round(point.y, 0))
                        coord_to_route[coord_key] = idx
                    except:
                        continue
        
        print(f"Built spatial index with {len(coord_to_route)} coordinate mappings")
        
        # Map geometries to edges using the spatial index
        edge_count = 0
        mapped_geometries = 0
        fallback_geometries = 0
        
        for u, v, data in G.edges(data=True):
            edge_count += 1
            
            # Try to find the best matching geometry
            edge_geom = None
            
            # Approach 1: Use momepy_idx if available
            if 'momepy_idx' in data and data['momepy_idx'] is not None:
                try:
                    idx = data['momepy_idx']
                    if 0 <= idx < len(gdf):
                        edge_geom = gdf.iloc[idx].geometry
                        mapped_geometries += 1
                        if edge_count <= 10:  # Log first 10 successful mappings
                            print(f"  Edge {edge_count}: Using momepy_idx {idx}")
                except Exception as e:
                    pass
            
            # Approach 2: Use spatial index lookup
            if edge_geom is None:
                # Try to find route by looking up coordinates in spatial index
                u_coord = (round(u[0], 0), round(u[1], 0))
                v_coord = (round(v[0], 0), round(v[1], 0))
                
                # Look up both coordinates
                for coord in [u_coord, v_coord]:
                    if coord in coord_to_route:
                        idx = coord_to_route[coord]
                        edge_geom = gdf.iloc[idx].geometry
                        mapped_geometries += 1
                        if edge_count <= 10:  # Log first 10 successful mappings
                            print(f"  Edge {edge_count}: Found via spatial index (idx {idx})")
                        break
            
            # Approach 3: Fallback to a reasonable route (not just the first one)
            if edge_geom is None:
                # Find a route that's geographically close to this edge
                edge_center = Point((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
                min_distance = float('inf')
                best_idx = 0
                
                for idx in range(min(50, len(gdf))):  # Check first 50 routes
                    route_geom = gdf.iloc[idx].geometry
                    try:
                        dist = edge_center.distance(route_geom)
                        if dist < min_distance:
                            min_distance = dist
                            best_idx = idx
                    except:
                        continue
                
                edge_geom = gdf.iloc[best_idx].geometry
                fallback_geometries += 1
                if edge_count <= 10:  # Log first 10 fallbacks
                    print(f"  Edge {edge_count}: Fallback to route {best_idx} (distance: {min_distance:.0f}m)")
            
            # Store the geometry
            data['geometry'] = edge_geom
            
            # Ensure we have proper weight for routing
            if 'length' in data and data['length'] is not None:
                data['weight'] = data['length']
            else:
                data['weight'] = edge_geom.length
        
        print(f"✅ Geometry mapping complete:")
        print(f"   - Total edges: {edge_count}")
        print(f"   - Successfully mapped: {mapped_geometries}")
        print(f"   - Fallback mappings: {fallback_geometries}")
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

def find_nearest_node(graph, point, max_search_distance=10000):
    """
    Finds the nearest point on the nearest route segment to the given coordinates.
    Uses spatial indexing for efficient nearest edge finding.
    """
    from shapely.geometry import Point as ShapelyPoint
    
    # Convert to tuple for consistency
    query_point = tuple(point)
    
    print(f"Searching for nearest point on route segments within {max_search_distance}m of {query_point}")
    print(f"Graph has {graph.number_of_edges()} edges to check")
    
    # Simple approach: find the closest edge by checking each one individually
    nearest_point = None
    min_dist = float('inf')
    
    for u, v, data in graph.edges(data=True):
        if 'geometry' in data:
            # Get the route geometry
            route_geometry = data['geometry']
            
            # Find the closest point on this route segment
            try:
                # Project the query point onto the route geometry
                closest_point = route_geometry.interpolate(route_geometry.project(ShapelyPoint(query_point)))
                dist = ShapelyPoint(query_point).distance(closest_point)
                
                if dist < min_dist and dist <= max_search_distance:
                    min_dist = dist
                    nearest_point = closest_point
                    print(f"  New closest: {closest_point.coords[0]} at {dist:.2f}m")
                        
            except Exception as e:
                print(f"  Error processing edge {u}->{v}: {e}")
                continue
        else:
            print(f"  Edge {u}->{v} has no geometry data")
    
    if nearest_point is not None:
        try:
            if nearest_point.geom_type == 'Point':
                coords = list(nearest_point.coords)[0]
            else:
                # Fallback: return the point as a tuple if possible
                coords = (nearest_point.x, nearest_point.y)
            
            print(f"✅ Found nearest point on route at {coords} (distance: {min_dist:.2f}m)")
            return coords
        except Exception as e:
            print(f"Error extracting coordinates from nearest point: {e}")
            return None
    else:
        print("❌ No route segments found within search distance")
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
        # Use a much higher weight to discourage routing through virtual edges
        # The routing algorithm should prefer actual route network edges
        virtual_weight = min_dist * 1000  # Make virtual edges 1000x more expensive
        virtual_geometry = LineString([point, nearest_node])
        graph.add_edge(point, nearest_node, weight=virtual_weight, geometry=virtual_geometry)
        print(f"Connected {point_type} point to route network node at distance {min_dist:.2f}m (virtual weight: {virtual_weight:.2f}m)")
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
                    print(f"  Edge has no geometry data - this is a virtual edge")
                    # Virtual edge (like start/end connections) - just add the nodes
                    # This is expected for start/end connections
                    route_geometry.append(current_node)
                    if i == len(path_nodes) - 2:  # Last segment
                        route_geometry.append(next_node)
            
            print(f"Final route geometry has {len(route_geometry)} coordinate points")
            
            # Filter out duplicate consecutive coordinates
            if len(route_geometry) > 1:
                filtered_geometry = [route_geometry[0]]
                for coord in route_geometry[1:]:
                    if coord != filtered_geometry[-1]:
                        filtered_geometry.append(coord)
                route_geometry = filtered_geometry
                print(f"Filtered to {len(route_geometry)} unique coordinate points")
            
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
    Main function that orchestrates the entire routing process using pre-processed networks.
    Now supports two network types: TLM3D (roads/trails) and ski touring.
    """
    print(f"=== SWISS {network_type.upper()} Route Optimization ===")
    print(f"Start point: {start_coords}")
    print(f"End point: {end_coords}")
    print(f"Buffer distance: {buffer_distance} meters")
    print(f"Network type: {network_type}")
    
    # Step 1: Load pre-processed network (no clipping needed)
    print("Loading pre-processed network...")
    network_gdf = load_preprocessed_network(network_type)
    
    if network_gdf is None:
        print("Failed to load pre-processed network")
        return None, None, 0
    
    segments_loaded = len(network_gdf)
    print(f"Loaded {segments_loaded} {network_type} segments")
    
    # Step 2: Convert to NetworkX graph using momepy
    print("Creating network graph using momepy...")
    full_network_graph = load_network_to_graph(network_gdf)
    
    if full_network_graph is None:
        print("Failed to create network graph with momepy")
        return None, None, segments_loaded
    
    # Step 3: Create clipped subgraph for routing (much more efficient)
    print("Creating clipped subgraph for routing...")
    clipped_graph = create_clipped_network_from_graph(
        full_network_graph, start_coords, end_coords, buffer_distance
    )
    
    if clipped_graph is None or clipped_graph.number_of_edges() == 0:
        print("Failed to create clipped subgraph")
        return None, None, segments_loaded
    
    # Debug: Check the weights of the clipped network edges
    print(f"\\n=== DEBUGGING CLIPPED NETWORK ===")
    print(f"Clipped network has {clipped_graph.number_of_edges()} edges")
    
    # Sample some edge weights to see what we're working with
    edge_weights = []
    for u, v, data in clipped_graph.edges(data=True):
        if 'weight' in data:
            edge_weights.append(data['weight'])
    
    if edge_weights:
        print(f"Edge weight statistics:")
        print(f"  Min weight: {min(edge_weights):.2f}m")
        print(f"  Max weight: {max(edge_weights):.2f}m")
        print(f"  Average weight: {sum(edge_weights)/len(edge_weights):.2f}m")
        
        # Show first few edges
        print(f"\\nFirst 5 edges:")
        for i, (u, v, data) in enumerate(list(clipped_graph.edges(data=True))[:5]):
            weight = data.get('weight', 'unknown')
            has_geom = 'geometry' in data
            print(f"  Edge {i}: {u} -> {v}, weight: {weight}, has_geometry: {has_geom}")
    else:
        print("No edge weights found in clipped network")

    # Step 4: Calculate the path using the clipped subgraph
    route_geometry, path_length = calculate_shortest_path(clipped_graph, start_coords, end_coords)
    
    return route_geometry, path_length, segments_loaded

# --- Main Execution ---
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "preprocess":
            # Create pre-processed networks
            print("🔄 Creating pre-processed network files...")
            create_preprocessed_networks()
            print("✅ Pre-processing complete! You can now run the routing application.")
        elif sys.argv[1] == "preprocess-ski":
            # Create pre-processed ski network only
            print("🔄 Creating pre-processed ski touring network file...")
            success = create_preprocessed_ski_network()
            if success:
                print("✅ Ski network pre-processing complete! You can now test ski routing.")
            else:
                print("❌ Ski network pre-processing failed.")
        else:
            print("Usage:")
            print("  python route_backend.py preprocess      # Create all pre-processed networks")
            print("  python route_backend.py preprocess-ski  # Create ski network only (faster)")
            print("  python route_backend.py                 # Test routing (requires pre-processed networks)")
    else:
        # Original test routing functionality
        print("Usage:")
        print("  python route_backend.py preprocess      # Create all pre-processed networks")
        print("  python route_backend.py preprocess-ski  # Create ski network only (faster)")
        print("  python route_backend.py                 # Test routing (requires pre-processed networks)")
        print()
        
        # Check if pre-processed networks exist
        if os.path.exists(PREPROCESSED_TLM3D_PATH) and os.path.exists(PREPROCESSED_SKI_PATH):
            print("✅ All pre-processed networks found. Testing routing...")
            
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
            route_geometry, path_length, segments_loaded = main_route_calculation(
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
        elif os.path.exists(PREPROCESSED_SKI_PATH):
            print("✅ Ski network found. Testing ski routing...")
            
            # Test with ski touring coordinates (Swiss Alps area)
            # Using the working coordinates from the web interface
            # WGS84: start:(46.34524751872321, 9.349758208605882), end:(46.359781458208204, 9.32144920284767)
            start_point_coords = (2747113, 1134443)  # Converted from WGS84
            end_point_coords = (2744895, 1136005)    # Converted from WGS84
            
            buffer_distance = 10000  # 10km buffer for ski routes (increased to find more routes)
            
            # Calculate the path using the optimized approach
            route_geometry, path_length, segments_loaded = main_route_calculation(
                start_point_coords, 
                end_point_coords, 
                buffer_distance,
                'ski_touring'
            )

            if route_geometry:
                print("\n=== Ski Route Summary ===")
                print(f"Total path length: {path_length:.2f} meters")
                print(f"Number of coordinate points: {len(route_geometry)}")
                print("✅ Ski routing is working!")
            else:
                print("Failed to calculate ski route.")
        else:
            print("❌ Pre-processed networks not found.")
            print("Run: python route_backend.py preprocess-ski  # For ski routes only (faster)")
            print("Or:  python route_backend.py preprocess      # For all networks")
            print("This will create the necessary pre-processed network files.")

