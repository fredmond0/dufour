import geopandas as gpd
import networkx as nx
import momepy
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np

print("=== ROUTING DEBUG SCRIPT ===\n")

# Load the cleaned network
print("1. Loading cleaned network...")
cleaned_gdf = gpd.read_file('cleaned_network.gpkg')
print(f"   Loaded {len(cleaned_gdf)} segments")
print(f"   CRS: {cleaned_gdf.crs}")

# Convert to NetworkX graph
print("\n2. Converting to NetworkX graph...")
G = momepy.gdf_to_nx(cleaned_gdf, approach='primal')
print(f"   Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Check graph connectivity
print("\n3. Analyzing graph connectivity...")
if nx.is_connected(G):
    print("   ✅ Graph is fully connected")
else:
    print("   ❌ Graph has disconnected components!")
    components = list(nx.connected_components(G))
    print(f"   Found {len(components)} disconnected components:")
    for i, comp in enumerate(components):
        print(f"     Component {i+1}: {len(comp)} nodes")

# Check for isolated nodes
isolated_nodes = list(nx.isolates(G))
if isolated_nodes:
    print(f"   ⚠️  Found {len(isolated_nodes)} isolated nodes")
else:
    print("   ✅ No isolated nodes")

# Analyze node degrees
degrees = [G.degree(node) for node in G.nodes()]
print(f"   Node degrees: min={min(degrees)}, max={max(degrees)}, avg={np.mean(degrees):.1f}")

# Check edge weights
print("\n4. Analyzing edge weights...")
weights = []
for u, v, data in G.edges(data=True):
    if 'length' in data:
        weights.append(data['length'])
    elif 'weight' in data:
        weights.append(data['weight'])
    else:
        print(f"   ⚠️  Edge {u}-{v} has no length/weight attribute")

if weights:
    print(f"   Edge lengths: min={min(weights):.1f}m, max={max(weights):.1f}m, avg={np.mean(weights):.1f}m")

# Test routing between some nodes
print("\n5. Testing routing between nodes...")
nodes = list(G.nodes())
if len(nodes) >= 2:
    # Test a few random routes
    for i in range(min(3, len(nodes)//2)):
        start = nodes[i*2]
        end = nodes[i*2 + 1]
        
        try:
            path = nx.shortest_path(G, start, end, weight='length')
            path_length = nx.shortest_path_length(G, start, end, weight='length')
            print(f"   Route {start} → {end}: {len(path)} nodes, {path_length:.1f}m")
            
            # Check if this is a direct connection
            if G.has_edge(start, end):
                direct_length = G[start][end].get('length', G[start][end].get('weight', 'unknown'))
                print(f"     Direct connection exists: {direct_length}m")
            else:
                print(f"     No direct connection - routing through {len(path)-2} intermediate nodes")
                
        except nx.NetworkXNoPath:
            print(f"   ❌ No path found from {start} to {end}")

# Visualize the graph
print("\n6. Creating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot 1: Original geometries
cleaned_gdf.plot(ax=ax1, color='blue', linewidth=1)
ax1.set_title('Cleaned Network Geometries')
ax1.set_aspect('equal')

# Plot 2: NetworkX graph structure
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=20, node_color='red')
nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='black', width=1, alpha=0.7)
ax2.set_title('NetworkX Graph Structure')
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()

print("\n=== DEBUG COMPLETE ===")
print("Check the visualization to see if the graph structure matches the geometries!")
print("If you see disconnected components or missing edges, that explains the routing issues.")
