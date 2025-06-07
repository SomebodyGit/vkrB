import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from shapely.geometry import LineString
import time as tm
from geopy.distance import great_circle

def get_min_edge_length(G, u, v):
    edges = G.get_edge_data(u, v)
    if not edges:
        return None
    lengths = [data['length'] for data in edges.values() if 'length' in data]
    return min(lengths) if lengths else None


def true_brute_force_iterative(G, start_node, target_node):
    stack = [(start_node, [start_node], {start_node}, 0)]
    min_length = float('inf')
    best_path = None
    min_length_to_node = {node: float('inf') for node in G.nodes}
    min_length_to_node[start_node] = 0

    while stack:
        current_node, path, path_set, current_length = stack.pop()
        if current_node == target_node:
            if current_length < min_length:
                min_length = current_length
                best_path = path
            continue
        for neighbor in G.neighbors(current_node):
            if neighbor not in path_set:
                edge_length = get_min_edge_length(G, current_node, neighbor)
                if edge_length is not None:
                    new_length = current_length + edge_length
                    if new_length < min_length_to_node[neighbor]:
                        min_length_to_node[neighbor] = new_length
                        new_path = path + [neighbor]
                        new_set = path_set.copy()
                        new_set.add(neighbor)
                        stack.append((neighbor, new_path, new_set, new_length))
    return best_path, min_length

start_lat = 55.737917
start_lon = 37.414975
target_lat = 55.814152
target_lon = 37.550787
start_coords = (start_lat, start_lon)
target_coords = (target_lat, target_lon)
start_time_all = tm.time()

center_point = ((start_lat + target_lat) / 2, (start_lon + target_lon) / 2)

dist = great_circle((start_lat, start_lon), (target_lat, target_lon)).meters * 1.05

G = ox.graph_from_point(center_point, dist=dist, network_type='drive')

strongly_connected_components = list(nx.strongly_connected_components(G))

largest_component = max(strongly_connected_components, key=len)
G_largest = G.subgraph(largest_component).copy()

start_node = ox.distance.nearest_nodes(G_largest, start_coords[1], start_coords[0])
target_node = ox.distance.nearest_nodes(G_largest, target_coords[1], target_coords[0])

nodes = list(G_largest.nodes)
node_to_index = {node: idx for idx, node in enumerate(nodes)}
index_to_node = {idx: node for node, idx in node_to_index.items()}
graph_dict = {
    node_to_index[node]: [node_to_index[neighbor] for neighbor in G_largest.neighbors(node)]
    for node in G_largest.nodes
}
if not nx.has_path(G_largest, start_node, target_node):
    exit()
start_time_brute = tm.time()
brute_path, brute_length = true_brute_force_iterative(G_largest, start_node, target_node)
if brute_path:
    print("Кратчайший путь (Brute Force):", brute_path)
    print(f"Длина кратчайшего пути (Brute Force): {brute_length:.2f} метров")
else:
    print("Путь не был найден с brute force.")
end_time_brute = tm.time()
print(f"Brute-force время выполнения: {end_time_brute - start_time_brute:.2f} секунд")
print("\n")


end_time_all = tm.time()
print(f"Полное время: {end_time_all - start_time_all:.2f} секунд")
print("\n")
