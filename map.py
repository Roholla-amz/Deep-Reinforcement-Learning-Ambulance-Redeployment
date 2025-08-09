import osmnx as ox
import networkx as nx
from shapely.geometry import Polygon
import time

coords = [
    (-75.6898882, 40.4019196),
    (-75.816231, 40.278404),
    (-75.7118609, 40.1756514),
    (-75.5360796, 40.1000604),
    (-75.4179766, 40.0412093),
    (-75.3328326, 39.9549427),
    (-75.2147296, 39.9296734),
    (-75.1350787, 40.0075573),
    (-75.0499346, 40.0349008),
    (-74.9126055, 40.192438),
    (-74.9510577, 40.2763086),
    (-75.080147, 40.3789084),
    (-75.2861407, 40.4520985),
    (-75.4399493, 40.5189454),
    (-75.5635455, 40.5001515),
    (-75.6898882, 40.40191)
]

polygon = Polygon(coords)

G = ox.graph_from_polygon(polygon, network_type='drive')
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

def travel_time(orig, dest):
    route = nx.shortest_path(G, orig, dest, weight='travel_time')

    edges = zip(route[:-1], route[1:])
    travel_time = 0
    for u, v in edges:
        data = G.get_edge_data(u, v)
        # In case of multiple edges between nodes (e.g. different directions)
        edge_data = data[0] if isinstance(data, dict) else data
        travel_time += edge_data.get("travel_time", 0)

    return travel_time / 60

start = time.time()
orig = ox.distance.nearest_nodes(G, -75.008731, 40.211379)  
dest = ox.distance.nearest_nodes(G, -75.3284004, 40.295271)
print('travel time:', travel_time(orig, dest))
print("Time taken: ", time.time() - start)

start = time.time()
orig = ox.distance.nearest_nodes(G, -75.341313, 40.113297)  
dest = ox.distance.nearest_nodes(G, -75.6604586, 40.254768) 
print('travel time:', travel_time(orig, dest))
print("Time taken: ", time.time() - start)

start = time.time()
orig = ox.distance.nearest_nodes(G, -75.407996, 40.2218045)  
dest = ox.distance.nearest_nodes(G, -75.0641492, 40.1228913) 
print('travel time:', travel_time(orig, dest))
print("Time taken: ", time.time() - start)


