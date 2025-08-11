from typing import List, Tuple
from haversine import haversine
from datetime import datetime, timedelta
# import osmnx as ox
# import networkx as nx

def travel_time(loc1: Tuple[float, float], loc2: Tuple[float, float], speed_kmh=40) -> timedelta:
    """
    Calculate the travel time between two locations.
    Args:
        loc1 (Tuple[float, float]): The first location (latitude, longitude).
        loc2 (Tuple[float, float]): The second location (latitude, longitude).
        speed_kmh (float): The speed in km/h.
    Returns:
        float: The travel time in minutes.
    """
    distance_km = distance(loc1, loc2)
    return timedelta(minutes = distance_km / speed_kmh * 60)

h_to_s_tt = {}
def travel_time_from_dict(h_loc: Tuple[float, float], s_loc: Tuple[float, float]):
    key = (h_loc, s_loc)
    if key not in h_to_s_tt:
        h_to_s_tt[key] = travel_time(h_loc, s_loc)
    return h_to_s_tt[key]

def distance(loc1: Tuple[float, float], loc2: Tuple[float, float]):
    return haversine(loc1, loc2)

# def travel_time_by_road(start: Tuple[float, float], end: Tuple[float, float], graph: nx.MultiDiGraph) -> timedelta:
#     orig = ox.distance.nearest_nodes(graph, Y=start[0], X=start[1])  
#     dest = ox.distance.nearest_nodes(graph, Y=end[0], X=end[1])

#     route = nx.shortest_path(graph, orig, dest, weight='travel_time')

#     edges = zip(route[:-1], route[1:])
#     travel_time = 0
#     for u, v in edges:
#         data = graph.get_edge_data(u, v)
#         # In case of multiple edges between nodes (e.g. different directions)
#         edge_data = data[0] if isinstance(data, dict) else data
#         travel_time += edge_data.get("travel_time", 0)

#     return timedelta(minutes = travel_time / 60)