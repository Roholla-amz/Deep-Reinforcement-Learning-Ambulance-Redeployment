from typing import List, Tuple
from haversine import haversine
from datetime import datetime, timedelta
import osmnx as ox
import networkx as nx
from shapely.geometry import Polygon
import time
import numpy as np
from scipy.stats import truncnorm

class util:
    
    def distance(loc1: Tuple[float, float], loc2: Tuple[float, float]):
        return haversine(loc1, loc2)
    
    def calculate_travel_time(loc1: Tuple[float, float], loc2: Tuple[float, float], speed_kmh=40) -> timedelta:
        """
        Calculate the travel time between two locations.
        Args:
            loc1 (Tuple[float, float]): The first location (latitude, longitude).
            loc2 (Tuple[float, float]): The second location (latitude, longitude).
            speed_kmh (float): The speed in km/h.
        Returns:
            float: The travel time in minutes.
        """
        distance_km = util.distance(loc1, loc2)
        return timedelta(minutes = distance_km / speed_kmh * 60)

    def add_noise(value: float, noise: float) -> float:
        if noise == 0:
            return value
        a, b = -0.5 / noise, 0.5 / noise
        sample = truncnorm.rvs(a, b, loc=1, scale=noise)
        # sample = np.random.normal(1, noise)
        result = value*sample
        # if result < value / 2:
        #     return value / 2
        # if result > value * 1.5:
        #     return value * 1.5
        return result
    
    h_to_s_tt = {}
    def travel_time(h_loc: Tuple[float, float], s_loc: Tuple[float, float], noise:float=None) -> timedelta:
        key = (h_loc, s_loc)
        if key not in util.h_to_s_tt:
            util.h_to_s_tt[key] = util.calculate_travel_time(h_loc, s_loc)
        result = util.h_to_s_tt[key]
        if noise:
            return timedelta(seconds= util.add_noise(result.total_seconds(), noise))
        return result

    

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
    graph: nx.MultiDiGraph = None


    def load_graph():
        print("Loading graph...")
        ox.settings.log_console = True
        util.graph = ox.graph_from_polygon(util.polygon, network_type='drive',simplify=True)
        util.graph = ox.add_edge_speeds(util.graph)
        util.graph = ox.add_edge_travel_times(util.graph)
        ox.settings.log_console = False
        print("Graph loaded")

    def calculate_travel_time_by_road(orig, dest) -> timedelta:
        route = nx.shortest_path(util.graph, orig, dest, weight='travel_time')

        edges = zip(route[:-1], route[1:])
        travel_time = 0
        for u, v in edges:
            data = util.graph.get_edge_data(u, v)
            edge_data = data[0] if isinstance(data, dict) else data
            travel_time += edge_data.get("travel_time", 0)

        return timedelta(seconds=travel_time)

    node_pair_travel_time = {}
    def travel_time_by_road(start: Tuple[float, float], end: Tuple[float, float]) -> timedelta:
        orig = ox.distance.nearest_nodes(util.graph, Y=start[0], X=start[1])  
        dest = ox.distance.nearest_nodes(util.graph, Y=end[0], X=end[1])
        key = (orig, dest)
        if key not in util.node_pair_travel_time:        
            util.node_pair_travel_time[key] = util.calculate_travel_time_by_road(orig, dest)
        return util.node_pair_travel_time[key]