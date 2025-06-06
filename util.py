from typing import List, Tuple
from haversine import haversine
from datetime import datetime, timedelta

def travel_time(loc1: Tuple[float, float], loc2: Tuple[float, float], speed_kmh: float = 40) -> timedelta:
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

def distance(loc1: Tuple[float, float], loc2: Tuple[float, float]):
    return haversine(loc1, loc2)