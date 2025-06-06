from typing import List, Tuple, Dict
from util import *
import random
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
import heapq

class LocationType(Enum):
    STATION = 1
    HOSPITAL = 2
    CALL = 3

    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)

class Station:
    def __init__(self, id: int, name: str = None, lat: float = None, lng: float = None):
        self.id = id
        self.name = name
        self.location : Tuple[float, float] = (lat, lng)
    
    def __str__(self):
        return f'({self.id}, {self.name})'
    
    def __repr__(self):
        return str(self)

class Ambulance:
    def __init__(self, id: int, location: Tuple[float, float] = None, at_station: bool = True):
        self.id = id
        self.at_station = at_station
        self.location = location
        self.destination : Tuple[float, float] = None
        self.destination_type : LocationType = None
        self.time_of_dispatch : datetime = None
        self.time_of_arrival : datetime = None
    
    def __str__(self):
        return f'({self.id}, {self.location}, {self.at_station})'
    
    def __repr__(self):
        return str(self)
    
class Hospital:
    def __init__(self, id: int, name: str = None, lat: float = None, lng: float = None):
        self.id = id
        self.name = name
        self.location : Tuple[float, float] = (lat, lng)

    def __str__(self):
        return f'({self.id}, {self.name})'
    
    def __repr__(self):
        return str(self)

class Call:
    def __init__(self, id: int, timestamp: datetime, lat: float = None, lng: float = None):
        self.id = id
        self.location : Tuple[float, float] = (lat, lng)
        self.timestamp = timestamp

    def __str__(self):
        return f'({self.id}, {self.timestamp})'
    
    def __repr__(self):
        return str(self)

class PayloadType(Enum):
    CALL = 1
    AMBULANCE = 2

@dataclass(order=True)
class TimedEvent:
    timestamp: datetime
    payload : Tuple[int, PayloadType] = field(compare=False)

State = List[List[float]]

class Environment:
    """"
    Environment class to hold the state of the simulation.
    Attributes:
        m (int): Number of time periods.
        k (int): Number of ambulances.
        ambulances (list): List of Ambulance objects.
        stations (list): List of Station objects.
    """
    def __init__(self, m: int, k: int, calls_size=1000, ambulance_count=35, verbose=False, normalize=True):
        self.m = m
        self.k = k
        self.reward : int = 0
        self.verbose = verbose
        self.ambulance_count = ambulance_count
        self.call_size = calls_size
        self.normalize = normalize
        self.time : datetime = None
        self.stations : List[Station] = []
        self.hospitals : List[Hospital] = []
        self.calls : List[Call] = []
        self.call_counts : Dict[Tuple[int, date, int], int] = {}
        self.ambulances: List[Ambulance] = []
        self.free_ambulance : int = None
        self.event_queue : List[TimedEvent] = []
        self.load_data()
    
    def load_data(self):
        if self.verbose:
            print("Loading data...")
        
        df_hospitals = pd.read_csv('./data/hospitals.csv')
        df_stations = pd.read_csv('./data/stations.csv')
        df_calls = pd.read_csv('./data/911_calls_no_outliers.csv')
        df_call_counts = pd.read_csv('./data/station_call_counts_per_hour.csv')

        for h in df_hospitals.to_dict(orient='records'):
            self.hospitals.append(Hospital(h['id'], h['hospital_name'], h['lat'], h['lng']))
        
        for s in df_stations.to_dict(orient='records'):
            self.stations.append(Station(s['id'], s['station_name'], s['lat'], s['lng']))

        df_calls['timeStamp'] = pd.to_datetime(df_calls['timeStamp'])
        for c in df_calls.to_dict(orient='records'):
            self.calls.append(Call(c['id'], c['timeStamp'], c['lat'], c['lng']))
        first_call_timestamp = min(self.calls, key=lambda x: x.timestamp).timestamp
        self.first_day = datetime(first_call_timestamp.year, first_call_timestamp.month, first_call_timestamp.day)

        df_call_counts['date'] = pd.to_datetime(df_call_counts['date']).dt.date
        for cc in df_call_counts.to_dict(orient='records'):
            self.call_counts[(cc['station_id'], cc['date'], cc['hour'])] = cc['call_count']
        
        data = np.load("feature_norm_stats.npz")
        self.mean_x = data['mu']
        self.std_x = data['sigma']
        
        if self.verbose:
            print("data loaded")
        
    def normalize_state(self, state: State) -> State:
        """
        Normalize the state using the precomputed mean and standard deviation.
        Args:
            state (State): The state to normalize.
        Returns:
            State: The normalized state.
        """
        raw_state = np.array(state)
        normalized_state = (raw_state - self.mean_x[None, :]) / (self.std_x[None, :] + 1e-8)
        return normalized_state
     
    def next_event(self) -> TimedEvent:
        if not self.event_queue:
            return None
        return heapq.heappop(self.event_queue)
    
    def peak_next_event(self) -> TimedEvent:
        if not self.event_queue:
            return None
        return self.event_queue[0]
    
    def add_event(self, event: TimedEvent):
        heapq.heappush(self.event_queue, event)
    
    def time_str(self):
        return self.time.strftime("%Y-%m-%d %H:%M:%S")
    
    def get_state(self) -> State:
        """
        Get the current state of the environment.
        Returns:
            List[StationParameters]: The current state of the environment.
        """
        if self.free_ambulance is None:
            return None
        
        state: State = []
        
        
        for i in range(len(self.stations)):
            x_i = []
            
            # lambda_i_m
            for m in range(self.m):
                hour = (self.time + timedelta(hours=m+1)).hour
                sum = 0.0
                for day in range(10):
                    date = (self.time - timedelta(days=day+1)).date()
                    sum += self.call_counts[(i+1, date, hour)] if (i+1, date, hour) in self.call_counts else 0
                x_i.append(sum / 10)
        
            # n_i
            n_i = 0
            for amb in self.ambulances:
                if amb.at_station and amb.location == self.stations[i].location:
                    n_i += 1
            x_i.append(n_i)
        
            # tt_i
            free_amb = self.ambulances[self.free_ambulance - 1]
            tt_i = travel_time(free_amb.location, self.stations[i].location)
            x_i.append(tt_i.total_seconds() / 60 / 60)
        
            # tt_i_j
            tt_i_j = []
            for j in range(self.ambulance_count):
                amb = self.ambulances[j]
                if amb.destination_type == LocationType.HOSPITAL:
                    tt = travel_time(amb.destination, self.stations[i].location)
                    time = (amb.time_of_arrival - self.time) + tt
                    tt_i_j.append(time.total_seconds() / 60 / 60)
            tt_i_j = sorted(tt_i_j)
            if len(tt_i_j) < self.k:
                tt_i_j += [2] * (self.k - len(tt_i_j))
            if len(tt_i_j) > self.k:
                tt_i_j = tt_i_j[:self.k]
            
            x_i.extend(tt_i_j)
            
            state.append(x_i)
        
        if self.normalize:
            return self.normalize_state(state)
        else:
            return state
    
    def run_until_decision_needed(self):
        while True:
            event = self.peak_next_event()
            if event is None:
                if self.verbose:
                    print(f"{self.time_str()} - No more events in the queue")
                return
            id, type = event.payload
            self.time = event.timestamp
            if type == PayloadType.CALL:
                call = self.calls[id - 1]
                amb, time = self.find_nearest_ambulance(call.location)
                if amb is None:
                    if self.verbose:
                        print(f"{self.time_str()} - No available ambulance for call {call.id}")
                else:
                    if self.verbose:
                        print(f"{self.time_str()} - Ambulance {amb.id} assigned to call {call.id}")
                    amb.at_station = False
                    amb.destination = call.location
                    amb.destination_type = LocationType.CALL
                    amb.time_of_dispatch = self.time
                    amb.time_of_arrival = self.time + time
                    self.add_event(TimedEvent(amb.time_of_arrival, (amb.id, PayloadType.AMBULANCE)))
            elif type == PayloadType.AMBULANCE:
                ambulance = self.ambulances[id - 1]
                if ambulance.destination_type == LocationType.CALL:
                    if self.verbose:
                        print(f"{self.time_str()} - Ambulance {ambulance.id} arrived to accident")
                    if self.time - ambulance.time_of_dispatch <= timedelta(minutes=30):
                        self.reward += 1
                    hosp, time = self.find_nearest_hospital(ambulance.destination)
                    ambulance.location = ambulance.destination
                    ambulance.destination = hosp.location
                    ambulance.time_of_arrival = self.time + time
                    ambulance.destination_type = LocationType.HOSPITAL
                    self.add_event(TimedEvent(ambulance.time_of_arrival, (ambulance.id, PayloadType.AMBULANCE)))
                elif ambulance.destination_type == LocationType.HOSPITAL:
                    if self.verbose:
                        print(f"{self.time_str()} - Ambulance {ambulance.id} arrived at hospital {ambulance.destination}")
                    ambulance.location = ambulance.destination
                    ambulance.destination = None
                    ambulance.destination_type = None
                    ambulance.time_of_arrival = None
                    self.free_ambulance = ambulance.id
                    return
                elif ambulance.destination_type == LocationType.STATION:
                    if self.verbose:
                        print(f"{self.time_str()} - Ambulance {ambulance.id} arrived at station {ambulance.destination}")
                    ambulance.at_station = True
                    ambulance.location = ambulance.destination
                    ambulance.destination = None
                    ambulance.destination_type = None
                    ambulance.time_of_arrival = None
            self.next_event()
    
    def reset(self, call_start=1) -> State:
        """
        Reset the environment to its initial state.
        """
        if self.verbose:
            print("Resetting environment...")
        self.ambulances = []
        for i in range(self.ambulance_count):
            station = random.choice(list(self.stations))
            self.ambulances.append(Ambulance(i+1, station.location))

        self.event_queue = [TimedEvent(call.timestamp, (call.id, PayloadType.CALL)) for call in self.calls if call_start <= call.id < call_start+self.call_size]
        heapq.heapify(self.event_queue)
        
        if self.verbose:
            print("environment reset")
            
        
        self.run_until_decision_needed()
        return self.get_state()             
                       
    def find_nearest_hospital(self, location: Tuple[float, float]) -> Tuple[Hospital, timedelta]:
        """
        Find the nearest hospital to a given location.
        Args:
            location (Tuple[float, float]): The location (latitude, longitude).
        Returns:
            Hospital: The nearest hospital.
            float: The travel time to the nearest hospital.
        """
        min_time = timedelta.max
        nearest_hospital = None
        for hospital in self.hospitals:
            time = travel_time(location, hospital.location)
            if time < min_time:
                min_time = time
                nearest_hospital = hospital
        return nearest_hospital, min_time
      
    def find_nearest_ambulance(self, location: Tuple[float, float]) -> Tuple[Ambulance, timedelta]:
        """
        Find the nearest station to a given location.
        Args:
            location (Tuple[float, float]): The location (latitude, longitude).
        Returns:
            Station: The nearest station.
            float: The travel time to the nearest station.
        """
        min_time = timedelta.max
        nearest_ambulance = None
        for amb in self.ambulances:
            if amb.at_station is False:
                continue
            time = travel_time(location, amb.location)
            if time < min_time:
                min_time = time
                nearest_ambulance = amb
        return nearest_ambulance, min_time
        
    def step(self, action: int) -> Tuple[State, float, bool]:
        """
        Take a step in the environment based on the action.
        Args:
            action: The action to take.
        Returns:
            Tuple: The next state, reward, and done flag.
        """
        self.next_event()
        station = self.stations[action - 1]
        ambulance = self.ambulances[self.free_ambulance - 1]
        
        
        time = travel_time(ambulance.location, station.location)
        ambulance.destination = station.location
        ambulance.destination_type = LocationType.STATION
        self.free_ambulance = None
        self.add_event(TimedEvent(self.time + time, (ambulance.id, PayloadType.AMBULANCE)))
        if self.verbose:
            print(f"{self.time_str()} - Ambulance {ambulance.id} dispatched to station {station.id}")
        
        self.reward = 0
        self.run_until_decision_needed()
        return self.get_state(), self.reward, len(self.event_queue) == 0
