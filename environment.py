from typing import List, Tuple, Dict
from util import util
import random
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
import heapq
import osmnx as ox
import networkx as nx
from shapely.geometry import Polygon

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
        self.patient_id : int = None
    
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
    def __init__(self, id: int, timestamp: datetime, lat: float = None, lng: float = None, priority: int = 0):
        self.id = id
        self.location : Tuple[float, float] = (lat, lng)
        self.timestamp = timestamp
        self.priority = priority

    def __str__(self):
        return f'({self.id}, {self.timestamp})'
    
    def __repr__(self):
        return str(self)

def threshold_for_pr(pr: int) -> int:
    if pr == 0:
        return 15
    if pr == 1:
        return 10
    if pr == 2:
        return 8

class Stats:
    def __init__(self):
        self.pt_per_pr = [[], [], []]
        self.pt_per_dist : List[List[Tuple[float, float]]] = [[], [], []]
        self.pickup_times_sum = [0, 0, 0]
        self.pickup_times_cnt = [0, 0, 0]
        self.pickup_times_in_time = [0, 0, 0]
    
    def add(self, pr:int, pickup: float, dist: float = None):
        self.pt_per_pr[pr].append(pickup)
        if dist:
            self.pt_per_dist[pr].append((dist, pickup))
        self.pickup_times_sum[pr] += pickup
        if pickup <= threshold_for_pr(pr):
            self.pickup_times_in_time[pr] += 1
        self.pickup_times_cnt[pr] += 1     
    
    def AvePT(self) -> Tuple[float, float, float]:
        return self.pickup_times_sum[0] / self.pickup_times_cnt[0], \
            self.pickup_times_sum[1] / self.pickup_times_cnt[1], \
            self.pickup_times_sum[2] / self.pickup_times_cnt[2]
            
    def RelaPT(self) -> Tuple[float, float, float]:
        return self.pickup_times_in_time[0]/self.pickup_times_cnt[0], \
            self.pickup_times_in_time[1]/self.pickup_times_cnt[1], \
            self.pickup_times_in_time[2]/self.pickup_times_cnt[2]
            
    def P90(self) -> Tuple[float, float, float]:
        arr0 = np.array(self.pt_per_pr[0])
        arr1 = np.array(self.pt_per_pr[1])
        arr2 = np.array(self.pt_per_pr[2])

        return np.percentile(arr0, 90).item(), \
            np.percentile(arr1, 90).item(), \
            np.percentile(arr2, 90).item()
            
    def pt_per_distance(self):
        return self.pt_per_dist
    
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
    def __init__(self, m: int, k: int, calls_size=1000, ambulance_count=35, verbose=False, use_map=False, noise: float=None):
        self.m = m
        self.k = k
        self.reward : float = 0
        self.verbose = verbose
        self.ambulance_count = ambulance_count
        self.call_size = calls_size
        self.time : datetime = None
        self.stations : List[Station] = []
        self.hospitals : List[Hospital] = []
        self.calls : Dict[int, Call] = {}
        self.call_ids : List[int] = []
        self.call_counts : Dict[Tuple[int, int, date, int], int] = {}
        self.ambulances: List[Ambulance] = []
        self.free_ambulance : int = None
        self.event_queue : List[TimedEvent] = []
        self.call_queue: Tuple[List[Call], List[Call], List[Call]] = ([], [], [])
        self.stats: Stats = Stats()
        self.use_map = use_map
        self.hold_time = [7, 2, 0]
        self.noise = noise
        self.load_data()
    
    def load_data(self):
        print("Loading environment variables...")
        
        df_hospitals = pd.read_csv('./data/hospitals_from_map.csv')
        df_stations = pd.read_csv('./data/stations_from_map.csv')
        df_calls = pd.read_csv('./data/911_calls_filtered_by_travel_time.csv')
        df_call_counts = pd.read_csv('./data/station_call_counts_per_priority_per_hour.csv')

        print("datasets loaded")
        
        for h in df_hospitals.to_dict(orient='records'):
            self.hospitals.append(Hospital(h['id'], h['hospital_name'], h['lat'], h['lng']))
        
        for s in df_stations.to_dict(orient='records'):
            self.stations.append(Station(s['id'], s['station_name'], s['lat'], s['lng']))

        df_calls['timeStamp'] = pd.to_datetime(df_calls['timeStamp'])
        for c in df_calls.to_dict(orient='records'):
            id = c['id']
            self.calls[id]= Call(c['id'], c['timeStamp'], c['lat'], c['lng'], c['priority'])
            self.call_ids.append(id)
        
        df_call_counts['date'] = pd.to_datetime(df_call_counts['date']).dt.date
        for cc in df_call_counts.to_dict(orient='records'):
            self.call_counts[(cc['station_id'], cc['priority'], cc['date'], cc['hour'])] = cc['call_count']
        
        if self.use_map:
            util.load_graph()
        
        print("environment variables loaded successfully")
     
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
                for pr in range(3):
                    sum = 0.0
                    for day in range(10):
                        date = (self.time - timedelta(days=day+1)).date()
                        key = (i+1, pr, date, hour)
                        sum += self.call_counts[key] if key in self.call_counts else 0
                    x_i.append(sum / 10)
        
            # n_i
            n_i = 0
            for amb in self.ambulances:
                if amb.at_station and amb.location == self.stations[i].location:
                    n_i += 1
            x_i.append(n_i)
        
            # tt_i
            free_amb = self.ambulances[self.free_ambulance - 1]
            start = free_amb.location
            end = self.stations[i].location
            tt_i = util.travel_time_by_road(start, end) if self.use_map else util.travel_time(start, end)
            x_i.append(tt_i.total_seconds() / 60 / 60)
        
            # tt_i_j
            tt_i_j = []
            for j in range(self.ambulance_count):
                amb = self.ambulances[j]
                if amb.destination_type == LocationType.HOSPITAL:
                    start = amb.destination
                    end = self.stations[i].location
                    tt = util.travel_time_by_road(start, end) if self.use_map else util.travel_time(start, end)
                    time = (amb.time_of_arrival - self.time) + tt
                    tt_i_j.append(time.total_seconds() / 60 / 60)
            tt_i_j = sorted(tt_i_j)
            if len(tt_i_j) < self.k:
                tt_i_j += [2] * (self.k - len(tt_i_j))
            if len(tt_i_j) > self.k:
                tt_i_j = tt_i_j[:self.k]
            
            x_i.extend(tt_i_j)
            
            state.append(x_i)
                
        return state
    
    def dispatch(self, call: Call):
        amb, time = self.find_nearest_ambulance(call.location)
        if amb is None:
            if self.verbose:
                print(f"{self.time_str()} - No ambulance available for call {call.id} with priority {call.priority} at {call.location}, call queued")
            self.call_queue[call.priority].append(call)
        else:
            # wait_time = (self.time - call.timestamp).total_seconds() / 60
            # hold_time = self.hold_time[call.priority]
            # if wait_time < hold_time:
            #     if self.verbose:
            #         print(f"{self.time_str()} - Call {call.id} with priority {call.priority} is being held due to hold time")
            #     self.add_event(TimedEvent(self.time + timedelta(minutes=hold_time), (call.id, PayloadType.CALL)))
            # else:
            if self.verbose:
                print(f"{self.time_str()} - Ambulance {amb.id} assigned to call {call.id} with priority {call.priority} at {call.location}")
            amb.at_station = False
            amb.destination = call.location
            amb.destination_type = LocationType.CALL
            amb.time_of_dispatch = self.time
            amb.time_of_arrival = self.time + util.add_noise(time, self.noise) if self.noise else self.time + time
            amb.patient_id = call.id
            
            self.add_event(TimedEvent(amb.time_of_arrival, (amb.id, PayloadType.AMBULANCE)))
    
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
                call = self.calls[id]
                self.dispatch(call)
            elif type == PayloadType.AMBULANCE:
                ambulance = self.ambulances[id - 1]
                if ambulance.destination_type == LocationType.CALL:
                    if self.verbose:
                        print(f"{self.time_str()} - Ambulance {ambulance.id} arrived to accident")
                    
                    call = self.calls[ambulance.patient_id]
                    pickup_time = (self.time - call.timestamp).total_seconds() / 60
                    pr = call.priority
                    
                    self.stats.add(pr, pickup_time)
                    
                    if pickup_time <= threshold_for_pr(pr):
                        self.reward -= pickup_time * (pr + 1)
                    else:
                        self.reward -= pickup_time * (pr + 1)**2
                    hosp, time = self.find_nearest_hospital(ambulance.destination)
                    
                    if self.verbose:
                        print(f"{self.time_str()} - Ambulance {ambulance.id} transporting patient to hospital {hosp.id} at {hosp.location}")
                    
                    ambulance.location = ambulance.destination
                    ambulance.destination = hosp.location
                    ambulance.time_of_arrival = self.time + util.add_noise(time, self.noise) if self.noise else self.time + time
                    ambulance.destination_type = LocationType.HOSPITAL
                    self.add_event(TimedEvent(ambulance.time_of_arrival, (ambulance.id, PayloadType.AMBULANCE)))
                elif ambulance.destination_type == LocationType.HOSPITAL:
                    if self.verbose:
                        print(f"{self.time_str()} - Ambulance {ambulance.id} arrived at hospital")
                    ambulance.location = ambulance.destination
                    ambulance.destination = None
                    ambulance.destination_type = None
                    ambulance.time_of_arrival = None
                    ambulance.patient_id = None
                    self.free_ambulance = ambulance.id
                    return
                elif ambulance.destination_type == LocationType.STATION:
                    if self.verbose:
                        print(f"{self.time_str()} - Ambulance {ambulance.id} arrived at station")

                    ambulance.at_station = True
                    ambulance.location = ambulance.destination
                    ambulance.destination = None
                    ambulance.destination_type = None
                    ambulance.time_of_arrival = None
                    ambulance.patient_id = None                    

                    for pr in reversed(range(3)):
                        if self.call_queue[pr]:
                            call = self.call_queue[pr].pop(0)
                            self.dispatch(call)
                            break
                        
            self.next_event()
    
    def reset(self, call_start=1, ambulance_count=None, m=None, k=None, noise=None) -> State:
        """
        Reset the environment to its initial state.
        """
        if self.verbose:
            print("Resetting environment...")
        if ambulance_count:
            self.ambulance_count = ambulance_count
        if m:
            self.m = m
        if k:
            self.k = k
        if noise:
            self.noise = noise
        self.ambulances = []
        for i in range(self.ambulance_count):
            station = random.choice(list(self.stations))
            self.ambulances.append(Ambulance(i+1, station.location))

        for i in range(call_start-1, call_start - 1 + self.call_size):
            call = self.calls[self.call_ids[i]]
            self.event_queue.append(TimedEvent(call.timestamp, (call.id, PayloadType.CALL)))
            i += 1
        heapq.heapify(self.event_queue)
        
        self.call_queue = ([], [], [])
        
        self.stats = Stats()
        
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
            time = util.travel_time_by_road(location, hospital.location) if self.use_map else util.travel_time(location, hospital.location)
            if time < min_time:
                min_time = time
                nearest_hospital = hospital
        return nearest_hospital, min_time
    
    def find_nearest_station(self, location: Tuple[float, float]) -> Tuple[Station, timedelta]:
        """
        Find the nearest hospital to a given location.
        Args:
            location (Tuple[float, float]): The location (latitude, longitude).
        Returns:
            Hospital: The nearest hospital.
            float: The travel time to the nearest hospital.
        """
        min_time = timedelta.max
        nearest_station = None
        for station in self.stations:
            time = util.travel_time_by_road(location, station.location) if self.use_map else util.travel_time(location, station.location)
            if time < min_time:
                min_time = time
                nearest_station = station
        return nearest_station, min_time
      
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
            start =location
            end = amb.location
            time = util.travel_time_by_road(start, end) if self.use_map else util.travel_time(start, end)
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
        station = self.stations[action]
        ambulance = self.ambulances[self.free_ambulance - 1]
        
        if self.verbose:
            print(f"{self.time_str()} - Ambulance {ambulance.id} dispatched to station {station.id} at {station.location}")
        
        start = ambulance.location
        end = station.location
        time = util.travel_time_by_road(start, end) if self.use_map else util.travel_time(start, end, self.noise)
        ambulance.destination = station.location
        ambulance.destination_type = LocationType.STATION
        self.free_ambulance = None
        self.add_event(TimedEvent(self.time + time, (ambulance.id, PayloadType.AMBULANCE)))
        
        self.reward = 0
        self.run_until_decision_needed()
        return self.get_state(), self.reward, len(self.event_queue) == 0
