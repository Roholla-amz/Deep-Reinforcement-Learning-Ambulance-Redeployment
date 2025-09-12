import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
import re
from collections import defaultdict
import numpy as np

class AmbulanceAnimation:
    def __init__(self, filename):
        self.events = []
        self.ambulances = {}
        self.stations = {}
        self.hospitals = {}
        self.calls = {}  # Track calls with their status
        
        # Color mapping for priorities
        self.priority_colors = {0: 'green', 1: 'yellow', 2: 'red'}
        
        # Animation settings - compressed time (1 minute = 1 second)
        self.time_compression = 1000  # 60 seconds real time = 1 second animation
        self.frame_rate = 30  # 30 FPS for smooth animation
        self.parse_file(filename)
        self.setup_plot()
        
    def parse_file(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse timestamp
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if not timestamp_match:
                    continue
                    
                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                
                # Parse different event types
                if 'assigned to call' in line:
                    match = re.search(r'Ambulance (\d+) assigned to call (\d+) with priority (\d+) at \(([^)]+)\)', line)
                    if match:
                        ambulance_id, call_id, priority, coords = match.groups()
                        lat, lon = map(float, coords.split(', '))
                        call_location = (lat, lon)
                        
                        # Initialize call info
                        self.calls[int(call_id)] = {
                            'location': call_location,
                            'priority': int(priority),
                            'created_time': timestamp,
                            'picked_up_time': None,
                            'assigned_ambulance': int(ambulance_id),
                            'active': True
                        }
                        
                        self.events.append({
                            'timestamp': timestamp,
                            'type': 'assigned',
                            'ambulance': int(ambulance_id),
                            'call': int(call_id),
                            'priority': int(priority),
                            'location': call_location
                        })
                        
                elif 'arrived to accident' in line:
                    match = re.search(r'Ambulance (\d+) arrived to accident', line)
                    if match:
                        ambulance_id = int(match.group(1))
                        
                        # Find which call this ambulance was responding to
                        for call_id, call_info in self.calls.items():
                            if (call_info['assigned_ambulance'] == ambulance_id and 
                                call_info['active'] and 
                                call_info['picked_up_time'] is None):
                                call_info['picked_up_time'] = timestamp
                                call_info['active'] = False
                                break
                        
                        self.events.append({
                            'timestamp': timestamp,
                            'type': 'arrived_accident',
                            'ambulance': ambulance_id
                        })
                        
                elif 'transporting patient to hospital' in line:
                    match = re.search(r'Ambulance (\d+) transporting patient to hospital (\d+) at \(([^)]+)\)', line)
                    if match:
                        ambulance_id, hospital_id, coords = match.groups()
                        lat, lon = map(float, coords.split(', '))
                        hospital_location = (lat, lon)
                        
                        self.hospitals[int(hospital_id)] = hospital_location
                        
                        self.events.append({
                            'timestamp': timestamp,
                            'type': 'transporting',
                            'ambulance': int(ambulance_id),
                            'hospital': int(hospital_id),
                            'location': hospital_location
                        })
                        
                elif 'arrived at hospital' in line:
                    match = re.search(r'Ambulance (\d+) arrived at hospital', line)
                    if match:
                        ambulance_id = int(match.group(1))
                        self.events.append({
                            'timestamp': timestamp,
                            'type': 'arrived_hospital',
                            'ambulance': ambulance_id
                        })
                        
                elif 'dispatched to station' in line:
                    match = re.search(r'Ambulance (\d+) dispatched to station (\d+) at \(([^)]+)\)', line)
                    if match:
                        ambulance_id, station_id, coords = match.groups()
                        lat, lon = map(float, coords.split(', '))
                        station_location = (lat, lon)
                        
                        self.stations[int(station_id)] = station_location
                        
                        self.events.append({
                            'timestamp': timestamp,
                            'type': 'to_station',
                            'ambulance': int(ambulance_id),
                            'station': int(station_id),
                            'location': station_location
                        })
                        
                elif 'arrived at station' in line:
                    match = re.search(r'Ambulance (\d+) arrived at station', line)
                    if match:
                        ambulance_id = int(match.group(1))
                        self.events.append({
                            'timestamp': timestamp,
                            'type': 'arrived_station',
                            'ambulance': ambulance_id
                        })
        
        # Sort events by timestamp
        self.events.sort(key=lambda x: x['timestamp'])
        
        if self.events:
            self.start_time = self.events[0]['timestamp']
            self.end_time = self.events[-1]['timestamp']
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
            self.animation_duration_seconds = self.total_duration_seconds / self.time_compression
            self.total_frames = int(self.animation_duration_seconds * self.frame_rate)
        
        # Initialize ambulance states - assume they start at station 2 (most common return station)
        default_station = list(self.stations.values())[0] if self.stations else (40.272578, -75.461352)
        
        for event in self.events:
            if event['ambulance'] not in self.ambulances:
                self.ambulances[event['ambulance']] = {
                    'position': default_station,
                    'destination': None,
                    'status': 'at_station',
                    'call_priority': None,
                    'call_id': None,
                    'move_start_time': None,
                    'move_end_time': None,
                    'move_start_pos': None,
                    'move_end_pos': None
                }
    
    def setup_plot(self):
        # Calculate bounds from all coordinates
        all_coords = []
        for event in self.events:
            if 'location' in event:
                all_coords.append(event['location'])
        
        if all_coords:
            lats, lons = zip(*all_coords)
            self.lat_min, self.lat_max = min(lats) - 0.01, max(lats) + 0.01
            self.lon_min, self.lon_max = min(lons) - 0.01, max(lons) + 0.01
        else:
            self.lat_min, self.lat_max = 40.0, 40.4
            self.lon_min, self.lon_max = -75.7, -75.0
        
        # Create figure with subplots for main plot and timeline
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main plot
        self.ax = plt.subplot2grid((20, 1), (0, 0), rowspan=18)
        self.ax.set_xlim(self.lon_min, self.lon_max)
        self.ax.set_ylim(self.lat_min, self.lat_max)
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Ambulance Dispatch Animation')
        self.ax.grid(True, alpha=0.3)
        
        # Timeline subplot
        self.timeline_ax = plt.subplot2grid((20, 1), (19, 0), rowspan=1)
        self.timeline_ax.set_xlim(0, 1)
        self.timeline_ax.set_ylim(0, 1)
        self.timeline_ax.set_xticks([])
        self.timeline_ax.set_yticks([])
        self.timeline_ax.set_ylabel('Timeline', rotation=0, ha='right')
        
        # Clock text
        self.clock_text = self.fig.text(0.02, 0.95, '', fontsize=14, weight='bold')
        
        # Create legend
        self.create_legend()
        
    def create_legend(self):
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Priority 0'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='Priority 1'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Priority 2'),
            plt.Line2D([0], [0], linestyle='--', color='blue', label='Destination path'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', markersize=8, label='Hospital'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=8, label='Station'),
            plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='purple', markersize=8, label='Active Call')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    def interpolate_position(self, start_pos, end_pos, progress):
        """Linear interpolation between two positions"""
        if start_pos is None or end_pos is None:
            return start_pos or end_pos
        
        lat = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        lon = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        return (lat, lon)
    
    def get_current_simulation_time(self, frame):
        """Convert frame number to simulation time"""
        if self.total_frames <= 0:
            return self.start_time
        
        time_progress = frame / self.total_frames
        simulation_seconds = time_progress * self.total_duration_seconds
        return self.start_time + timedelta(seconds=simulation_seconds)
    
    def get_ambulance_position_at_time(self, ambulance_id, current_time):
        """Calculate ambulance position at specific time with smooth interpolation"""
        ambulance = self.ambulances[ambulance_id]
        
        if (ambulance['move_start_time'] and ambulance['move_end_time'] and
            ambulance['move_start_time'] <= current_time <= ambulance['move_end_time']):
            
            # Calculate progress through the movement
            total_move_time = (ambulance['move_end_time'] - ambulance['move_start_time']).total_seconds()
            elapsed_time = (current_time - ambulance['move_start_time']).total_seconds()
            
            if total_move_time > 0:
                progress = min(1.0, elapsed_time / total_move_time)
                return self.interpolate_position(
                    ambulance['move_start_pos'], 
                    ambulance['move_end_pos'], 
                    progress
                )
        
        return ambulance['position']
    
    def update_ambulance_states(self, current_time):
        """Update ambulance states based on events up to current time"""
        
        for event in self.events:
            if event['timestamp'] > current_time:
                break
                
            ambulance_id = event['ambulance']
            ambulance = self.ambulances[ambulance_id]
            
            if event['type'] == 'assigned':
                # Start moving to accident location
                ambulance['move_start_time'] = event['timestamp']
                ambulance['move_start_pos'] = ambulance['position']
                ambulance['move_end_pos'] = event['location']
                ambulance['destination'] = event['location']
                ambulance['status'] = 'to_accident'
                ambulance['call_priority'] = event['priority']
                ambulance['call_id'] = event['call']
                
                # Find when they arrive at accident
                for future_event in self.events:
                    if (future_event['ambulance'] == ambulance_id and 
                        future_event['type'] == 'arrived_accident' and
                        future_event['timestamp'] > event['timestamp']):
                        ambulance['move_end_time'] = future_event['timestamp']
                        break
                
            elif event['type'] == 'arrived_accident':
                ambulance['position'] = ambulance['move_end_pos']
                ambulance['status'] = 'at_accident'
                ambulance['move_start_time'] = None
                ambulance['move_end_time'] = None
                
            elif event['type'] == 'transporting':
                # Start moving to hospital
                ambulance['move_start_time'] = event['timestamp']
                ambulance['move_start_pos'] = ambulance['position']
                ambulance['move_end_pos'] = event['location']
                ambulance['destination'] = event['location']
                ambulance['status'] = 'transporting'
                
                # Find when they arrive at hospital
                for future_event in self.events:
                    if (future_event['ambulance'] == ambulance_id and 
                        future_event['type'] == 'arrived_hospital' and
                        future_event['timestamp'] > event['timestamp']):
                        ambulance['move_end_time'] = future_event['timestamp']
                        break
                
            elif event['type'] == 'arrived_hospital':
                ambulance['position'] = ambulance['move_end_pos']
                ambulance['status'] = 'at_hospital'
                ambulance['call_priority'] = None
                ambulance['call_id'] = None
                ambulance['move_start_time'] = None
                ambulance['move_end_time'] = None
                
            elif event['type'] == 'to_station':
                # Start moving to station
                ambulance['move_start_time'] = event['timestamp']
                ambulance['move_start_pos'] = ambulance['position']
                ambulance['move_end_pos'] = event['location']
                ambulance['destination'] = event['location']
                ambulance['status'] = 'to_station'
                
                # Find when they arrive at station
                for future_event in self.events:
                    if (future_event['ambulance'] == ambulance_id and 
                        future_event['type'] == 'arrived_station' and
                        future_event['timestamp'] > event['timestamp']):
                        ambulance['move_end_time'] = future_event['timestamp']
                        break
                
            elif event['type'] == 'arrived_station':
                ambulance['position'] = ambulance['move_end_pos']
                ambulance['status'] = 'at_station'
                ambulance['destination'] = None
                ambulance['move_start_time'] = None
                ambulance['move_end_time'] = None
    
    def draw_fixed_locations(self):
        """Draw hospitals and stations that stay on the map"""
        
        # Draw hospitals
        for hospital_id, location in self.hospitals.items():
            self.ax.plot(location[1], location[0], 's', markersize=12, 
                        color='lightblue', markeredgecolor='navy', markeredgewidth=2)
            self.ax.annotate(f'H{hospital_id}', (location[1], location[0]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, weight='bold', color='navy')
        
        # Draw stations
        for station_id, location in self.stations.items():
            self.ax.plot(location[1], location[0], '^', markersize=12, 
                        color='orange', markeredgecolor='darkorange', markeredgewidth=2)
            self.ax.annotate(f'S{station_id}', (location[1], location[0]), 
                           xytext=(5, -15), textcoords='offset points', 
                           fontsize=9, weight='bold', color='darkorange')
    
    def draw_active_calls(self, current_time):
        """Draw calls that are currently active (created but not yet picked up)"""
        
        for call_id, call_info in self.calls.items():
            # Show call if it's been created but not yet picked up
            if (call_info['created_time'] <= current_time and 
                (call_info['picked_up_time'] is None or call_info['picked_up_time'] > current_time)):
                
                location = call_info['location']
                priority = call_info['priority']
                color = self.priority_colors[priority]
                
                # Draw call location
                self.ax.plot(location[1], location[0], 'X', markersize=12, 
                            color=color, markeredgewidth=3, markeredgecolor='black')
                self.ax.annotate(f'Call {call_id}\nP{priority}', (location[1], location[0]), 
                               xytext=(-15, 10), textcoords='offset points', 
                               fontsize=8, weight='bold', color='black',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    def update_timeline(self, frame):
        """Update the timeline display"""
        if self.total_frames <= 0:
            return
            
        progress = frame / self.total_frames
        
        # Clear and redraw timeline
        self.timeline_ax.clear()
        self.timeline_ax.set_xlim(0, 1)
        self.timeline_ax.set_ylim(0, 1)
        self.timeline_ax.set_xticks([])
        self.timeline_ax.set_yticks([])
        self.timeline_ax.set_ylabel('Timeline', rotation=0, ha='right')
        
        # Draw timeline bar
        self.timeline_ax.barh(0.5, 1, height=0.4, color='lightgray', alpha=0.5)
        self.timeline_ax.barh(0.5, progress, height=0.4, color='blue', alpha=0.7)
        self.timeline_ax.axvline(progress, color='red', linewidth=2)
        
        # Add time labels
        start_str = self.start_time.strftime('%H:%M')
        end_str = self.end_time.strftime('%H:%M')
        self.timeline_ax.text(0, -0.3, start_str, ha='left', va='top', fontsize=8)
        self.timeline_ax.text(1, -0.3, end_str, ha='right', va='top', fontsize=8)
    
    def animate(self, frame):
        if frame >= self.total_frames:
            frame = self.total_frames - 1
        
        # Get current simulation time
        current_time = self.get_current_simulation_time(frame)
        
        # Clear and redraw main plot
        self.ax.clear()
        self.ax.set_xlim(self.lon_min, self.lon_max)
        self.ax.set_ylim(self.lat_min, self.lat_max)
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Ambulance Dispatch Animation')
        self.ax.grid(True, alpha=0.3)
        
        # Draw fixed locations first
        self.draw_fixed_locations()
        
        # Draw active calls (appear when created, disappear when picked up)
        self.draw_active_calls(current_time)
        
        # Update ambulance states
        self.update_ambulance_states(current_time)
        
        # Draw ambulances
        for amb_id, ambulance in self.ambulances.items():
            current_pos = self.get_ambulance_position_at_time(amb_id, current_time)
            
            if current_pos:
                # Determine color based on call priority
                color = 'blue'
                if ambulance['call_priority'] is not None:
                    color = self.priority_colors[ambulance['call_priority']]
                
                # Draw ambulance
                self.ax.plot(current_pos[1], current_pos[0], 'o', markersize=10, 
                           color=color, markeredgecolor='black', markeredgewidth=2)
                
                # Label ambulance
                label = f"A{amb_id}"
                if ambulance['call_id']:
                    label += f"\nC{ambulance['call_id']}"
                
                self.ax.annotate(label, (current_pos[1], current_pos[0]), 
                               xytext=(8, 8), textcoords='offset points', 
                               fontsize=9, weight='bold', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Draw destination line if moving
                if (ambulance['destination'] and 
                    ambulance['move_start_time'] and ambulance['move_end_time'] and
                    ambulance['move_start_time'] <= current_time < ambulance['move_end_time']):
                    
                    dest = ambulance['destination']
                    self.ax.plot([current_pos[1], dest[1]], 
                               [current_pos[0], dest[0]], 
                               '--', color='blue', alpha=0.7, linewidth=2)
        
        # Update clock
        self.clock_text.set_text(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update timeline
        self.update_timeline(frame)
        
        # Recreate legend
        self.create_legend()
    
    def run_animation(self):
        # Calculate interval for smooth animation
        interval = 1000 // 20  # 20 FPS for smooth but visible animation
        
        anim = animation.FuncAnimation(
            self.fig, self.animate, frames=self.total_frames,
            interval=interval, repeat=True, blit=False
        )
        plt.tight_layout()
        plt.show()
        return anim

# Usage
if __name__ == "__main__":
    # Save your data to a file called 'ambulance_data.txt'
    animator = AmbulanceAnimation('data\sample log.txt')
    anim = animator.run_animation()