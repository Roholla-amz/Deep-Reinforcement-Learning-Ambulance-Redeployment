import pandas as pd
from util import util
from tqdm import tqdm
from shapely.geometry import Point
import numpy as np

def assign_priorities_to_calls():
    calls_df = pd.read_csv("./data/911_calls_no_outliers.csv")

    calls_df['title'] = calls_df['title'].str.replace("EMS:", "").str.strip().str.upper()
    
    priority_map = {
        "CARDIAC EMERGENCY": 2,
        "CVA/STROKE": 2,
        "RESPIRATORY EMERGENCY": 2,
        "HEMORRHAGING": 2,
        "SEIZURES": 1,
        "DIABETIC EMERGENCY": 1,
        "SYNCOPAL EPISODE": 1,
        "HEAD INJURY": 1,
        "ABDOMINAL PAINS": 0,
        "BACK PAINS/INJURY": 0,
        "NAUSEA/VOMITING": 0
    }
    
    calls_df['priority'] = calls_df['title'].map(priority_map).fillna(0).astype(int)
    
    calls_df.to_csv("./data/911_calls_with_priority.csv", index=False)

def assign_the_nearest_station_to_a_call():
    calls_df = pd.read_csv("./data/911_calls_with_priority.csv")
    stations_df = pd.read_csv("./data/stations_from_map.csv")

    station_locations = stations_df[['id', 'lat', 'lng']].copy()
    station_locations['coord'] = list(zip(station_locations['lat'], station_locations['lng']))

    def find_nearest_station(call_lat: float, call_lng: float) -> int:
        call_loc = (call_lat, call_lng)
        min_dist = float('inf')
        assigned_id = None
        for _, row in station_locations.iterrows():
            dist = util.distance(call_loc, row['coord'])
            if dist < min_dist:
                min_dist = dist
                assigned_id = row['id']
        return assigned_id

    tqdm.pandas(desc="Assigning nearest stations")

    calls_df['assigned_station_id'] = calls_df.progress_apply(
        lambda row: find_nearest_station(row['lat'], row['lng']),
        axis=1
    )

    calls_df.to_csv("./data/911_calls_with_station.csv", index=False)

def filter_calls_by_travel_time_threshold(threshold_minutes_by_priority=None):
    """
    Filter 911 calls based on travel time to nearest station exceeding threshold based on priority.
    
    Args:
        threshold_minutes_by_priority (dict): Dictionary mapping priority levels to maximum allowed travel time in minutes.
                                            Default: {0: 15, 1: 10, 2: 8} (lower priority = higher threshold)
    
    Returns:
        pd.DataFrame: Filtered dataframe with calls that meet travel time requirements
    """
    if threshold_minutes_by_priority is None:
        threshold_minutes_by_priority = {0: 15, 1: 10, 2: 8}  # Default thresholds in minutes
    
    # Read the data
    calls_df = pd.read_csv("./data/911_calls_with_station.csv")
    stations_df = pd.read_csv("./data/stations_from_map.csv")
    
    # Create station coordinates mapping
    station_coords = stations_df.set_index('id')[['lat', 'lng']].to_dict('index')
    
    def calculate_travel_time_to_station(row):
        """Calculate travel time from call location to assigned station"""
        call_coord = (row['lat'], row['lng'])
        station_id = row['assigned_station_id']
        
        if pd.isna(station_id) or station_id not in station_coords:
            return float('inf')
        
        station_coord = (station_coords[station_id]['lat'], station_coords[station_id]['lng'])
        travel_time = util.calculate_travel_time(call_coord, station_coord)
        return travel_time.total_seconds() / 60  # Convert to minutes
    
    print("Calculating travel times to assigned stations...")
    calls_df['travel_time_minutes'] = calls_df.apply(
        calculate_travel_time_to_station, axis=1
    )
    
    # Filter calls based on priority thresholds
    print("Filtering calls based on travel time thresholds...")
    filtered_calls = []
    
    for priority, max_time in threshold_minutes_by_priority.items():
        priority_calls = calls_df[calls_df['priority'] == priority]
        valid_calls = priority_calls[priority_calls['travel_time_minutes'] <= max_time]
        filtered_calls.append(valid_calls)
        print(f"Priority {priority}: {len(priority_calls)} total calls, {len(valid_calls)} within {max_time} min threshold")
    
    # Combine all filtered calls
    final_df = pd.concat(filtered_calls, ignore_index=True)
    
    # Save filtered data
    output_file = "./data/911_calls_filtered_by_travel_time.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")
    print(f"Total calls after filtering: {len(final_df)} (removed {len(calls_df) - len(final_df)} calls)")
    
    return final_df

def get_station_call_counts_per_hour():
    df = pd.read_csv("./data/911_calls_filtered_by_travel_time.csv")

    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    df['date'] = df['timeStamp'].dt.date
    df['hour'] = df['timeStamp'].dt.hour

    group_keys = list(df.groupby(['assigned_station_id', 'priority', 'date', 'hour']).groups.keys())

    result = []

    for (station_id, priority, date, hour) in tqdm(group_keys, desc="Counting calls per station per priority per hour per day"):
        count = df[(df['assigned_station_id'] == station_id) &
                   (df['priority'] == priority) &
                   (df['date'] == date) &
                   (df['hour'] == hour)].shape[0]
        result.append({
            'station_id': station_id,
            'priority': priority,
            'date': date,
            'hour': hour,
            'call_count': count
        })

    call_counts = pd.DataFrame(result)

    call_counts.to_csv("./data/station_call_counts_per_priority_per_hour.csv", index=False)

def sort_calls_by_id():
    """
    Read the filtered 911 calls CSV and sort all records by ID.
    
    Returns:
        pd.DataFrame: Sorted dataframe with calls ordered by ID
    """
    print("Reading filtered 911 calls data...")
    calls_df = pd.read_csv("./data/911_calls_filtered_by_travel_time.csv")
    
    print(f"Total records: {len(calls_df)}")
    print(f"Columns: {list(calls_df.columns)}")
    
    # Check if 'id' column exists
    if 'id' not in calls_df.columns:
        print("Warning: 'id' column not found. Available columns:")
        for col in calls_df.columns:
            print(f"  - {col}")
        return calls_df
    
    # Sort by ID
    print("Sorting records by ID...")
    sorted_df = calls_df.sort_values('id').reset_index(drop=True)
    
    # Save sorted data
    output_file = "./data/911_calls_filtered_by_travel_time.csv"
    sorted_df.to_csv(output_file, index=False)
    print(f"Sorted data saved to {output_file}")
    
    # Display first few sorted records
    print("\nFirst 5 records after sorting:")
    print(sorted_df[['id', 'title', 'priority', 'assigned_station_id']].head())
    
    return sorted_df


if __name__ == '__main__':
    # assign_the_nearest_station_to_a_call()
    # filter_calls_by_travel_time_threshold()
    # get_station_call_counts_per_hour()
    sort_calls_by_id()