import pandas as pd
from util import *
from tqdm import tqdm

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
    calls_df = pd.read_csv("./data/911_calls_no_outliers.csv")
    stations_df = pd.read_csv("./data/stations.csv")

    station_locations = stations_df[['id', 'lat', 'lng']].copy()
    station_locations['coord'] = list(zip(station_locations['lat'], station_locations['lng']))

    def find_nearest_station(call_lat: float, call_lng: float) -> int:
        call_loc = (call_lat, call_lng)
        min_dist = float('inf')
        assigned_id = None
        for _, row in station_locations.iterrows():
            dist = distance(call_loc, row['coord'])
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

def get_station_call_counts_per_hour():
    df = pd.read_csv("./data/911_calls_with_priority.csv")

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

if __name__ == '__main__':
    get_station_call_counts_per_hour()