import pandas as pd
from util import *
from tqdm import tqdm


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
    df = pd.read_csv("./data/911_calls_with_station.csv")

    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    df['date'] = df['timeStamp'].dt.date
    df['hour'] = df['timeStamp'].dt.hour

    group_keys = list(df.groupby(['assigned_station_id', 'date', 'hour']).groups.keys())

    result = []

    for (station_id, date, hour) in tqdm(group_keys, desc="Counting calls per station per hour per day"):
        count = df[(df['assigned_station_id'] == station_id) &
                (df['date'] == date) &
                (df['hour'] == hour)].shape[0]
        result.append({
            'station_id': station_id,
            'date': date,
            'hour': hour,
            'call_count': count
        })

    call_counts = pd.DataFrame(result)

    call_counts.to_csv("./data/station_call_counts_per_hour.csv", index=False)

if __name__ == '__main__':
    get_station_call_counts_per_hour()