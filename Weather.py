import pickle
import time
from meteostat import Point, Stations, Hourly
from datetime import datetime, timedelta
import pandas as pd


def read(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df = df.fillna(0)
    data = df.to_numpy()

    # truncating to 8750 columns (divisible by 50)
    data = data[:, :8750]  # shape: (50, 8750)

    # reshape to (50, 50, 175)
    reshaped_data = data.reshape(50, 50, -1)
    return reshaped_data


def get_data():
    cities = [
        {'name': 'Beijing', 'lat': 39.9042, 'lon': 116.4074},
        {'name': 'Shanghai', 'lat': 31.2304, 'lon': 121.4737},
        {'name': 'Guangzhou', 'lat': 23.1291, 'lon': 113.2644},
        {'name': 'Shenzhen', 'lat': 22.5431, 'lon': 114.0579},
        {'name': 'Chengdu', 'lat': 30.5728, 'lon': 104.0668},
        {'name': 'Hangzhou', 'lat': 30.2741, 'lon': 120.1551},
        {'name': 'Wuhan', 'lat': 30.5928, 'lon': 114.3055},
        {'name': 'Xi\'an', 'lat': 34.3416, 'lon': 108.9398},
        {'name': 'Chongqing', 'lat': 29.5630, 'lon': 106.5516},
        {'name': 'Tianjin', 'lat': 39.3434, 'lon': 117.3616},
        {'name': 'Nanjing', 'lat': 32.0603, 'lon': 118.7969},
        {'name': 'Suzhou', 'lat': 31.2989, 'lon': 120.5853},
        {'name': 'Qingdao', 'lat': 36.0671, 'lon': 120.3826},
        {'name': 'Dalian', 'lat': 38.9140, 'lon': 121.6147},
        {'name': 'Shenyang', 'lat': 41.8057, 'lon': 123.4315},
        {'name': 'Harbin', 'lat': 45.8038, 'lon': 126.5349},
        {'name': 'Jinan', 'lat': 36.6512, 'lon': 117.1201},
        {'name': 'Fuzhou', 'lat': 26.0745, 'lon': 119.2965},
        {'name': 'Zhengzhou', 'lat': 34.7466, 'lon': 113.6254},
        {'name': 'Changsha', 'lat': 28.2282, 'lon': 112.9388},
        {'name': 'Kunming', 'lat': 25.0389, 'lon': 102.7183},
        {'name': 'Hefei', 'lat': 31.8206, 'lon': 117.2272},
        {'name': 'Nanchang', 'lat': 28.6820, 'lon': 115.8582},
        {'name': 'Xiamen', 'lat': 24.4798, 'lon': 118.0894},
        {'name': 'Urumqi', 'lat': 43.8256, 'lon': 87.6168},
        {'name': 'Lanzhou', 'lat': 36.0611, 'lon': 103.8343},
        {'name': 'Hohhot', 'lat': 40.8426, 'lon': 111.7492},
        {'name': 'Yinchuan', 'lat': 38.4872, 'lon': 106.2309},
        {'name': 'Taiyuan', 'lat': 37.8706, 'lon': 112.5489},
        {'name': 'Guiyang', 'lat': 26.6477, 'lon': 106.6302},
        {'name': 'Nanning', 'lat': 22.8170, 'lon': 108.3669},
        {'name': 'Haikou', 'lat': 20.0440, 'lon': 110.1999},
        {'name': 'Lhasa', 'lat': 29.6520, 'lon': 91.1721},
        {'name': 'Macau', 'lat': 22.1987, 'lon': 113.5439},
        {'name': 'Hong Kong', 'lat': 22.3193, 'lon': 114.1694},
        {'name': 'Sanya', 'lat': 18.2528, 'lon': 109.5119},
        {'name': 'Zhuhai', 'lat': 22.2760, 'lon': 113.5675},
        {'name': 'Wuxi', 'lat': 31.5747, 'lon': 120.2960},
        {'name': 'Tangshan', 'lat': 39.6309, 'lon': 118.1802},
        {'name': 'Weifang', 'lat': 36.7069, 'lon': 119.1618},
        {'name': 'Changchun', 'lat': 43.8171, 'lon': 125.3235},
        {'name': 'Baotou', 'lat': 40.6574, 'lon': 109.8403},
        {'name': 'Xining', 'lat': 36.6171, 'lon': 101.7782},
        {'name': 'Linyi', 'lat': 35.1047, 'lon': 118.3564},
        {'name': 'Yantai', 'lat': 37.4638, 'lon': 121.4479},
        {'name': 'Yangzhou', 'lat': 32.3936, 'lon': 119.4127},
        {'name': 'Luoyang', 'lat': 34.6197, 'lon': 112.4540},
        {'name': 'Datong', 'lat': 40.0900, 'lon': 113.2910},
        {'name': 'Jilin', 'lat': 43.8378, 'lon': 126.5496},
        {'name': 'Zibo', 'lat': 36.8131, 'lon': 118.0549}
    ]

    result = {}
    for city in cities:
        print(f"\n=== Fetching city now: {city['name']} ===")
        data = weather_info_obtain(
            longitude=city['lon'],
            latitude=city['lat'],
            time1=['2024-05-01 00:00:00'],
            timezone='UTC'
        )

        if not data.empty and 'temp' in data.columns:
            result[city['name']] = data['temp']
        else:
            print(f"City {city['name']} has no tempareture data.")
            result[city['name']] = None
    # convert dict to DataFrame (columns are cities, rows are time)
    temp_df = pd.DataFrame(result)

    # transpose to get shape (50, T)
    temp_matrix = temp_df.T
    temp_matrix.to_csv('city_hourly_temperature.csv', index=True)# 即 temp_df.transpose()
    return temp_matrix

def weather_info_obtain(longitude, latitude, time1, timezone='UTC', max_retries=5):
    """
    Given GPS coordinates and a start time, return hourly weather data for the past year.

    Parameters:
        longitude: Longitude (float)
        latitude: Latitude (float)
        time1: List of start time strings in ['YYYY-MM-DD HH:MM:SS'] format
        timezone: Timezone (default 'UTC')
        max_retries: Maximum number of retries on failure

    Returns:
        DataFrame: Contains hourly weather data
    """
    start_time = datetime.strptime(time1[0], '%Y-%m-%d %H:%M:%S')
    end_time = start_time + timedelta(days=365)

    # defined by latitude and longitude
    location = Point(latitude, longitude)

    # search weather stations nearby
    stations = Stations().nearby(latitude, longitude).fetch(10)

    for station_id in stations.index:
        for attempt in range(max_retries + 1):
            print(f"Try {attempt + 1} times: request station {station_id} data, from {start_time} to {end_time}...")

            try:
                data_hourly = Hourly(station_id, start_time, end_time, timezone=timezone)
                data = data_hourly.fetch()

                if not data.empty:
                    print(f"Successfully fetch data from {station_id} !")
                    return data

                else:
                    print(f"Station {station_id} has no data, try next stattion in queue.")
                    break

            except Exception as e:
                print(f"Error when fetching data: {e}")
                time.sleep(2 * (attempt + 1))  # exponential backoff waiting time
    print("Did not get any data from all stations.")
    return pd.DataFrame()
