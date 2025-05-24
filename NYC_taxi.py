import pandas as pd
import glob
import os
import numpy as np


def read(data_path, number):
    # match all .parquet files
    parquet_files = glob.glob(os.path.join(data_path, '*.parquet'))

    # read and combine all parquet files
    df_list = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(df_list, ignore_index=True)

    # convert pickup_datetime to datetime format
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    # extrac date and hour
    df['pickup_date'] = df['pickup_datetime'].dt.date  # extract date
    df['pickup_hour'] = df['pickup_datetime'].dt.hour  # extrac hour

    # calculate the passenger flow of each PULocationID every day and hour
    pu_traffic = df.groupby(['pickup_date', 'pickup_hour', 'PULocationID']).size().reset_index(name='passenger_count')

    # calculate the passenger flow of each DOLocationID every day and hour
    do_traffic = df.groupby(['pickup_date', 'pickup_hour', 'DOLocationID']).size().reset_index(name='passenger_count')

    # combine date and hour
    pu_traffic['date_hour'] = pu_traffic['pickup_date'].astype(str) + '_' + pu_traffic['pickup_hour'].astype(
        str).str.zfill(2)

    # create pivot table
    pu_traffic_pivot = pu_traffic.pivot_table(
        index='PULocationID',  # y-axis：PULocationID
        columns='date_hour',  # x-axis: date_hour
        values='passenger_count',  # value: passenger_count
        fill_value=0  # filling na 0
    )

    # convert NumPy array
    numpy_array = pu_traffic_pivot.values[:, 4:]

    segments = []
    for row in numpy_array:
        split = row.reshape(16, 321)  # each row is reshaped to (16, 321)
        segments.append(split)

    # combine all segments: 262 × 16 = 4192 rows
    reshaped_array = np.vstack(segments)

    # calculate the sum of each row
    row_sums = reshaped_array.sum(axis=1)

    # use boolean indexing to filter out rows less than 100
    filtered_matrix = reshaped_array[row_sums >= number]

    return filtered_matrix
