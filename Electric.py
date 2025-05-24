import pandas as pd


def read(data_path, number, split_factor):
    # 1. read data
    df = pd.read_csv(data_path, sep=';', decimal=',', index_col=0, parse_dates=True)
    df.index.name = 'timestamp'

    ## 2. Resample by hour (aggregate hourly electricity usage)
    df_hourly = df.resample('H').sum()

    # 3. Transpose (move meter_id to row index)
    df_transposed = df_hourly.T
    df_transposed.index.name = 'meter_id'

    # 4 processing data (fill missing values with 0 and keep decimal)
    pivot = df_transposed.fillna(0).astype(float)

    # 5. cut the data (e.g., from row 173 to the end, first 800 time points) 
    result_array = pivot.values[173:, 2000:10000]

    # 6. filter the meters with total consumption greater than number
    row_sums = result_array.sum(axis=1)
    filtered_matrix = result_array[row_sums > number]

    # ontain original dimensions
    rows, cols = filtered_matrix.shape

    # calculate the number of columns that can be retained
    valid_cols = (cols // split_factor) * split_factor

    # drop the extra columns
    data_trimmed = filtered_matrix[:, :valid_cols]

    # calculate new dimensions
    new_rows = rows * split_factor
    new_cols = cols // split_factor

    # perform reshape
    reshaped_data_param = data_trimmed.reshape(new_rows, new_cols)
    return reshaped_data_param