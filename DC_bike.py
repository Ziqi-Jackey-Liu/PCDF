import pandas as pd
import glob


def read(data_path, number, fold_factor):
    # read data
    # df = pd.read_csv(data_path)
    data_path = glob.glob(data_path)

    dfs = []
    for path in data_path:
        df = pd.read_csv(path, usecols=['Start station number', 'Start date'])
        dfs.append(df)

    # merge all DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # keep only the relevant columns
    df = df[['Start station number', 'Start date']]

    # covert time format
    df['Start date'] = pd.to_datetime(df['Start date'])

    # extract date (by day)
    df['date'] = df['Start date'].dt.date  # or .dt.floor('D')

    # Group counting
    grouped = df.groupby(['Start station number', 'date']).size().reset_index(name='ride_count')

    # Generate a pivot table (rows are station IDs, columns are dates, values are ride counts)
    pivot = grouped.pivot(index='Start station number', columns='date', values='ride_count')

    # fill na with 0，covert to int
    pivot = pivot.fillna(0).astype(int)

    # covert to NumPy array
    result_array = pivot.values

    # Compute the sum of each row
    row_sums = result_array.sum(axis=1)

    # Use boolean indexing to filter out rows with a row sum less than 100
    filtered_matrix = result_array[row_sums >= number]

    num_rows, num_cols = filtered_matrix.shape

    # Ensure the number of columns is divisible by fold_factor
    usable_cols = (num_cols // fold_factor) * fold_factor
    matrix_trimmed = filtered_matrix[:, :usable_cols]

    # reshape and transpose
    reshaped = matrix_trimmed.reshape(num_rows, -1, fold_factor)  # -> (num_rows, new_cols, fold_factor)
    reshaped = reshaped.transpose(0, 2, 1)  # -> (num_rows, fold_factor, new_cols)
    result = reshaped.reshape(num_rows * fold_factor, -1)  # -> (num_rows * fold_factor, new_cols)

    return result