import pandas as pd
import glob
import os

def read(data_path, number, split_factor):
    # use glob+os to filter the csv file name that contains "actual" (case insensitive)
    all_files = glob.glob(os.path.join(data_path, '*.csv'))
    actual_files = [f for f in all_files if 'actual' in os.path.basename(f).lower()]

    # read all actual files into a big DataFrame
    df_list = [pd.read_csv(f) for f in actual_files]
    df = pd.concat(df_list, axis=1, ignore_index=True)

    resampled_df = pd.DataFrame()
    # assume every two columns are a group (0+1, 2+3, 4+5,...)
    for i in range(0, df.shape[1], 2):
        # take time column and value column
        time_col = pd.to_datetime(df[i], format='%m/%d/%y %H:%M')
        value_col = df[i + 1]

        # create a temporary DataFrame
        temp = pd.DataFrame({'timestamp': time_col, 'value': value_col})
        temp = temp.set_index('timestamp')

        # resample and sum by hour
        resampled = temp.resample('h').sum()

        # select data between 06:00 and 17:59
        resampled = resampled.between_time('06:00', '17:59')

        # rename columns to avoid duplicates
        resampled.columns = [f'series_{i // 2 + 1}']

        # combine into the big DataFrame
        resampled_df = pd.concat([resampled_df, resampled], axis=1)

    # transpose and covert to numpy array
    result_array = resampled_df.T.values

    # obtain original dimensions
    rows, cols = result_array.shape

    # calculate the number of columns that can be kept
    valid_cols = (cols // split_factor) * split_factor

    # drop the extra columns
    data_trimmed = result_array[:, :valid_cols]

    # recalculate new dimensions
    new_rows = rows * split_factor
    new_cols = cols // split_factor

    # reshape
    reshaped_data_param = data_trimmed.reshape(new_rows, new_cols)
    return reshaped_data_param
