import pandas as pd
import glob
import os


def read(data_path, number, fold_factor):
    # 1. scan all batch files (*.dat)
    file_list = glob.glob(os.path.join(data_path, "batch*.dat"))

    df_list = []

    for file in file_list:
        # read data
        df = pd.read_csv(file, sep='\s+', header=None)

        # split data by column 0
        split_df = df[0].str.split(';', expand=True)
        split_df.columns = ['gas_type', 'concentration']
        split_df['gas_type'] = split_df['gas_type'].astype(int)
        split_df['concentration'] = split_df['concentration'].astype(float)

        # delete the first column
        df = df.drop(columns=0)

        # remove the prefix n: and convert to float
        value_df = df.apply(lambda col: col.str.split(':', expand=True)[1].astype(float))

        # concatenate sensor data, label, batch column
        batch_name = os.path.basename(file).split('.')[0]  # 'batch1'
        batch_col = pd.Series([batch_name] * len(df), name='batch')
        full_df = pd.concat([split_df, batch_col, value_df], axis=1)

        df_list.append(full_df)

    # combine all batches
    all_data = pd.concat(df_list, ignore_index=True)

    # reorder columns
    meta_cols = ['gas_type', 'concentration', 'batch']
    feature_cols = [col for col in all_data.columns if col not in meta_cols]
    all_data = all_data[feature_cols]

    # original data shape = (13910, 128)
    data = all_data.values if isinstance(all_data, pd.DataFrame) else all_data  # (13910, 128)
    data = data.T  # transpose to (128, 13910)

    # split data into 50 batches
    batch_size = 50
    total_time = data.shape[1]
    time_per_batch = total_time // batch_size  # each batch time length

    # Truncate to make divisible
    truncate_len = batch_size * time_per_batch
    data = data[:, :truncate_len]  # shape: (128, truncate_len)

    # split into batch
    batches = data.reshape(128, batch_size, time_per_batch)  # shape: (128, 50, time)
    batches = batches.transpose(1, 0, 2)  # shape: (50, 128, time)

    return batches
