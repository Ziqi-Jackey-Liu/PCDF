import pandas as pd
import glob
import os


def read(data_path, number, fold_factor):
    # 1. 扫描所有 batch 文件（*.dat）
    file_list = glob.glob(os.path.join(data_path, "batch*.dat"))

    df_list = []

    for file in file_list:
        # 读取数据
        df = pd.read_csv(file, sep='\s+', header=None)

        # 拆分第0列
        split_df = df[0].str.split(';', expand=True)
        split_df.columns = ['gas_type', 'concentration']
        split_df['gas_type'] = split_df['gas_type'].astype(int)
        split_df['concentration'] = split_df['concentration'].astype(float)

        # 删除原始第0列
        df = df.drop(columns=0)

        # 去除每列的 n: 前缀并转为 float
        value_df = df.apply(lambda col: col.str.split(':', expand=True)[1].astype(float))

        # 拼接 sensor 数据 + 标签 + batch 列
        batch_name = os.path.basename(file).split('.')[0]  # 'batch1' 之类
        batch_col = pd.Series([batch_name] * len(df), name='batch')
        full_df = pd.concat([split_df, batch_col, value_df], axis=1)

        df_list.append(full_df)

    # 合并所有 batch
    all_data = pd.concat(df_list, ignore_index=True)

    # 调整列顺序
    meta_cols = ['gas_type', 'concentration', 'batch']
    feature_cols = [col for col in all_data.columns if col not in meta_cols]
    all_data = all_data[feature_cols]

    # 原始数据 shape = (13910, 128)
    data = all_data.values if isinstance(all_data, pd.DataFrame) else all_data  # (13910, 128)
    data = data.T  # 转为 (128, 13910)

    # 拆分成 50 段（batch 数）
    batch_size = 50
    total_time = data.shape[1]
    time_per_batch = total_time // batch_size  # 每个 batch 的时间长度

    # 截断以整除
    truncate_len = batch_size * time_per_batch
    data = data[:, :truncate_len]  # shape: (128, truncate_len)

    # 分成 batch
    batches = data.reshape(128, batch_size, time_per_batch)  # shape: (128, 50, time)
    batches = batches.transpose(1, 0, 2)  # shape: (50, 128, time)

    return batches
