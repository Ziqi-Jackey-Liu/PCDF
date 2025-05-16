import pandas as pd
import glob
import os
import numpy as np


def read(data_path, number):
    # 匹配所有 .parquet 文件
    parquet_files = glob.glob(os.path.join(data_path, '*.parquet'))

    # 读取并合并所有 parquet 文件
    df_list = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(df_list, ignore_index=True)

    # 将 pickup_datetime 转换为 datetime 类型
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    # 提取日期和小时
    df['pickup_date'] = df['pickup_datetime'].dt.date  # 提取日期
    df['pickup_hour'] = df['pickup_datetime'].dt.hour  # 提取小时

    # 计算每天的每小时、每个 PULocationID 的客流量
    pu_traffic = df.groupby(['pickup_date', 'pickup_hour', 'PULocationID']).size().reset_index(name='passenger_count')

    # 计算每天的每小时、每个 DOLocationID 的客流量
    do_traffic = df.groupby(['pickup_date', 'pickup_hour', 'DOLocationID']).size().reset_index(name='passenger_count')

    # 合并日期和小时
    pu_traffic['date_hour'] = pu_traffic['pickup_date'].astype(str) + '_' + pu_traffic['pickup_hour'].astype(
        str).str.zfill(2)

    # 创建透视表
    pu_traffic_pivot = pu_traffic.pivot_table(
        index='PULocationID',  # 纵轴：PULocationID
        columns='date_hour',  # 横轴：日期_小时
        values='passenger_count',  # 表内容：客流量
        fill_value=0  # 填充缺失值为 0
    )

    # 转换为 NumPy 数组
    numpy_array = pu_traffic_pivot.values[:, 4:]

    segments = []
    for row in numpy_array:
        split = row.reshape(16, 321)  # 每行变成 (16, 321)
        segments.append(split)

    # 合并所有段：262 × 16 = 4192 行
    reshaped_array = np.vstack(segments)

    # 计算每行的和
    row_sums = reshaped_array.sum(axis=1)

    # 使用布尔索引过滤掉行和小于 100 的行
    filtered_matrix = reshaped_array[row_sums >= number]

    return filtered_matrix
