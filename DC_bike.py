import pandas as pd
import glob


def read(data_path, number, fold_factor):
    # 读取数据
    # df = pd.read_csv(data_path)
    data_path = glob.glob(data_path)

    dfs = []
    for path in data_path:
        df = pd.read_csv(path, usecols=['Start station number', 'Start date'])
        dfs.append(df)

    # 合并所有 DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # 只保留需要的两列
    df = df[['Start station number', 'Start date']]

    # 转换时间格式
    df['Start date'] = pd.to_datetime(df['Start date'])

    # 提取日期（按天）
    df['date'] = df['Start date'].dt.date  # 或 .dt.floor('D')

    # 分组计数
    grouped = df.groupby(['Start station number', 'date']).size().reset_index(name='ride_count')

    # 生成透视表（行是站点编号，列是日期，值是骑行数量）
    pivot = grouped.pivot(index='Start station number', columns='date', values='ride_count')

    # 填补缺失值为 0，并转换为整数
    pivot = pivot.fillna(0).astype(int)

    # 转换为 NumPy 数组
    result_array = pivot.values

    # 计算每行的和
    row_sums = result_array.sum(axis=1)

    # 使用布尔索引过滤掉行和小于 100 的行
    filtered_matrix = result_array[row_sums >= number]

    num_rows, num_cols = filtered_matrix.shape

    # 保证列数能整除 fold_factor
    usable_cols = (num_cols // fold_factor) * fold_factor
    matrix_trimmed = filtered_matrix[:, :usable_cols]

    # reshape 和变换
    reshaped = matrix_trimmed.reshape(num_rows, -1, fold_factor)  # -> (num_rows, new_cols, fold_factor)
    reshaped = reshaped.transpose(0, 2, 1)  # -> (num_rows, fold_factor, new_cols)
    result = reshaped.reshape(num_rows * fold_factor, -1)  # -> (num_rows * fold_factor, new_cols)

    return result