import pandas as pd


def read(data_path, number, split_factor):
    # 1. 读取数据
    df = pd.read_csv(data_path, sep=';', decimal=',', index_col=0, parse_dates=True)
    df.index.name = 'timestamp'

    # 2. 按小时重采样（汇总每小时用电量）
    df_hourly = df.resample('H').sum()

    # 3. 转置（将 meter_id 移为行索引）
    df_transposed = df_hourly.T
    df_transposed.index.name = 'meter_id'

    # 4. 处理数据（缺失值填 0 + 保留小数）
    pivot = df_transposed.fillna(0).astype(float)

    # 5. 截取部分数据（例如第 173 行到末尾，前 800 个时间点）
    result_array = pivot.values[173:, 2000:10000]

    # 6. 过滤用电量总和大于 number 的电表
    row_sums = result_array.sum(axis=1)
    filtered_matrix = result_array[row_sums > number]

    # 获取原始维度
    rows, cols = filtered_matrix.shape

    # 计算可保留的整除部分列数
    valid_cols = (cols // split_factor) * split_factor

    # 丢弃多余的列
    data_trimmed = filtered_matrix[:, :valid_cols]

    # 计算新维度
    new_rows = rows * split_factor
    new_cols = cols // split_factor

    # 执行 reshape
    reshaped_data_param = data_trimmed.reshape(new_rows, new_cols)
    return reshaped_data_param