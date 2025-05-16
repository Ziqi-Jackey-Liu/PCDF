import pandas as pd
import glob
import os

def read(data_path, number, split_factor):
    # 使用 glob + os 过滤文件名中包含 "actual"（不区分大小写）的 CSV 文件
    all_files = glob.glob(os.path.join(data_path, '*.csv'))
    actual_files = [f for f in all_files if 'actual' in os.path.basename(f).lower()]

    # 读取所有 actual 文件到一个大 DataFrame
    df_list = [pd.read_csv(f) for f in actual_files]
    df = pd.concat(df_list, axis=1, ignore_index=True)

    resampled_df = pd.DataFrame()
    # 假设每两列为一组（0+1, 2+3, 4+5,...）
    for i in range(0, df.shape[1], 2):
        # 取时间列和数值列
        time_col = pd.to_datetime(df[i], format='%m/%d/%y %H:%M')
        value_col = df[i + 1]

        # 创建一个临时 DataFrame
        temp = pd.DataFrame({'timestamp': time_col, 'value': value_col})
        temp = temp.set_index('timestamp')

        # 按小时重采样并求和
        resampled = temp.resample('h').sum()

        # 筛选每天 06:00 到 17:59 的数据
        resampled = resampled.between_time('06:00', '17:59')

        # 重命名列，防止重复
        resampled.columns = [f'series_{i // 2 + 1}']

        # 合并到总 DataFrame 中
        resampled_df = pd.concat([resampled_df, resampled], axis=1)

    # 转置（行变列，列变行）并转换为 numpy 数组
    result_array = resampled_df.T.values

    # 获取原始维度
    rows, cols = result_array.shape

    # 计算可保留的整除部分列数
    valid_cols = (cols // split_factor) * split_factor

    # 丢弃多余的列
    data_trimmed = result_array[:, :valid_cols]

    # 计算新维度
    new_rows = rows * split_factor
    new_cols = cols // split_factor

    # 执行 reshape
    reshaped_data_param = data_trimmed.reshape(new_rows, new_cols)
    return reshaped_data_param
