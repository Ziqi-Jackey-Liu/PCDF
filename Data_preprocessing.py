import numpy as np


def process_trip_dict(data, min_size=20):
    grouped = data.groupby('trip_id')[['latitude', 'longitude', 'timestamp']]
    trip_dict = {trip_id: group[['latitude', 'longitude', 'timestamp']].values.tolist() for trip_id, group in grouped}

    new_trip_dict = {}
    new_trip_id = 1

    for trip_id, coords_list in trip_dict.items():
        # 如果元素少于 min_size，直接跳过
        if len(coords_list) < min_size:
            continue

        # 将每个 trip_id 的列表分组处理
        while len(coords_list) >= min_size:
            # 前 min_size 个作为一组，添加到结果中
            new_trip_dict[new_trip_id] = coords_list[:min_size]
            coords_list = coords_list[min_size:]  # 截断剩余部分
            new_trip_id += 1  # 更新 trip_id

            # 如果剩余部分不足 min_size，直接舍弃
            if len(coords_list) < min_size:
                break

    rows = []
    for coords_list in new_trip_dict.values():
        rows.append(np.array(coords_list).T)

    # 转换为 NumPy 数组
    result_array = np.array(rows, dtype=object)
    return result_array


def normalized(vector):
    mean_value = np.mean(vector)
    normalized_vector = vector - mean_value
    range_value = np.max(normalized_vector) - np.min(normalized_vector)

    if range_value == 0:
        range_value = 1

    result = 2 * normalized_vector / range_value
    return result


def test(period, cycles, multiplier):
    # 创建一个周期向量
    base_pattern = np.arange(1, period + 1)  # 从1开始的周期

    # 根据周期数和倍数生成完整向量
    vector = np.concatenate([base_pattern * (multiplier ** i) for i in range(cycles)])
    return vector
