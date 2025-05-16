import torch
import numpy as np


def sampling(data, sample_rate=0.5, cycle_size=5):
    # 分组并采样
    num_cycles = (len(data) + cycle_size - 1) // cycle_size  # 周期数
    sampled_indices = []

    for cycle_idx in range(num_cycles):
        # 获取当前周期的数据索引范围
        start_idx = cycle_idx * cycle_size
        end_idx = min(start_idx + cycle_size, len(data))
        cycle_indices = np.arange(start_idx, end_idx)

        # 按采样率随机采样
        num_samples = int(sample_rate * cycle_size)
        sampled = np.random.choice(cycle_indices, size=num_samples, replace=False)

        # 添加到采样结果中
        sampled_indices.extend(sampled)

    # 按时间顺序排序采样的索引
    sampled_indices.sort()

    # 获取采样结果
    sampled_data = data[sampled_indices]
    return sampled_data

def seasonal_trend_decomposition(data, kernel_size = 5, padding_size = 2):
    avg_pool1d = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding_size)
    output_trend = avg_pool1d(data)
    output_seasonal = data - output_trend
    return output_seasonal

def fft_transform(signal):
    fft_result = torch.fft.fft(signal)
    return fft_result

def js_divergence(vector_1, vector_2):
    # 步骤1：归一化处理（转为概率分布）
    vector_1_prob = torch.nn.functional.softmax(vector_1, dim=-1)
    vector_2_prob = torch.nn.functional.softmax(vector_2, dim=-1)
    aligned_vector_2 = torch.nn.functional.interpolate(vector_2_prob.unsqueeze(0), size=vector_1_prob.size(1), mode="linear").squeeze(0)
    """计算两个概率分布的JS散度"""
    p = vector_1_prob
    q = aligned_vector_2
    m = 0.5 * (p + q)  # 平均分布
    kl_p_m = torch.nn.functional.kl_div(m.log(), p, reduction='batchmean', log_target=True)
    kl_q_m = torch.nn.functional.kl_div(m.log(), q, reduction='batchmean', log_target=True)
    return 0.5 * (kl_p_m + kl_q_m)

def lcm(a, b):
    """计算两个数的最小公倍数"""
    return abs(a * b) // torch.gcd(a, b)

def lcm_of_vector(vector):
    """计算向量中所有元素的最小公倍数"""
    result = vector[0]
    for element in vector[1:]:
        result = lcm(result, element)
    return result

def selection_sampling(period_vector, data):
    period_lcm = lcm_of_vector(period_vector)
    sampled_data_comba = []
    for period in period_vector:
        sampled_data = sampling(data, cycle_size=period)
        sampled_data_comba.append(sampled_data)
    return torch.stack(sampled_data_comba), period_lcm

def similarity():
    return 1

def comparison_selection(data, altered_data, period):
    num_cycles = (len(data) + period - 1) // period
    best_altered_data = []
    for cycle_idx in range(num_cycles):
        start_idx = cycle_idx * period
        end_idx = min(start_idx + period, len(data))
        best_js_value = None  # 存储最优的 js_divergence 值
        best_row = None  # 存储对应的 altered_data 行
        for row_idx, row in enumerate(altered_data):
            cycle_indices = js_divergence(data[start_idx:end_idx], row[start_idx:end_idx])
            # 比较最小值或最大值（这里默认寻找最大值）
            if best_js_value is None or cycle_indices > best_js_value:
                best_js_value = cycle_indices
                best_row = row_idx  # 更新最优行数据
        best_altered_data.append(altered_data[best_row, start_idx:end_idx])
    final_vector = torch.cat(best_altered_data)
    return final_vector

def combination(vector1, vector2):
    interleaved_vector = torch.empty(len(vector1) + len(vector2), dtype=vector1.dtype)
    interleaved_vector[0::2] = vector1  # 将 vector1 放在偶数位置
    interleaved_vector[1::2] = vector2  # 将 vector2 放在奇数位置
    return interleaved_vector

def model_key_compress(input_1, input_2):
    output_seasonal_1= seasonal_trend_decomposition(input_1)
    output_seasonal_2 = seasonal_trend_decomposition(input_2)
    fft_result1 = fft_transform(output_seasonal_1)
    fft_result2 = fft_transform(output_seasonal_2)
    sampling_result1, period_lcm1 = selection_sampling(torch.topk(fft_result1, k=3, largest=True, sorted=True), input_1)
    sampling_result2, period_lcm2 = selection_sampling(torch.topk(fft_result2, k=3, largest=True, sorted=True), input_2)
    selected_result1 = comparison_selection(input_1, sampling_result1, period_lcm1)
    selected_result2 = comparison_selection(input_2, sampling_result2, period_lcm2)
    combination_result = combination(selected_result1, selected_result2)
    return combination_result

