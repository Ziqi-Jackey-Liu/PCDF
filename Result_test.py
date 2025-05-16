import numpy as np
from scipy.stats import pearsonr
import torch
import matplotlib.pyplot as plt


def similar_test(data_new_12, period):
    """
    计算数据的皮尔逊相关性矩阵，仅考虑能够完整分割的部分。

    :param data_new_12: 输入数据（1D 数组或列表）
    :param period: 分割周期的长度
    :return: 皮尔逊相关性矩阵
    """
    # 确保 data_new_12 是一个 NumPy 数组
    data_new_12 = np.array(data_new_12)

    # 计算可以完整分割的长度
    valid_length = (len(data_new_12) // period) * period

    # 截取可分割的部分
    data_new_12 = data_new_12[:valid_length]

    # 分割数据为周期长度的子数组
    cycles_new_12 = [data_new_12[i:i + period] for i in range(0, valid_length, period)]

    # 初始化相关性矩阵
    num_cycles = len(cycles_new_12)
    results_new_12 = np.zeros((num_cycles, num_cycles))

    # 计算皮尔逊相关性
    for i in range(num_cycles):
        for j in range(num_cycles):
            if i != j:
                results_new_12[i, j], _ = pearsonr(cycles_new_12[i], cycles_new_12[j])
            else:
                results_new_12[i, j] = 1.0  # 自相关为 1

    return results_new_12


def key_matrix_test(vector1, vector2):
    pseudo_inverse = torch.linalg.pinv(torch.matmul(vector1.unsqueeze(1), vector1.unsqueeze(0)))
    matrix = torch.matmul(torch.matmul(pseudo_inverse, vector1.unsqueeze(1)), vector2.unsqueeze(0))
    return matrix

def compute_m3(m1, m2):
    """
    Compute M3 based on the formula:
    M3 = M1^-1 - M2^\dagger (M2 M1^-1)

    Args:
        m1: torch.Tensor, a square matrix (n x n), assumed to be invertible.
        m2: torch.Tensor, a matrix (n x n), potentially non-invertible.

    Returns:
        M3: torch.Tensor, the computed matrix (n x n).
    """
    # Check if M1 is invertible
    # if torch.det(m1) == 0:
    #     raise ValueError("M1 is not invertible")

    # Compute M1 inverse
    m1_inv = torch.linalg.pinv(m1)

    # Compute M2 pseudo-inverse
    m2_pseudo_inv = torch.linalg.pinv(m2)

    # Compute the correction term: -M2^\dagger (M2 M1^-1)
    correction_term = -torch.matmul(m2_pseudo_inv, torch.matmul(m2, m1_inv))

    # Compute M3: M1^-1 + correction_term
    m3 = m1_inv + correction_term

    return m3

def gram_schmidt_list(X):
    """
    对矩阵 X 进行 Gram-Schmidt 正交化，使其列向量正交。

    参数:
    - X: 输入矩阵 (rows, cols)

    返回:
    - 正交化后的矩阵
    """
    Q = np.zeros_like(X, dtype=float)
    for i in range(X.shape[1]):
        Q[:, i] = X[:, i]
        for j in range(i):
            Q[:, i] -= np.dot(Q[:, j], X[:, i]) / np.dot(Q[:, j], Q[:, j]) * Q[:, j]
        Q[:, i] /= np.linalg.norm(Q[:, i])
    return Q

def gram_schmidt_line(X):
    """
    对矩阵 X 进行 Gram-Schmidt 正交化，使其行向量正交。

    参数:
    - X: 输入矩阵 (rows, cols)

    返回:
    - 正交化后的矩阵
    """
    Q = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):  # 修改为按行处理
        Q[i, :] = X[i, :]
        for j in range(i):
            Q[i, :] -= np.dot(Q[j, :], X[i, :]) / np.dot(Q[j, :], Q[j, :]) * Q[j, :]
        Q[i, :] /= np.linalg.norm(Q[i, :])
    return Q


def generate_matrix_and_unit_vector(rows, cols):
    # 生成随机矩阵，包含随机整数
    random_matrix = np.random.randint(-10, 10, (rows, cols)).astype(float)

    # 使用QR分解获得正交矩阵
    orthogonal_matrix = gram_schmidt_line(random_matrix)
    ortho_sum = np.matmul(orthogonal_matrix[0], orthogonal_matrix[2].T)

    # 生成随机矩阵，元素在 [0,1) 之间
    matrix = np.random.randint(1, 5, (rows, cols))

    # 生成单位向量，方向随机
    random_vector = np.random.rand(rows)
    vector_norm = np.linalg.norm(random_vector)

    # 归一化向量以确保其为单位向量
    unit_vector = random_vector / vector_norm if vector_norm != 0 else random_vector

    unit_sum = np.sum(np.square(unit_vector))

    return matrix, unit_vector, orthogonal_matrix

def data_test(data, row_index):
    if row_index is None:
        row_index = 119  # 第120行 → 索引是119
    wave = data[row_index]

    # 绘制波形图（折线图）
    plt.figure(figsize=(16, 6))
    plt.plot(wave, linewidth=2)  # 不加marker，更光滑
    plt.title(f'Waveform of Row {row_index + 1}')
    plt.xlabel('Time Step')  # 或 Column Index
    plt.ylabel('Amplitude')  # 可根据含义修改
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 1
