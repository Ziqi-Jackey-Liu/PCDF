import numpy as np
from scipy.stats import pearsonr
import torch
import matplotlib.pyplot as plt


def similar_test(data_new_12, period):
    """
    Calculate the Pearson correlation matrix of the data, considering only the fully divisible parts.

    :param data_new_12: Input data (1D array or list)
    :param period: Length of the segmentation period
    :return: Pearson correlation matrix
    """
    # Ensure data_new_12 is a NumPy array
    data_new_12 = np.array(data_new_12)

    # calculate the length that can be fully divided
    valid_length = (len(data_new_12) // period) * period

    # truncate the data to the valid length
    data_new_12 = data_new_12[:valid_length]

    # split the data into subarrays of length period
    cycles_new_12 = [data_new_12[i:i + period] for i in range(0, valid_length, period)]

    # initialize the correlation matrix
    num_cycles = len(cycles_new_12)
    results_new_12 = np.zeros((num_cycles, num_cycles))

    # calculate the Pearson correlation coefficient
    for i in range(num_cycles):
        for j in range(num_cycles):
            if i != j:
                results_new_12[i, j], _ = pearsonr(cycles_new_12[i], cycles_new_12[j])
            else:
                results_new_12[i, j] = 1.0  # self-correlation set to 1.0

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
    Perform Gram-Schmidt orthogonalization on matrix X to make its column vectors orthogonal.

    Parameters:
    - X: Input matrix (rows, cols)

    Returns:
    - Orthogonalized matrix
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
    Perform Gram-Schmidt orthogonalization on matrix X to make its row vectors orthogonal.

    Parameters:
    - X: Input matrix (rows, cols)

    Returns:
    - Orthogonalized matrix
    """
    Q = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]): # modify to process by row
        Q[i, :] = X[i, :]
        for j in range(i):
            Q[i, :] -= np.dot(Q[j, :], X[i, :]) / np.dot(Q[j, :], Q[j, :]) * Q[j, :]
        Q[i, :] /= np.linalg.norm(Q[i, :])
    return Q


def generate_matrix_and_unit_vector(rows, cols):
    # generate random metrix with random integers
    random_matrix = np.random.randint(-10, 10, (rows, cols)).astype(float)

    # use QR decomposition to get orthogonal matrix
    orthogonal_matrix = gram_schmidt_line(random_matrix)
    ortho_sum = np.matmul(orthogonal_matrix[0], orthogonal_matrix[2].T)

    # generate random matrix with elements in [0,1)
    matrix = np.random.randint(1, 5, (rows, cols))

    # generate random unit vector with random direction
    random_vector = np.random.rand(rows)
    vector_norm = np.linalg.norm(random_vector)

    # normalize the vector to ensure it is a unit vector
    unit_vector = random_vector / vector_norm if vector_norm != 0 else random_vector

    unit_sum = np.sum(np.square(unit_vector))

    return matrix, unit_vector, orthogonal_matrix

def data_test(data, row_index):
    if row_index is None:
        row_index = 119  # row 120 the index is 119
    wave = data[row_index]

    # Plot the waveform (line chart)
    plt.figure(figsize=(16, 6))
    plt.plot(wave, linewidth=2)  # don't add marker, more smooth
    plt.title(f'Waveform of Row {row_index + 1}')
    plt.xlabel('Time Step')  # Column Index
    plt.ylabel('Amplitude')  # modify by y data
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 1
