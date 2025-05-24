import torch.nn as nn
import numpy as np
import Data_preprocessing
import torch


class CyclicVectorGenerator:
    def __init__(self, factor):
        """
        # Initialize the vector generator.
        # :param factor: The ratio factor between elements in each cycle.
        """
        self.factor = factor

    @staticmethod
    def generate_periodic_vector(period, num_vectors):
        """
        # Generate a single cycle vector with elements of random sizes, normalized to 1.
        """
        # periodic_vector = torch.randn(period)  # Elements within a single cycle are of random sizes
        # periodic_vector /= periodic_vector.sum()  # Normalize to ensure the sum is 1

        random_matrix = torch.randn(period, num_vectors)
        orthogonal_vectors, _ = torch.linalg.qr(random_matrix, mode='reduced')
        return orthogonal_vectors.T

    def generate_final_vector(self, length, period, num_vectors):
        """
        # Generate a vector of the target length, where the element values in each cycle are smaller than those in the previous cycle by a factor.
        """
        periodic_vector = self.generate_periodic_vector(period, num_vectors)
        vectors = []

        # Cycles are concatenated in a loop, decreasing proportionally
        current_vector = periodic_vector.clone()
        total_length = 0
        while total_length < length:
            vectors.append(current_vector)
            total_length += period
            current_vector = current_vector * self.factor  # Scale down by a factor of m

        # Concatenate all complete cycles
        final_vector = torch.cat(vectors, dim=1)[:, :length]  # Trim to the target length

        # # Normalize to ensure the sum of all elements is 1
        # final_vector /= final_vector.sum(dim=1, keepdim=True)
        # final_vector = final_vector - final_vector.mean(dim=1, keepdim=True)
        return periodic_vector, final_vector


class CyclicVectorGeneratorD(nn.Module):
    def __init__(self):
        super(CyclicVectorGeneratorD, self).__init__()

    @staticmethod
    def generate_circular_periodic_vector(period):
        """
        # Generate an orthogonal circulant matrix through two FFTs.
        """
        # Generate a random vector
        random_vector = torch.randn(period, dtype=torch.complex64)

        # First FFT
        random_vector_fft = torch.fft.fft(random_vector)

        # Normalize the vector to have a magnitude of 1
        random_vector_fft_normalized = random_vector_fft / torch.abs(random_vector_fft)

        # Inverse FFT to obtain the first column of the orthogonal circulant matrix
        c_prime = torch.fft.ifft(random_vector_fft_normalized).real

        # Create an orthogonal circulant matrix (each row is a cyclic vector)
        cyclic_matrix = torch.stack([torch.roll(c_prime, i) for i in range(period)])

        # Transpose the matrix so that each row is an orthogonal vector
        # cyclic_matrix = (cyclic_matrix - cyclic_matrix.min()) / (cyclic_matrix.max() - cyclic_matrix.min())
        return cyclic_matrix.T

    @staticmethod
    def generate_non_periodic_vector(period, num_vectors=1):
        return torch.randn(period, period)

    def forward(self, period, mode):
        if mode == 'periodic':
            periodic_vector = self.generate_circular_periodic_vector(period)
        else:
            periodic_vector = self.generate_non_periodic_vector(period)
        return periodic_vector


class CircularConvolution1(nn.Module):
    def __init__(self):
        super(CircularConvolution1, self).__init__()
        self.none = None

    def forward(self, key, input_x, factor):
        none = self.none
        n = input_x.size(0)  # Length of the input vector
        m = key.size(0)      # Length of the convolution kernel

        # Initialize the result tensor with the same shape as input_x
        result = torch.zeros_like(input_x)
        key_values = []

        # Perform circular convolution by iterating over each element position
        for j in range(n):
            temp_key = []
            for k in range(m):  # input_x.size(0)
                result[j] += input_x[k] * key[(j - k) % m]
                temp_key.append(key[(j - k) % m].item())
            # key = torch.multiply(key, factor)  # Amplify by a factor of k after each convolution
            key_values.append(temp_key)
        return result, torch.tensor(key_values)


class CircularConvolution(nn.Module):
    def __init__(self):
        super(CircularConvolution, self).__init__()
        self.none = None

    def forward(self, key, input_x, factor):
        none = self.none
        n = input_x.size(0)  # Length of the input vector
        m = key.size(0)      # Length of the convolution kernel

        # Initialize the result tensor with the same shape as input_x
        result = torch.zeros_like(input_x)
        key_values = []

        # Perform circular convolution by iterating through each element position
        for j in range(n):
            temp_key = []
            for k in range(m):  # Iterate over the elements of the convolution kernel
                result[j] += input_x[(j + k) % n] * key[k]
                temp_key.append(key[k].item())
            key_values.append(temp_key)
            key = torch.multiply(key, factor)  # Amplify by a factor of k after each convolution
        # for k in range(input_x.size(0)):  # input_x.size(0)
        #     result[j] += input_x[k] * key[(j + input_x.size(0) - k) % key.size(0)]
        #     key_index = (j + input_x.size(0) - k) % key.size(0)
        #     temp_key.append(key[key_index].item())
        # key_values.append(temp_key)
        return result, torch.tensor(key_values)


class CircularCorrelation(nn.Module):
    def __init__(self):
        super(CircularCorrelation, self).__init__()
        self.none = None
        self.key_matrix1 = None
        self.key_matrix2 = None

    def forward(self, key, input_x, key_matrix1, input_o1, key_matrix2, input_o2):
        self.key_matrix1 = key_matrix1
        self.key_matrix2 = key_matrix2
        """
        Perform circular convolution between vectors c and x.
        Args:
            key: torch.Tensor, first vector of shape (1, 3n)
            input_x: torch.Tensor, second vector of shape (1, n)
        Returns:
            torch.Tensor: Result of circular convolution of shape (1, n)
        """
        self.none = None
        n = input_x.size(0)  # Length of the input vector
        m = key.size(0)  # Length of the convolution kernel
        key_values = []

        result = torch.zeros_like(input_x)
        for j in range(n):
            temp_key = []
            for k in range(m):
                result[j] += input_x[k] * key[(k - j) % m]
                temp_key.append(key[(k - j) % m].item())
            key_values.append(temp_key)
            key_current = torch.tensor(torch.tensor(temp_key))
            redun1, redun2 = self.coefficient(self.key_matrix1, key_current, input_o1, j)
            redun3, redun4 = self.coefficient(self.key_matrix2, key_current, input_o2, j)
            result[j] = torch.div(result[j] - redun2 - redun4, redun1)
        # for j in range(input_x.size(0)):
        #     for k in range(input_x.size(0)):
        #         result[j] += input_x[k] * key[(k - j + input_x.size(0)) % key.size(0)]
        return result, redun3

    @staticmethod
    def summ(vector, window_size=20):
        squared_sums = torch.conv1d(vector.view(1, 1, -1) ** 2, torch.ones(1, 1, window_size), stride=1).view(-1)
        return torch.flip(squared_sums, dims=[0])

    @staticmethod
    def coefficient(key_matrix, key_current, input_x, index):
        input_x = torch.cat((input_x[:index], input_x[index + 1:]))
        r_1 = torch.mv(key_matrix, key_current)
        r_3 = torch.cat((r_1[:index], r_1[index + 1:]))
        r_2 = torch.dot(r_3.double(), input_x.double())
        return r_1[index], r_2


class PeriodFinder:
    def __init__(self, k):
        """
        Initialize the PeriodFinder class.

        Args:
            k (int): Number of most significant periods to find.
        """
        self.k = k

    def find_periods(self, signal):
        """
        Find the k most significant periods in the time series using FFT.

        Args:
            signal (torch.Tensor): 1D tensor representing the time series data.

        Returns:
            List[float]: The k most significant periods sorted by significance.
        """
        if not isinstance(signal, torch.Tensor):
            raise ValueError("Input time_series must be a torch.Tensor")

        if signal.dim() != 1:
            raise ValueError("Input time_series must be a 1D tensor")

        # Compute the FFT of the time series
        fft_result = torch.fft.fft(signal)

        # Compute the power spectrum (magnitude squared of FFT results)
        power_spectrum = torch.abs(fft_result) ** 2

        # Exclude the zero-frequency component
        power_spectrum[0] = 0

        # Get the frequencies corresponding to FFT components
        n = len(signal)
        frequencies = torch.fft.fftfreq(n)

        # Convert frequencies to periods (1/frequency)
        positive_frequencies = frequencies[frequencies > 0]
        periods = 1 / positive_frequencies

        # Get the corresponding power spectrum for positive frequencies
        positive_power_spectrum = power_spectrum[frequencies > 0]

        # Find the indices of the top k power spectrum values
        top_k_indices = torch.topk(positive_power_spectrum, self.k).indices

        # Retrieve the periods corresponding to the top k indices
        significant_periods = periods[top_k_indices]

        # Sort the periods by their power spectrum value (descending)
        sorted_periods = significant_periods[torch.argsort(positive_power_spectrum[top_k_indices], descending=True)]

        return sorted_periods.tolist()


class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the fully connected network.

        Parameters:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        """
        super(FullyConnectedModel, self).__init__()

        # Define a single fully connected layer
        self.activation = nn.Tanh()
        self.fc = nn.Linear(input_size, output_size)

        # Custom parameter initialization
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')  # Use Kaiming initialization
        nn.init.zeros_(self.fc.bias)  # Initialize the bias to 0

    def forward(self, x):
        """
        Forward propagation logic.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Model output.
        """
        x = self.fc(x)
        x = self.activation(x)
        return x


def calculate_elementwise_lcm(list1, list2):
    """
    Compute the least common multiple (LCM) for each pair of elements between two tensors.
    :param: 1D PyTorch tensor
    :param: 1D PyTorch tensor
    :return: A 2D tensor where each element is the LCM of the corresponding elements in tensor1 and tensor2
    """
    # Ensure both tensors are of integer type
    # Convert lists to PyTorch tensors

    tensor1 = torch.tensor(list1, dtype=torch.long)
    tensor2 = torch.tensor(list2, dtype=torch.long)

    # Use broadcasting to compute the GCD for each pair of elements
    gcd = torch.gcd(tensor1[:, None], tensor2[None, :])

    # Use LCM(a, b) = abs(a * b) // GCD(a, b)
    lcm = torch.div((tensor1[:, None] * tensor2[None, :]).abs(), gcd, rounding_mode='trunc')

    # Find the least common multiple and its index
    min_lcm, flat_index = torch.min(lcm.flatten(), dim=0)  # Flatten the matrix and find the minimum value and its index
    min_i, min_j = divmod(flat_index.item(), lcm.size(1))  # Compute 2D indices

    return list1[min_i], list2[min_j], min_lcm.item()


class AdaptiveEncoding:
    def __init__(self):
        # Create a period finder object (k=2 means finding the top 2 most significant periods)
        self.ad_finder = PeriodFinder(k=3)
        # Create a cyclic vector generator object (factor=2 means the generated vectors will be scaled by a factor of 2)
        self.ad_key = CyclicVectorGenerator(factor=1)
        # Create a circular convolution object and perform circular convolution on the normalized data
        self.ad_circular_conv = CircularConvolution1()

    def encode_process(self, input_x, input_y):
        # Data preprocessing: Normalize the first and second trajectory data in processed_trip_dict and convert them to tensors
        normalized_trip1 = torch.tensor(
            np.array(Data_preprocessing.normalized(input_x), dtype=float))  # 1st trajectory data
        normalized_trip2 = torch.tensor(
            np.array(Data_preprocessing.normalized(input_y), dtype=float))  # 2nd trajectory data

        # Find the main periods of the two trajectories
        top_period1_ = self.ad_finder.find_periods(normalized_trip1)
        top_period2_ = self.ad_finder.find_periods(normalized_trip2)
        top_period1, top_period2, min_gcd = calculate_elementwise_lcm(top_period1_, top_period2_)

        # Compute the least common multiple (LCM) of the two periods and return the corresponding periods and the LCM
        top_period1, top_period2, min_gcd = calculate_elementwise_lcm(top_period1_, top_period2_)

        # Generate two key vectors based on the least common period
        key_vector, key_vector_whole = self.ad_key.generate_final_vector(length=len(normalized_trip1),
                                                                         period=int(min_gcd),
                                                                         num_vectors=2)

        cir_vector1, key_matrix1 = self.ad_circular_conv(key_vector_whole[0, :], normalized_trip1, 1)
        cir_vector2, key_matrix2 = self.ad_circular_conv(key_vector_whole[1, :], normalized_trip2, 1)
        cir_vector3 = cir_vector1 + cir_vector2
        return cir_vector3


class PeriodAdaptation(nn.Module):
    def __init__(self, compress_size):
        super(PeriodAdaptation, self).__init__()
        self.fc1 = nn.Linear(compress_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, data):
        device = self.fc1.weight.device  # Get the device on which this layer is located
        data = data.to(device)  # Move the input to the same device
        x = self.fc1(data)
        mean = self.activation(x)
        variance = self.r_mse(x, mean)
        return mean, variance

    @staticmethod
    def r_mse(x, mean):
        mse_b = torch.mean((x - mean) ** 2, dim=2, keepdim=True)
        x_b = torch.sqrt(mse_b)
        return x_b


class PeriodAdaptationOutput(nn.Module):
    def __init__(self, compress_size):
        super(PeriodAdaptationOutput, self).__init__()
        self.fc1 = nn.Linear(compress_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, data):
        x = self.fc1(data)
        mean = self.activation(x)
        variance = self.r_mse(x, mean)
        return mean, variance

    @staticmethod
    def r_mse(x, mean):
        mse_b = torch.mean((x - mean) ** 2, dim=1, keepdim=True)
        x_b = torch.sqrt(mse_b)
        return x_b


class ElementWiseProductLayer(nn.Module):
    def __init__(self, input_size, out_channel):
        super(ElementWiseProductLayer, self).__init__()
        # Define trainable parameters 'weights', with the same size as the input
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channel, kernel_size=1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)  # 30% ramdom drop
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)
        self.linear2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        # light weight
        x = self.conv1(x)
        x = self.gelu(x)
        # x = self.dropout(x)
        x_in = self.conv2(x)
        x = x + x_in
        x = self.linear2(x)
        with torch.no_grad():
            x_in = self.linear2(x_in)
        return x, x_in
