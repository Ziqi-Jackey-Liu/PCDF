import torch.nn as nn
import numpy as np
import Data_preprocessing
import torch


class CyclicVectorGenerator:
    def __init__(self, factor):
        """
        初始化向量生成器。
        :param factor: 每个周期中元素之间的比例因子。
        """
        self.factor = factor

    @staticmethod
    def generate_periodic_vector(period, num_vectors):
        """
        生成单个周期的向量，元素为随机大小，归一化为1。
        """
        # periodic_vector = torch.randn(period)  # 单周期内元素为随机大小
        # periodic_vector /= periodic_vector.sum()  # 归一化

        random_matrix = torch.randn(period, num_vectors)
        orthogonal_vectors, _ = torch.linalg.qr(random_matrix, mode='reduced')
        return orthogonal_vectors.T

    def generate_final_vector(self, length, period, num_vectors):
        """
        生成目标长度的向量，满足每个周期的元素值比前一个周期小factor倍。
        """
        periodic_vector = self.generate_periodic_vector(period, num_vectors)
        vectors = []

        # 循环拼接周期，按照比例递减
        current_vector = periodic_vector.clone()
        total_length = 0
        while total_length < length:
            vectors.append(current_vector)
            total_length += period
            current_vector = current_vector * self.factor  # 缩小m倍

        # 拼接所有完整周期
        final_vector = torch.cat(vectors, dim=1)[:, :length]  # 截取到目标长度

        # # 归一化以确保所有元素和为1
        # final_vector /= final_vector.sum(dim=1, keepdim=True)
        # final_vector = final_vector - final_vector.mean(dim=1, keepdim=True)
        return periodic_vector, final_vector


class CyclicVectorGeneratorD(nn.Module):
    def __init__(self):
        super(CyclicVectorGeneratorD, self).__init__()

    @staticmethod
    def generate_circular_periodic_vector(period):
        """
        通过两次 FFT 生成正交的循环矩阵。
        """
        # 生成随机向量
        random_vector = torch.randn(period, dtype=torch.complex64)

        # 第一次 FFT
        random_vector_fft = torch.fft.fft(random_vector)

        # 调整模长为 1
        random_vector_fft_normalized = random_vector_fft / torch.abs(random_vector_fft)

        # 反向 FFT 得到正交循环矩阵的第一列
        c_prime = torch.fft.ifft(random_vector_fft_normalized).real

        # 创建正交的循环矩阵（每一行都是一个周期向量）
        cyclic_matrix = torch.stack([torch.roll(c_prime, i) for i in range(period)])

        # 转置矩阵，使每一行是一个正交向量
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
        n = input_x.size(0)  # 输入向量的长度
        m = key.size(0)      # 卷积核的长度

        # 初始化结果张量，形状与 input_x 相同
        result = torch.zeros_like(input_x)
        key_values = []

        # 遍历每个元素位置进行循环卷积
        for j in range(n):
            temp_key = []
            for k in range(m):  # input_x.size(0)
                result[j] += input_x[k] * key[(j - k) % m]
                temp_key.append(key[(j - k) % m].item())
            # key = torch.multiply(key, factor)  # 每次卷积后放大 k 倍
            key_values.append(temp_key)
        return result, torch.tensor(key_values)


class CircularConvolution(nn.Module):
    def __init__(self):
        super(CircularConvolution, self).__init__()
        self.none = None

    def forward(self, key, input_x, factor):
        none = self.none
        n = input_x.size(0)  # 输入向量的长度
        m = key.size(0)      # 卷积核的长度

        # 初始化结果张量，形状与 input_x 相同
        result = torch.zeros_like(input_x)
        key_values = []

        # 遍历每个元素位置进行循环卷积
        for j in range(n):
            temp_key = []
            for k in range(m):  # 遍历卷积核的元素
                result[j] += input_x[(j + k) % n] * key[k]
                temp_key.append(key[k].item())
            key_values.append(temp_key)
            key = torch.multiply(key, factor)  # 每次卷积后放大 k 倍
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
        n = input_x.size(0)  # 输入向量的长度
        m = key.size(0)  # 卷积核的长度
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
        初始化全连接网络。

        参数：
        input_size (int): 输入层的大小。
        output_size (int): 输出层的大小。
        """
        super(FullyConnectedModel, self).__init__()

        # 定义单层全连接
        self.activation = nn.Tanh()
        self.fc = nn.Linear(input_size, output_size)

        # 自定义参数初始化
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')  # 使用Kaiming初始化
        nn.init.zeros_(self.fc.bias)  # 将偏置初始化为0

    def forward(self, x):
        """
        前向传播逻辑。

        参数：
        x (torch.Tensor): 输入张量。

        返回：
        torch.Tensor: 模型输出。
        """
        x = self.fc(x)
        x = self.activation(x)
        return x


def calculate_elementwise_lcm(list1, list2):
    """
            计算两个张量之间每对元素的最小公倍数
            :param: 1D PyTorch 张量
            :param: 1D PyTorch 张量
            :return: 一个 2D 张量，其中每个元素是 tensor1 和 tensor2 中对应元素的最小公倍数
            """
    # 确保两个张量是整型
    # 将列表转换为 PyTorch 张量
    tensor1 = torch.tensor(list1, dtype=torch.long)
    tensor2 = torch.tensor(list2, dtype=torch.long)

    # 使用广播机制计算每对元素的 GCD
    gcd = torch.gcd(tensor1[:, None], tensor2[None, :])

    # 使用公式 LCM(a, b) = abs(a * b) // GCD(a, b)
    lcm = torch.div((tensor1[:, None] * tensor2[None, :]).abs(), gcd, rounding_mode='trunc')

    # 找到最小公倍数及其索引
    min_lcm, flat_index = torch.min(lcm.flatten(), dim=0)  # 展平矩阵并找到最小值及其索引
    min_i, min_j = divmod(flat_index.item(), lcm.size(1))  # 计算二维索引

    return list1[min_i], list2[min_j], min_lcm.item()


class AdaptiveEncoding:
    def __init__(self):
        # 创建周期查找器对象 (k=2 表示寻找前 2 个最重要的周期)
        self.ad_finder = PeriodFinder(k=3)
        # 创建循环向量生成器对象 (factor=2 表示生成的向量会按因子 2 缩放)
        self.ad_key = CyclicVectorGenerator(factor=1)
        # 创建循环卷积对象并对归一化数据进行循环卷积
        self.ad_circular_conv = CircularConvolution1()

    def encode_process(self, input_x, input_y):
        # 数据预处理：将 processed_trip_dict 中的第一个和第二个轨迹数据归一化并转换为张量
        normalized_trip1 = torch.tensor(
            np.array(Data_preprocessing.normalized(input_x), dtype=float))  # 第一个轨迹数据
        normalized_trip2 = torch.tensor(
            np.array(Data_preprocessing.normalized(input_y), dtype=float))  # 第二个轨迹数据

        # 查找两个轨迹的主要周期
        top_period1_ = self.ad_finder.find_periods(normalized_trip1)
        top_period2_ = self.ad_finder.find_periods(normalized_trip2)
        top_period1, top_period2, min_gcd = calculate_elementwise_lcm(top_period1_, top_period2_)

        # 计算两个周期的最小公倍数并返回相应的周期和最小公倍数
        top_period1, top_period2, min_gcd = calculate_elementwise_lcm(top_period1_, top_period2_)

        # 根据最小周期生成两个关键向量
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
        device = self.fc1.weight.device  # 🔧 获取该层所在设备
        data = data.to(device)  # ✅ 把输入放到同设备上
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
        # 定义可训练的参数 weights, 大小与输入一致
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channel, kernel_size=1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)  # 30% 随机丢弃
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
