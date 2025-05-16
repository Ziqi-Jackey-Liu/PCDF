import Adaptive_encoding
import Data_preprocessing
from tqdm import tqdm
import torch
import numpy as np


def encoding_process(data):
    results = []
    keys = []
    keys_matrix = []
    for trip_series in tqdm(range(50)):  # len(processed_trip_dict)
        normalized_trip1 = torch.tensor(
            np.array(Data_preprocessing.normalized(data[trip_series, 0]), dtype=float))
        normalized_trip2 = torch.tensor(
            np.array(Data_preprocessing.normalized(data[trip_series, 1]), dtype=float))

        # normalized_trip = torch.tensor(Data_preprocessing.normalized(Data_preprocessing.test(period=5, cycles=4, multiplier=2)))
        Ad_finder = Adaptive_encoding.PeriodFinder(k=3)
        top_period1_ = Ad_finder.find_periods(normalized_trip1)
        top_period2_ = Ad_finder.find_periods(normalized_trip2)
        top_period1, top_period2, min_gcd = Adaptive_encoding.calculate_elementwise_lcm(top_period1_, top_period2_)

        Ad_key = Adaptive_encoding.CyclicVectorGenerator(factor=1)
        key_vector, key_vector_whole = Ad_key.generate_final_vector(length=len(normalized_trip1), period=int(min_gcd),
                                                                    num_vectors=2)

        Ad_circular_conv = Adaptive_encoding.CircularConvolution1()
        cir_vector1, key_matrix1 = Ad_circular_conv(key_vector_whole[0, :], normalized_trip1, 1)
        cir_vector2, key_matrix2 = Ad_circular_conv(key_vector_whole[1, :], normalized_trip2, 1)
        cir_vector3 = cir_vector1 + cir_vector2

        # Ad_circular_corr = Adaptive_encoding.CircularCorrelation()
        # cir_vector_corr1, hhh1_ = Ad_circular_corr(key_vector_whole[0, :], cir_vector3, key_matrix1.T, normalized_trip1,
        #                                            key_matrix2.T, normalized_trip2)
        # cir_vector_corr2, hhh2_ = Ad_circular_corr(key_vector_whole[1, :], cir_vector3, key_matrix2.T, normalized_trip2,
        #                                            key_matrix1.T, normalized_trip1)

        results.append([normalized_trip1, normalized_trip2, cir_vector3])
        keys.append(key_vector_whole)
        keys_matrix.append([key_matrix1, key_matrix2])
    return results, keys, keys_matrix