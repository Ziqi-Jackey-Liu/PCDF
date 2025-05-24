import Adaptive_encoding
import Data_compress
import Electric
import Result_test
import Data_preprocessing
import Prediction_model
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import Temp
import Turbofan
import Visualization
import NYC_taxi
import DC_bike
import Electric
import Solar_energy
import Training_process
import os
import Weather


def set_seed(seed=42):
    torch.manual_seed(seed)  # Fix the PyTorch random seed (CPU)
    torch.cuda.manual_seed(seed)  # Fix the PyTorch random seed on CUDA (GPU)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs, fix all CUDA devices
    np.random.seed(seed)  # Fix the NumPy random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

_ = Visualization.scatter()

Wea_data = Weather.read('./Data/WeatherBench/city_hourly_temperature.csv')
processed_trip_dict_wea = Wea_data

# Gas_data = Turbofan.read('./Data/GasSensor/', 0, 0)
# processed_trip_dict_gas = Gas_data

# Sol_data = Solar_energy.read('./Data/Solar_energy', 0, 20)
# Sol_data_s = Sol_data[:Sol_data.shape[0] - Sol_data.shape[0] % 40, :]  # Ensure the number of rows is even
# processed_trip_dict_sol = Sol_data_s.reshape(Sol_data_s.shape[0] // 40, 40, Sol_data_s.shape[1])

# Ele_data = Electric.read('./Data/Electric/LD2011_2014.txt', 0, 12)
# Ele_data_s = Ele_data[:Ele_data.shape[0] - Ele_data.shape[0] % 40, :]
# processed_trip_dict_ele = Ele_data_s.reshape(Ele_data_s.shape[0] // 40, 40, Ele_data_s.shape[1])

# DC_data = DC_bike.read('./Data/DC_bike/20[1][2-7]*.csv', 48000, 10)
# DC_data_s = DC_data[:DC_data.shape[0] - DC_data.shape[0] % 40, :]  # Ensure the number of rows is even
# processed_trip_dict_dc = DC_data_s.reshape(DC_data_s.shape[0] // 40, 40, DC_data_s.shape[1])
#
# NYC_data = NYC_taxi.read('./Data/NYC_taxi', 250)
# # _ = Result_test.data_test(NYC_data, row_index=40)
# NYC_data_s = NYC_data[:NYC_data.shape[0] - NYC_data.shape[0] % 40, :]  # Ensure the number of rows is even
# processed_trip_dict_nyc = NYC_data_s.reshape(NYC_data_s.shape[0] // 40, 40, NYC_data_s.shape[1])

set_seed(1200)
hhh_ch = []
hhh_what = []
hhh_time = []
for chanel_number in [5, 10, 20, 30, 40]:
    hhh_t = []
    remote_p = 0
    o_ = 0
    for remote_time in range(3):
        losses = []
        for times in range(1):
            _, o_ = Training_process.train_stage2(processed_trip_dict_wea[:, 0:chanel_number, :], 0,
                                              remote_p=remote_p,
                                              channel_=chanel_number)
            # _ = Training_process.Ablation(processed_trip_dict_gas[:, 0:chanel_number, :], 0,
            #                                       remote_p=remote_p,
            #                                       channel_=chanel_number)
            losses.append(_)
        remote_p = remote_p + 1
        array = np.array(losses)
        column_means = np.mean(array, axis=0)
        hhh_t.append(column_means)
        hhh_what.append(array)
    hhh_time.append(np.array(o_))
    hhh_t_means = np.mean(hhh_t, axis=0)
    hhh_ch.append(hhh_t_means)
    print(1)

# remote server info
server_user = "xxx"
server_host = "xxx"
server_path = "xxx"  # path to save the CSV file on remote server

# generate CSV and save it on remote server
df = pd.DataFrame(zip(*hhh_t))
df.to_csv(server_path, index=False, header=False)
print(f"# The CSV file has been saved on the remote server: {server_path}")

