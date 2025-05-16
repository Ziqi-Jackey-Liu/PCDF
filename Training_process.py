from collections import Counter
from ftplib import MSG_OOB

from tqdm import tqdm
import torch
import Dlinear
import Autoformer
import Data_preprocessing
import numpy as np
import Adaptive_encoding
import HDMixer_
import PatchTST_
import Prediction_model
import MSGNET
import TSMixer


def train_stage2(data, parameter_alpha, remote_p, channel_):
    normalized_data = []
    for trip_series in range(data.shape[0]):
        # len(processed_trip_dict)
        normalized_data_temp = []
        for channel_hhh in range(data.shape[1]):
            normalized_data1 = torch.tensor(
                np.array(Data_preprocessing.normalized(data[trip_series, channel_hhh, :]), dtype=float))
            normalized_data_temp.append(normalized_data1)
        normalized_data.append(torch.stack((normalized_data_temp), dim=0))

    data_period1 = []
    data_period2 = []
    for trip_series in tqdm(range(len(normalized_data))):
        ad_finder = Adaptive_encoding.PeriodFinder(k=3)

        # 对每个 trip 的两个部分分别找周期
        top_period1 = ad_finder.find_periods(normalized_data[trip_series][0])
        top_period2 = ad_finder.find_periods(normalized_data[trip_series][1])

        # 将结果堆叠并添加到列表中
        data_period1.append(top_period1[0])
        data_period2.append(top_period2[0])
    # 将所有结果合并为一个张量
    counter1 = [Counter(data_period1).most_common(1)[0][0], Counter(data_period1).most_common(1)[0][0]]
    counter2 = [Counter(data_period2).most_common(1)[0][0], Counter(data_period2).most_common(1)[0][0]]
    data_period1_c, data_period2_c, min_gcd = Adaptive_encoding.calculate_elementwise_lcm(counter1, counter2)

    data_length = min_gcd

    data_normalized_tensor = np.array(normalized_data)
    data_normalized_tensor = torch.tensor(data_normalized_tensor)

    data_length = 25
    ones_tensor = torch.ones_like(data_normalized_tensor[:, :, :data_length])

    # predict_model1 = Prediction_model.TimeSeriesCNN(sequence_length=data_length*3, num_classes=data_length, compress_channel=1)
    # train_process1 = Prediction_model.Trainer(learning_rate=0.01, epochs=120, model=predict_model1, sample_length=data_length)
    # loss_test1, outputs_test1, time1 = train_process1.train(
    #     train_data=data_normalized_tensor[:, 0, data_length * (remote_p):data_length * (3 + remote_p)],
    #     train_label=data_normalized_tensor[:, 0, data_length * (3 + remote_p):data_length * (4 + remote_p)],
    #     test_data=data_normalized_tensor[:, 0, data_length * (remote_p + 1):data_length * (remote_p + 4)],
    #     test_label=data_normalized_tensor[:, 0, data_length * (remote_p + 4):data_length * (remote_p + 5)],
    #     pattern=None)
    predict_model1 = Prediction_model.TimeSeriesCNNWhole(sequence_length=data_length * 3, num_classes=data_length,
                                                         compress_period=data_length, mode='periodic',
                                                         t_pattern='mlp', channel=channel_, ablation=[0, 0])
    train_process1 = Prediction_model.TrainerCom(learning_rate=0.05, epochs=120, model=predict_model1,
                                                 sample_length=data_length)
    loss_test1, outputs_test1, time1, mem1 = train_process1.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (4 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=torch.cat(
            [data_normalized_tensor[:, :, data_length * (remote_p + 1): data_length * (remote_p + 4)], ones_tensor],
            dim=2),
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        alpha_=parameter_alpha)

    predict_model2 = Prediction_model.TimeSeriesCNNWhole(sequence_length=data_length*3, num_classes=data_length, compress_period=data_length, mode='periodic', t_pattern='convolution', channel = channel_, ablation=[0, 0])
    train_process2 = Prediction_model.TrainerCom(learning_rate=0.05, epochs=120, model=predict_model2, sample_length=data_length)
    loss_test2, outputs_test2, time2, mem2 = train_process2.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (4 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=torch.cat(
            [data_normalized_tensor[:, :, data_length * (remote_p + 1): data_length * (remote_p + 4)], ones_tensor],
            dim=2),
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        alpha_=parameter_alpha)

    predict_model3 = Prediction_model.TimeSeriesCNNWhole(sequence_length=data_length * 3, num_classes=data_length, compress_period=data_length, mode='periodic', t_pattern='transformer', channel = channel_, ablation=[0, 0])
    train_process3 = Prediction_model.TrainerCom(learning_rate=0.05, epochs=120, model=predict_model3, sample_length=data_length)
    loss_test3, outputs_test3, time3, mem3 = train_process3.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (4 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=torch.cat(
            [data_normalized_tensor[:, :, data_length * (remote_p + 1): data_length * (remote_p + 4)], ones_tensor],
            dim=2),
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        alpha_=parameter_alpha)

    predict_model4 = Prediction_model.TimeSeriesCNNWhole(sequence_length=data_length * 3, num_classes=data_length, compress_period=data_length, mode='periodic', t_pattern='linear', channel = channel_, ablation=[0, 0])
    train_process4 = Prediction_model.TrainerCom(learning_rate=0.05, epochs=120, model=predict_model4, sample_length=data_length)
    loss_test4, outputs_test4, time4, mem4 = train_process4.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (4 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=torch.cat(
            [data_normalized_tensor[:, :, data_length * (remote_p + 1): data_length * (remote_p + 4)], ones_tensor],
            dim=2),
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        alpha_=parameter_alpha)

    patch_tst = PatchTST_.ModelPTST(c_in=channel_, context_window=data_length * 3, target_window=data_length, patch_len=data_length // 12, n_heads=1)
    train_process5 = Prediction_model.Trainer(learning_rate=0.01, epochs=120, model=patch_tst, sample_length=data_length)
    loss_test5, outputs_test5, time5, mem5 = train_process5.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (3 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=data_normalized_tensor[:, :, data_length * (remote_p + 1):data_length * (remote_p + 4)],
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        pattern=None)

    auto_former = Autoformer.Model(enc_in=channel_, dec_in=channel_, c_out=channel_, label_len=data_length * 3 // 2, pred_len=data_length)
    train_process6 = Prediction_model.Trainer(learning_rate=0.01, epochs=120, model=auto_former, sample_length=data_length)
    loss_test6, outputs_test6, time6, mem6 = train_process6.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (3 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=data_normalized_tensor[:, :, data_length * (remote_p + 1):data_length * (remote_p + 4)],
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        pattern=None)

    hdm_mixer = HDMixer_.Model(enc_in=channel_, c_in=channel_, context_window=data_length * 3, target_window=data_length, patch_len=data_length//12)
    train_process7 = Prediction_model.Trainer(learning_rate=0.01, epochs=120, model=hdm_mixer, sample_length=data_length)
    loss_test7, outputs_test7, time7, mem7 = train_process7.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (3 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=data_normalized_tensor[:, :, data_length * (remote_p + 1):data_length * (remote_p + 4)],
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        pattern='hdm_mixer')

    d_linear = Dlinear.Model(seq_len=data_length*3, pred_len=data_length, individual=True, enc_in=channel_)
    train_process8 = Prediction_model.Trainer(learning_rate=0.01, epochs=120, model=d_linear, sample_length=data_length)
    loss_test8, outputs_test8, time8, mem8 = train_process8.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (3 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=data_normalized_tensor[:, :, data_length * (remote_p + 1):data_length * (remote_p + 4)],
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        pattern=None)

    MseNet = MSGNET.Model(channel=channel_,  seq_len=data_length*3, label_len=data_length * 3 // 2, pred_len=data_length)
    train_process9 = Prediction_model.Trainer(learning_rate=0.01, epochs=120, model=MseNet, sample_length=data_length)
    loss_test9, outputs_test9, time9, mem9 = train_process9.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (3 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=data_normalized_tensor[:, :, data_length * (remote_p + 1):data_length * (remote_p + 4)],
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        pattern=None)

    TSmixer = TSMixer.PatchTSMixerForPrediction(context_length=data_length * 3, in_features=channel_,
                                                out_features=channel_, prediction_length=data_length)
    train_process10 = Prediction_model.Trainer(learning_rate=0.01, epochs=120, model=TSmixer, sample_length=data_length)
    loss_test10, outputs_test10, time10, mem10 = train_process10.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (3 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=data_normalized_tensor[:, :, data_length * (remote_p + 1):data_length * (remote_p + 4)],
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        pattern=None)

    # # 收集到一个list
    # mem_list = [mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8, mem9, mem10]
    #
    # # 拆分float和int
    # float_array = np.array([item[0] for item in mem_list])
    # int_array = np.array([item[1] for item in mem_list])
    return ([loss_test1[0].item(), loss_test2[0].item(), loss_test3[0].item(), loss_test4[0].item(), loss_test5.item(),
            loss_test6.item(), loss_test7.item(), loss_test8.item(), loss_test9.item(), loss_test10.item()],
            [time1[1], time2[1], time3[1], time4[1], time5[1], time6[1], time7[1], time8[1], time9[1], time10[1]])


def Ablation(data, parameter_alpha, remote_p, channel_):
    normalized_data = []
    for trip_series in range(data.shape[0]):
        # len(processed_trip_dict)
        normalized_data_temp = []
        for channel_hhh in range(data.shape[1]):
            normalized_data1 = torch.tensor(
                np.array(Data_preprocessing.normalized(data[trip_series, channel_hhh, :]), dtype=float))
            normalized_data_temp.append(normalized_data1)
        normalized_data.append(torch.stack((normalized_data_temp), dim=0))

    data_period1 = []
    data_period2 = []
    for trip_series in tqdm(range(len(normalized_data))):
        ad_finder = Adaptive_encoding.PeriodFinder(k=3)

        # 对每个 trip 的两个部分分别找周期
        top_period1 = ad_finder.find_periods(normalized_data[trip_series][0])
        top_period2 = ad_finder.find_periods(normalized_data[trip_series][1])

        # 将结果堆叠并添加到列表中
        data_period1.append(top_period1[0])
        data_period2.append(top_period2[0])
    # 将所有结果合并为一个张量
    counter1 = [Counter(data_period1).most_common(1)[0][0], Counter(data_period1).most_common(1)[0][0]]
    counter2 = [Counter(data_period2).most_common(1)[0][0], Counter(data_period2).most_common(1)[0][0]]
    data_period1_c, data_period2_c, min_gcd = Adaptive_encoding.calculate_elementwise_lcm(counter1, counter2)

    data_length = min_gcd

    data_normalized_tensor = np.array(normalized_data)
    data_normalized_tensor = torch.tensor(data_normalized_tensor)

    data_length = 30
    ones_tensor = torch.ones_like(data_normalized_tensor[:, :, :data_length])

    predict_model5 = Prediction_model.TimeSeriesCNNWhole(sequence_length=data_length * 3, num_classes=data_length,
                                                         compress_period=data_length, mode='periodic',
                                                         t_pattern='mlp', channel=channel_, ablation=[1, 0])
    train_process5 = Prediction_model.TrainerCom(learning_rate=0.05, epochs=120, model=predict_model5,
                                                 sample_length=data_length)
    loss_test5, outputs_test5, time5, mem5 = train_process5.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (4 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=torch.cat(
            [data_normalized_tensor[:, :, data_length * (remote_p + 1): data_length * (remote_p + 4)], ones_tensor],
            dim=2),
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        alpha_=parameter_alpha)

    predict_model6 = Prediction_model.TimeSeriesCNNWhole(sequence_length=data_length * 3, num_classes=data_length,
                                                         compress_period=data_length, mode='periodic',
                                                         t_pattern='convolution', channel=channel_, ablation=[1, 0])
    train_process6 = Prediction_model.TrainerCom(learning_rate=0.05, epochs=120, model=predict_model6,
                                                 sample_length=data_length)
    loss_test6, outputs_test6, time6, mem6 = train_process6.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (4 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=torch.cat(
            [data_normalized_tensor[:, :, data_length * (remote_p + 1): data_length * (remote_p + 4)], ones_tensor],
            dim=2),
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        alpha_=parameter_alpha)

    predict_model7 = Prediction_model.TimeSeriesCNNWhole(sequence_length=data_length * 3, num_classes=data_length,
                                                         compress_period=data_length, mode='periodic',
                                                         t_pattern='transformer', channel=channel_, ablation=[1, 0])
    train_process7 = Prediction_model.TrainerCom(learning_rate=0.05, epochs=120, model=predict_model7,
                                                 sample_length=data_length)
    loss_test7, outputs_test7, time7, mem7 = train_process7.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (4 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=torch.cat(
            [data_normalized_tensor[:, :, data_length * (remote_p + 1): data_length * (remote_p + 4)], ones_tensor],
            dim=2),
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        alpha_=parameter_alpha)

    predict_model8 = Prediction_model.TimeSeriesCNNWhole(sequence_length=data_length * 3, num_classes=data_length,
                                                         compress_period=data_length, mode='periodic',
                                                         t_pattern='linear', channel=channel_, ablation=[1, 0])
    train_process8 = Prediction_model.TrainerCom(learning_rate=0.05, epochs=120, model=predict_model8,
                                                 sample_length=data_length)
    loss_test8, outputs_test8, time8, mem8 = train_process8.train(
        train_data=data_normalized_tensor[:, :, data_length * (remote_p):data_length * (4 + remote_p)],
        train_label=data_normalized_tensor[:, :, data_length * (3 + remote_p):data_length * (4 + remote_p)],
        test_data=torch.cat(
            [data_normalized_tensor[:, :, data_length * (remote_p + 1): data_length * (remote_p + 4)], ones_tensor],
            dim=2),
        test_label=data_normalized_tensor[:, :, data_length * (remote_p + 4):data_length * (remote_p + 5)],
        alpha_=parameter_alpha)

    return [loss_test5[0].item(), loss_test6[0].item(), loss_test7[0].item(), loss_test8[0].item()]
