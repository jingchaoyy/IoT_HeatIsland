"""
Created on  9/21/20
@author: Jingchao Yang
"""
import pandas as pd
from numpy import isnan
import numpy as np
import random
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import json
import os
from cnn_lstm import get_data
from sklearn.metrics import r2_score
import math
from sklearn.metrics import mean_squared_error


def gen_train_and_test_data(data_array=None, input_length=100,
                            test_ratio=0.25, shuffle=True, cut_bin=True, x_is_percentage=False, y_is_percentage=False):
    # df = pd.read_csv(csv_path, header=None)
    # s = df[0]
    # data_array = np.array(s)

    # 对温度进行分箱处理
    if cut_bin:
        # 分箱宽度
        bin_length = 0.5
        label_array = np.round(data_array / bin_length)
        label_array_min = label_array.min()
        data_array = label_array - label_array.min()

        print('Cut bin Done! Bin length is: %s, and the transform equation is : T = %s * label + %s'
              % (bin_length, bin_length, label_array_min))

    all_x_y = []
    train_x_y = []
    test_x_y = []
    for i in range(0, data_array.shape[0] - input_length):
        all_x_y.append(data_array[i:input_length + i + 1])

    # 按照test_ratio分配test数量，取全部数据里面的最后一部分作为test数据
    test_num = int(len(all_x_y) * test_ratio)
    # 取all_x_y中的后面作为test
    test_indexs = np.arange(len(all_x_y) - test_num, len(all_x_y))

    for i in range(len(all_x_y)):
        if i in test_indexs:
            test_x_y.append(all_x_y[i])
        else:
            train_x_y.append((all_x_y[i]))

    # 因为要保证test数据是连续的，因此只能取完test数据之后，对train数据进行shuffle
    if shuffle:
        random.shuffle(train_x_y)

    train_x_y = np.array(train_x_y)
    test_x_y = np.array(test_x_y)

    train_x = train_x_y[:, 0:input_length]
    train_y = train_x_y[:, input_length]
    test_x = test_x_y[:, 0:input_length]
    test_y = test_x_y[:, input_length]

    # if x_is_percentage:

    # 将输出的y转化成变化的百分比
    if y_is_percentage:
        train_y = (train_y - train_x[:, -1]) / train_x[:, -1] * 100.0
        test_y = (test_y - test_x[:, -1]) / test_x[:, -1] * 100.0

    return train_x, train_y, test_x, test_y


def fill_missing(values):
    """
    fill missing values with a value at the same time one day ago
    :param values:
    :return:
    """
    one_day = 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]


def plot_results(predicted_data, true_data, model_type='xgboost'):
    '''plot result'''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Ori')
    plt.plot(predicted_data, label='Pred')
    plt.xlabel('time (hour)')
    plt.ylabel('temperature (F)')
    plt.legend()
    plt.show()

    '''result evaluation'''
    testScore = math.sqrt(mean_squared_error(true_data[:len(predicted_data)], predicted_data))
    print('Test Score: %.2f RMSE' % (testScore))

    lstm_score = r2_score(true_data[:len(predicted_data)], predicted_data)
    print("R^2 Score of model = ", lstm_score)

    '''save to csv'''
    save_path = f'/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA/exp_data/result_single_point_prediction/{model_type}/'
    testscore_dict = {'ori': true_data[:len(predicted_data)],
                      'pred': predicted_data}
    testscore_df = pd.DataFrame(data=testscore_dict)
    testscore_df.to_csv(save_path + 'pred.csv')


def get_data():
    geohash_df = pd.read_csv(
        '/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA/dataHarvest/merged/nodes_missing_5percent.csv',
        usecols=['Geohash'])
    iot_sensors = geohash_df.values.reshape(-1)
    iot_df = pd.read_csv('/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA/dataHarvest/merged/tempMatrix_LA_2019_20.csv',
                         usecols=['datetime'] + iot_sensors.tolist(), index_col=['datetime'])
    fill_missing(iot_df.values)

    return iot_sensors, iot_df
