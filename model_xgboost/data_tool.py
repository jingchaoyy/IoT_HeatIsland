"""
Created on  2019-09-24
@author: Jingchao Yang
"""
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


def gen_train_test_data(y_is_center_point, alldata):
    """

    :param y_is_center_point:
    :return:
    """
    configs = json.load(open('../config_lpy.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    coor = ['9q5csxx', '9q5csz8', '9q5csz9', '9q5cszd', '9q5csze', '9q5cszs', '9q5cszt', '9q5csxr', '9q5csz2',
            '9q5csz3', '9q5csz6', '9q5csz7', '9q5cszk', '9q5cszm', '9q5csxp', '9q5csz0', '9q5csz1', '9q5csz4',
            '9q5csz5', '9q5cszh', '9q5cszj', '9q5cswz', '9q5csyb', '9q5csyc', '9q5csyf', '9q5csyg', '9q5csyu',
            '9q5csyv', '9q5cswx', '9q5csy8', '9q5csy9', '9q5csyd', '9q5csye', '9q5csys', '9q5csyt', '9q5cswr',
            '9q5csy2', '9q5csy3', '9q5csy6', '9q5csy7', '9q5csyk', '9q5csym', '9q5cswp', '9q5csy0', '9q5csy1',
            '9q5csy4', '9q5csy5', '9q5csyh', '9q5csyj']

    data = pd.read_csv('../../IoT_HeatIsland_Data/data/LA/tempMatrix_LA.csv', usecols=coor)
    for c in range(len(coor)):
        coor[c] = data[coor[c]]

    coor = np.asarray(coor)
    coor = coor.reshape(7, 7, coor.shape[-1])
    print('time-series matrix data processed')
    print(coor.shape)
    train = coor[:, :, :int(coor.shape[-1] * configs['data']['train_test_split'])]
    test = coor[:, :, int(coor.shape[-1] * configs['data']['train_test_split']):]

    x_train, y_train, train_nor = get_train_data(train, configs['data']['sequence_length'],
                                                 configs['data']['normalise'])
    x_test, y_test, test_nor = get_all_data(test, configs['data']['sequence_length'], configs['data']['normalise'])

    if y_is_center_point:
        y_train = y_train[:, y_train.shape[1] // 2, y_train.shape[2] // 2]
        y_test = y_test[:, y_test.shape[1] // 2, y_test.shape[2] // 2]
    return x_train, y_train, x_test, y_test


def get_all_data(data, seq_len, normalise):
    """
    Create x, y test/train data windows
    Warning: batch method, not generative, make sure you have enough memory to
    load data, otherwise reduce size of the training split.

    :param data:
    :param seq_len:
    :param normalise:
    :return:
    """
    data_windows, x, y = [], [], []

    for i in range(data.shape[-1] - seq_len):
        data_windows.append(data[:, :, i:i + seq_len])

    data_windows = np.array(data_windows).astype(float)
    data_windows, data_normalizers = normalise_windows(data_windows, single_window=False, norm=normalise)

    # x = data_windows[:, :-1]
    # y = data_windows[:, -1]
    # y = y[:, [int(y.shape[1] / 2)], [int(y.shape[-1] / 2)]]
    #
    # temp_normalizers = data_normalizers[:, 0]
    return data_windows


def get_test_data(corner, train_all, test_all):
    """

    :param corner:
    :param train_x:
    :return:
    """
    if corner == 0:
        # getting the first element (top-left corner of the last of the sequence as y data
        train_x = train_all[:, -1, 0, [0]]
        test_x = test_all[:, -1, 0, [0]]

    return train_x, test_x


def normalise_windows(window_data, single_window=False, norm=True):
    """
    Normalise window with a base value of zero

    :param window_data:
    :param single_window:
    :param norm:
    :return:
    """
    normalised_data, all_normalizers = [], []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window, normalizers = [], []
        for col_i in range(window.shape[-1]):
            # normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
            normalised = []
            curr_window = window[:, :, col_i]
            if norm == True:
                normalizer = curr_window[0, 0]
            else:
                normalizer = 1
            for q in curr_window:
                nor_col = []
                for p in q:
                    if p != -1:
                        nor = (float(p) / normalizer) - 1
                    else:
                        nor = p
                    nor_col.append(nor)
                normalised.append(nor_col)
            normalizers.append(normalizer)
            normalised_window.append(normalised)

        # reshape and transpose array back into original multidimensional format
        normalised_window = np.array(normalised_window)
        normalizers = np.array(normalizers)
        normalised_data.append(normalised_window)
        all_normalizers.append(normalizers)

    return np.array(normalised_data), np.array(all_normalizers)


def plot_results(predicted_data, true_data):
    """

    :param predicted_data:
    :param true_data:
    :return:
    """
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def rect_detect(leftdown, length, width, coor_path):
    """

    :param left_down:
    :param length:
    :param width:
    :return:
    """
    coor = pd.read_csv(coor_path, usecols=['x', 'y', 'geohash'])
    print('leng:width:')
    geohash_list = coor['geohash'].tolist()
    x_list = coor['x'].tolist()
    y_list = coor['y'].tolist()
    coor_list = []
    for i in range(len(x_list)):
        coor_list.append([x_list[i], y_list[i]])
    print('%s points found!' % i)

    valid_rect_flag = True
    rect_geohashs = []
    for i in range(width):
        for j in range(length):
            if [leftdown[0] + j, leftdown[1] + i] not in coor_list:
                print('relative coordinate (%s,%s) not exist!' % (i, j))
                valid_rect_flag = False
            else:
                rect_geohashs.append(geohash_list[coor_list.index([leftdown[0] + j, leftdown[1] + i])])

    if valid_rect_flag:
        print('valid rectangle detected!')
        ret = rect_geohashs
    else:
        print('Error!invalid rectangle area!')
        ret = None

    assert ret is not None
    return ret
