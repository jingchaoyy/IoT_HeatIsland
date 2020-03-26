import pandas as pd
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

    train_x = train_x_y[:, 0:100]
    train_y = train_x_y[:, 100]
    test_x = test_x_y[:, 0:100]
    test_y = test_x_y[:, 100]

    # if x_is_percentage:

    # 将输出的y转化成变化的百分比
    if y_is_percentage:
        train_y = (train_y - train_x[:, -1]) / train_x[:, -1] * 100.0
        test_y = (test_y - test_x[:, -1]) / test_x[:, -1] * 100.0

    return train_x, train_y, test_x, test_y


# 为CNN提供数据
# x_train.shape=[1740,95,7,7]
# y_train.shape=[1740,7,7] (if not y_is_center_point)
def gen_cnn_data(y_is_center_point):
    configs = json.load(open('../config_lpy.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    # coor = ['9q5csmp', '9q5cst0', '9q5xxxx', '9q5cst4', '9q5cst5', '9q5csth', '9q5cstj', '9q5cskz', '9q5cssb',
    #         '9q5cssc',
    #         '9q5cssf', '9q5cssg', '9q5cssu', '9q5cssv', '9q5cskx', '9q5css8', '9q5css9', '9q5cssd', '9q5csse',
    #         '9q5csss',
    #         '9q5csst', '9q5cskr', '9q5css2', '9q5css3', '9q5css6', '9q5css7', '9q5cssk', '9q5cssm', '9q5xxxx',
    #         '9q5css0',
    #         '9q5css1', '9q5css4', '9q5css5', '9q5cssh', '9q5cssj', '9q5cs7z', '9q5xxxx', '9q5csec', '9q5csef',
    #         '9q5cseg',
    #         '9q5cseu', '9q5csev', '9q5cs7x', '9q5cse8', '9q5xxxx', '9q5csed', '9q5csee', '9q5cses', '9q5cset']
    #
    # data = pd.read_csv('../../data/joined_49_fillna_1.csv', usecols=coor)
    coor = ['9q5csxx', '9q5csz8', '9q5csz9', '9q5cszd', '9q5csze', '9q5cszs', '9q5cszt', '9q5csxr', '9q5csz2',
            '9q5csz3', '9q5csz6', '9q5csz7', '9q5cszk', '9q5cszm', '9q5csxp', '9q5csz0', '9q5csz1', '9q5csz4',
            '9q5csz5', '9q5cszh', '9q5cszj', '9q5cswz', '9q5csyb', '9q5csyc', '9q5csyf', '9q5csyg', '9q5csyu',
            '9q5csyv', '9q5cswx', '9q5csy8', '9q5csy9', '9q5csyd', '9q5csye', '9q5csys', '9q5csyt', '9q5cswr',
            '9q5csy2', '9q5csy3', '9q5csy6', '9q5csy7', '9q5csyk', '9q5csym', '9q5cswp', '9q5csy0', '9q5csy1',
            '9q5csy4', '9q5csy5', '9q5csyh', '9q5csyj']

    # data = pd.read_csv('/Users/jc/Documents/GitHub/IoT_HeatIsland_Data/data/LA/joined_49_fillna_1.csv', usecols=coor)
    data = pd.read_csv('../../IoT_HeatIsland_Data/data/LA/tempMatrix_LA.csv', usecols=coor)
    for c in range(len(coor)):
        coor[c] = data[coor[c]]

    coor = np.asarray(coor)
    coor = coor.reshape(7, 7, coor.shape[-1])
    print('time-series matrix data processed')
    print(coor.shape)
    train = coor[:, :, :int(coor.shape[-1] * configs['data']['train_test_split'])]
    test = coor[:, :, int(coor.shape[-1] * configs['data']['train_test_split']):]

    x_train, y_train, train_nor = get_data(train, configs['data']['sequence_length'], configs['data']['normalise'])
    x_test, y_test, test_nor = get_data(test, configs['data']['sequence_length'], configs['data']['normalise'])

    if y_is_center_point:
        y_train = y_train[:, y_train.shape[1] // 2, y_train.shape[2] // 2]
        y_test = y_test[:, y_test.shape[1] // 2, y_test.shape[2] // 2]
    return x_train, y_train, x_test, y_test


def reshape_as_image(arrays):
    return (n.reshape((n.shape[0], n.shape[1], 1, 1)) for n in arrays)


def plot_results(predicted_data, true_data):
    '''plot result'''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

    '''result evaluation'''
    # testScore = math.sqrt(mean_squared_error(true_data[:len(predicted_data)], predicted_data))
    # print('Test Score: %.2f RMSE' % (testScore))
    #
    # lstm_score = r2_score(true_data[:len(predicted_data)], predicted_data)
    # print("R^2 Score of model = ", lstm_score)


def plot_results_multiple(test_x, test_y, prediction_len, model, model_type, filename):
    assert model_type in ['gbdt', 'xgboost', 'torch_dnn']
    # fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(111)
    # ax.plot(test_y, label='True Data')
    pred_multiple_all = []
    for i in range(test_x.shape[0] // prediction_len):
        pred_multiple = []
        current_x = test_x[i * prediction_len]
        for j in range(prediction_len):
            if j == 0:
                if model_type == 'xgboost':
                    current_x_xgb = xgb.DMatrix([current_x])
                    pred_multiple.append(model.predict(current_x_xgb))
                if model_type == 'gbdt':
                    pred_multiple.append(model.predict([current_x]))
                if model_type == 'torch_dnn':
                    with torch.no_grad():
                        pred_torch = model(torch.from_numpy(current_x).float()).numpy()
                    pred_multiple.append(pred_torch)
            else:
                current_x = np.delete(current_x, 0)
                current_x = np.append(current_x, pred_multiple[-1])
                if model_type == 'xgboost':
                    current_x_xgb = xgb.DMatrix([current_x])
                    pred_multiple.append(model.predict(current_x_xgb))
                if model_type == 'gbdt':
                    pred_multiple.append(model.predict([current_x]))
                if model_type == 'torch_dnn':
                    with torch.no_grad():
                        pred_torch = model(torch.from_numpy(current_x).float()).numpy()
                    pred_multiple.append(pred_torch)

        pred_multiple_all.append(pred_multiple)

    '''plot result'''
    # # plot_results(pred_multiple,true_data)
    # fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(111)
    # ax.plot(test_y, label='True Data')
    # plt.xlabel("Time")
    # plt.ylabel("Temperature (F)")
    # for i in range(len(pred_multiple_all)):
    #     padding = [None for p in range(i * prediction_len)]
    #     plt.plot(padding + pred_multiple_all[i], label='Prediction')

    # fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(111)
    # plt.plot(test_y, label='True Data')
    # # plt.plot(pred_multiple_all_merge, label='Prediction')
    # plt.legend()
    # plt.show()
    # plt.legend()
    # plt.show()
    #
    # print('multiple Printed!')

    '''result evaluation'''
    # pred_multiple_all_merge = [j for i in pred_multiple_all for j in i]

    # testScore = math.sqrt(mean_squared_error(test_y[:len(pred_multiple_all_merge)], pred_multiple_all_merge))
    # print('Test Score: %.2f RMSE' % (testScore))
    #
    # lstm_score = r2_score(test_y[:len(pred_multiple_all_merge)], pred_multiple_all_merge)
    # print("R^2 Score of model = ", lstm_score)

    '''save to csv'''
    result = pd.DataFrame(index=np.arange(max(len(test_y), len(pred_multiple_all))))
    result.index.name = 'time'
    result['tru'] = pd.Series(test_y.reshape(test_y.size))
    result['prediction'] = pd.Series([j.item() for i in pred_multiple_all for j in i])
    result.to_csv('../../IoT_HeatIsland_Data/data/LA/exp_data/result_single_point_prediction_GBDT/'
                  + filename + '.csv')
