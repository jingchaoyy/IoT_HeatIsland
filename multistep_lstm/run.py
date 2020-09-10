"""
Created on  8/27/20
@author: Jingchao Yang
"""
import pandas as pd
import matplotlib.pyplot as plt
import time
from platform import python_version
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statistics import mean
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, date, timedelta
# from multistep_lstm import multistep_lstm_keras
from multistep_lstm import multistep_lstm_pytorch
import random
from sklearn import preprocessing


def min_max_scaler(df):
    """

    :param df:
    :return:
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    df_np = df.values
    df_np_scaled = min_max_scaler.fit_transform(df_np)
    df_scaled = pd.DataFrame(df_np_scaled)

    df_scaled.index = df.index
    df_scaled.columns = df.columns

    return df_scaled



'''pytorch'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
multi_variate_mode = True
'''data'''
# aggr_df = pd.read_csv(r'D:\1_GitHub\Fresh-Air-LA\data\aggr_la_aq_preprocessed.csv', index_col=False)
# vars = list(set(aggr_df.columns[1:]) - set(['datetime']))

geohash_df = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\merged\nodes_missing_5percent.csv',
                         usecols=['Geohash'])
iot_sensors = geohash_df.values.reshape(-1)
iot_df = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\merged\tempMatrix_LA_2019_20.csv',
                     usecols=['datetime'] + iot_sensors.tolist(), index_col=['datetime'])

if multi_variate_mode:
    ext_data_path = r'E:\IoT_HeatIsland_Data\data\LA\weather_underground\WU_preprocessed_LA\processed\byAttributes'
    humidity_df = pd.read_csv(ext_data_path + r'\humidity.csv', index_col=['datetime'])
    pressure_df = pd.read_csv(ext_data_path + r'\pressure.csv', index_col=['datetime'])
    windSpeed_df = pd.read_csv(ext_data_path + r'\windSpeed.csv', index_col=['datetime'])
    ext_data = [humidity_df, pressure_df, windSpeed_df]
    ext_data_scaled = []
    for ext in ext_data:
        ext_data_scaled.append(min_max_scaler(ext))
    iot_wu_match_df = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\merged\iot_wu_colocate.csv', index_col=0)

iot_df = iot_df.dropna()

'''all stations'''
# selected_vars = [var for var in vars if var.split('_')[1] in ['PM2.5', 'OZONE', 'NO2']]
# dataset = aggr_df[selected_vars]
# selected_vars = random.choices(all_sensors, k=int(len(all_sensors)*0.7))

selected_vars = iot_sensors
dataset = iot_df

print('selected sensors', dataset.columns)

dataset = dataset.values
dataset[dataset < 0] = 0
print('size', dataset.shape)

# find max and min values for normalization
norm_min = dataset.min()
norm_max = dataset.max()
print('dataset min, max', norm_min, norm_max)

# normalize the data
dataset = (dataset - norm_min) / (norm_max - norm_min)
print('normalized dataset min, max', dataset.min(), dataset.max())

# separate train and test stations
train_stations = selected_vars[:int(len(selected_vars) * 0.7)]
test_stations = selected_vars[int(len(selected_vars) * 0.7):]

train_data_raw = iot_df[train_stations]
test_data_raw = iot_df[test_stations]

print(train_data_raw.shape)
print(test_data_raw.shape)
print(train_data_raw.columns)

train_window = 72
output_size = 12

if not multi_variate_mode:
    train_data = multistep_lstm_pytorch.Dataset(train_data_raw,
                                                (norm_min, norm_max),
                                                train_window, output_size)
    test_data = multistep_lstm_pytorch.Dataset(test_data_raw,
                                               (norm_min, norm_max),
                                               train_window,
                                               output_size,
                                               test_station=True)
else:
    train_data = multistep_lstm_pytorch.Dataset_multivariate(train_data_raw,
                                                             (norm_min, norm_max),
                                                             train_window,
                                                             output_size,
                                                             ext_data_scaled,
                                                             iot_wu_match_df)
    test_data = multistep_lstm_pytorch.Dataset_multivariate(test_data_raw,
                                                            (norm_min, norm_max),
                                                            train_window,
                                                            output_size,
                                                            ext_data_scaled,
                                                            iot_wu_match_df,
                                                            test_station=True)

print('Number of stations in training data: ', len(train_data))
print('Number of stations in testing data: ', len(test_data))

print("Training input and output for each station: %s, %s" % (train_data[0][0].shape, train_data[0][1].shape))
print("Validation input and output for each station: %s, %s" % (train_data[0][2].shape, train_data[0][3].shape))
print("Testing input and output for each station: %s, %s" % (test_data[0][0].shape, test_data[0][1].shape))

# initialize the model
num_epochs = 6
epoch_interval = 1
loss_func, model, optimizer = multistep_lstm_pytorch.initial_model(input_size=train_data[0][0].shape[-1],
                                                                   output_size=output_size,
                                                                   learning_rate=0.001)
train_loss, test_loss = [], []

start = time.time()
# train the model
for epoch in range(num_epochs):
    running_loss_train = []
    running_loss_val = []
    for idx in range(len(train_data)):
        train_loader = DataLoader(TensorDataset(train_data[idx][0][:, :, 0, :].to(device),
                                                train_data[idx][1][:, :, 0, :].to(device)),
                                  shuffle=True, batch_size=1000, drop_last=True)
        val_loader = DataLoader(TensorDataset(train_data[idx][2][:, :, 0, :].to(device),
                                              train_data[idx][3][:, :, 0, :].to(device)),
                                shuffle=True, batch_size=400, drop_last=True)
        loss1 = multistep_lstm_pytorch.train_LSTM(train_loader, model, loss_func, optimizer,
                                                  epoch)  # calculate train_loss
        loss2 = multistep_lstm_pytorch.test_LSTM(val_loader, model, loss_func, optimizer, epoch)  # calculate test_loss
        running_loss_train.append(sum(loss1))
        running_loss_val.append(sum(loss2))
        train_loss.extend(loss1)
        test_loss.extend(loss2)

    if epoch % epoch_interval == 0:
        print(
            "Epoch: %d, train_loss: %1.5f, val_loss: %1.5f" % (epoch, mean(running_loss_train), mean(running_loss_val)))

end = time.time()
print(end - start)

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()

# save trained model
torch.save(model.state_dict(), r'D:\1_GitHub\IoT_HeatIsland\multistep_lstm\saved_models'
                               f'\\multivariate_epoch{num_epochs}.pt')

# Predict the training dataset of training stations and testing dataset of testing stations
train_pred_orig_dict = dict()
for idx in range(len(train_data)):
    station = train_data.keys[idx]
    with torch.no_grad():
        train_pred = model(train_data[idx][0][:, :, 0, :].to(device))
        train_pred_trans = train_pred * (norm_max - norm_min) + norm_min

        train_orig = train_data[idx][1][:, :, 0, :].reshape(train_pred.shape).to(device)
        train_orig_trans = train_orig * (norm_max - norm_min) + norm_min

        train_pred_orig_dict[station] = (train_pred_trans, train_orig_trans)

test_pred_orig_dict = dict()
for idx in range(len(test_data)):
    station = test_data.keys[idx]
    with torch.no_grad():
        test_pred = model(test_data[idx][0][:, :, 0, :].to(device))
        test_pred_trans = test_pred * (norm_max - norm_min) + norm_min

        test_orig = test_data[idx][1][:, :, 0, :].reshape(test_pred.shape).to(device)
        test_orig_trans = test_orig * (norm_max - norm_min) + norm_min

        test_pred_orig_dict[station] = (test_pred_trans, test_orig_trans)

print(list(test_pred_orig_dict.keys())[0])

# plot baseline and predictions
plt.plot(test_pred_orig_dict[list(test_pred_orig_dict.keys())[0]][0][:, 0].data.tolist(), label='pred')  # predicted
plt.plot(test_pred_orig_dict[list(test_pred_orig_dict.keys())[0]][1][:, 0].data.tolist(), label='Ori')  # original
plt.show()

# calculate root mean squared error
trainScores_stations = dict()
testScores_stations = dict()

for key in train_data.keys:
    trainScores_stations[key] = math.sqrt(mean_squared_error(train_pred_orig_dict[key][0].data.tolist(),
                                                             train_pred_orig_dict[key][1].data.tolist()))

for key in test_data.keys:
    testScores_stations[key] = math.sqrt(mean_squared_error(test_pred_orig_dict[key][0].data.tolist(),
                                                            test_pred_orig_dict[key][1].data.tolist()))

print(max(trainScores_stations.values()))
print(min(trainScores_stations.values()))

print(max(testScores_stations.values()))
print(min(testScores_stations.values()))

# using 3-sigma for selecting high loss stations
trainScores_stations_df = pd.DataFrame.from_dict(trainScores_stations, orient='index', columns=['value'])
sigma_3 = (3 * trainScores_stations_df.std() + trainScores_stations_df.mean()).values[0]
anomaly_iot_train = trainScores_stations_df.loc[trainScores_stations_df['value'] >= sigma_3]
print('High loss stations (train):', anomaly_iot_train)
testScores_stations_df = pd.DataFrame.from_dict(testScores_stations, orient='index', columns=['value'])
sigma_3 = (3 * testScores_stations_df.std() + testScores_stations_df.mean()).values[0]
anomaly_iot_test = testScores_stations_df.loc[testScores_stations_df['value'] >= sigma_3]
print('High loss stations (test):', anomaly_iot_test)

'''single station'''
# # extract only one variable
# variable = '060371103_PM2.5'
# uni_data = aggr_df[variable].values.reshape(-1, 1)
# uni_data[uni_data < 0] =0
# print(uni_data.shape)
#
# # find max and min values for normalization
# norm_min = min(uni_data)
# norm_max = max(uni_data)
# print(norm_min, norm_max)
#
# # normalize the data
# uni_data = (uni_data - norm_min) / (norm_max - norm_min)
# print(uni_data.min(), uni_data.max())
#
# # split into train and test
# TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)
#
# past_history = 72
# output_size = 12
#
# x_train, y_train = multistep_lstm_pytorch.univariate_data(uni_data, 0, TRAIN_SPLIT, past_history, output_size)
# x_test, y_test = multistep_lstm_pytorch.univariate_data(uni_data, TRAIN_SPLIT, None, past_history, output_size)
#
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
#
# num_epochs = 120
# epoch_interval = 20
# loss_func, model, optimizer = multistep_lstm_pytorch.initial_model(output_size=output_size)
# train_loss, test_loss = [], []
#
# train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=1000)
# test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=True, batch_size=400)
#
# for idx, data in enumerate(train_loader):
#     print(idx, data[0].shape, data[1].shape)
#
# for epoch in range(num_epochs):
#     loss1 = multistep_lstm_pytorch.train_LSTM(train_loader, model, loss_func, optimizer, epoch)  # calculate train_loss
#     loss2 = multistep_lstm_pytorch.test_LSTM(test_loader, model, loss_func, optimizer, epoch)  # calculate test_loss
#
#     train_loss.extend(loss1)
#     test_loss.extend(loss2)
#
#     if epoch % epoch_interval == 0:
#         print("Epoch: %d, train_loss: %1.5f, test_loss: %1.5f" % (epoch, sum(loss1), sum(loss2)))
#
# plt.plot(train_loss)
# plt.plot(test_loss)
# plt.show()

'''keras'''
# aggr_df = pd.read_csv('/Users/jc/Documents/GitHub/Fresh-Air-LA/data/aggr_la_aq_preprocessed.csv', index_col=False)
# print(aggr_df.head())
#
# vars = list(set(aggr_df.columns[1:]) - set(['datetime']))
#
# sensors = pd.read_csv('/Users/jc/Documents/GitHub/Fresh-Air-LA/data/sensors_la_preprocessed.csv',
#                       index_col=False, dtype=str)
# print(sensors.head())
#
# selected_vars = [var for var in vars if var.split('_')[1] == 'PM2.5']
# print(selected_vars)
#
# # plot the timeseries to have a general view
# selected_df = aggr_df[selected_vars]
# selected_df.index = aggr_df['datetime']
# if selected_df.shape[1] > 5:
#     for i in range(0, selected_df.shape[1], 5):
#         selected_df_plot = selected_df[selected_df.columns[i:(i+5)]]
#         selected_df_plot.plot(subplots=True)
#         plt.show()

# variable = '060371201_PM2.5'
# start = time.time()
#
# multistep_lstm_keras.encoder_decoder_LSTM_univariate(variable)
# # multistep_lstm_keras.encoder_decoder_LSTM_multivariate(variable)
#
# end = time.time()
# print(end - start)
