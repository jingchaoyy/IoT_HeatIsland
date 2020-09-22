"""
Created on  8/27/20
@author: Jingchao Yang
"""
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
from sklearn.metrics import mean_squared_error
from statistics import mean
import torch
from torch.utils.data import TensorDataset, DataLoader
from multistep_lstm import multistep_lstm_pytorch
from sklearn import preprocessing
import numpy as np
from numpy import isnan
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


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
geohash_df = pd.read_csv(r'D:\IoT_HeatIsland\exp_data_bak\merged\nodes_missing_5percent.csv',
                         usecols=['Geohash'])
iot_sensors = geohash_df.values.reshape(-1)
iot_df = pd.read_csv(r'D:\IoT_HeatIsland\exp_data_bak\merged\tempMatrix_LA_2019_20.csv',
                     usecols=['datetime'] + iot_sensors.tolist(), index_col=['datetime'])

ext_name = ['humidity', 'windSpeed']
# ext_name = ['humidity', 'windSpeed', 'dewPoint']
ext_data_scaled = []
if multi_variate_mode:
    ext_data_path = r'D:\IoT_HeatIsland\exp_data_bak\WU_preprocessed_LA\processed\byAttributes'
    for ext in ext_name:
        ext_df = pd.read_csv(ext_data_path + f'\{ext}.csv', index_col=['datetime'])
        fill_missing(ext_df.values)
        ext_data_scaled.append(min_max_scaler(ext_df))
    iot_wu_match_df = pd.read_csv(r'D:\IoT_HeatIsland\exp_data_bak\merged\iot_wu_colocate.csv', index_col=0)

fill_missing(iot_df.values)

'''all stations'''
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
train_stations = set(np.random.choice(selected_vars, int(len(selected_vars) * 0.7), replace=False))
test_stations = set(selected_vars) - train_stations

train_data_raw = iot_df[train_stations]
test_data_raw = iot_df[test_stations]

print(train_data_raw.shape)
print(test_data_raw.shape)
print(train_data_raw.columns)

train_window = 24
output_size = 1


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
                                                             ext_name,
                                                             iot_wu_match_df)
    test_data = multistep_lstm_pytorch.Dataset_multivariate(test_data_raw,
                                                            (norm_min, norm_max),
                                                            train_window,
                                                            output_size,
                                                            ext_data_scaled,
                                                            ext_name,
                                                            iot_wu_match_df,
                                                            test_station=True)

print('Number of stations in training data: ', len(train_data))
print('Number of stations in testing data: ', len(test_data))

print("Training input and output for each station: %s, %s" % (train_data[0][0].shape, train_data[0][1].shape))
print("Validation input and output for each station: %s, %s" % (train_data[0][2].shape, train_data[0][3].shape))
print("Testing input and output for each station: %s, %s" % (test_data[0][0].shape, test_data[0][1].shape))

# initialize the model
num_epochs = 15
epoch_interval = 1
# https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
# hidden_size = int((2/3)*(train_window*len(ext_data_scaled)+1))
hidden_size = 6
loss_func, model, optimizer = multistep_lstm_pytorch.initial_model(input_size=train_data[0][0].shape[-1],
                                                                   hidden_size=hidden_size,
                                                                   output_size=output_size,
                                                                   learning_rate=0.001)
train_loss, test_loss, mean_loss_train, mean_test_loss = [], [], [], []
min_val_loss, mean_min_val_loss = np.Inf, np.Inf
n_epochs_stop = 3
epochs_no_improve = 0
early_stop = False

start = time.time()
# train the model
for epoch in range(num_epochs):
    running_loss_train = []
    running_loss_val = []
    loss2 = 0
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

        if mean(loss2) < min_val_loss:
            # Save the model
            # torch.save(model)
            epochs_no_improve = 0
            min_val_loss = mean(loss2)

        else:
            epochs_no_improve += 1

        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            early_stop = True
            break
        else:
            continue

    mean_loss_train.append(mean(running_loss_train))
    mean_test_loss.append(mean(running_loss_val))
    if epoch % epoch_interval == 0:
        print(
            "Epoch: %d, train_loss: %1.5f, val_loss: %1.5f" % (epoch, mean(running_loss_train), mean(running_loss_val)))
        if mean(running_loss_val) < mean_min_val_loss:
            mean_min_val_loss = mean(running_loss_val)
        else:
            print('Early stopping!')
            early_stop = True
    if early_stop:
        print("Stopped")
        break

end = time.time()
print(end - start)

print(model)

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()

plt.plot(mean_loss_train)
plt.plot(mean_test_loss)
plt.show()

# save trained model
modelName = int(time.time())
torch.save(model.state_dict(), r'D:\1_GitHub\IoT_HeatIsland\multistep_lstm\saved_models'
                               f'\\ep{num_epochs}_neu{hidden_size}_pred{output_size}_{modelName}.pt')
print('model saved')

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
d = {'ori': test_pred_orig_dict[list(test_pred_orig_dict.keys())[0]][1][:, 0].data.tolist(),
     'pred': test_pred_orig_dict[list(test_pred_orig_dict.keys())[0]][0][:, 0].data.tolist()}
pred_df = pd.DataFrame(data=d)
pred_df.to_csv(r'D:\1_GitHub\IoT_HeatIsland\multistep_lstm\saved_models\pred.csv')
pred_df.plot()
plt.xlabel('time (hour)')
plt.ylabel('temperature (F)')
plt.show()

# getting r2 score for mode evaluation
model_score = r2_score(pred_df.pred, pred_df.ori)
print("R^2 Score: ", model_score)

# calculate root mean squared error
trainScores_stations, trainScores_stations_mae = dict(), dict()
testScores_stations, testScores_stations_mae = dict(), dict()

for key in train_data.keys:
    trainScores_stations[key] = math.sqrt(mean_squared_error(train_pred_orig_dict[key][0].data.tolist(),
                                                             train_pred_orig_dict[key][1].data.tolist()))
    testScores_stations_mae[key] = mean_absolute_error(train_pred_orig_dict[key][0].data.tolist(),
                                                       train_pred_orig_dict[key][1].data.tolist())

for key in test_data.keys:
    testScores_stations[key] = math.sqrt(mean_squared_error(test_pred_orig_dict[key][0].data.tolist(),
                                                            test_pred_orig_dict[key][1].data.tolist()))
    testScores_stations_mae[key] = mean_absolute_error(test_pred_orig_dict[key][0].data.tolist(),
                                                       test_pred_orig_dict[key][1].data.tolist())

print('max train RMSE', max(trainScores_stations.values()))
print('min train RMSE', min(trainScores_stations.values()))
score_df = pd.DataFrame(trainScores_stations.values())
score_df.to_csv(r'D:\1_GitHub\IoT_HeatIsland\multistep_lstm\saved_models\trainScores.csv')

print('max test RMSE', max(testScores_stations.values()))
print('min test RMSE', min(testScores_stations.values()))
score_df = pd.DataFrame(testScores_stations.values())
score_df.to_csv(r'D:\1_GitHub\IoT_HeatIsland\multistep_lstm\saved_models\testScores.csv')

print('max train MAE', max(testScores_stations_mae.values()))
print('min train MAE', min(testScores_stations_mae.values()))
print('max test MAE', max(testScores_stations_mae.values()))
print('min test MAE', min(testScores_stations_mae.values()))

# using 3-sigma for selecting high loss stations
trainScores_stations_df = pd.DataFrame.from_dict(trainScores_stations, orient='index', columns=['value'])
sigma_3 = (3 * trainScores_stations_df.std() + trainScores_stations_df.mean()).values[0]
anomaly_iot_train = trainScores_stations_df.loc[trainScores_stations_df['value'] >= sigma_3]
print('High loss stations (train):', anomaly_iot_train)
testScores_stations_df = pd.DataFrame.from_dict(testScores_stations, orient='index', columns=['value'])
sigma_3 = (3 * testScores_stations_df.std() + testScores_stations_df.mean()).values[0]
anomaly_iot_test = testScores_stations_df.loc[testScores_stations_df['value'] >= sigma_3]
print('High loss stations (test):', anomaly_iot_test)


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
