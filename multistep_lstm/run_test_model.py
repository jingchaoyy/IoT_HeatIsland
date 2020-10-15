"""
Created on  9/24/2020
@author: Jingchao Yang

Train in La, test in NYC
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
geohash_df = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\NYC\dataHarvest_NYC_202001_03\processed\nodes_missing_10percent.csv',
                         usecols=['Geohash'])
iot_sensors = geohash_df.values.reshape(-1)
iot_df = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\NYC\dataHarvest_NYC_202001_03\processed\preInt_matrix_full.csv',
                     usecols=['datetime'] + iot_sensors.tolist(), index_col=['datetime'])

ext_name = ['humidity', 'windSpeed']
# ext_name = ['humidity', 'windSpeed', 'dewPoint']
ext_data_scaled = []
if multi_variate_mode:
    ext_data_path = r'E:\IoT_HeatIsland_Data\data\NYC\WU_preprocessed_NYC_manzhu\processed\byAttributes'
    for ext in ext_name:
        ext_df = pd.read_csv(ext_data_path + f'\{ext}.csv', index_col=['datetime'])
        while ext_df.isnull().values.any():
            fill_missing(ext_df.values)
        print(f'NaN value in {ext} df?', ext_df.isnull().values.any())
        ext_data_scaled.append(min_max_scaler(ext_df))
    iot_wu_match_df = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\NYC\exp_data\iot_wu_colocate.csv', index_col=0)

while iot_df.isnull().values.any():
    fill_missing(iot_df.values)
print('NaN value in IoT df?', iot_df.isnull().values.any())


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

train_window = 24
output_size = 8


if not multi_variate_mode:
    test_data = multistep_lstm_pytorch.Dataset(iot_df,
                                               (norm_min, norm_max),
                                               train_window,
                                               output_size,
                                               test_station=True)
else:
    test_data = multistep_lstm_pytorch.Dataset_multivariate(iot_df,
                                                            (norm_min, norm_max),
                                                            train_window,
                                                            output_size,
                                                            ext_data_scaled,
                                                            ext_name,
                                                            iot_wu_match_df,
                                                            test_station=True)

print('Number of stations in testing data: ', len(test_data))

print("Testing input and output for each station: %s, %s" % (test_data[0][0].shape, test_data[0][1].shape))

'''initialize the model'''
num_epochs = 15
epoch_interval = 1
# https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
# hidden_size = int((2/3)*(train_window*len(ext_data_scaled)+1))
hidden_size = 6
loss_func, model, optimizer = multistep_lstm_pytorch.initial_model(input_size=test_data[0][0].shape[-1],
                                                                   hidden_size=hidden_size,
                                                                   output_size=output_size,
                                                                   learning_rate=0.001)

model_path = r'E:\IoT_HeatIsland_Data\data\LA\exp_data\result_multi_point_prediction' \
             r'\fillmiss_humidity_windSpeed_6neurons_epoch7_24_8\ep15_neu6_pred8_1600447456.pt'
model.load_state_dict(torch.load(model_path))


# Predict testing dataset of testing stations
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
pred_df.to_csv(r'E:\IoT_HeatIsland_Data\data\NYC\exp_data\output_temp\pred.csv')
pred_df.plot()
plt.xlabel('time (hour)')
plt.ylabel('temperature (F)')
plt.show()

# getting r2 score for mode evaluation
model_score = r2_score(pred_df.pred, pred_df.ori)
print("R^2 Score: ", model_score)

# calculate root mean squared error
testScores_stations, testScores_stations_mae = dict(), dict()


for key in test_data.keys:
    testScores_stations[key] = math.sqrt(mean_squared_error((test_pred_orig_dict[key][0].data.cpu().numpy() - 32) * 5.0/9.0,
                                                            (test_pred_orig_dict[key][1].data.cpu().numpy() - 32) * 5.0/9.0))
    testScores_stations_mae[key] = mean_absolute_error((test_pred_orig_dict[key][0].data.cpu().numpy() - 32) * 5.0/9.0,
                                                       (test_pred_orig_dict[key][1].data.cpu().numpy() - 32) * 5.0/9.0)

print('max test RMSE', max(testScores_stations.values()))
print('min test RMSE', min(testScores_stations.values()))
score_df = pd.DataFrame(testScores_stations.values())
score_df.to_csv(r'E:\IoT_HeatIsland_Data\data\NYC\exp_data\output_temp\testScores.csv')

print('max test MAE', max(testScores_stations_mae.values()))
print('min test MAE', min(testScores_stations_mae.values()))

# using 3-sigma for selecting high loss stations
testScores_stations_df = pd.DataFrame.from_dict(testScores_stations, orient='index', columns=['value'])
sigma_3 = (3 * testScores_stations_df.std() + testScores_stations_df.mean()).values[0]
anomaly_iot_test = testScores_stations_df.loc[testScores_stations_df['value'] >= sigma_3]
print('High loss stations (test):', anomaly_iot_test)