"""
Created on  2019-07-17
@author: Jingchao Yang

Standard Average model and Exponential Moving Average model 
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(os.path.join('tempRecord_byCol.csv'), usecols=['time', '0'])
df = df.sort_values('time')
print(df.head())

# plt.figure(figsize=(18, 9))
# plt.plot(range(df.shape[0]), (df['0']))
# plt.xticks(range(0, df.shape[0], 500), df['time'].loc[::500], rotation=45)
# plt.xlabel('Time', fontsize=18)
# plt.ylabel('Temp. (F)', fontsize=18)
# plt.show()

# example using sensor 0
sensor0 = df.loc[:, '0'].as_matrix()

# split the training data and test data
split = int(len(sensor0) / 10)
train_data = sensor0[:-split]
test_data = sensor0[-split:]

# Scale the data to be between 0 and 1
# When scaling remember! You normalize both test and train data with respect to training data
# When scaling remember! You normalize both test and train data with respect to training data
# Because you are not supposed to have access to test data
scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

# Train the Scaler with training data and smooth data
smoothing_window_size = 150
for di in range(0, 600, smoothing_window_size):
    scaler.fit(train_data[di:di + smoothing_window_size, :])
    train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

# You normalize the last bit of remaining data
scaler.fit(train_data[di + smoothing_window_size:, :])
train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)

# Now perform exponential moving average (EMA) smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(train_data.size):
    EMA = gamma * train_data[ti] + (1 - gamma) * EMA
    train_data[ti] = EMA

# Used for visualization and test purposes
all_data = np.concatenate([train_data, test_data], axis=0)

'''Standard Average Model'''
window_size = 6  # using the average of every previous 6 hours for the nex hour prediction
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size, N):
    hour = pd.to_datetime(df.time, format="%Y%m%d%H")
    std_avg_predictions.append(np.mean(train_data[pred_idx - window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx]) ** 2)
    std_avg_x.append(hour)

print('MSE error for standard averaging: %.5f' % (0.5 * np.mean(mse_errors)))

# plt.figure(figsize=(18, 9))
# plt.plot(range(df.shape[0]), all_data, color='b', label='True')
# plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Prediction')
# # plt.xticks(range(0,df.shape[0],50),df['time'].loc[::50],rotation=45)
# plt.xlabel('Time')
# plt.ylabel('Temp. (F)')
# plt.legend(fontsize=18)
# plt.show()

'''Exponential Moving Average'''
N = train_data.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

# Calculates the exponential moving average from t+1 time step and uses that as the one step ahead prediction
for pred_idx in range(1, N):
    running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx]) ** 2)
    run_avg_x.append(hour)

print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))

plt.figure(figsize=(18, 9))
plt.plot(range(df.shape[0]), all_data, color='b', label='True')
plt.plot(range(0, N), run_avg_predictions, color='orange', label='Prediction')
# plt.xticks(range(0, df.shape[0], 50), df['time'].loc[::50], rotation=45)
plt.title('Exponential Moving Average')
plt.xlabel('Time')
plt.ylabel('Temp. (F)')
plt.legend(fontsize=18)
plt.show()
