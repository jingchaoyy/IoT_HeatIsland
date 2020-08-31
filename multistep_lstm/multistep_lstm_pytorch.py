"""
Created on  8/27/20
@author: Jingchao Yang
"""
from platform import python_version
import matplotlib
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

aggr_df = pd.read_csv('/Users/jc/Documents/GitHub/Fresh-Air-LA/data/aggr_la_aq_preprocessed.csv', index_col=False)
vars = list(set(aggr_df.columns[1:]) - set(['datetime']))
sensors = pd.read_csv('/Users/jc/Documents/GitHub/Fresh-Air-LA/data/sensors_la_preprocessed.csv', index_col=False,
                      dtype=str)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def reset_hidden_state(self):
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, x):
        # input shape: (batch, seq_len, input_size) (how many sequences, train window, how many inputs)
        # output shape: (seq_len, output_size, input_size)
        self.batch_size = x.size(0)
        self.reset_hidden_state()
        output, self.hidden = self.lstm(x, self.hidden)
        # Decode the hidden state of the last time step
        y_pred = self.linear(output)[:, -1, :]
        return y_pred  # (seq_len, output_size)


def initial_model(hidden_size=30, num_layers=2, learning_rate=0.05):
    loss_func = torch.nn.MSELoss()  # mean-squared error for regression
    model = LSTM(1, hidden_size, num_layers, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_func, model, optimizer


def train_LSTM(dataloader, model, loss_func, optimizer, epoch):
    model.train()
    loss_list = []
    for idx, data in enumerate(dataloader):
        y_pred = model(data[0])
        optimizer.zero_grad()
        # obtain the loss function
        loss = loss_func(y_pred, data[1].reshape(y_pred.shape))
        loss.backward()
        optimizer.step()
        # record loss
        loss_list.append(loss.item())
    return loss_list


def test_LSTM(dataloader, model, loss_func, optimizer, epoch):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            y_pred = model(data[0])
            loss = loss_func(y_pred, data[1].reshape(y_pred.shape))
            loss_list.append(loss.item())
    return loss_list


def univariate_data(dataset, start_index, end_index, history_size, target_size, tensor=True):
    # The parameter history_size is the size of the past window of information.
    # The target_size is how far in the future does the model need to learn to predict.
    # The target_size is the label that needs to be predicted.
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i: i + target_size])

    data = np.array(data)
    labels = np.array(labels)

    if tensor:
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).float()

    return data, labels


# extract only one variable
variable = '060371103_PM2.5'
uni_data = aggr_df[variable].values.reshape(-1, 1)
uni_data[uni_data < 0] =0
print(uni_data.shape)

# find max and min values for normalization
norm_min = min(uni_data)
norm_max = max(uni_data)
print(norm_min, norm_max)

# normalize the data
uni_data = (uni_data - norm_min) / (norm_max - norm_min)
print(uni_data.min(), uni_data.max())

# split into train and test
TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)

past_history = 72
output_size = 12

x_train, y_train = univariate_data(uni_data, 0, TRAIN_SPLIT, past_history, output_size)
x_test, y_test = univariate_data(uni_data, TRAIN_SPLIT, None, past_history, output_size)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

num_epochs = 300
epoch_interval = 20
loss_func, model, optimizer = initial_model()
train_loss, test_loss = [], []

train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=1000)
test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=True, batch_size=400)

for idx, data in enumerate(train_loader):
    print(idx, data[0].shape, data[1].shape)

for epoch in range(num_epochs):
    loss1 = train_LSTM(train_loader, model, loss_func, optimizer, epoch)  # calculate train_loss
    loss2 = test_LSTM(test_loader, model, loss_func, optimizer, epoch)  # calculate test_loss

    train_loss.extend(loss1)
    test_loss.extend(loss2)

    if epoch % epoch_interval == 0:
        print("Epoch: %d, train_loss: %1.5f, test_loss: %1.5f" % (epoch, sum(loss1), sum(loss2)))
