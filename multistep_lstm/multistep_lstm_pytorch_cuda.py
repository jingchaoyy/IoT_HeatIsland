"""
Created on  9/02/20
@author: Jingchao Yang
"""
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
from prettytable import PrettyTable
from torch import optim
from tqdm import tqdm

aggr_df = pd.read_csv(r'D:\1_GitHub\Fresh-Air-LA\data\aggr_la_aq_preprocessed.csv', index_col=False)
vars = list(set(aggr_df.columns[1:]) - set(['datetime']))
sensors = pd.read_csv(r'D:\1_GitHub\Fresh-Air-LA\data\sensors_la_preprocessed.csv', index_col=False,
                      dtype=str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))

    def forward(self, x):
        # input shape: (batch, seq_len, input_size) (how many sequences, train window, how many inputs)
        # output shape: (seq_len, output_size, input_size)
        self.batch_size = x.size(0)
        self.reset_hidden_state()
        # x = self.dropout(x) # add drop out value
        output, self.hidden = self.lstm(x, self.hidden)
        # Decode the hidden state of the last time step
        y_pred = self.linear(output)[:, -1, :]
        return y_pred  # (seq_len, output_size)


def summary(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(model)
    print(table)
    print(f"Total Trainable Params: {total_params}")

    return total_params


def reshape_add_1d(array):
    array = array.reshape(array.shape[0], array.shape[1], 1)
    return array


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

    # data = reshape_add_1d(data)
    # labels = reshape_add_1d(labels)

    if tensor:
        data = torch.from_numpy(data).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)

    return data, labels


# extract only one variable
variable = '060371103_PM2.5'
uni_data = aggr_df[variable].values.reshape(-1, 1)
uni_data[uni_data < 0] = 0
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

'''model training'''
model = LSTM(input_size=x_train.shape[2],
             hidden_size=45,
             num_layers=2,
             output_size=y_train.shape[1]).to(device)
summary(model)

epochs = 300
loss_func = nn.MSELoss()
LEARNING_RATE = 0.05
MOMENTUM = 0.9

# Define a message template
msg_template = 'Epoch #{: >3}/{: <3}: [train loss mean {:.4} std {:.4}] [validate loss mean {:.4} std {:.4}]'

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

predictions_list = []
train_loss_list, validate_loss_list = [], []

# Model training and validation
for epoch in range(epochs):

    # Train the model for one epoch
    # Turn on the model training mode
    model.train()

    # Initialize a loss list
    train_loss = []
    num_samples = x_train.shape[0]

    samples_index = list(range(num_samples))

    for sample_index in tqdm(range(num_samples)):
        # Get sample
        sample_x = x_train[[samples_index[sample_index]]]
        sample_y = y_train[[samples_index[sample_index]]]

        # Reset gradient
        optimizer.zero_grad()

        # Forward propagation
        y_hat = model(sample_x)

        # Calculate the reconstruction loss
        loss = loss_func(y_hat.unsqueeze(-1), sample_y)

        # Back propagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Append loss
        train_loss.append(loss.item())

    # Calculate the validation loss
    # Turn on the evaluation mode
    model.eval()

    # Initialize a loss list and a list for predictions
    validate_loss = []
    predictions = []
    num_samples = x_test.shape[0]

    with torch.no_grad():
        for sample_index in tqdm(range(num_samples)):
            # Get sample
            sample_x = x_test[[sample_index]]
            sample_y = y_test[[sample_index]]

            # Forward propagation
            y_hat = model(sample_x)

            # Calculate loss
            loss = loss_func(y_hat.unsqueeze(-1), sample_y).item()

            # Append results
            predictions.append(y_hat.unsqueeze(-1).detach().cpu().numpy().reshape(1, y_test.shape[1]))
            validate_loss.append(loss)

    predictions_list.append(predictions)
    train_loss_list.append(np.mean(train_loss))
    validate_loss_list.append(np.mean(validate_loss))

    # Print progress message
    print('Univariate GTN', msg_template.format(
        epoch + 1, int(epochs), np.mean(train_loss), np.std(train_loss),
        np.mean(validate_loss), np.std(validate_loss)))
