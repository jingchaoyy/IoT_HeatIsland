"""
Created on  9/22/20
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
from singlestep_all import get_data
from sklearn.metrics import mean_absolute_error

# extract only one variable
# aggr_df = pd.read_csv('/Users/jc/Documents/GitHub/Fresh-Air-LA/data/aggr_la_aq_preprocessed.csv', index_col=False)
# variable = '060371103_PM2.5'
# uni_data = aggr_df[variable].values.reshape(-1, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iot_sensors, iot_df = get_data.get_data()

for coor in [iot_sensors[0]]:
    # s = iot_df[coor]
    # data = np.array(s)
    uni_data = iot_df[coor].values.reshape(-1, 1)

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
    TRAIN_SPLIT = int(iot_df.shape[0] * 0.75)

    past_history = 24
    output_size = 1

    x_train, y_train = multistep_lstm_pytorch.univariate_data(uni_data, 0, TRAIN_SPLIT, past_history, output_size)
    x_test, y_test = multistep_lstm_pytorch.univariate_data(uni_data, TRAIN_SPLIT, None, past_history, output_size)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    '''Train model'''
    num_epochs = 1000
    epoch_interval = 20
    hidden_size = 6
    loss_func, model, optimizer = multistep_lstm_pytorch.initial_model(input_size=1,
                                                                       hidden_size=hidden_size,
                                                                       output_size=output_size,
                                                                       learning_rate=0.001)
    train_loss, test_loss = [], []

    train_loader = DataLoader(TensorDataset(x_train.to(device), y_train.to(device)), shuffle=True, batch_size=1000)
    test_loader = DataLoader(TensorDataset(x_test.to(device), y_test.to(device)), shuffle=True, batch_size=400)

    for idx, data in enumerate(train_loader):
        print(idx, data[0].shape, data[1].shape)

    for epoch in range(num_epochs):
        loss1 = multistep_lstm_pytorch.train_LSTM(train_loader, model, loss_func, optimizer,
                                                  epoch)  # calculate train_loss
        loss2 = multistep_lstm_pytorch.test_LSTM(test_loader, model, loss_func, optimizer, epoch)  # calculate test_loss

        train_loss.extend(loss1)
        test_loss.extend(loss2)

        if epoch % epoch_interval == 0:
            print("Epoch: %d, train_loss: %1.5f, test_loss: %1.5f" % (epoch, sum(loss1), sum(loss2)))

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.show()

    '''the model is in eval() mode at the end of training process'''
    # model.training
    # model.eval()
    with torch.no_grad():
        train_pred = model(x_train)
        train_pred_trans = ((train_pred.cpu().detach().numpy() * (norm_max - norm_min) + norm_min) - 32) * 5.0/9.0
        trainY_trans = ((y_train.reshape(train_pred.shape).cpu().detach().numpy() * (norm_max - norm_min) + norm_min) - 32) * 5.0/9.0

        test_pred = model(x_test)
        test_pred_trans = ((test_pred.cpu().detach().numpy() * (norm_max - norm_min) + norm_min) - 32) * 5.0/9.0
        testY_trans = ((y_test.reshape(test_pred.shape).cpu().detach().numpy() * (norm_max - norm_min) + norm_min) - 32) * 5.0/9.0

    # plot baseline and predictions
    plt.plot(trainY_trans[:, 0])
    plt.plot(train_pred_trans[:, 0])
    plt.show()

    # plot baseline and predictions
    plt.plot(testY_trans[:, 0])
    plt.plot(test_pred_trans[:, 0])
    plt.show()

    '''calculate root mean squared error'''
    trainScore = math.sqrt(mean_squared_error(trainY_trans, train_pred_trans))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY_trans, test_pred_trans))
    print('Test Score: %.2f RMSE' % (testScore))

    trainScore_mae = mean_absolute_error(trainY_trans, train_pred_trans)
    print('Train Score: %.2f MAE' % (trainScore_mae))
    testScore_mae = mean_absolute_error(testY_trans, test_pred_trans)
    print('Test Score: %.2f MAE' % (testScore_mae))

    '''save to csv'''
    save_path = '/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA/exp_data/result_single_point_prediction/lstm/'
    testscore_dict = {'ori': testY_trans[:,0],
                      'pred': test_pred_trans[:,0]}
    testscore_df = pd.DataFrame(data=testscore_dict)
    testscore_df.to_csv(save_path + 'pred.csv')
