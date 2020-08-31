"""
Created on  8/27/20
@author: Jingchao Yang
"""
from platform import python_version

import pandas as pd
import numpy as np
import sklearn
import tensorflow
import matplotlib
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, ConvLSTM2D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.optimizers import RMSprop

aggr_df = pd.read_csv('/Users/jc/Documents/GitHub/Fresh-Air-LA/data/aggr_la_aq_preprocessed.csv', index_col=False)
print(aggr_df.head())

vars = list(set(aggr_df.columns[1:]) - set(['datetime']))

sensors = pd.read_csv('/Users/jc/Documents/GitHub/Fresh-Air-LA/data/sensors_la_preprocessed.csv', index_col=False,
                      dtype=str)
print(sensors.head())

selected_vars = [var for var in vars if var.split('_')[1] == 'PM2.5']
print(selected_vars)


def univariate_data(dataset, start_index, end_index, history_size, target_size):
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
    return np.array(data), np.array(labels)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    # The below function performs the same windowing task as below,
    # however, here it samples the past observation based on the step size given.
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def build_model(x_train_uni, y_train_uni, x_val_uni, y_val_uni):
    # define parameters
    verbose, epochs, batch_size = 0, 10, 16
    n_timesteps, n_features, n_outputs = x_train_uni.shape[1], x_train_uni.shape[2], y_train_uni.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae')
    # fit network
    history = model.fit(x_train_uni, y_train_uni, validation_data=(x_val_uni, y_val_uni), epochs=epochs,
                        batch_size=batch_size, verbose=verbose)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    return model


def build_encoder_decoder_LSTM_model(x_train_uni, y_train_uni, x_val_uni, y_val_uni):
    # define parameters
    verbose, epochs, batch_size = 0, 10, 8
    n_timesteps, n_features, n_outputs = x_train_uni.shape[1], x_train_uni.shape[2], y_train_uni.shape[1]
    # reshape output into [samples, timesteps, features]
    y_train_uni = y_train_uni.reshape((y_train_uni.shape[0], y_train_uni.shape[1], 1))
    y_val_uni = y_val_uni.reshape((y_val_uni.shape[0], y_val_uni.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(y_val_uni.shape[1]))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae')
    # fit network
    history = model.fit(x_train_uni, y_train_uni, validation_data=(x_val_uni, y_val_uni), epochs=epochs,
                        batch_size=batch_size, verbose=verbose)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    return model


def build_CNN_LSTM_model(x_train_uni, y_train_uni, x_val_uni, y_val_uni):
    # define parameters
    verbose, epochs, batch_size = 0, 20, 16
    n_timesteps, n_features, n_outputs = x_train_uni.shape[1], x_train_uni.shape[2], y_train_uni.shape[1]
    # reshape output into [samples, timesteps, features]
    y_train_uni = y_train_uni.reshape((y_train_uni.shape[0], y_train_uni.shape[1], 1))
    y_val_uni = y_val_uni.reshape((y_val_uni.shape[0], y_val_uni.shape[1], 1))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(x_train_uni, y_train_uni, validation_data=(x_val_uni, y_val_uni), epochs=epochs,
                        batch_size=batch_size, verbose=verbose)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    return model


def build_ConvLSTM_model(x_train_uni, y_train_uni, x_val_uni, y_val_uni):
    # define parameters
    verbose, epochs, batch_size = 0, 20, 16
    n_timesteps, n_features, n_outputs = x_train_uni.shape[1], x_train_uni.shape[2], y_train_uni.shape[1]
    # reshape output into [samples, timesteps, features]
    y_train_uni = y_train_uni.reshape((y_train_uni.shape[0], y_train_uni.shape[1], 1))
    y_val_uni = y_val_uni.reshape((y_val_uni.shape[0], y_val_uni.shape[1], 1))
    # define model
    model = Sequential()
    model.add(
        ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(
            x_train_uni.shape[1], x_train_uni.shape[2], x_train_uni.shape[3], x_train_uni.shape[4])))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(x_train_uni, y_train_uni, validation_data=(x_val_uni, y_val_uni), epochs=epochs,
                        batch_size=batch_size, verbose=verbose)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    return model


def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each target batch
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def smape(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)

    return 2 * np.mean(
        np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def vanilla_LSTM_univariate(variable='060371103_PM2.5'):
    uni_data = aggr_df[variable]
    uni_data = uni_data.values.reshape(-1, 1)
    # create scaler
    scaler = MinMaxScaler()
    # fit scaler on data
    scaler.fit(uni_data)
    # apply transform
    uni_data = scaler.transform(uni_data)

    # split into train and test
    TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)

    univariate_past_history = 72
    univariate_future_target = 12

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    # evaluate model and get scores
    model = build_model(x_train_uni, y_train_uni.reshape((y_train_uni.shape[0], y_train_uni.shape[1])), x_val_uni,
                        y_val_uni.reshape((y_val_uni.shape[0], y_val_uni.shape[1])))
    predictions = model.predict(x_val_uni)
    predictions = scaler.inverse_transform(predictions.reshape((predictions.shape[0], predictions.shape[1])))
    score, scores = evaluate_forecasts(y_val_uni, predictions)
    smape_score = smape(y_val_uni.reshape((predictions.shape[0], predictions.shape[1])), predictions)
    print('vanilla_LSTM_univariate_smape', smape_score)

    # summarize scores
    summarize_scores('vanilla_LSTM_univariate_rmse', score, scores)

    # plot scores
    hours = np.arange(1, 13)
    plt.plot(hours, scores, marker='o', label='vanilla_LSTM_univariate_rmse')
    plt.show()


def vanilla_LSTM_multivariate(variable='060371103_PM2.5'):
    selected_vars = [var for var in vars if variable.split('_')[0] in var]
    dataset = aggr_df[selected_vars]

    dataset = dataset.values
    # create scaler
    x_scaler = MinMaxScaler()
    # fit scaler on data
    x_scaler.fit(dataset)
    # apply transform
    dataset = x_scaler.transform(dataset)

    y = aggr_df[variable].values.reshape(-1, 1)
    # create scaler
    y_scaler = MinMaxScaler()
    # fit scaler on data
    y_scaler.fit(y)
    # apply transform
    y = y_scaler.transform(y)

    # split into train and test
    TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)

    past_history = 3 * 24
    future_target = 12
    STEP = 1

    x_train, y_train = multivariate_data(dataset, y, 0,
                                         TRAIN_SPLIT, past_history,
                                         future_target, STEP,
                                         single_step=False)
    x_val, y_val = multivariate_data(dataset, y,
                                     TRAIN_SPLIT, None, past_history,
                                     future_target, STEP,
                                     single_step=False)

    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1])

    # evaluate model and get scores
    model = build_model(x_train, y_train, x_val, y_val)
    predictions = model.predict(x_val)
    predictions = y_scaler.inverse_transform(predictions.reshape(predictions.shape[0], predictions.shape[1]))
    score, scores = evaluate_forecasts(y_val, predictions)
    smape_score = smape(y_val.reshape((predictions.shape[0], predictions.shape[1])), predictions)
    print('vanilla_LSTM_multivariate_smape', smape_score)

    # summarize scores
    summarize_scores('vanilla_LSTM_multivariate_rmse', score, scores)

    # plot scores
    hours = np.arange(1, 13)
    plt.plot(hours, scores, marker='o', label='lstm')
    plt.show()


def encoder_decoder_LSTM_univariate(variable='060371103_PM2.5'):
    uni_data = aggr_df[variable]
    uni_data = uni_data.values.reshape(-1, 1)
    # create scaler
    scaler = MinMaxScaler()
    # fit scaler on data
    scaler.fit(uni_data)
    # apply transform
    uni_data = scaler.transform(uni_data)

    # split into train and test
    TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)

    univariate_past_history = 72
    univariate_future_target = 12

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    # evaluate model and get scores
    model = build_encoder_decoder_LSTM_model(x_train_uni, y_train_uni, x_val_uni, y_val_uni)
    predictions = model.predict(x_val_uni)
    predictions = scaler.inverse_transform(predictions.reshape((predictions.shape[0], predictions.shape[1])))
    score, scores = evaluate_forecasts(y_val_uni, predictions)
    smape_score = smape(y_val_uni.reshape((predictions.shape[0], predictions.shape[1])), predictions)
    print('encoder_decoder_LSTM_univariate_smape', smape_score)

    # summarize scores
    summarize_scores('encoder_decoder_LSTM_univariate_rmse', score, scores)

    # plot scores
    hours = np.arange(1, 13)
    plt.plot(hours, scores, marker='o', label='lstm')
    plt.show()


def encoder_decoder_LSTM_multivariate(variable='060371103_PM2.5'):
    selected_vars = [var for var in vars if variable.split('_')[0] in var]
    dataset = aggr_df[selected_vars]

    dataset = dataset.values
    # create scaler
    x_scaler = MinMaxScaler()
    # fit scaler on data
    x_scaler.fit(dataset)
    # apply transform
    dataset = x_scaler.transform(dataset)

    y = aggr_df[variable].values.reshape(-1, 1)
    # create scaler
    y_scaler = MinMaxScaler()
    # fit scaler on data
    y_scaler.fit(y)
    # apply transform
    y = y_scaler.transform(y)

    # split into train and test
    TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)

    past_history = 3 * 24
    future_target = 12
    STEP = 1

    x_train, y_train = multivariate_data(dataset, y, 0,
                                         TRAIN_SPLIT, past_history,
                                         future_target, STEP,
                                         single_step=False)
    x_val, y_val = multivariate_data(dataset, y,
                                     TRAIN_SPLIT, None, past_history,
                                     future_target, STEP,
                                     single_step=False)

    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1])

    # evaluate model and get scores
    model = build_encoder_decoder_LSTM_model(x_train, y_train, x_val, y_val)
    predictions = model.predict(x_val)
    predictions = y_scaler.inverse_transform(predictions.reshape(predictions.shape[0], predictions.shape[1]))
    score, scores = evaluate_forecasts(y_val, predictions)
    smape_score = smape(y_val.reshape((predictions.shape[0], predictions.shape[1])), predictions)
    print('encoder_decoder_LSTM_multivariate_smape', smape_score)

    # summarize scores
    summarize_scores('encoder_decoder_LSTM_multivariate_rmse', score, scores)

    # plot scores
    hours = np.arange(1, 13)
    plt.plot(hours, scores, marker='o', label='lstm')
    plt.show()


def CNN_LSTM_univariate(variable='060371103_PM2.5'):
    uni_data = aggr_df[variable]
    uni_data = uni_data.values.reshape(-1, 1)
    # create scaler
    scaler = MinMaxScaler()
    # fit scaler on data
    scaler.fit(uni_data)
    # apply transform
    uni_data = scaler.transform(uni_data)

    # split into train and test
    TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)

    univariate_past_history = 72
    univariate_future_target = 12

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    # evaluate model and get scores
    model = build_CNN_LSTM_model(x_train_uni, y_train_uni, x_val_uni, y_val_uni)
    predictions = model.predict(x_val_uni)
    predictions = scaler.inverse_transform(predictions.reshape((predictions.shape[0], predictions.shape[1])))
    score, scores = evaluate_forecasts(y_val_uni, predictions)
    smape_score = smape(y_val_uni.reshape((predictions.shape[0], predictions.shape[1])), predictions)
    print('CNN_LSTM_univariate_smape', smape_score)

    # summarize scores
    summarize_scores('CNN_LSTM_univariate_rmse', score, scores)

    # plot scores
    hours = np.arange(1, 13)
    plt.plot(hours, scores, marker='o', label='lstm')
    plt.show()


def CNN_LSTM_multivariate(variable='060371103_PM2.5'):
    selected_vars = [var for var in vars if variable.split('_')[0] in var]
    dataset = aggr_df[selected_vars]

    dataset = dataset.values
    # create scaler
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    # fit scaler on data
    x_scaler.fit(dataset)
    # apply transform
    dataset = x_scaler.transform(dataset)

    y = aggr_df[variable].values.reshape(-1, 1)
    # create scaler
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    # fit scaler on data
    y_scaler.fit(y)
    # apply transform
    y = y_scaler.transform(y)

    # split into train and test
    TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)

    past_history = 3 * 24
    future_target = 12
    STEP = 1

    x_train, y_train = multivariate_data(dataset, y, 0,
                                         TRAIN_SPLIT, past_history,
                                         future_target, STEP,
                                         single_step=False)
    x_val, y_val = multivariate_data(dataset, y,
                                     TRAIN_SPLIT, None, past_history,
                                     future_target, STEP,
                                     single_step=False)

    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1])

    # evaluate model and get scores
    model = build_CNN_LSTM_model(x_train, y_train, x_val, y_val)
    # model = build_model(x_train, y_train, x_val, y_val)
    predictions = model.predict(x_val)
    predictions = y_scaler.inverse_transform(predictions.reshape(predictions.shape[0], predictions.shape[1]))
    score, scores = evaluate_forecasts(y_val, predictions)
    smape_score = smape(y_val.reshape((predictions.shape[0], predictions.shape[1])), predictions)
    print('CNN_LSTM_multivariate_smape', smape_score)

    # summarize scores
    summarize_scores('CNN_LSTM_multivariate_rmse', score, scores)

    # plot scores
    hours = np.arange(1, 13)
    plt.plot(hours, scores, marker='o', label='lstm')
    plt.show()


def ConvLSTM_univariate(variable='060371103_PM2.5'):
    uni_data = aggr_df[variable]
    uni_data = uni_data.values.reshape(-1, 1)
    # create scaler
    scaler = MinMaxScaler()
    # fit scaler on data
    scaler.fit(uni_data)
    # apply transform
    uni_data = scaler.transform(uni_data)

    # split into train and test
    TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)

    univariate_past_history = 72
    univariate_future_target = 12

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    x_train_uni = x_train_uni.reshape((x_train_uni.shape[0], 6, 1, 12, 1))
    x_val_uni = x_val_uni.reshape((x_val_uni.shape[0], 6, 1, 12, 1))

    # evaluate model and get scores
    model = build_ConvLSTM_model(x_train_uni, y_train_uni, x_val_uni, y_val_uni)
    predictions = model.predict(x_val_uni)
    predictions = scaler.inverse_transform(predictions.reshape((predictions.shape[0], predictions.shape[1])))
    score, scores = evaluate_forecasts(y_val_uni, predictions)
    smape_score = smape(y_val_uni.reshape((predictions.shape[0], predictions.shape[1])), predictions)
    print('ConvLSTM_univariate_smape', smape_score)

    # summarize scores
    summarize_scores('ConvLSTM_univariate_rmse', score, scores)

    # plot scores
    hours = np.arange(1, 13)
    plt.plot(hours, scores, marker='o', label='lstm')
    plt.show()


def ConvLSTM_multivariate(variable='060371103_PM2.5'):
    selected_vars = [var for var in vars if variable.split('_')[0] in var]
    dataset = aggr_df[selected_vars]

    dataset = dataset.values
    # create scaler
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    # fit scaler on data
    x_scaler.fit(dataset)
    # apply transform
    dataset = x_scaler.transform(dataset)

    y = aggr_df[variable].values.reshape(-1, 1)
    # create scaler
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    # fit scaler on data
    y_scaler.fit(y)
    # apply transform
    y = y_scaler.transform(y)

    # split into train and test
    TRAIN_SPLIT = int(aggr_df.shape[0] * 0.7)

    past_history = 3 * 24
    future_target = 12
    STEP = 1

    x_train, y_train = multivariate_data(dataset, y, 0,
                                         TRAIN_SPLIT, past_history,
                                         future_target, STEP,
                                         single_step=False)
    x_val, y_val = multivariate_data(dataset, y,
                                     TRAIN_SPLIT, None, past_history,
                                     future_target, STEP,
                                     single_step=False)

    x_train = x_train.reshape((x_train.shape[0], 6, 1, 12, x_train.shape[2]))
    x_val = x_val.reshape((x_val.shape[0], 6, 1, 12, x_val.shape[2]))

    # evaluate model and get scores
    model = build_ConvLSTM_model(x_train, y_train, x_val, y_val)
    predictions = model.predict(x_val)
    predictions = y_scaler.inverse_transform(predictions.reshape(predictions.shape[0], predictions.shape[1]))
    score, scores = evaluate_forecasts(y_val, predictions)
    smape_score = smape(y_val.reshape((predictions.shape[0], predictions.shape[1])), predictions)
    print('ConvLSTM_multivariate_smape', smape_score)

    # summarize scores
    summarize_scores('ConvLSTM_multivariate_rmse', score, scores)

    # plot scores
    hours = np.arange(1, 13)
    plt.plot(hours, scores, marker='o', label='lstm')
    plt.show()
