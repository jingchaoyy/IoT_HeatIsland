"""
Created on  2019-08-22
@author: Jingchao Yang
"""
import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import datetime as dt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import newaxis
import numpy as np
import pandas as pd
from keras.layers import Masking
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.title('prediction length ' + str(prediction_len))
        plt.legend()
    plt.show()


def predict_sequences_multiple(in_model, test_data, window_size, prediction_len):
    # Predict sequence of n steps before shifting prediction run forward by n steps
    print('[Model] Predicting Sequences Multiple...')
    prediction_seqs = []
    for i in range(int(len(test_data) / prediction_len)):
        curr_frame = test_data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(in_model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def cnn_lstm_predict(in_model, test_data, window_size, prediction_len):
    print('[Model] Predicting CNN_LSTM Multiple...')
    prediction_seqs = []


def predict_point_by_point(in_model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('[Model] Predicting Point-by-Point...')
    predicted = in_model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def get_data(data, seq_len, normalise):
    '''
    Create x, y test/train data windows
    Warning: batch method, not generative, make sure you have enough memory to
    load data, otherwise reduce size of the training split.
    '''
    data_windows, x, y = [], [], []

    for i in range(data.shape[-1] - seq_len):
        data_windows.append(data[:, :, i:i + seq_len])

    data_windows = np.array(data_windows).astype(float)
    data_windows, data_normalizers = normalise_windows(data_windows, single_window=False, norm=normalise)

    x = data_windows[:, :-1]
    y = data_windows[:, -1]
    # y = y[:, [int(y.shape[1] / 2)], [int(y.shape[-1] / 2)]]  # Getting only the centroid point value

    temp_normalizers = data_normalizers[:, 0]
    return x, y, temp_normalizers


def normalise_windows(window_data, single_window=False, norm=True):
    '''Normalise window with a base value of zero'''
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


def main():
    configs = json.load(open('config.json', 'r'))
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
    coor = ['9q5csxx', '9q5csz8', '9q5csz9', '9q5cszd', '9q5csze', '9q5cszs', '9q5cszt', '9q5csxr', '9q5csz2',
            '9q5csz3', '9q5csz6', '9q5csz7', '9q5cszk', '9q5cszm', '9q5csxp', '9q5csz0', '9q5csz1', '9q5csz4',
            '9q5csz5', '9q5cszh', '9q5cszj', '9q5cswz', '9q5csyb', '9q5csyc', '9q5csyf', '9q5csyg', '9q5csyu',
            '9q5csyv', '9q5cswx', '9q5csy8', '9q5csy9', '9q5csyd', '9q5csye', '9q5csys', '9q5csyt', '9q5cswr',
            '9q5csy2', '9q5csy3', '9q5csy6', '9q5csy7', '9q5csyk', '9q5csym', '9q5cswp', '9q5csy0', '9q5csy1',
            '9q5csy4', '9q5csy5', '9q5csyh', '9q5csyj']

    # data = pd.read_csv('/Users/jc/Documents/GitHub/IoT_HeatIsland_Data/data/LA/joined_49_fillna_1.csv', usecols=coor)
    data = pd.read_csv('/Users/jc/Documents/GitHub/IoT_HeatIsland_Data/data/LA/tempMatrix_LA.csv', usecols=coor)

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

    x_train = x_train[:, :, :, :, np.newaxis]
    x_test = x_test[:, :, :, :, np.newaxis]

    shape = x_train.shape[1:]
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, 3, 3, border_mode='same'), input_shape=shape))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, 3, 3, border_mode='same')))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512)))

    model.add(TimeDistributed(Dense(35, name="first_dense")))

    model.add(LSTM(20, return_sequences=True, name="lstm_layer"))

    # %%
    model.add(TimeDistributed(Dense(1), name="time_distr_dense_one"))
    model.add(GlobalAveragePooling1D(name="global_avg"))
    # plot_model(model.model, to_file='/Users/jc/Desktop/model_plot.png', show_shapes=True, show_layer_names=True)

    # %%
    model.compile(loss='mse', optimizer='adam')

    save_fname = os.path.join(configs['model']['save_dir'], '%s-e%s.h5' % (
        dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(configs['training']['epochs'])))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2),
        ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
    ]

    model.fit(
        x_train,
        y_train,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        callbacks=callbacks
    )
    model.save(save_fname)

    '''next n time-stamp prediction'''
    predictions = predict_sequences_multiple(model, x_test, configs['data']['sequence_length'],
                                             configs['data']['prediction_length'])

    '''reverse normalization'''
    for yt in range(len(y_test)):
        nor = test_nor[yt]
        y_test[yt][0] = nor * (y_test[yt][0] + 1)

    for prdt in range(len(predictions)):
        nor_ = test_nor[prdt * configs['data']['prediction_length']]
        predictions[prdt] = [nor_ * (j + 1) for j in predictions[prdt]]

    plot_results_multiple(predictions, y_test, configs['data']['prediction_length'])


if __name__ == '__main__':
    main()
