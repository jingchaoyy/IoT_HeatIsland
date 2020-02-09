"""
Created on  2019-08-02
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
from keras.layers import Masking
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import TimeDistributed
from sklearn.metrics import r2_score
import math
from sklearn.metrics import mean_squared_error


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
        plt.xlabel("Time")
        plt.ylabel("Temperature (F)")
        # plt.title('prediction length ' + str(prediction_len))
        # plt.legend()
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


def predict_point_by_point(in_model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('[Model] Predicting Point-by-Point...')
    predicted = in_model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join(configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    x_train, y_train, train_nor = data.get_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'],
        tbd='train'
    )

    x_test, y_test, test_nor = data.get_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'],
        tbd='test'
    )

    model = Sequential()
    # model.add(Masking(mask_value=-1, input_shape=(
    #     configs['model']['layers'][0]['input_timesteps'], configs['model']['layers'][0]['input_dim'])))
    model.add(LSTM(configs['model']['layers'][0]['neurons'], input_shape=(
        configs['model']['layers'][0]['input_timesteps'], configs['model']['layers'][0]['input_dim']),
                   return_sequences=configs['model']['layers'][0]['return_seq']))
    model.add(Dropout(configs['model']['layers'][1]['rate']))
    model.add(
        LSTM(configs['model']['layers'][2]['neurons'], return_sequences=configs['model']['layers'][2]['return_seq']))
    model.add(
        LSTM(configs['model']['layers'][3]['neurons'], return_sequences=configs['model']['layers'][3]['return_seq']))
    model.add(Dropout(configs['model']['layers'][4]['rate']))
    model.add(Dense(configs['model']['layers'][5]['neurons'], activation=configs['model']['layers'][5]['activation']))
    # plot_model(model.model, to_file='model_plot_test.png', show_shapes=True, show_layer_names=True)

    model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

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
    '''next single time-stamp prediction'''
    # predictions = predict_point_by_point(model, x_test)

    for yt in range(len(y_test)):
        nor = test_nor[yt]
        y_test[yt][0] = nor * (y_test[yt][0] + 1)

    for prdt in range(len(predictions)):
        nor_ = test_nor[prdt * configs['data']['prediction_length']]
        predictions[prdt] = [nor_ * (j + 1) for j in predictions[prdt]]

    '''result plot'''
    plot_results_multiple(predictions, y_test, configs['data']['prediction_length'])

    # model = Model()
    # model.build_lstm(configs)
    # plot_model(model.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #
    # # in-memory training
    # model.train(
    #     x_train,
    #     y_train,
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     save_dir=configs['model']['save_dir']
    # )
    #
    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
    #                                                configs['data']['sequence_length'])
    #
    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])

    '''result evaluation'''
    # pred_multiple_all_merge = [j for i in predictions for j in i]
    #
    # testScore = math.sqrt(mean_squared_error(y_test[:len(pred_multiple_all_merge)], pred_multiple_all_merge))
    # print('Test Score: %.2f RMSE' % (testScore))
    #
    # lstm_score = r2_score(y_test[:len(pred_multiple_all_merge)], pred_multiple_all_merge)
    # print("R^2 Score of model = ", lstm_score)

    # plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
