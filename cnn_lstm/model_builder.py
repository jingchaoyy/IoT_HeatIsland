"""
Created on  2019-07-06
@author: Jingchao Yang

https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/

Using CNN and LSTM for simple time series prediction only, as the model can support very long input sequences that can
be read as blocks or subsequences by the CNN model, then pieced together by the LSTM model.
The CNN model interprets each sub-sequence and the LSTM pieces together the interpretations from the subsequences
"""
import cnn_lstm.data_prep as input
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''define dataset'''
temp = pd.read_csv("tempRecord_byCol_test.csv")
col_name = '0'
temp_s0 = temp[col_name]

input_number, output_num = 24, 2
X, y = input.split_sequence(temp_s0, input_number, output_num)
X = array(X)
y = array(y)

'''reshape from [samples, timesteps] into [samples, timesteps, features]'''
X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1], 1))

'''setting training data'''
# train_size = int(len(X) / 10)
# X_train = X[:-train_size]
# y_train = y[:-train_size]
# x_val = X[-train_size:]
# y_val = y[-train_size:]

'''fit model'''
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(input_number, 1)))
model.add(RepeatVector(2))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# model.summary()
# model.fit(X_train, y_train, validation_data=(x_val, y_val), epochs=200, verbose=0)
history = model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([59.2, 58.7, 58.7, 58.91502415, 58.82705249, 58.98935525, 59.7, 56.9, 59.5, 60.3,
                 61.5, 64.8, 65.6, 68.6, 69.5, 71.5, 68.8, 68.8, 68.0, 63.9, 64.2, 63.4, 62.2, 65.1])
x_input = x_input.reshape(1, input_number, 1)
yhat = model.predict(x_input, verbose=0)
print(yhat)
# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
