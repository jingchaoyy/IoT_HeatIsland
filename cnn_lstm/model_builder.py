"""
Created on  2019-07-06
@author: Jingchao Yang

https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/

Using CNN and LSTM for simple time series prediction only, as the model can support very long input sequences that can
be read as blocks or subsequences by the CNN model, then pieced together by the LSTM model.
The CNN model interprets each sub-sequence and the LSTM pieces together the interpretations from the subsequences
"""
# univariate cnn example
import cnn_lstm.data_prep as input
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import LSTM
import pandas as pd

# define dataset
temp = pd.read_csv("tempRecord_byCol_test.csv")
col_name = '0'
temp_s0 = temp[col_name]

input_number, output_num = 4, 1
X, y, Xy = input.to_timeseries(temp_s0, input_number, output_num)
X = array(X)
y = array(y)
print(Xy)

# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
X = X.reshape((X.shape[0], 2, 2, 1))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 2, 1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([63.4, 62.2, 65.1, 64.7])
x_input = x_input.reshape((1, 2, 2, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)
