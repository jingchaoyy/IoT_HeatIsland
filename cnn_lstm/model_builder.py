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

input_number, output_num = 24, 2
X, y, Xy = input.to_timeseries(temp_s0, input_number, output_num)
X = array(X)
y = array(y)
print(Xy)

# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
samples = X.shape[0]
subsequences = 2
timesteps = int(X.shape[1] / 2)
features = 1
X = X.reshape(samples, subsequences, timesteps, features)
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                          input_shape=(None, timesteps, features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(output_num))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([59.2, 58.7, 58.7, 58.91502415, 58.82705249, 58.98935525, 59.7, 56.9, 59.5, 60.3,
                 61.5, 64.8, 65.6, 68.6, 69.5, 71.5, 68.8, 68.8, 68.0, 63.9, 64.2, 63.4, 62.2, 65.1])
x_input = x_input.reshape(1, subsequences, timesteps, features)
yhat = model.predict(x_input, verbose=0)
print(yhat)
