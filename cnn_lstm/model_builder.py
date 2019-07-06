"""
Created on  2019-07-06
@author: Jingchao Yang
"""
# univariate cnn example
import cnn_lstm.data_prep as input
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM

# define dataset
# X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
# y = array([40, 50, 60, 70])
X = array(input.X)
y = array(input.y)

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
# define model
model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(50, activation='relu'))
model.add(LSTM(50, activation='relu', input_shape=(4, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([37, 52, 48, 55])
x_input = x_input.reshape((1, 4, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)
