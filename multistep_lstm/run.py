"""
Created on  8/27/20
@author: jc
"""
import pandas as pd
import matplotlib.pyplot as plt
import time
from multistep_lstm import multistep_lstm_keras

# aggr_df = pd.read_csv('/Users/jc/Documents/GitHub/Fresh-Air-LA/data/aggr_la_aq_preprocessed.csv', index_col=False)
# print(aggr_df.head())
#
# vars = list(set(aggr_df.columns[1:]) - set(['datetime']))
#
# sensors = pd.read_csv('/Users/jc/Documents/GitHub/Fresh-Air-LA/data/sensors_la_preprocessed.csv', index_col=False, dtype=str)
# print(sensors.head())
#
# selected_vars = [var for var in vars if var.split('_')[1] == 'PM2.5']
# print(selected_vars)
#
# # plot the timeseries to have a general view
# selected_df = aggr_df[selected_vars]
# selected_df.index = aggr_df['datetime']
# if selected_df.shape[1] > 5:
#     for i in range(0, selected_df.shape[1], 5):
#         selected_df_plot = selected_df[selected_df.columns[i:(i+5)]]
#         selected_df_plot.plot(subplots=True)
#         plt.show()

variable = '060371201_PM2.5'
start = time.time()

multistep_lstm_keras.encoder_decoder_LSTM_multivariate(variable)

end = time.time()
print(end - start)