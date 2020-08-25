"""
Created on  7/27/20
@author: Jingchao Yang

https://www.kaggle.com/sumi25/understand-arima-and-tune-p-d-q
"""
import json
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import itertools
from sklearn.metrics import r2_score
import math
from sklearn.metrics import mean_squared_error
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')


def test_stationarity(timeseries, window=12, cutoff=0.01):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag=20)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    print(dfoutput)


def main():
    configs = json.load(open('config.json', 'r'))
    spike_bins = ['spike_bin3_count', 'spike_bin5_count']
    data_re = pd.read_csv('/Volumes/Samsung_T5/covid/death.csv')
    data_re[['Year_of_Death', 'Month_of_Death']].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S.%f')
    data_re['recording_datetime'] = pd.to_datetime(data_re.Year_of_Death.astype(str) + ' '
                                                   + data_re.Month_of_Death.astype(str))

    '''decomposition'''
    data_re = data_re.set_index('recording_datetime')
    decomposition = sm.tsa.seasonal_decompose(data_re.Heart_Disease_Deaths, model='additive', freq=12)
    fig = decomposition.plot()
    plt.show()
    data_re.reset_index(inplace=True)

    '''differencing data transform'''
    first_diff = data_re.Heart_Disease_Deaths - data_re.Heart_Disease_Deaths.shift(1)
    first_diff = first_diff.dropna(inplace=False)
    test_stationarity(first_diff, window=12)

    '''check autocorrelation, before data transform'''
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data_re.Heart_Disease_Deaths, lags=40, ax=ax1)  #
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data_re.Heart_Disease_Deaths, lags=40, ax=ax2)  #
    plt.show()

    '''check autocorrelation, after data transform'''
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(first_diff, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(first_diff, lags=40, ax=ax2)
    plt.show()

    '''ARIMA'''
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    data_re = data_re.set_index('recording_datetime')

    '''testing different (p,d,q))'''
    # for param in pdq:
    #     for param_seasonal in seasonal_pdq:
    #         try:
    #             mod = sm.tsa.statespace.SARIMAX(data_re.value,
    #                                             order=param,
    #                                             seasonal_order=param_seasonal,
    #                                             enforce_stationarity=False,
    #                                             enforce_invertibility=False)
    #             results = mod.fit()
    #             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
    #         except:
    #             continue

    '''pick to top combination that yields lowest AIC'''
    mod = sm.tsa.statespace.SARIMAX(data_re.Heart_Disease_Deaths,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])

    '''plot diagnose'''
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()

    '''validating forecast'''
    pred = results.get_prediction(start=pd.to_datetime('2018-08-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = data_re.Heart_Disease_Deaths['2018-02-01':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Avg_Power')
    plt.legend()
    plt.show()

    '''evaluating result'''
    testScore = math.sqrt(mean_squared_error(data_re.Heart_Disease_Deaths['2018-08-01':], pred.predicted_mean))
    print('Test Score: %.2f RMSE' % (testScore))

    lstm_score = r2_score(data_re.Heart_Disease_Deaths['2018-08-01':], pred.predicted_mean)
    print("R^2 Score of model = ", lstm_score)


main()
