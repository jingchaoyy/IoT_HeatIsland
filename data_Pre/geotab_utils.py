"""
Created on  2019-07-30
@author: Jingchao Yang
"""
import pandas as pd
import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import glob
import sys
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pykrige.rk import RegressionKriging
from pykrige.compat import train_test_split
import os


def uniNodes(fileList):
    """

    :param fileList:
    :return:
    """
    nodes = pd.DataFrame()
    for file in fileList:
        data = pd.read_csv(file)
        nodes = nodes.append(data[['Geohash', 'Latitude_SW', 'Longitude_SW', 'Latitude_NE', 'Longitude_NE']],
                             ignore_index=True)
        print('\nstart', nodes.shape)
        nodes = nodes.drop_duplicates()
        print('end', nodes.shape)

    return nodes


def datehour_split(infile, out_path):
    """

    :param infile:
    :param out_path:
    :return:
    """
    data = pd.read_csv(infile)
    time = ['LocalDate', 'LocalHour']
    LocalTime = data[time]
    # print(LocalTime)
    regroup = data.groupby(time)
    groups = regroup.groups
    for i in groups:
        print(i)
        name = [str(x) for x in i]
        fname = '-'.join(name)
        g = regroup.get_group(i)
        # print(g)
        g.to_csv(out_path + fname + '.csv')


def anomaly_detect(file):
    """

    :param file:
    :return:
    """




def ordinary_kriging(file, target_file):
    """
    data = np.array([[0.3, 1.2, 0.47],
                 [1.9, 0.6, 0.56],
                 [1.1, 3.2, 0.74],
                 [3.3, 4.4, 1.47],
                 [4.7, 3.8, 1.74]])
    gridx = np.arange(0.0, 5.5, 0.5)
    gridy = np.arange(0.0, 5.5, 0.5)
    Create the ordinary kriging object. Required inputs are the X-coordinates of
    the data points, the Y-coordinates of the data points, and the Z-values of the
    data points. If no variogram model is specified, defaults to a linear variogram
    model. If no variogram model parameters are specified, then the code automatically
    calculates the parameters by fitting the variogram model to the binned
    experimental semivariogram. The verbose kwarg controls code talk-back, and
    the enable_plotting kwarg controls the display of the semivariogram.

    :param file:
    :param target_file:
    :return:
    """
    data = pd.read_csv(file)
    lat = data['Latitude_SW']
    lng = data['Longitude_SW']
    temp = data['Temperature_F']

    allpoints = target_file
    alldata = pd.read_csv(allpoints)
    geohash = alldata['Geohash']
    all_lat = alldata['Latitude_SW']
    all_lng = alldata['Longitude_SW']

    OK = OrdinaryKriging(lng, lat, temp, variogram_model='linear',
                         verbose=False, enable_plotting=False)

    # Creates the kriged grid and the variance grid. Allows for kriging on a rectangular
    # grid of points, on a masked rectangular grid of points, or with arbitrary points.
    # (See OrdinaryKriging.__doc__ for more information.)
    z, ss = OK.execute('points', all_lng, all_lat)

    output = {'geohash': geohash, 'lng': all_lng, 'lat': all_lat, 'temp_F': z}
    df = pd.DataFrame(output)

    return df


def regression_kriging(file):
    """
    https://pykrige.readthedocs.io/en/latest/examples/regression_kriging2d.html

    :param file:
    :return:
    """
    svr_model = SVR(C=0.1)
    rf_model = RandomForestRegressor(n_estimators=100)
    lr_model = LinearRegression(normalize=True, copy_X=True, fit_intercept=False)

    models = [svr_model, rf_model, lr_model]

    for m in models:
        print('=' * 40)
        print('regression model:', m.__class__.__name__)
        m_rk = RegressionKriging(regression_model=m, n_closest_points=10)
        m_rk.fit(p_train, x_train, target_train)
        print('Regression Score: ', m_rk.regression_model.score(p_test, target_test))
        print('RK score: ', m_rk.score(p_test, x_test, target_test))


def fname_change(fpath):
    """

    :param fpath:
    :return:
    """
    fList = glob.glob(fpath + '*.csv')
    for filename in fList:
        try:
            print(filename)
            c = ''.join([n for n in filename if n.isdigit()])
            if len(c) < 10:
                c = c[:8] + '0' + c[8]
            print(c)
            os.rename(filename, fpath + c + '.csv')
        except:
            print('skip')


def my_csv_reader(path):
    """

    :param path: file list
    :return:
    """

    fname = path.split('/')[-1].split('.')[0]

    d = pd.read_csv(path, index_col='geohash')
    d = d.rename(index=str, columns={"temp_F": fname})
    temp = d[fname]  # certain column to be added to each dataframe

    return temp
