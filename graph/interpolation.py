"""
Created on  2019-06-05
@author: Jingchao Yang
"""
import pandas as pd
import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import glob


def kriging(file):
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
    :return:
    """
    data = pd.read_csv(file)
    lat = data['Latitude_SW']
    lng = data['Longitude_SW']
    temp = data['Temperature_F']

    allpoints = '../dataHarvest_NYC/processed/uniqueNodes.csv'
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


# Read data from CSV
# file = '../dataHarvest_NYC/processed/2019-05-30-10.csv'
file_list = glob.glob('../dataHarvest_NYC/processed/2019*.csv')
size = len(file_list)
count = 1
for i in range(len(file_list)):
    if i > 510:
        f = file_list[i]
        fname = f.split('\\')[1]
        print(fname, str(count) + '/' + str(size))
        result = kriging(f)
        result.to_csv('interpolated/int-' + fname)
        print('success')
        count += 1
