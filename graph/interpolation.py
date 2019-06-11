"""
Created on  2019-06-05
@author: Jingchao Yang
"""
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
import numpy as np
import pandas as pd

"""
# # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
# sp = samplingplan(2)
# X = sp.optimallhc(20)
#
# # Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin
# y = testfun(X)

# Read data from CSV
file = '../data/NYC/dataHarvest_NYC/processed/2019-05-30-10.csv'
data = pd.read_csv(file)
X = data[['Latitude_SW', 'Longitude_SW']]
y = data['Temperature_F']

# Now that we have our initial data, we can create an instance of a Kriging model
k = kriging(X, y, testfunction=testfun, name='simple')
k.train()

# Now, five infill points are added. Note that the model is re-trained after each point is added
# numiter = 5
# for i in range(numiter):
#     print ('Infill iteration {0} of {1}....'.format(i + 1, numiter))
#     newpoints = k.infill(1)
#     for point in newpoints:
#         k.addPoint(point, testfun(point)[0])
#     k.train()

# And plot the results
k.plot()
"""

import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

# Read data from CSV
file = '../data/NYC/dataHarvest_NYC/processed/2019-05-30-10.csv'
data = pd.read_csv(file)
lat = data['Latitude_SW']
lng = data['Longitude_SW']
temp = data['Temperature_F']

allpoints = '../data/NYC/dataHarvest_NYC/processed/uniqueNodes.csv'
alldata = pd.read_csv(allpoints)
geohash = alldata['Geohash']
all_lat = alldata['Latitude_SW']
all_lng = alldata['Longitude_SW']

# data = np.array([[0.3, 1.2, 0.47],
#                  [1.9, 0.6, 0.56],
#                  [1.1, 3.2, 0.74],
#                  [3.3, 4.4, 1.47],
#                  [4.7, 3.8, 1.74]])
# gridx = np.arange(0.0, 5.5, 0.5)
# gridy = np.arange(0.0, 5.5, 0.5)
# Create the ordinary kriging object. Required inputs are the X-coordinates of
# the data points, the Y-coordinates of the data points, and the Z-values of the
# data points. If no variogram model is specified, defaults to a linear variogram
# model. If no variogram model parameters are specified, then the code automatically
# calculates the parameters by fitting the variogram model to the binned
# experimental semivariogram. The verbose kwarg controls code talk-back, and
# the enable_plotting kwarg controls the display of the semivariogram.
OK = OrdinaryKriging(lng, lat, temp, variogram_model='linear',
                     verbose=False, enable_plotting=False)
# Creates the kriged grid and the variance grid. Allows for kriging on a rectangular
# grid of points, on a masked rectangular grid of points, or with arbitrary points.
# (See OrdinaryKriging.__doc__ for more information.)
z, ss = OK.execute('points', all_lng, all_lat)
# Writes the kriged grid to an ASCII grid file.
# print(z)
# kt.write_asc_grid(all_lng, all_lat, z, filename="output.asc")
output = {'geohash': geohash, 'lng': all_lng, 'lat': all_lat, 'temp_F': z}
df = pd.DataFrame(output)
df.to_csv('output.csv')
