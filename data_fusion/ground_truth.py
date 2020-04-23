"""
Created on  1/14/20
@author: Jingchao Yang

Ground truth data correlation and alignment test
"""
import pandas as pd
from datetime import datetime, timedelta

iot_path = '/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA/exp_data/tempMatrix_LA.csv'
wu_path = '/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA/weather_underground/LA/processed_updated/KCALOSAN764.txt.csv'

iot_data = pd.read_csv(iot_path, usecols=['time', '9q5csyd'])
iot_data['time'] = pd.to_datetime(iot_data['time'], format='%Y%m%d%H')
wu_data = pd.read_csv(wu_path, usecols=['time', 'temperature'])
wu_data['time'] = pd.DatetimeIndex(wu_data['time']) - timedelta(hours=7)
wu_data['time'] = pd.to_datetime(wu_data['time'], format='%Y%m%d%H')

joined = iot_data.set_index('time').join(wu_data.set_index('time'))

print()