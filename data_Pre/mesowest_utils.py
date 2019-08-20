"""
Created on  2019-08-12
@author: Jingchao Yang
"""
import pandas as pd

data = pd.read_csv('/Users/jc/Documents/GitHub/IoT_HeatIsland_Data/data/LA/mesowest/processed/concat.csv')
data['Date_Time'] = pd.to_datetime(data.Date_Time)
agg = data.groupby(
    [data['Date_Time'].dt.year, data['Date_Time'].dt.month, data['Date_Time'].dt.day, data['Date_Time'].dt.hour]).mean()
agg.to_csv('/Users/jc/Documents/GitHub/IoT_HeatIsland_Data/data/LA/9q5css6_meso.csv')

weather = '/Users/jc/Documents/GitHub/IoT_HeatIsland_Data/data/LA/9q5css6_meso.csv'
weather_data = pd.read_csv(weather)
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'], format='%Y%m%d%H')
temp = '/Users/jc/Documents/GitHub/IoT_HeatIsland_Data/data/LA/9q5css6.csv'
temp_data = pd.read_csv(temp)
temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y%m%d%H')

rng = pd.DataFrame()
start = temp_data['datetime'][0]
end = temp_data['datetime'][temp_data.shape[0]-1]
rng['datetime'] = pd.date_range(start=start, end=end, freq='H')

joined = rng.set_index('datetime').join(temp_data.set_index('datetime'))
joined = joined.join(weather_data.set_index('datetime'))

joined.to_csv('/Users/jc/Documents/GitHub/IoT_HeatIsland_Data/data/LA/joined.csv')
joined_interpo = joined.interpolate(method='time')
joined_interpo.to_csv('/Users/jc/Documents/GitHub/IoT_HeatIsland_Data/data/LA/joined_interpo.csv')