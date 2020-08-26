"""
Created on  8/26/20
@author: jingchao yang
"""
import pandas as pd
import glob
import timeit
from tqdm import tqdm

filepath = '/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA/weather_underground/WU_preprocessed_LA_manzhu/'
files = glob.glob(filepath + '20200506*.csv')

fsorted = sorted(files)
data_tb = pd.DataFrame()
for f in tqdm(fsorted):
    data = pd.read_csv(f)
    data_tb = pd.concat([data_tb, data])

start = timeit.timeit()
grouped_df = data_tb.groupby(['lon', 'lat'])
end = timeit.timeit()
print(end - start)

coor_keys = []
ind = 0
for key, item in grouped_df:
    coor_keys.append([key[0], key[1]])
    item.to_csv('/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA'
                f'/weather_underground/WU_preprocessed_LA_manzhu/processed/{ind}.csv')
    ind += 1

coor_keys = pd.DataFrame(coor_keys, columns=['lon', 'lat'])
coor_keys.to_csv('/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA'
                 '/weather_underground/WU_preprocessed_LA_manzhu/processed/coor_keys.csv')