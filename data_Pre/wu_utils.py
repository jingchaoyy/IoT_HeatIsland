"""
Created on  8/26/20
@author: jingchao yang
"""
import pandas as pd
import glob
import timeit
from tqdm import tqdm
from pathlib import Path


def records_by_sensor():
    """
    organizing data by sensor
    :return:
    """
    filepath = r'E:\IoT_HeatIsland_Data\data\LA\weather_underground\WU_preprocessed_LA'
    files = glob.glob(filepath + '/20*.csv')

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
        item.to_csv(r'E:\IoT_HeatIsland_Data\data\LA\weather_underground\WU_preprocessed_LA\processed'
                    f'\{ind}.csv')
        ind += 1

    coor_keys = pd.DataFrame(coor_keys, columns=['lon', 'lat'])
    coor_keys.to_csv(r'E:\IoT_HeatIsland_Data\data\LA\weather_underground\WU_preprocessed_LA\processed\coor_keys.csv')


def attr_matrix(attr):
    """
    create attribute matrix
    :param attr:
    :return:
    """
    sensor_path = r'E:\IoT_HeatIsland_Data\data\LA\weather_underground\WU_preprocessed_LA\processed\sensors'
    sensor_file = glob.glob(sensor_path + r'\*.csv')

    # attr = 'windSpeed'

    # establishing a full time period
    date_range_df = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\merged\tempMatrix_LA_2019_20.csv',
                                usecols=['datetime'],
                                index_col=['datetime'])

    # dataframe join by datetime
    for f in tqdm(sensor_file):
        fname = Path(f).stem
        data = pd.read_csv(f, usecols=['local_datetime', attr], index_col=['local_datetime'])
        data = data.rename(columns={attr: fname})
        date_range_df = date_range_df.join(data)

    date_range_df.index.name = 'datetime'
    date_range_df = date_range_df.reset_index()
    date_range_df = date_range_df.sort_values(by=['datetime'])
    date_range_df.to_csv(r'E:\IoT_HeatIsland_Data\data\LA\weather_underground\WU_preprocessed_LA\processed\byAttributes'
                         f'\\{attr}.csv')

# records_by_sensor()
# attr_matrix('windSpeed')
