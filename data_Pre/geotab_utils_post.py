"""
Created on  8/31/2020
@author: no281
"""
import glob
import pandas as pd


def my_csv_reader(path, uni_ind):
    """
    Processing pre-interpolation matrix using unique nodes
    :param path: file list
    :return:
    """
    fname = path.split('\\')[-1].split('.')[0]
    fname = fname.replace('-', '')

    d = pd.read_csv(path, index_col='Geohash', usecols=['Geohash', 'Temperature_F'])
    d = d.rename(columns={"Temperature_F": fname})
    # temp = d[fname]  # certain column to be added to each dataframe

    temp_ = uni_ind.join(d)
    if not temp_.shape[0] == uni_ind.shape[0]:
        print('duplicated Geohash', fname)
        d = d.reset_index()
        d = d.drop_duplicates(subset=['Geohash'])
        temp_ = uni_ind.join(d.set_index('Geohash'))
    return temp_


# files1 = glob.glob(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\dataHarvest_LA_201904_07\processed' + '\\20*.csv')
# files2 = glob.glob(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\dataHarvest_LA_201907_12\processed' + '\\20*.csv')
# files3 = glob.glob(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\dataHarvest_LA_202001_04\processed' + '\\20*.csv')
#
# df1 = pd.concat([my_csv_reader(f) for f in files1], axis=1, join='inner')
# df1.T.to_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\dataHarvest_LA_201904_07\processed\preInt_matrix.csv')
# print('finished 1')
#
# df2 = pd.concat([my_csv_reader(f) for f in files2], axis=1, join='inner')
# df2.T.to_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\dataHarvest_LA_201907_12\processed\preInt_matrix.csv')
# print('finished 2')
#
# df3 = pd.concat([my_csv_reader(f) for f in files3], axis=1, join='inner')
# df3.T.to_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\dataHarvest_LA_202001_04\processed\preInt_matrix.csv')
# print('finished 3')

# '''merge data'''
# con_df1 = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\dataHarvest_LA_201904_07\processed\interpolated_LA\tempMatrix_LA.csv')
# con_df2 = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\dataHarvest_LA_201907_12\processed\interpolated_LA\tempMatrix_LA.csv')
# con_df3 = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\LA\dataHarvest\dataHarvest_LA_202001_04\processed\interpolated_LA\tempMatrix_LA.csv')
#
# con_df = pd.concat([con_df1, con_df2, con_df3], ignore_index=True)
# con_df.rename(columns={'Unnamed: 0': "datetime"}, inplace=True)
# con_df['datetime'] = pd.to_datetime(con_df['datetime'], format='%Y%m%d%H')
# con_df = con_df.sort_values(by=['datetime'])


process_path = r'E:\IoT_HeatIsland_Data\data\NYC\dataHarvest_NYC\processed'

'''establishing preInt matrix'''
uni_ind_load = pd.read_csv(process_path + r'\uniqueNodes.csv', index_col='Geohash', usecols=['Geohash'])
files = glob.glob(process_path + r'\20*.csv')
df = pd.concat([my_csv_reader(f, uni_ind_load) for f in files], axis=1, join='inner')
df.T.to_csv(process_path + r'\preInt_matrix.csv')
print('preInt matrix established')

# '''load preInt matrix'''
con_df = pd.read_csv(process_path + r'\preInt_matrix.csv')
con_df.rename(columns={'Unnamed: 0': "datetime"}, inplace=True)
con_df['datetime'] = pd.to_datetime(con_df['datetime'], format='%Y%m%d%H')
con_df = con_df.sort_values(by=['datetime'])

'''establishing a full time period'''
rng = pd.DataFrame()
start = con_df['datetime'].iloc[0]
end = con_df['datetime'].iloc[-1]
rng['datetime'] = pd.date_range(start=start, end=end, freq='H')

joined = rng.set_index('datetime').join(con_df.set_index('datetime'))
# joined = joined.reset_index()
joined.reset_index().to_csv(process_path + r'\preInt_matrix_full.csv')
print('full preInt matrix established')

# '''missing data percentage'''
# joined = pd.read_csv(process_path + r'\preInt_matrix_full.csv', index_col=0)
data_missing = joined.isna().mean().round(4) * 100
selected = joined.loc[:, data_missing < 5].columns.values.tolist()
data_all = pd.read_csv(process_path + r'\preInt_matrix_full.csv', usecols=selected)
selected_df = pd.DataFrame({'Geohash': selected[1:]})
coor = pd.read_csv(process_path + r'\uniqueNodes.csv')
selected_coor = selected_df.set_index('Geohash').join(coor.set_index('Geohash'))
selected_coor = selected_coor.reset_index()
selected_coor.to_csv(process_path + r'\nodes_missing_5percent.csv')
print('station with missing data < 5% selected')
