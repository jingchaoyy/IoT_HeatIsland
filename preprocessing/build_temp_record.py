"""
Created on  2019-06-27
@author: Jingchao Yang
"""
import glob
import pandas as pd


def my_csv_reader(path):
    """

    :param path: file list
    :return:
    """

    fname = path.split('/')[-1].split('.')[0]

    d = pd.read_csv(path)
    d = d.rename(index=str, columns={"temp_F": fname})
    temp = d[fname]  # certain column to be added to each dataframe

    return temp


path = '../data/NYC/dataHarvest_NYC/processed/interpolated/0527_0531/'
files = glob.glob(path + '*.csv')
lsorted = sorted(files)  # sort by date
print(lsorted)

'''2D matrix (sensor * time)'''
df = pd.concat([my_csv_reader(f) for f in lsorted], axis=1, join='inner')
print(df)

'''2D matrix (time * sensor)'''
df1_transposed = df.T
print(df1_transposed)
# df1_transposed.to_csv('tempRecord_byCol.csv')

'''reorganize to hour * day * sensor'''
stack_by_row = df.stack()
print(stack_by_row)
# stack_by_row.to_csv('tempRecord.csv')
