"""
Created on  2019-07-30
@author: Jingchao Yang
"""
import os
import glob
import data_Pre.geotab_utils as utils
import pandas as pd


def main():
    # setting basic environment
    loc = "NYC"
    harvested_temp_dir = "../../IoT_HeatIsland_Data/data/" + loc + "/dataHarvest_" + loc + "/"
    try:
        # Create target Directory
        os.mkdir(harvested_temp_dir + "processed")
        print("Directory ", harvested_temp_dir + "processed", " Created ")
    except FileExistsError:
        print("Directory ", harvested_temp_dir + "processed", " already exists")

    processed_dir = harvested_temp_dir + "processed/"
    uniNodes_dir = processed_dir + "uniqueNodes.csv"
    try:
        # Create target Directory
        os.mkdir(processed_dir + "interpolated")
        print("Directory ", processed_dir + "interpolated", " Created ")
    except FileExistsError:
        print("Directory ", processed_dir + "interpolated", " already exists")

    interpolated_dir = processed_dir + "interpolated/"

    # collecting all possible sensor locations
    dir = glob.glob(harvested_temp_dir + '*.csv')
    print(dir, '\ntotal harvested files', len(dir))
    all_nodes = utils.uniNodes(dir)
    all_nodes.to_csv(uniNodes_dir)

    # separating harvested data by date hour
    print('\nseparating harvested data by date hour')
    filePath = glob.glob(harvested_temp_dir + '*.csv')
    for f in filePath:
        print('\n', f)
        utils.datehour_split(f, processed_dir)

    # interpolation
    file_list = glob.glob(processed_dir + '2019*.csv')
    size = len(file_list)
    print('\nstart interpreting files total', size)
    count = 1
    for i in range(size):
        f = file_list[i]
        fname = f.split('/')[-1]
        print(fname, str(count) + '/' + str(size))
        result = utils.ordinary_kriging(f, uniNodes_dir)
        result.to_csv(interpolated_dir + 'int-' + fname)
        print('success')
        count += 1

    # build m*n temp matrix (m:hour(time), n:sensors)
    print('\n##### start building temp matrix #####')
    utils.fname_change(interpolated_dir)  # change files name to sort by date
    files = glob.glob(interpolated_dir + '*.csv')
    lsorted = sorted(files)  # sort by date
    print(lsorted)

    '''2D matrix (sensor * time)'''
    df = pd.concat([utils.my_csv_reader(f) for f in lsorted], axis=1, join='inner')
    print(df)

    '''2D matrix (time * sensor)'''
    df1_transposed = df.T
    print(df1_transposed)
    df1_transposed.to_csv('tempMatrix_LA.csv')

    '''reorganize to hour * day * sensor'''
    # stack_by_row = df.stack()
    # print(stack_by_row)
    # stack_by_row.to_csv('tempRecord.csv')


if __name__ == '__main__':
    main()
