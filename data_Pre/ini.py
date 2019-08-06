"""
Created on  2019-07-30
@author: Jingchao Yang
"""
import os
import glob
import data_Pre.geotab_utils as utils
import pandas as pd
import threading
import time

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
    result_matrix = "../../IoT_HeatIsland_Data/data/" + loc + "/tempMatrix_" + loc + ".csv"

    # collecting all possible sensor locations
    print('\nstart collecting unique sensors from all records')
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
        # fname = f.split('/')[-1]
        head, tail = os.path.split(f)
        fname = tail
        print(fname, str(count) + '/' + str(size))
        result = utils.ordinary_kriging(f, uniNodes_dir)
        result.to_csv(interpolated_dir + 'int-' + fname)
        print('success')
        count += 1

    # multi thread interpolation
    # def interpolation_job(file_list, thread_no):
    #     count = 1
    #     size = len(file_list)
    #     for i in range(size):
    #         f =file_list[i]
    #         head, tail = os.path.split(f)
    #         fname = tail
    #         print(fname, str(count) + '/' + str(size))
    #         result = utils.ordinary_kriging(f, uniNodes_dir)
    #         result.to_csv(interpolated_dir + 'int-' + fname)
    #         print('Thread No.%s, progressed %s / %s' % (thread_no, count, size))
    #         count += 1
    #
    # # Define numbers of Threads and split the file list evenly according to thread_num
    # thread_num = 1
    # file_list = glob.glob(processed_dir + '2019*.csv')
    # file_num_each_thread = (len(file_list)//thread_num)+1
    # file_list_splits = [file_list[i:i+file_num_each_thread] for i in range(0,len(file_list),file_num_each_thread)]
    #
    # thread_list=[]
    # for i in range(thread_num):
    #     t = threading.Thread(target=interpolation_job,name='interpolation_job'+str(i),args=(file_list_splits[i],i))
    #     thread_list.append(t)
    #
    # for t in thread_list:
    #     t.start()

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
    df1_transposed.to_csv(result_matrix)

    '''reorganize to hour * day * sensor'''
    # stack_by_row = df.stack()
    # print(stack_by_row)
    # stack_by_row.to_csv('tempRecord.csv')


if __name__ == '__main__':
    main()
