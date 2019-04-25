"""
Created on  4/15/2019
@author: Jingchao Yang
"""
import sys
import pandas as pd


def split_file_by_line(path, fname, n_split):
    """

    :param path:
    :param fname:
    :param n_split:
    :return:
    """

    # fil = sys.argv[1]
    csvfilename = open(path + fname, 'r').readlines()
    file = 1
    for j in range(len(csvfilename)):
        if j % n_split == 0:
            open(str(path) + "\\split_line\\" + str(file) + '.csv', 'w+').writelines(csvfilename[j:j + n_split])
            file += 1


def split_file_by_date(path, fname):
    """

    :param path:
    :param fname:
    :return:
    """
    datePool, dateGroup, dataFlag = [], [], '1900-0-0'
    with open(path + fname, 'r') as file:
        for cnt, line in enumerate(file):
            tstamp = line
            date = tstamp.split()[0]
            if date in datePool:
                dateGroup.append(line)
            else:
                open(str(path) + "\\split_date\\" + str(dataFlag) + '.csv', 'w+').writelines(dateGroup)
                dataFlag = date
                dateGroup = []
                datePool.append(date)
                dateGroup.append(line)


def split_file_by_hour(path, fname):
    """

    :param path:
    :param fname:
    :return:
    """
    datePool, dateGroup, dataFlag = [], [], '1900-0-0'
    with open(path + fname, 'r') as file:
        for cnt, line in enumerate(file):
            if cnt > 0:
                tstamp = line
                date_hour = tstamp.split(':')[0]
                if date_hour in datePool:
                    dateGroup.append(line)
                else:
                    open(str(path) + "\\split_hour\\" + str(dataFlag) + '.csv', 'w+').writelines(dateGroup)
                    dataFlag = date_hour
                    dateGroup = []
                    datePool.append(date_hour)
                    dateGroup.append(line)


def split_file_by_minute(path, fname):
    """

    :param path:
    :param fname:
    :return:
    """
    datePool, dateGroup, dataFlag = [], [], '1900-0-0'
    with open(path + fname, 'r') as file:
        for cnt, line in enumerate(file):
            if cnt > 0:
                tstamp = line
                datetime_tmp = tstamp.split(':')[0]
                minute = tstamp.split(':')[1]
                datetime = datetime_tmp + ':' + minute
                if datetime in datePool:
                    dateGroup.append(line)
                else:
                    id = str(dataFlag)
                    id = id.replace('/', '-')
                    id = id.replace(':', '-')
                    open(str(path) + "\\splitOriginal\\split_time\\" + id + '.csv', 'w+').writelines(
                        dateGroup)
                    dataFlag = datetime
                    dateGroup = []
                    datePool.append(datetime)
                    dateGroup.append(line)


path = 'D:\\IoT_HeatIsland\\AoT_data\\arrayOfThings_Chicago\\chicago-complete.monthly.2018-08-01-to-2018-08-31\\data.csv\\'
fileName = 'data.csv'
# fileName = 'data.csv'
# split_file_by_line(path, fileName, 1000000)
split_file_by_minute(path, fileName)
