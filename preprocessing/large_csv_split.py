"""
Created on  4/15/2019
@author: Jingchao Yang
"""
import sys


def split_file(path, fname, n_split):
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
            open(str(path) + "\\split\\" + str(file) + '.csv', 'w+').writelines(csvfilename[j:j + n_split])
            file += 1


path = 'D:\\IoT_HeatIsland\\AoT_data\\arrayOfThings_Chicago\\zzLatest\\AoT_Chicago.complete.latest\\AoT_Chicago.complete.2019-03-30\\data.csv\\'
# path = 'D:\\IoT_HeatIsland\AoT_data\\arrayOfThings_Chicago\\chicago-complete.daily.2019-04-14\\data.csv\\'
fileName = 'data.csv'
split_file(path, fileName, 1000000)
