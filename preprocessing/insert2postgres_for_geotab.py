"""
Created on 4/15/2019
@author: Jingchao Yang
"""

import psycopg2.extras
from os import listdir
from os.path import join
import pandas as pd


def get_file_from_dir(dir):
    """

    :param dir:
    :return:
    """
    onlyCSV = []
    # this is the extension you want to detect
    extension = '.csv'

    onlyCSV += [each for each in listdir(dir) if each.endswith(extension)]
    return onlyCSV


def get_data_fom_csv(file):
    """

    :param file:
    :return:
    """
    data = pd.read_csv(file)
    return data


tb_in_Name = 'geotab_Las_Vegas_201804_test'
path = '../dataHarvest_Las_Vegas/'
allCSVs = get_file_from_dir(path)

try:
    conn = psycopg2.connect("dbname='arrayOfThings' user='postgres' host='localhost' password='123456'")
except:
    print("I am unable to connect to the database")

cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

try:
    cur.execute("drop table " + tb_in_Name)
    conn.commit()
    print("drop table succeeded " + tb_in_Name)
except:
    print("drop table failed " + tb_in_Name)
    conn.rollback()  # when command fail, the transaction will be aborted and no further command will be executed
    # until a call to the rollback(). This except will prevent such abort when table is new and cannot be found and drop

try:
    cur.execute("create table " + tb_in_Name + "("
                                               "eID int PRIMARY KEY NOT NULL,"
                                               "Geohash Text,"
                                               "Latitude_SW double precision,"
                                               "Longitude_SW double precision,"
                                               "Latitude_NE double precision,"
                                               "Longitude_NE double precision,"
                                               "LocalDate Date,"
                                               "LocalHour int,"
                                               "Temperature_C double precision,"
                                               "Temperature_F double precision"
                                               ");")
    conn.commit()
    print("create table succeeded " + tb_in_Name)
except:
    print("create table failed " + tb_in_Name)

sql = "insert into " + tb_in_Name + " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

id = 0
for c in allCSVs:
    print('start processing', c)
    allData = get_data_fom_csv(join(path, c))
    # print(allData)
    Geohash = allData['Geohash']
    Latitude_SW = allData['Latitude_SW']
    Longitude_SW = allData['Longitude_SW']
    Latitude_NE = allData['Latitude_NE']
    Longitude_NE = allData['Longitude_NE']
    LocalDate = allData['LocalDate']
    LocalHour = allData['LocalHour']
    Temperature_C = allData['Temperature_C']
    Temperature_F = allData['Temperature_F']

    for i in range(Geohash.shape[0]):
        v = (
            id, Geohash[i], Latitude_SW[i], Longitude_SW[i], Latitude_NE[i], Longitude_NE[i], LocalDate[i],
            int(LocalHour[i]), Temperature_C[i], Temperature_F[i])

        try:
            cur.execute(sql, v)
            conn.commit()
            id += 1
        except:
            print("I can't insert into " + tb_in_Name)

conn.close()
