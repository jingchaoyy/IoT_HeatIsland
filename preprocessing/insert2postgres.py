"""
Created on 3/30/2019
@author: Jingchao Yang
"""

import csv
import psycopg2.extras

tb_in_Name = 'sensordata_chicago_201808'
fileName = 'D:\\IoT_HeatIsland\AoT_data\\arrayOfThings_Chicago\\chicago-complete.monthly.2018-08-01-to-2018-08-31\\data.csv\\data.csv'

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
                                               "timestamp TIMESTAMP,"
                                               "node_id Text,"
                                               "sensor Text,"
                                               "parameter Text,"
                                               "value_row double precision,"
                                               "value_hrf double precision"
                                               ");")
    conn.commit()
    print("create table succeeded " + tb_in_Name)
except:
    print("create table failed " + tb_in_Name)

sql = "insert into " + tb_in_Name + " values (%s, %s, %s, %s, %s, %s, %s)"

with open(fileName, newline='') as csvfile:
    csvreader = csv.reader((line.replace('\0', '') for line in csvfile))
    next(csvreader)
    count = 0
    for row in csvreader:
        try:
            timestamp = row[0]
            node_id = row[1]
            sensor = row[3]
            parameter = row[4]
            # if parameter == 'temperature':
            try:
                try:
                    value_row = float(row[5])
                except:
                    value_row = None
                try:
                    value_hrf = float(row[6])
                except:
                    value_hrf = None
                data = (count, timestamp, node_id, sensor, parameter, value_row, value_hrf)

                try:
                    cur.execute(sql, data)
                    conn.commit()
                except:
                    print("I can't insert into " + tb_in_Name)

                count += 1
            except:
                print('skip record:', ', '.join(row))
        except:
            print('NA row value', row)

conn.close()
