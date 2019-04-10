"""
Created on 4/10/2019
@author: Jingchao Yang
"""

import csv
import psycopg2.extras

tb_in_Name = 'sensor_Chicago_all'
fileName = 'D:\\IoT_HeatIsland\\AoT_data\\arrayOfThings_Chicago\\zzLatest\\AoT_Chicago.complete.latest\\AoT_Chicago.complete.2019-03-30\\nodes.csv'

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
    print("create table " + tb_in_Name + "("
                                               "eID int PRIMARY KEY NOT NULL,"
                                               "node_id Text,"
                                               "address Text,"
                                               "lat double precision,"
                                               "lng double precision"
                                               ");")
    cur.execute("create table " + tb_in_Name + "("
                                               "eID int PRIMARY KEY NOT NULL,"
                                               "node_id Text,"
                                               "address Text,"
                                               "lat double precision,"
                                               "lng double precision"
                                               ");")
    conn.commit()
    print("create table succeeded " + tb_in_Name)
except:
    print("create table failed " + tb_in_Name)

sql = "insert into " + tb_in_Name + " values (%s, %s, %s, %s, %s)"

with open(fileName, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    count = 0
    for row in csvreader:
        node_id = row[0]
        address = row[3]
        lat = row[4]
        lng = row[5]
        data = (count, node_id, address, lat, lng)

        try:
            print(sql, data)
            cur.execute(sql, data)
            conn.commit()
        except:
            print("I can't insert into " + tb_in_Name)

        count += 1

conn.close()
