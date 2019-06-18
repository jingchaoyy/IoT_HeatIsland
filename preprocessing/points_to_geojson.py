"""
Created on  2019-06-17
@author: Jingchao Yang

Designed for https://data.cityofnewyork.us/Transportation/Real-Time-Traffic-Speed-Data/qkm5-nuaq
"""
import pandas as pd
import json


def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup


def to_geojson_polyline(line_points, output_file):
    """

    :param line_points:
    :param output_file:
    :return:
    """
    ID = line_points['ID']
    SPEED = line_points['SPEED']
    TRAVEL_TIME = line_points['TRAVEL_TIME']
    DATA_AS_OF = line_points['DATA_AS_OF']
    LINK_ID = line_points['LINK_ID']
    LINK_POINTS = line_points['LINK_POINTS']
    BOROUGH = line_points['BOROUGH']

    features = []
    for i in range(len(LINK_POINTS)):
        geometry = {}
        properties = {}
        feature = {}

        point_list = LINK_POINTS[i].split()
        point_list = [x.split(',') for x in point_list]
        try:
            point_list = [[float(x) for x in tup] for tup in point_list]
            point_list = [Reverse(tup) for tup in point_list]

            feature = {"type": "Feature",
                       "properties": {
                           "id": i,
                           "LINK_ID": int(LINK_ID[i])
                       },
                       "geometry": {
                           "type": "LineString",
                           "coordinates": point_list}}
            features.append(feature)
        except:
            print(ID[i], point_list)

    json_data = {"type": "FeatureCollection",
                 "features": features}

    with open(output_file, 'w') as outfile:
        json.dump(json_data, outfile)


file = '../data/NYC/NYC_traffic/Data_Emmanuel/trafficdatacoded.csv'
data = pd.read_csv(file)
jfile = '../data/NYC/NYC_traffic/Data_Emmanuel/geojson/'
jname = 'test.geojson'
to_geojson_polyline(data, jfile + jname)
