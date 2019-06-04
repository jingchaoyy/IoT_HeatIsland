"""
Created on  2019-06-04
@author: Jingchao Yang
"""
import pandas as pd
import glob


def uniNodes(fileList):
    """

    :param fileList:
    :return:
    """
    nodes = pd.DataFrame()
    for file in fileList:
        data = pd.read_csv(file)
        nodes = nodes.append(data[['Geohash', 'Latitude_SW', 'Longitude_SW', 'Latitude_NE', 'Longitude_NE']],
                             ignore_index=True)
        print('\nstart', nodes.shape)
        nodes = nodes.drop_duplicates()
        print('end', nodes.shape)

    return nodes


dir = glob.glob("../data/NYC/dataHarvest_NYC/*.csv")
print(dir, '\ntotal files', len(dir))
all_nodes = uniNodes(dir)
all_nodes.to_csv("../data/NYC/dataHarvest_NYC/processed/uniqueNodes.csv")
