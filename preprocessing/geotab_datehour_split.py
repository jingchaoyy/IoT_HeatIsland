"""
Created on  2019-06-02
@author: Jingchao Yang
"""
import pandas as pd
import glob


def datehour_split(file):
    """

    :param file:
    :return:
    """
    data = pd.read_csv(file)
    time = ['LocalDate', 'LocalHour']
    LocalTime = data[time]
    # print(LocalTime)
    regroup = data.groupby(time)
    groups = regroup.groups
    for i in groups:
        print(i)
        name = [str(x) for x in i]
        fname = '-'.join(name)
        g = regroup.get_group(i)
        # print(g)
        g.to_csv('../dataHarvest_NYC/processed/' + fname + '.csv')


filePath = glob.glob("../dataHarvest_NYC/*.csv")
print(filePath, '\ntotal files', len(filePath))
for f in filePath:
    print('\n', f)
    datehour_split(f)
