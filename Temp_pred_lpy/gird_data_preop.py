"""
Created on 8/28/2019
@author: no281
"""
import pandas as pd
import numpy as np
import csv

csv_path='../../data/2019042910.csv'
coordinates_csv_to_write = csv_path.split('.csv')[0]+'_coordinate.csv'

raw_csv = pd.read_csv(csv_path)
raw_lngs = np.array(raw_csv.lng)
sorted_lngs = np.sort(raw_lngs)
unique_sorted_lngs = np.unique(sorted_lngs)
dif_lngs = []
for i in range(unique_sorted_lngs.shape[0]-1):
    dif_lngs.append(unique_sorted_lngs[i+1]-unique_sorted_lngs[i])
dif_lngs = np.array(dif_lngs)


raw_lats = np.array(raw_csv.lat)
unique_sorted_lats = np.unique(np.sort(raw_lats))
# !!!!补充了一个中间不连续的值
unique_sorted_lats = np.insert(unique_sorted_lats,99,33.844755)
dif_lats = []
for i in range(unique_sorted_lats.shape[0]-1):
    dif_lats.append((unique_sorted_lats[i+1]-unique_sorted_lats[i]))
dif_lats = np.array(dif_lats)

print(dif_lats.max(),dif_lats.min(),dif_lngs.max(),dif_lngs.min())

# list_uni_sorted_lng = unique_sorted_lngs.tolist()
# list_uni_sorted_lat = unique_sorted_lats.tolist()
f2w = open(coordinates_csv_to_write,'w',newline='')
csv_write = csv.writer(f2w,dialect='excel')
coordinates=[]
for i in range(raw_lngs.shape[0]):
    for j in range(unique_sorted_lngs.shape[0]):
        if abs(raw_lngs[i] - unique_sorted_lngs[j])<0.00005:
            for k in range(unique_sorted_lats.shape[0]):
                if abs(raw_lats[i] - unique_sorted_lats[k])<0.00005:
                    coordinates.append([i,j,k])
                    if i > 23456:
                        print('>23456')
                    csv_write.writerow([i, j, k])
    if i%100==0:
        print('%s point done' % i)
f2w.close()
print('Coordinates calc Done!')