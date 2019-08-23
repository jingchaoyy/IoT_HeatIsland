"""
Created on 8/23/2019
@author: no281
"""
import json
import numpy as np
import pandas as pd
import os
from cnn_lstm import get_data
from Temp_pred_lpy.data_helper import *

train_x,train_y,test_x,test_y = gen_cnn_data(y_is_center_point=True)
print('data done!')



# configs = json.load(open('../../data/config.json', 'r'))
# if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
#
# coor = ['9q5csmp', '9q5cst0', '9q5xxxx', '9q5cst4', '9q5cst5', '9q5csth', '9q5cstj', '9q5cskz', '9q5cssb',
#         '9q5cssc',
#         '9q5cssf', '9q5cssg', '9q5cssu', '9q5cssv', '9q5cskx', '9q5css8', '9q5css9', '9q5cssd', '9q5csse',
#         '9q5csss',
#         '9q5csst', '9q5cskr', '9q5css2', '9q5css3', '9q5css6', '9q5css7', '9q5cssk', '9q5cssm', '9q5xxxx',
#         '9q5css0',
#         '9q5css1', '9q5css4', '9q5css5', '9q5cssh', '9q5cssj', '9q5cs7z', '9q5xxxx', '9q5csec', '9q5csef',
#         '9q5cseg',
#         '9q5cseu', '9q5csev', '9q5cs7x', '9q5cse8', '9q5xxxx', '9q5csed', '9q5csee', '9q5cses', '9q5cset']
#
# data = pd.read_csv('../../data/joined_49_fillna_1.csv', usecols=coor)
# for c in range(len(coor)):
#     coor[c] = data[coor[c]]
#
# coor = np.asarray(coor)
# coor = coor.reshape(7, 7, coor.shape[-1])
# print('time-series matrix data processed')
# print(coor.shape)
# train = coor[:, :, :int(coor.shape[-1] * configs['data']['train_test_split'])]
# test = coor[:, :, int(coor.shape[-1] * configs['data']['train_test_split']):]
#
# x_train, y_train, train_nor = get_data(train, configs['data']['sequence_length'], configs['data']['normalise'])
# x_test, y_test, test_nor = get_data(test, configs['data']['sequence_length'], configs['data']['normalise'])
