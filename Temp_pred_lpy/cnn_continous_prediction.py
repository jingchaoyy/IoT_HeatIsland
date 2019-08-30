"""
Created on 8/28/2019
@author: no281
"""
import pandas as pd
import numpy as np
from Temp_pred_lpy.cnn import train_cnn
from torch.utils.data import TensorDataset
import torch

# 区域的右下角坐标
leftdown = [287,236]
# 区域的长宽
length = 18
width = 18

coor_path = '../../data/2019042910withCoordinate.csv'
data_path = '../../data/tempMatrix_LA.csv'

def rect_detect(left_down,length,width):
    coor = pd.read_csv(coor_path,usecols=['x','y','geohash'])
    print('leng:width:')
    geohash_list = coor['geohash'].tolist()
    x_list = coor['x'].tolist()
    y_list = coor['y'].tolist()
    coor_list = []
    for i in range(len(x_list)):
        coor_list.append([x_list[i],y_list[i]])
    print('%s points found!' % i)

    valid_rect_flag = True
    rect_geohashs = []
    for i in range(width):
        for j in range(length):
            if [leftdown[0]+j,leftdown[1]+i] not in coor_list:
                print('relative coordinate (%s,%s) not exist!' % (i,j))
                valid_rect_flag = False
            else:
                rect_geohashs.append(geohash_list[coor_list.index([leftdown[0]+j,leftdown[1]+i])])

    if valid_rect_flag:
        print('valid rectangle detected!')
        ret = rect_geohashs
    else:
        print('Error!invalid rectangle area!')
        ret = None

    assert ret is not None
    return ret


def padding(pred_frame):
    input_frame = np.zeros(shape=[width, length])
    input_frame[cnn_kernel // 2:width - cnn_kernel // 2, cnn_kernel // 2:length - cnn_kernel // 2] = pred_frame

    # 上下三行补齐
    for i in range(cnn_kernel//2):
        input_frame[i,:] = input_frame[cnn_kernel//2,:]
    for i in range(width-cnn_kernel//2,width):
        input_frame[i,:] = input_frame[width-cnn_kernel//2-1,:]

    # 左右三行补齐
    for i in range(cnn_kernel//2):
        input_frame[:,i] = input_frame[:,cnn_kernel//2]
    for i in range(length-cnn_kernel//2,length):
        input_frame[:,i] = input_frame[:,length-cnn_kernel//2-1]

    return input_frame


if __name__ == '__main__':
    # 得到一个合法矩形区域的全部geohash值（按照从左到右，从下到上的顺序排列）
    rect_geohashs = rect_detect(leftdown,length,width)
    data = pd.read_csv(data_path,usecols=rect_geohashs)
    temperature = []
    for c in range(len(rect_geohashs)):
        temperature.append(data[rect_geohashs[c]])
    temperature = np.array(temperature)
    temperature = temperature.reshape(width,length,temperature.shape[-1])
    # 此时得到的temperature.shape=(18,18,2088)
    print(temperature.shape)

    input_time_length = 99
    output_time_length = 20
    pred_times_num = 6
    # 此处写死cnn的处理大小为7*7
    cnn_kernel = 7
    # 训练模型or载入模型
    model = train_cnn()
    # model = torch.load('./CNN_model.ckpt')

    all_pred = []
    for t in range(pred_times_num):
        pred_frames_1time = []
        for i in range(output_time_length):
            if i == 0:
                # init_input.shape = (18,18,99)
                init_input = temperature[:,:,t*input_time_length:(t+1)*input_time_length]
                # 将init_input reshape变为(1,99,18,18)
                init_input = init_input.transpose(2,0,1)
                init_input = init_input[np.newaxis,:]
                input_frame = init_input
            else:
                input_frame[:,:-1,:,:] = input_frame[:,1:,:,:]
                input_frame[:,-1,:,:] = padding(pred_frame)

            pred_frame = []
            for dx in range(temperature.shape[0]-cnn_kernel+1):
                for dy in range(temperature.shape[1]-cnn_kernel+1):
                    cur_input = torch.from_numpy(input_frame[:,:,dx:dx+cnn_kernel,dy:dy+cnn_kernel]).float()
                    with torch.no_grad():
                        pred_point = model(cur_input).numpy()[0,0]
                        pred_frame.append(pred_point)
            # pred_frame.shape=(12,12) 18 - 7//2
            pred_frame = np.array(pred_frame).reshape(width-cnn_kernel+1,length-cnn_kernel+1)
            pred_frames_1time.append(padding(pred_frame))
        pred_frames_1time = np.array(pred_frames_1time)
        all_pred.append(pred_frames_1time)
    all_pred = np.array(all_pred)
    all_pred = all_pred.reshape(all_pred.shape[0]*all_pred.shape[1],all_pred.shape[2],all_pred.shape[3])
    np.save('../../data/all_pred.npy',all_pred)
    all_truth = temperature[:,:,1:1+output_time_length*pred_times_num]
    all_truth = all_truth.transpose(2,0,1)
    np.save('../../data/all_truth.npy',all_truth)
    print('npy saved!')