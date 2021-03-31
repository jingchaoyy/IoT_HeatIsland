"""
Created on  9/28/20
@author: Jingchao Yang
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
import os.path
from os import path


def all_RMSE(result_path, exp_list):
    df = pd.DataFrame()
    for exp in tqdm(range(len(exp_list))):
        train_window, output_size = exp_list[exp]

        rmse_path = glob.glob(result_path + f'\{str(train_window)}_{str(output_size)}' + r'\testScores_C.csv')
        try:
            df[f'{str(train_window)}_{str(output_size)}'] = pd.read_csv(rmse_path[0],
                                                                             usecols=['0']).values.flatten()
        except:
            df[f'{str(train_window)}_{str(output_size)}'] = np.NaN

    return df


def all_r2(result_path, exp_list):
    df = pd.DataFrame()
    for exp in tqdm(range(len(exp_list))):
        train_window, output_size = exp_list[exp]

        pred_path = glob.glob(result_path + f'\{str(train_window)}_{str(output_size)}' + r'\pred.csv')
        try:
            pred_result = pd.read_csv(pred_path[0])
            df[f'{str(train_window)}_{str(output_size)}'] = [r2_score(pred_result.pred, pred_result.ori)]
        except:
            df[f'{str(train_window)}_{str(output_size)}'] = np.NaN

    return df


def box_plot(df, name):
    df.boxplot(boxprops= dict(color="blue", linewidth=3.0),
               whiskerprops=dict(linewidth=3.0),
               medianprops=dict(color="green", linewidth=2))
    plt.xlabel('input-->forecast (hour)', size=30)
    plt.ylabel('Temperature (C) RMSE', size=30)
    plt.ylim(0, 6)
    plt.xticks(size=30, rotation=20)
    plt.yticks(size=30)
    # plt.xticks(rotation=18)
    plt.title(f'Local Testing RMSE (CHI)', size=40)
    # plt.legend(fontsize=40, loc=4)  # lower right
    plt.show()
    # plt.savefig('image.png', dpi=300)



def line_plot(fpath):
    # df = pd.DataFrame()
    # df = pd.read_csv(fpath, index_col=['exp'])
    # df[f'{name}_min'] = rmse_df.min().tolist()
    # df[f'{name}_max'] = rmse_df.max().tolist()
    # df.to_csv(r'E:\IoT_HeatIsland_Data\data\zmodel_evaluation\local_RMSE\all_minmax.csv', index=False)

    # color_dict = {'LA_min': 'blue', 'NYC_min': 'orange', 'Atlanta_min': 'green', 'Chicago_min': 'red',
    #               'LA_max': 'blue', 'NYC_max': 'orange', 'Atlanta_max': 'green', 'Chicago_max': 'red'}
    # color_dict = {'LA': 'blue', 'NYC': 'orange', 'Atlanta': 'green', 'Chicago': 'red'}

    df = pd.read_csv(fpath)
    color_dict = {'LA_CHI': 'blue', 'NYC_CHI': 'orange', 'AT_CHI': 'green', 'CHI_CHI': 'red'}

    # use get to specify dark gray as the default color.
    df.plot(color=[color_dict.get(x, '#333333') for x in df.columns], linewidth=5.0)
    plt.xlabel('hour', size=40)
    plt.ylabel('Temperature (C) RMSE', size=40)
    plt.xticks(size=40)
    plt.yticks(size=40)
    plt.legend(fontsize=30, loc=4)
    # plt.xticks(rotation=18)
    plt.title(f'Transfer Learning Hourly RMSE (12h)', size=40)
    plt.show()


def save_RMSE(name, df, fpath):
    if path.exists(fpath):
        df_tem = pd.read_csv(fpath, index_col=['exp'])
        df_tem[f'{name}_avg'] = df.median()
        df_tem.to_csv(fpath)
    else:
        df_tem = pd.DataFrame()
        # df[f'{name}_min'] = rmse_df.min().tolist()
        # df[f'{name}_max'] = rmse_df.max().tolist()
        df_tem[f'{name}_avg'] = df.median()
        df_tem.index.name = 'exp'
        df_tem.to_csv(fpath)


city = 'Chicago'
model_load_path = f'E:\IoT_HeatIsland_Data\data\{city}\exp_data\{city}_{city}_12neuron'

experiments = [(24, 1), (24, 4), (24, 8), (24, 12), (36, 12), (48, 12), (72, 12), (48, 24), (72, 36), (96, 48),
               (144, 72), (240, 120)]

'''RMSE box plot'''
rmse_df = all_RMSE(model_load_path, experiments)
box_plot(rmse_df, city)
#
# '''Save RMSE'''
# # rmse_df = all_RMSE(model_load_path, experiments)
# # save_RMSE(city, rmse_df, r'E:\IoT_HeatIsland_Data\data\zmodel_evaluation\local_RMSE\all_avg.csv')
#
# '''RMSE city comparison'''
# # rmse_df = all_RMSE(model_load_path, experiments)
# # line_plot(city, rmse_df)
#
# rmse_avg = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\zmodel_evaluation\local_RMSE\all_avg.csv')
# rmse_max = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\zmodel_evaluation\local_RMSE\all_max.csv')
# rmse_min = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\zmodel_evaluation\local_RMSE\all_min.csv')
#
# cities = ['LA', 'NYC', 'AT', 'CHI']
# color = ['blue', 'orange', 'green', 'red']
#
# fig, ax = plt.subplots()
# for city in range(len(cities)):
#     ax.plot(rmse_avg['exp'],
#             rmse_avg[cities[city]],
#             '-',
#             label=cities[city],
#             linewidth=5.0)
#     ax.fill_between(rmse_avg['exp'],
#                     rmse_min[cities[city]],
#                     rmse_max[cities[city]],
#                     alpha=0.2,
#                     facecolor=color[city])
# plt.legend(fontsize=30)
# plt.xlabel('input-->forecast (hour)', size=30)
# plt.ylabel('Temperature (C) RMSE', size=30)
# plt.xticks(size=30)
# for label in ax.get_xaxis().get_ticklabels()[::2]:
#     label.set_visible(False)
# plt.yticks(size=30)
# plt.title(f'RMSE by City', size=30)
#
# plt.show()
#
# '''R2 city comparison'''
# # r2_df = all_r2(model_load_path, experiments)
# # line_plot(r'E:\IoT_HeatIsland_Data\data\zmodel_evaluation\local_RMSE\All_r2.csv')
#
# r2 = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\zmodel_evaluation\local_RMSE\All_r2.csv')
#
# cities = ['LA', 'NYC', 'AT', 'CHI']
# color = ['blue', 'orange', 'green', 'red']
#
# fig, ax = plt.subplots()
# for city in range(len(cities)):
#     ax.plot(r2['exp'],
#             r2[cities[city]],
#             '-',
#             label=cities[city],
#             linewidth=5.0)
#
# plt.legend(fontsize=30)
# plt.xlabel('input-->forecast (hour)', size=30)
# plt.ylabel('R-square', size=30)
# plt.xticks(size=30)
# for label in ax.get_xaxis().get_ticklabels()[::2]:
#     label.set_visible(False)
# plt.yticks(size=30)
# plt.title(f'R-square by City', size=30)
#
# plt.show()
#
# '''hourly RMSE plot'''
# # data = pd.read_csv(r'E:\IoT_HeatIsland_Data\data\zmodel_evaluation\transfer_learning\chicago_48_12_hourly_RMSE.csv')
# line_plot(r'E:\IoT_HeatIsland_Data\data\zmodel_evaluation\transfer_learning\chicago_48_12_hourly_RMSE.csv')