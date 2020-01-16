"""
Created on  10/24/2019
@author: Jingchao Yang
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(target):
    """

    :param target:
    :return:
    """
    np.random.seed(20180316)
    # x = np.random.randn(4, 4)
    x = np.load('../../IoT_HeatIsland_Data/data/LA/exp_data/xgboost_models/result/prediction.npy')
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
    y = np.load('../../IoT_HeatIsland_Data/data/LA/exp_data/xgboost_models/result/truth.npy')
    if 'truth' in target:
        for i in range(x.shape[0]):
            f, (ax1) = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(y[i], annot=False, linewidths=0.05, ax=ax1, vmax=80, vmin=50)
            # sns.heatmap(x, annot=True, ax=ax2, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
            plt.savefig('../../IoT_HeatIsland_Data/data/LA/exp_data/xgboost_models/result/plot/tru' + str(i) + '.png')
            print('truth %s saved!' % i)
            # plt.show()
    if 'pred' in target:
        for i in range(x.shape[0]):
            f, (ax1) = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(x[i], annot=False, linewidths=0.05, ax=ax1, vmax=80, vmin=50)
            # sns.heatmap(x[i], annot=False, linewidths=0.05, ax=ax1)
            # sns.heatmap(x, annot=True, ax=ax2, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
            plt.savefig('../../IoT_HeatIsland_Data/data/LA/exp_data/xgboost_models/result/plot/pred' + str(i) + '.png')
            print('pred %s saved!' % i)
            # plt.show()


plot_heatmap('truth')
