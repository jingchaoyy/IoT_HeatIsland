import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import imageio
from Temp_pred_lpy.data_helper import plot_results
import time

def plot_heatmap(target):
    np.random.seed(20180316)
    # x = np.random.randn(4, 4)
    x = np.load('../../data/all_pred.npy')
    y = np.load('../../data/all_truth.npy')
    if 'truth' in target:
        for i in range(x.shape[0]):
            f, (ax1) = plt.subplots(figsize=(6,6),nrows=1)
            sns.heatmap(y[i], annot=False,linewidths = 0.05, ax=ax1,vmax=80,vmin=50)
            # sns.heatmap(x, annot=True, ax=ax2, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
            plt.savefig('../../data/visualize/truth_%s.png' % i)
            print('truth %s saved!' % i)
            # plt.show()
    if 'pred' in target:
        for i in range(x.shape[0]):
            f, (ax1) = plt.subplots(figsize=(6,6),nrows=1)
            sns.heatmap(x[i], annot=False,linewidths = 0.05, ax=ax1,vmax=80,vmin=50)
            # sns.heatmap(x[i], annot=False, linewidths=0.05, ax=ax1)
            # sns.heatmap(x, annot=True, ax=ax2, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
            plt.savefig('../../data/visualize/pred_%s.png' % i)
            print('pred %s saved!' % i)
            # plt.show()


def gif():
    def create_gif(image_list, gif_name, duration=1.0):
        frames = []
        for image_name in image_list:
            frames.append(imageio.imread(image_name))

        imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
        return
    pred_image_list = ['../../data/visualize/pred_' + str(i) + '.png' for i in range(120)]
    truth_image_list= ['../../data/visualize/truth_' + str(i) + '.png' for i in range(120)]
    create_gif(pred_image_list,'pred.gif',0.1)
    create_gif(truth_image_list,'truth.gif',0.1)


def plot_single_point(coor):
    pred = np.load('../../data/all_pred.npy')
    truth = np.load('../../data/all_truth.npy')
    plot_results(pred[:,coor[0],coor[1]],truth[:,coor[0],coor[1]])


if __name__ == '__main__':
    # plot_heatmap(['pred','truth'])
    # time.sleep(1)
    # gif()
    plot_single_point((9,9))
