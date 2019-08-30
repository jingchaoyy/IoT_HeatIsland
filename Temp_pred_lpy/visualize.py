import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot_heatmap():
    np.random.seed(20180316)
    # x = np.random.randn(4, 4)
    x = np.load('../../data/all_pred.npy')
    y = np.load('../../data/all_truth.npy')
    for i in range(x.shape[0]):
        f, (ax1) = plt.subplots(figsize=(6,6),nrows=1)
        sns.heatmap(y[i], annot=False, ax=ax1,vmax=80,vmin=50)
    # sns.heatmap(x, annot=True, ax=ax2, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
        plt.savefig('../../data/visualize/truth_%s.png' % i)
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

if __name__ == '__main__':
    gif()