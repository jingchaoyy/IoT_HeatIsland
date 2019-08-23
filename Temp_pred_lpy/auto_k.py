import autokeras as ak
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from data_helper import *


def plot_raw_data():
    df = pd.read_csv('./LSTM/data/tempMatrix_LA.csv')
    cols = df.columns
    s = df[cols[2]]
    s.to_csv('./df2.csv',index=False)
    # s[:300].plot()
    # plt.show()


def train():
    time_limit = [60 * 60 * 4]
    model = ak.ImageClassifier(verbose=True)
    train_x,train_y,test_x,test_y=gen_train_and_test_data()

    train_x, test_x = reshape_as_image((train_x, test_x))

    model.fit(train_x,train_y,time_limit=time_limit[0])
    model.final_fit(train_x,train_y,test_x,test_y,retrain=True)

    score = model.evaluate(test_x,test_y)
    predictions = model.predict(test_x)

if __name__ == '__main__':
    train()
    # train_x, train_y, test_x, test_y = gen_train_and_test_data()
    # ((trainX, trainY), (testX, testY)) = cifar10.load_data()