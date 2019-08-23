import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import xgboost as xgb


def gen_train_and_test_data(csv_path='./df2.csv', input_length=100,
                            test_ratio=0.1, shuffle=True, cut_bin=True,
                            x_is_percentage=False,y_is_percentage=False):
    df = pd.read_csv(csv_path,header=None)
    s = df[0]
    data_array = np.array(s)

    # 对温度进行分箱处理
    if cut_bin:
        # 分箱宽度
        bin_length = 0.5
        label_array = np.round(data_array/bin_length)
        label_array_min = label_array.min()
        data_array = label_array - label_array.min()

        print('Cut bin Done! Bin length is: %s, and the transform equation is : T = %s * label + %s'
              % (bin_length,bin_length,label_array_min))

    all_x_y = []
    train_x_y = []
    test_x_y = []
    for i in range(0,data_array.shape[0]-input_length):
        all_x_y.append(data_array[i:input_length + i + 1])

    # 按照test_ratio分配test数量，取全部数据里面的最后一部分作为test数据
    test_num = int(len(all_x_y) * test_ratio)
    # 取all_x_y中的后面作为test
    test_indexs = np.arange(len(all_x_y)-test_num,len(all_x_y))

    for i in range(len(all_x_y)):
        if i in test_indexs:
            test_x_y.append(all_x_y[i])
        else:
            train_x_y.append((all_x_y[i]))

    # 因为要保证test数据是连续的，因此只能取完test数据之后，对train数据进行shuffle
    if shuffle:
        random.shuffle(train_x_y)

    train_x_y = np.array(train_x_y)
    test_x_y = np.array(test_x_y)

    train_x = train_x_y[:,0:100]
    train_y = train_x_y[:,100]
    test_x = test_x_y[:,0:100]
    test_y = test_x_y[:,100]

    # if x_is_percentage:

    # 将输出的y转化成变化的百分比
    if y_is_percentage:
        train_y = (train_y - train_x[:,-1])/train_x[:,-1]*100.0
        test_y = (test_y - test_x[:,-1])/test_x[:,-1]*100.0

    return train_x, train_y, test_x, test_y


def reshape_as_image(arrays):
    return (n.reshape((n.shape[0],n.shape[1],1,1)) for n in arrays)


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(test_x, true_data, prediction_len,model,is_for_xgb=False):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    pred_multiple_all = []
    for i in range(test_x.shape[0]//prediction_len):
        pred_multiple = []
        current_x = test_x[i*prediction_len]
        for j in range(prediction_len):
            if j == 0:
                if is_for_xgb:
                    current_x_xgb = xgb.DMatrix([current_x])
                    pred_multiple.append(model.predict(current_x_xgb))
                else:
                    pred_multiple.append(model.predict([current_x]))
            else:
                current_x = np.delete(current_x, 0)
                current_x = np.append(current_x, pred_multiple[-1])
                if is_for_xgb:
                    current_x_xgb = xgb.DMatrix([current_x])
                    pred_multiple.append(model.predict(current_x_xgb))
                else:
                    pred_multiple.append(model.predict([current_x]))
        pred_multiple_all.append(pred_multiple)

    # plot_results(pred_multiple,true_data)
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i in range(len(pred_multiple_all)):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + pred_multiple_all[i], label='Prediction')
        plt.legend()
    plt.show()

    print('multiple Printed!')
