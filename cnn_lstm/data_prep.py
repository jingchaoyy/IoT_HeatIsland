"""
Created on  2019-07-06
@author: Jingchao Yang
"""
import pandas as pd


def to_timeseries(data, ip_num, op_num):
    """

    :param data: time series data
    :param ip_num: the number of inputs for prediction, for example: [10, 20, 30] as input and [40] as output
    :param op_num: expected number of predicted outputs
    :return:
    """
    X_ans = []
    Y_ans = []
    for i in range(len(data) - ip_num - op_num + 1):
        try:
            X = list(data)[i:i + ip_num]
            Y = list(data)[i + ip_num: i + ip_num + op_num]
            X_ans.append(X)
            Y_ans.append(Y)
            in_ = pd.DataFrame([str(x) for x in X_ans], columns=['input'])
            out = pd.DataFrame([str(x) for x in Y_ans], columns=['output'])
        except:
            print('skip the last pair')
    ans_1 = pd.concat([in_, out], axis=1)
    return X_ans, Y_ans, ans_1
