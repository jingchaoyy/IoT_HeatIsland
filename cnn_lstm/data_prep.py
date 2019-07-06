"""
Created on  2019-07-06
@author: Jingchao Yang
"""
import pandas as pd


def to_timeseries(data, input_num):
    """

    :param d:
    :param input_num:
    :return:
    """
    X_ans = []
    Y_ans = []
    for i in range(len(data["Births"]) - input_num + 1):
        try:
            X = list(data["Births"])[i:i + input_num]
            Y = list(data["Births"])[i + input_num]
            X_ans.append(X)
            Y_ans.append(Y)
            in_ = pd.DataFrame([str(x) for x in X_ans], columns=['input'])
            out = pd.DataFrame([str(x) for x in Y_ans], columns=['output'])
        except:
            print('skip the last pair')
    ans_1 = pd.concat([in_, out], axis=1)
    print(ans_1)
    return X_ans, Y_ans, ans_1


birth = pd.read_csv("/Users/jc/Desktop/daily-total-female-births.csv")
birth = birth[:len(birth) - 1]
X, y, Xy = to_timeseries(birth, 4)
