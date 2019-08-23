"""
Created on 8/20/2019
@author: no281
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from Temp_pred_lpy.data_helper import *


def train_gbdt():
    gbdt=GradientBoostingRegressor(
      loss='ls'
    , learning_rate=0.1
    , n_estimators=100
    , subsample=1
    , min_samples_split=2
    , min_samples_leaf=1
    , max_depth=10
    , init=None
    , random_state=None
    , max_features=None
    , alpha=0.9
    , verbose=0
    , max_leaf_nodes=None
    , warm_start=False
    )

    train_x,train_y,test_x,test_y=gen_train_and_test_data(shuffle=True,
                                                          cut_bin=False,
                                                          y_is_percentage=False)

    gbdt.fit(train_x,train_y)
    pred = gbdt.predict(test_x)
    total_err = 0
    correct_num = 0
    tend_correct_num = 0
    for i in range(pred.shape[0]):
        print(pred[i],test_y[i])
        err=(pred[i]-test_y[i])/test_y[i]

        wrong_thresh = 2.0
        if abs(pred[i]-test_y[i])<wrong_thresh:
            correct_num += 1

        if abs(pred[i]-test_y[i]) < abs(test_x[i,-1]-test_y[i]):
            tend_correct_num += 1

        total_err += err*err
    print(total_err/pred.shape[0])
    print("wrong threh = %s. correct ratio is %s" % (wrong_thresh, correct_num / test_y.shape[0]))
    print("tend correct ratio is %s" % (tend_correct_num/test_y.shape[0]))
    # plot_results(pred,test_y)
    plot_results_multiple(test_x,test_y,20,gbdt,model_type='gbdt')

    return gbdt


if __name__ == '__main__':
    gbdt_model = train_gbdt()
