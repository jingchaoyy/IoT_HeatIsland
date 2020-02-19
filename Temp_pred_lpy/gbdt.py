"""
Created on 8/20/2019
@author: no281
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from Temp_pred_lpy.data_helper import *


def train_gbdt(data, output):
    gbdt = GradientBoostingRegressor(
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

    train_x, train_y, test_x, test_y = gen_train_and_test_data(data_array=data,
                                                               shuffle=True,
                                                               cut_bin=False,
                                                               y_is_percentage=False)

    gbdt.fit(train_x, train_y)
    pred = gbdt.predict(test_x)
    total_err = 0
    correct_num = 0
    tend_correct_num = 0
    for i in range(pred.shape[0]):
        print(pred[i], test_y[i])
        err = (pred[i] - test_y[i]) / test_y[i]

        wrong_thresh = 2.0
        if abs(pred[i] - test_y[i]) < wrong_thresh:
            correct_num += 1

        if abs(pred[i] - test_y[i]) < abs(test_x[i, -1] - test_y[i]):
            tend_correct_num += 1

        total_err += err * err
    print(total_err / pred.shape[0])
    print("wrong threh = %s. correct ratio is %s" % (wrong_thresh, correct_num / test_y.shape[0]))
    print("tend correct ratio is %s" % (tend_correct_num / test_y.shape[0]))
    # plot_results(pred,test_y)
    plot_results_multiple(test_x, test_y, 20, gbdt, model_type='gbdt', filename=output)

    return gbdt


if __name__ == '__main__':

    coors = ['9q5csxx', '9q5csz8', '9q5csz9', '9q5cszd', '9q5csze', '9q5cszs', '9q5cszt', '9q5csxr', '9q5csz2',
             '9q5csz3', '9q5csz6', '9q5csz7', '9q5cszk', '9q5cszm', '9q5csxp', '9q5csz0', '9q5csz1', '9q5csz4',
             '9q5csz5', '9q5cszh', '9q5cszj', '9q5cswz', '9q5csyb', '9q5csyc', '9q5csyf', '9q5csyg', '9q5csyu',
             '9q5csyv', '9q5cswx', '9q5csy8', '9q5csy9', '9q5csyd', '9q5csye', '9q5csys', '9q5csyt', '9q5cswr',
             '9q5csy2', '9q5csy3', '9q5csy6', '9q5csy7', '9q5csyk', '9q5csym', '9q5cswp', '9q5csy0', '9q5csy1',
             '9q5csy4', '9q5csy5', '9q5csyh', '9q5csyj']

    df = pd.read_csv('../../IoT_HeatIsland_Data/data/LA/exp_data/tempMatrix_LA_selected.csv')
    for coor in coors:
        s = df[coor]
        data_input = np.array(s)
        gbdt_model = train_gbdt(data_input, coor)
