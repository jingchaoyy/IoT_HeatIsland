import xgboost as xgb
from Temp_pred_lpy.data_helper import *

coors = ['9q5csxx', '9q5csz8', '9q5csz9', '9q5cszd', '9q5csze', '9q5cszs', '9q5cszt', '9q5csxr', '9q5csz2',
         '9q5csz3', '9q5csz6', '9q5csz7', '9q5cszk', '9q5cszm', '9q5csxp', '9q5csz0', '9q5csz1', '9q5csz4',
         '9q5csz5', '9q5cszh', '9q5cszj', '9q5cswz', '9q5csyb', '9q5csyc', '9q5csyf', '9q5csyg', '9q5csyu',
         '9q5csyv', '9q5cswx', '9q5csy8', '9q5csy9', '9q5csyd', '9q5csye', '9q5csys', '9q5csyt', '9q5cswr',
         '9q5csy2', '9q5csy3', '9q5csy6', '9q5csy7', '9q5csyk', '9q5csym', '9q5cswp', '9q5csy0', '9q5csy1',
         '9q5csy4', '9q5csy5', '9q5csyh', '9q5csyj']

df = pd.read_csv('../../IoT_HeatIsland_Data/data/LA/exp_data/tempMatrix_LA_selected.csv')
for coor in coors:
    s = df[coor]
    data = np.array(s)

    train_x, train_y, test_x, test_y = gen_train_and_test_data(data_array=data,
                                                               shuffle=True,
                                                               cut_bin=False,
                                                               y_is_percentage=False)

    # train_x, train_y, test_x, test_y = gen_cnn_data(y_is_center_point=False)
    # train_x = train_x.reshape(-1, 95 * 49)
    # test_x = test_x.reshape(-1, 95 * 49)

    data_train = xgb.DMatrix(train_x, label=train_y)
    data_test = xgb.DMatrix(test_x, label=test_y)
    # data_test = xgb.DMatrix(test1[feature_use].fillna(-1), label=test1['target'])
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 5, 'eta': 0.01, 'silent': 1, 'objective': 'reg:linear', 'num_boost_round': 950,
             'subsample': 0.8,
             'colsample_bytree': 0.2319, 'min_child_weight': 11}
    bst = xgb.train(param, data_train, num_boost_round=900, evals=watch_list)
    y_pred = bst.predict(data_test)
    # plot_results(y_pred, test_y)
    plot_results_multiple(test_x, test_y, 18, bst, model_type='xgboost', filename=coor)
