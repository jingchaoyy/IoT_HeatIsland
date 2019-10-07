import xgboost as xgb
from Temp_pred_lpy.data_helper import *

# train_x,train_y,test_x,test_y=gen_train_and_test_data(shuffle=True,
#                                                       cut_bin=False,
#                                                       y_is_percentage=False)

train_x, train_y, test_x, test_y = gen_cnn_data(y_is_center_point=False)

train_x = train_x.reshape(-1, 95 * 49)
test_x = test_x.reshape(-1, 95 * 49)

data_train = xgb.DMatrix(train_x, label=train_y)
data_test = xgb.DMatrix(test_x, label=test_y)
# data_test = xgb.DMatrix(test1[feature_use].fillna(-1), label=test1['target'])
watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth': 5, 'eta': 0.01, 'silent': 1, 'objective': 'reg:linear', 'num_boost_round': 950, 'subsample': 0.8,
         'colsample_bytree': 0.2319, 'min_child_weight': 11}
bst = xgb.train(param, data_train, num_boost_round=900, evals=watch_list)
y_pred = bst.predict(data_test)
plot_results(y_pred, test_y)
# plot_results_multiple(test_x,test_y,18,bst,model_type='xgboost')
