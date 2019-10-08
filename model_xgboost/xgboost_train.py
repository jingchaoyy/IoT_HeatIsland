"""
Created on  2019-09-24
@author: Jingchao Yang
"""
from model_xgboost.data_tool import *


class xgboost_tree:
    def __init__(self, model, test_data_y):
        self.model = model
        # self.test_data_x = test_data_x
        self.test_data_y = test_data_y


def data_ready(start, shape_l, shape_w, p_coor, p_data, conf):
    """

    :param start:
    :param shape_l:
    :param shape_w:
    :param p_coor:
    :param p_data:
    :param conf:
    :return:
    """
    # 得到一个合法矩形区域的全部geohash值（按照从左到右，从下到上的顺序排列）
    rect_geohashs = rect_detect(start, shape_l, shape_w, p_coor)
    data = pd.read_csv(p_data, usecols=rect_geohashs)

    for c in range(len(rect_geohashs)):
        rect_geohashs[c] = data[rect_geohashs[c]]

    temperature = np.asarray(rect_geohashs)
    temperature = temperature.reshape(shape_l, shape_w, temperature.shape[-1])
    # (18, 18, 2088)
    print('time-series matrix data processed')
    print(temperature.shape)
    train = temperature[:, :, :int(temperature.shape[-1] * conf['data']['train_test_split'])]
    test = temperature[:, :, int(temperature.shape[-1] * conf['data']['train_test_split']):]

    train_x = get_all_data(train, conf['data']['sequence_length'], conf['data']['normalise'])
    test_x = get_all_data(test, conf['data']['sequence_length'], conf['data']['normalise'])

    return train_x, test_x


def xgboost_model(trainX, trainy, testX, testy):
    """

    :param trainX:
    :param trainy:
    :param testX:
    :param testy:
    :return:
    """
    data_train = xgb.DMatrix(trainX, label=trainy)
    data_test = xgb.DMatrix(testX, label=testy)
    # data_test = xgb.DMatrix(test1[feature_use].fillna(-1), label=test1['target'])
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 5, 'eta': 0.01, 'silent': 1, 'objective': 'reg:linear', 'num_boost_round': 950,
             'subsample': 0.8, 'colsample_bytree': 0.2319, 'min_child_weight': 11}
    bst = xgb.train(param, data_train, num_boost_round=900, evals=watch_list)
    # y_pred = bst.predict(data_test)
    # plot_results(y_pred, testy)
    # plot_results_multiple(test_x,test_y,18,bst,model_type='xgboost')

    return bst


def get_trees(kernal, timestep, trainALL, testALL):
    """

    :param kernal:
    :param timestep:
    :param trainX:
    :param testX:
    :return:
    """
    print('start building trees')
    corner = [0, 1, 2, 3]  # 0,1,2,3 are corner of topleft, topright, botleft, botright respectively
    forest = []
    all_test_x = []
    for cor in corner:
        print('trees for corner', cor)
        trees = []
        for dx in range(trainALL.shape[-1] - kernal + 1):
            for dy in range(trainALL.shape[-1] - kernal + 1):
                cur_train_all = trainALL[:, :, dy:dy + kernal, dx:dx + kernal]
                cur_test_all = testALL[:, :, dy:dy + kernal, dx:dx + kernal]
                cur_train_y, cur_test_y = get_test_data(cor, cur_train_all, cur_test_all)
                cur_train_x = cur_train_all[:, :-1, :, :]
                cur_test_x = cur_test_all[:, :-1, :, :]
                cur_train_x = cur_train_x.reshape(-1, timestep * kernal * kernal)
                cur_test_x = cur_test_x.reshape(-1, timestep * kernal * kernal)

                cur_model = xgboost_model(cur_train_x, cur_train_y, cur_test_x, cur_test_y)
                if cor == 0:
                    all_test_x.append(cur_test_x)  # to avoid saving duplicated test_x data

                tree = xgboost_tree(cur_model, cur_test_y)
                trees.append(tree)
        forest.append(trees)

    return forest, all_test_x


def predict_multiple(boosters, testdata, prediction_len):
    """

    :param boosters:
    :param testdata:
    :param prediction_len:
    :return:
    """
    prediction_seqs_all = []
    for i in range(int(len(testdata[0]) / prediction_len)):
        prediction_seqs = []
        for k in range(prediction_len):
            predicted = []
            for j in range(len(boosters)):
                if j == 0:
                    curr_frame = testdata[j][i * prediction_len]

                data_test = xgb.DMatrix(curr_frame, label=boosters[j].test_data_y)
                y_pred = boosters[j].model.predict(data_test)
                predicted.append(y_pred)

            curr_frame = curr_frame[int(curr_frame / prediction_len):]  # remove the t0 data (7*7)
            curr_frame = curr_frame + predicted  # add the newly predicted (t+1) to update the curr_frame
            prediction_seqs.append(predicted)  # store each predicted 18*18 frame for a prediction_len
        prediction_seqs_all.append(prediction_seqs)
    return prediction_seqs_all


if __name__ == '__main__':
    coor_path = '../../IoT_HeatIsland_Data/data/LA/exp_data/2019042910withCoordinate.csv'
    data_path = '../../IoT_HeatIsland_Data/data/LA/exp_data/tempMatrix_LA.csv'
    configs = json.load(open('../config.json', 'r'))

    # 区域的右下角坐标
    leftdown = [287, 236]
    # 区域的长宽
    length = 18
    width = 18

    # # organizing data
    # train_all, test_all = data_ready(leftdown, length, width, coor_path, data_path, configs)
    # np.save('../../IoT_HeatIsland_Data/data/LA/exp_data/processed/train_x.npy', train_all)
    # np.save('../../IoT_HeatIsland_Data/data/LA/exp_data/processed/test_x.npy', test_all)

    # load directly if file exist
    train_all = np.load('../../IoT_HeatIsland_Data/data/LA/exp_data/processed/train_x.npy')
    test_all = np.load('../../IoT_HeatIsland_Data/data/LA/exp_data/processed/test_x.npy')

    print("trainX %s, testX data %s ready" % (train_all.shape, test_all.shape))

    input_timesteps = configs['data']['sequence_length'] - 1
    prediction_length = configs['data']['prediction_length']
    pred_times_num = 6

    # 此处写死cnn的处理大小为7*7
    cnn_kernel = 7
    all_models, model_test_x = get_trees(cnn_kernel, input_timesteps, train_all, test_all)

    # model reorganize to remove overlays (from 4 different 12*12 to 1 18*18)

    predictions = predict_multiple(all_models, model_test_x, prediction_length)
