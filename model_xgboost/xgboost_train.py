"""
Created on  2019-09-24
@author: Jingchao Yang
"""
from model_xgboost.data_tool import *
from joblib import dump
from joblib import load
import time
import glob
import datetime as dt


class xgboost_tree:
    def __init__(self, model, test_data_x, test_data_y):
        self.model = model
        self.test_data_x = test_data_x
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


def get_trees(kernal, timestep, trainALL, testALL, model_path, models_test_x_path, models_test_y_path):
    """

    :param kernal:
    :param timestep:
    :param trainALL:
    :param testALL:
    :param model_path:
    :param models_test_x_path:
    :param models_test_y_path:
    :return:
    """
    print('start building trees')
    corner = [0, 1, 2, 3]  # 0,1,2,3 are corner of topleft, topright, botleft, botright respectively
    forest = []
    for cor in corner:
        print('trees for corner', cor)
        # trees = []
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

                tree = xgboost_tree(cur_model, cur_test_x, cur_test_y)
                forest.append(tree)

                # save models and data
                date = dt.datetime.now().strftime('%d%m%Y-%H%M%S')
                sname = date + '-' + str(cor) + '-' + str(dx) + '-' + str(dy)
                dump(cur_model, model_path + sname + ".dat")
                np.save(models_test_x_path + sname + '.npy', cur_test_x)
                np.save(models_test_y_path + sname + '.npy', cur_test_y)

    # forest.append(trees)
    forest = np.array(forest)
    forest = forest.reshape(len(corner), timestep + 1, timestep + 1)  # 4*12*12

    return forest


def load_models(model_path, models_test_x_path, models_test_y_path):
    """

    :param model_path:
    :param models_test_x_path:
    :param models_test_y_path:
    :return:
    """
    all_models = []
    models = [f for f in glob.glob(model_path + '*.dat')]
    models_test_x = [f for f in glob.glob(models_test_x_path + '*.npy')]
    models_test_y = [f for f in glob.glob(models_test_y_path + '*.npy')]

    for i in range(len(models)):
        model = load(models[i])
        test_x = np.load(models_test_x[i])
        test_y = np.load(models_test_y[i])

        all_models.append(xgboost_tree(model, test_x, test_y))

    return all_models


def padding(kernal, pad_length, pad_width, models):
    """
    padding (organizing) a 18*18 grid with four 12*12 models

    :param kernal: 12
    :param pad_length: 18
    :param pad_width: 18
    :param models: len(models) == 4
    :return:
    """
    model_length = pad_length - kernal + 1
    model_width = pad_width - kernal + 1
    model_length_overlay = model_length * 2 - pad_length
    model_width_overlay = model_width * 2 - pad_width

    # org_models = np.arange(pad_length * pad_width)
    # org_models = org_models.reshape(pad_length, pad_width)
    org_models = np.empty([pad_length, pad_width], dtype=xgboost_tree)

    for i in range(len(models)):
        m = models[i]
        m = np.array(m)
        m.reshape(model_length, model_width)  # 12*12
        if i == 0:  # all model in models (12*12) will be applied to the org_models (18*18)
            for j0 in range(model_length):
                for k0 in range(model_width):
                    org_models[j0][k0] = m[j0][k0]
        elif i == 1:
            for j1 in range(model_length):
                for k1 in range(pad_width - model_width):
                    org_models[j1][model_width + k1] = m[j1][model_width_overlay + k1]
        elif i == 2:
            for j2 in range(pad_length - model_length):
                for k2 in range(model_width):
                    org_models[model_length + j2][k2] = m[model_length_overlay + j2][k2]
        elif i == 3:
            for j3 in range(pad_length - model_length):
                for k3 in range(pad_width - model_width):
                    org_models[model_length + j3][model_width + k3] = \
                        m[model_length_overlay + j3][model_width_overlay + k3]

    return org_models.reshape(pad_length, pad_width)


def predict_multiple(boosters, prediction_len, kernal, win_l, win_w):
    """

    :param boosters:
    :param prediction_len:
    :param kernal:
    :param win_l:
    :param win_w:
    :return:
    """
    prediction_seqs_all = []
    for i in range(int(len(boosters[0].test_data_x.shape[0]) / prediction_len)):
        prediction_seqs = []
        for k in range(prediction_len):
            predicted = []
            for j in range(len(boosters)):
                rolling_x = boosters[j].test_data_x[i * prediction_len]
                rolling_y = boosters[j].test_data_y[i * prediction_len]

                data_test = xgb.DMatrix(rolling_x, label=rolling_y)
                y_pred = boosters[j].model.predict(data_test)
                predicted.append(y_pred)

            boosters = np.array(boosters)
            boosters.reshape(win_l, win_w)
            predicted = np.array(predicted)
            predicted.reshape(win_l, win_w)
            for dx in range(boosters.shape[-1] - kernal + 1):
                for dy in range(boosters.shape[-1] - kernal + 1):
                    # get each 7*7 predicted from 18*18
                    pre = predicted[dy:dy + kernal, dx:dx + kernal]
                    # remove the t0 data (7*7)
                    boosters[dy:dy + kernal, dx:dx + kernal].test_data_x[i * prediction_len] = \
                        boosters[dy:dy + kernal, dx:dx + kernal].test_data_x[i * prediction_len][1:]
                    # add the newly predicted (t+1) to update the boosters
                    boosters[dy:dy + kernal, dx:dx + kernal].test_data_x[i * prediction_len] = \
                        boosters[dy:dy + kernal, dx:dx + kernal].test_data_x[i * prediction_len] + [pre]

            boosters.reshape(1, win_l * win_w)
            predicted.reshape(1, kernal * kernal)
            prediction_seqs.append(predicted)  # store each predicted 18*18 frame for a prediction_len
        prediction_seqs_all.append(prediction_seqs)
    return prediction_seqs_all


if __name__ == '__main__':
    coor_path = '../../IoT_HeatIsland_Data/data/LA/exp_data/2019042910withCoordinate.csv'
    data_path = '../../IoT_HeatIsland_Data/data/LA/exp_data/tempMatrix_LA.csv'
    configs = json.load(open('../config.json', 'r'))

    m_path = '../../IoT_HeatIsland_Data/data/LA/exp_data/xgboost_models/models/'
    m_test_x_path = '../../IoT_HeatIsland_Data/data/LA/exp_data/xgboost_models/models_test_x/'
    m_test_y_path = '../../IoT_HeatIsland_Data/data/LA/exp_data/xgboost_models/models_test_y/'

    # 区域的右下角坐标
    leftdown = [287, 236]
    # 区域的长宽
    length = 18
    width = 18

    input_timesteps = configs['data']['sequence_length'] - 1
    prediction_length = configs['data']['prediction_length']
    pred_times_num = 6

    # 此处写死cnn的处理大小为7*7
    cnn_kernel = 7

    if os.listdir(m_path) == []:
        # # organizing data
        # train_all, test_all = data_ready(leftdown, length, width, coor_path, data_path, configs)
        # np.save('../../IoT_HeatIsland_Data/data/LA/exp_data/processed/train_x.npy', train_all)
        # np.save('../../IoT_HeatIsland_Data/data/LA/exp_data/processed/test_x.npy', test_all)

        # load directly if file exist
        train_all = np.load('../../IoT_HeatIsland_Data/data/LA/exp_data/processed/train_x.npy')
        test_all = np.load('../../IoT_HeatIsland_Data/data/LA/exp_data/processed/test_x.npy')

        print("trainX %s, testX data %s ready" % (train_all.shape, test_all.shape))

        start = time.time()
        all_models = get_trees(cnn_kernel, input_timesteps, train_all, test_all, m_path, m_test_x_path, m_test_y_path)
        end = time.time()
        print('model building time:', end - start)
    else:
        # load models directly
        all_models = load_models(m_path, m_test_x_path, m_test_y_path)
        all_models = np.array(all_models)
        all_models = all_models.reshape(4, length - cnn_kernel + 1, width - cnn_kernel + 1)  # 4*12*12

    # model reorganize to remove overlays (from 4 different 12*12 to 1 18*18)
    all_models = padding(cnn_kernel, length, width, all_models)
    all_models = all_models.flatten()

    predictions = predict_multiple(all_models, prediction_length, cnn_kernel, length, width)
