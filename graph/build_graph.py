"""
Created on  2019-06-25
@author: Jingchao Yang

Building a universal graph from unique nodes
https://github.com/jingchaoyy/STGCN_IJCAI-18/blob/master/utils/math_graph.py
"""
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix


def weight_matrix(coors, sigma2=0.1, epsilon=0.5):
    '''
    Load weight matrix function.
    :param coors: all coordinates.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :return: np.ndarray, [n_route, n_route].
    '''

    # building distance matrix
    W = pd.DataFrame(distance_matrix(coors.values, coors.values), index=coors.index, columns=coors.index)
    print('Distance Matrix Built:', W.shape)

    n = W.shape[0]
    # W = W / 10000.
    W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)

    # refer to Eq.10
    W_result = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    print('Weighted Matrix Built:', W_result)
    return W_result


file = 'uniqueNodes.csv'
uni = pd.read_csv(file)
uni_coor = uni[['Latitude_SW', 'Longitude_SW']]
# print(uni_coor)

weight_matrix(uni_coor)
