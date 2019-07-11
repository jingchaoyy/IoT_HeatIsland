"""
Created on  2019-07-06
@author: Jingchao Yang
"""
from numpy import array


def split_sequence(sequence, n_steps_in, n_steps_out):
    """

    :param sequence: e.g., [10, 20, 30, 40, 50, 60, 70, 80, 90]
    :param n_steps_in: 3, e.g., [10, 20, 30]
    :param n_steps_out: 2, e.g., [40, 50]
    :return:
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        print(seq_x, seq_y)
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
