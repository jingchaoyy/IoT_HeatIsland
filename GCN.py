"""
Created on  2/27/2019
@author: Jingchao Yang
"""
import keras
from keras_gcn import GraphConv


DATA_DIM = 3

data_layer = keras.layers.Input(shape=(None, DATA_DIM))
edge_layer = keras.layers.Input(shape=(None, None))
conv_layer = GraphConv(
    units=32,
    step_num=1,
)([data_layer, edge_layer])