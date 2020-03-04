# make the model graph and then return the last layer
# A simple 3 convolution layer cnn

import tensorflow as tf
import numpy as np
import pandas as pd


class Model():

    def __init__(weights, bias, learning_rate, image_size, dropout):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.dropout = dropout


    def build():



        return out, predictions
