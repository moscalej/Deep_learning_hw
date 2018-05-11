######################
# CORE DNN CLASS ###
######################


import time
import numpy as np
import math
from Macros import *
from Layer import Layer


class MyDNN:
    def __init__(self, architecture, loss, weight_decay=0):
        """
        :param architecture: A list of dictionaries used to initialize Layer objects
        :param loss: a string denoting the type of loss function we should use for this network
        :param weight_decay: a float denoting the weight decay within this network. Default is 0.
        """

        # Assertions

        assert isinstance(architecture, list)
        assert loss in LOSS_OPTIONS, (loss + " is not a valid loss function")
        assert isinstance(weight_decay, float), \
            ("weight_decay should be a float, not a " + str(type(weight_decay)))

        # Attribute setting

        self.layers = []

        for layer in architecture:
            layer_input = layer[INPUT]
            layer_output = layer[OUTPUT]
            non_linearity = layer[NON_LINEAR]
            regularization = layer[REGULARIZATION]
            learning_rate = layer[LEARNING_RATE]
            new_layer = Layer(layer_input, layer_output, non_linearity, regularization, learning_rate)
            self.layers.append(new_layer)

        self.loss = loss
        self.weight_decay = weight_decay

    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, x_val=None, y_val=None):
        """

        :param x_train: a Numpy nd-array where each row is a sample
        :param y_train: a 2d array, the labels of X in one-hot representation for classification
                        or a value for each sample for regression.
        :param epochs: number of epochs to run
        :param batch_size: batch size for SGD
        :param learning_rate:  float, a fixed learning rate that will be used in SGD.
        :param x_val: the validation x data (same structure as train data) default is None.
                      When validation data is given, evaluation over this data will be made at the end of every epoch.
        :param y_val: the corresponding validation y data (labels) whose structure is identical to y_train.
        :return: history - intermediate optimization results, which is a list of dictionaries,
                such that each epoch has a corresponding dictionary containing all relevant
                results. These dictionaries do not contain formatting information (you will
                later use the history to print various things including plots of learning and
                convergence curves for your networks).
        """

        num_samples, num_dimensions = x_train.shape
        pass

    def _forward(self, in_put):
        current_input = in_put  # something
        for layer in self.layers:
            current_input = layer.forward(current_input)

        return current_input

    def _backward(self, gradiand):
        current_gradiant = gradiand
        for layer in self.layers:
            current_gradiant = layer.forward(current_gradiant)
