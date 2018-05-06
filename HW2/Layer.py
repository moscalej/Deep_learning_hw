
#################
#  Layer class  #
#################


from Macros import *
from Nodes import node_factory
import math
import numpy as np


class Layer:

    def __init__(self, layer_input, layer_output, non_linearity, regularization):
        """

        :param layer_input:
        :param layer_output:
        :param non_linearity:
        :param regularization:
        """

        # Assertions

        assert isinstance(layer_input, int)
        assert isinstance(layer_output, int)
        assert non_linearity in NON_LINEAR_OPTIONS, \
            (non_linearity + " is not a valid non-linear option")
        assert regularization in REGULARIZATION_OPTIONS, \
            (regularization + " is not a valid regularization option")

        # Attribute setting

        self.input = layer_input
        self.output = layer_output
        self.non_linearity = non_linearity
        self.regularization = regularization
        self.weights = self.initialize_weights()
        self.bias = self.initialize_biases()

    def initialize_weights(self):
        val = 1 / (math.sqrt(self.input))
        return np.random.uniform(-val, val)

    def initialize_biases(self):
        return np.zeros(self.input)