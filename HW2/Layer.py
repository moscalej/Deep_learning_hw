
#################
#  Layer class  #
#################


from Macros import *
from Nodes import node_factory
import math
import numpy as np


class Layer:

    def __init__(self, layer_input, layer_output, non_linearity, regularization, learning_rate):
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
        self.learning_rate = learning_rate
        self.non_linearity = node_factory(non_linearity)
        self.regularization = node_factory(regularization)  # Todo need to check this part
        self.multiplication = node_factory('add')
        self.addition = node_factory('multi')
        self.weights = self._initialize_weights()
        self.bias = self._initialize_biases()

    def forward(self, input):
        '''
        Calculate the forward value

        :param input: [prevuis_layer_dim,1]
        :return: values of the non linear [this layer dim,1]
        '''
        return self.non_linearity(self.addition(self.multiplication(input, self.weights), self.addition(self.bias)))

    def backward(self, det_in):
        """
        Control the backward of the layer and will update the values of W and b
        :param det_in:
        :return: derivatives to the previous layer [ ]
        """
        pass

    def _initialize_weights(self):
        val = 1 / (math.sqrt(self.input))
        return np.random.uniform(-val, val)

    def _initialize_biases(self):
        return np.zeros(self.input)