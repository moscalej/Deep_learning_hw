
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

        :param layer_input: An int. The dimension of our input
        :param layer_output: An int. The dimension of our output
        :param non_linearity: “nonlinear” string, whose possible values are: “relu”, “sigmoid”, “sotmax” or “none”
        :param regularization: “regularization” string, whose possible values are: “l1” (L1 norm), or “l2” (L2 norm)
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
        # self.regularization = node_factory(regularization)  # Todo need to check this part
        self.multiplication = node_factory('multi')
        self.addition = node_factory('add')
        self.weights = self._initialize_weights()
        self.bias = self._initialize_biases()

    def forward(self, input):
        '''
        Calculate the forward value

        :param input: [prevuis_layer_dim,1]
        :return: values of the non linear [this layer dim,1]
        '''

        forward_mult = self.multiplication.forward(input, self.weights)
        forward_add = self.addition.forward(forward_mult, self.bias)
        return self.non_linearity.forward(forward_add)

    def backward(self, det_in):
        """
        Control the backward of the layer and will update the values of W and b
        :param det_in:
        :return: derivatives to the previous layer [ ]
        """
        pass

    def _initialize_weights(self):
        """

        :return: a vector of dimensions w X n where w is the size of this layer's output
        and n is the size of this layers's input
        """
        val = 1 / (math.sqrt(self.input))
        return np.random.uniform(-val, val, [self.output, self.input])

    def _initialize_biases(self):
        """

        :return: a vector of dimension n, where n is the size of this layer's input.
        """
        return np.zeros([self.output, 1])


if __name__ == "__main__":
    layer1 = Layer(9, 3, "sigmoid", "l1", 0.2)
    layer_input = np.ones([9, 1])
    out = layer1.forward(layer_input)
