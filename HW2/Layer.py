from Macros import *
from Nodes import node_factory
import math
import numpy as np


#################
#  Layer class  #
#################

class Layer:

    def __init__(self, layer_input, layer_output, non_linearity, regularization, learning_rate, weight_decay=0):
        """

        :param weight_decay:
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
        self.weight_decay = weight_decay
        self.input = layer_input
        self.output = layer_output
        self.learning_rate = learning_rate
        self.non_linearity = node_factory(non_linearity)
        self.regularization = regularization
        self.multiplication = node_factory('multi')
        self.addition = node_factory('add')
        self.weights = self._initialize_weights()
        self.bias = self._initialize_biases()
        self.weights_norm = np.sum(np.square(self.weights))
        if self.regularization == REGULARIZATION_OPTIONS[1]:  # L2
            self.weights_norm = np.square(self.weights_norm)

    def forward(self, input):
        """
        Recalcutate the norm of the weights matrix to be used later for regularization.
        Calculate the forward value

        :param input: [prevuis_layer_dim,1]
        :return: values of the non linear [this layer dim,1]
        """
        self.weights_norm = np.linalg.norm(self.weights)
        if self.regularization == REGULARIZATION_OPTIONS[0]:
            self.weights_norm = np.square(self.weights_norm)

        forward_mult = self.multiplication.forward(input, self.weights)
        forward_add = self.addition.forward(forward_mult, self.bias)
        return self.non_linearity.forward(forward_add)

    def backward(self, gradiant_in):
        """
        Control the backward of the layer and will update the values of W and b
        :param gradiant_in:
        :return: (wights, bias) tuple. derivatives to the previous layer [ ]
        """
        backward_non_linearity = self.non_linearity.backward(gradiant_in)
        backward_add, grad_b = self.addition.backward(backward_non_linearity)
        backward_mult_x, backward_mult_w = self.multiplication.backward(backward_add)

        # TODO Re check this part
        self.bias -= self.learning_rate * np.mean(grad_b, axis=1).reshape(self.bias.shape)
        if self.regularization == REGULARIZATION_OPTIONS[1]:    # L2
            self.weights -= self.learning_rate * (backward_mult_w + 2 * self.weight_decay * self.weights)
        else:                                                   # L1
            self.weights -= self.learning_rate * (backward_mult_w + self.weight_decay * self.weights)
        return backward_mult_x

    def _initialize_weights(self):
        """
        :return: a vector of dimensions w X n where w is the size of this layer's output
        and n is the size of this layer's input
        """
        val = 1 / (math.sqrt(self.input))
        return np.random.uniform(-val, val, [self.output, self.input])

    def _initialize_biases(self):
        """

        :return: a vector of dimension n, where n is the size of this layer's input.
        """
        return np.zeros([self.output, 1])


if __name__ == "__main__":
    for non_linear in ["relu", "sigmoid", "softmax", "none"]:
        for loss in ['l1', 'l2']:
            layer1 = Layer(9, 3, non_linear, loss, 0.2, 0.1)
            layer_input = np.ones([9, 8])
            forward_1 = layer1.forward(layer_input)
            backward_1_bias = layer1.backward(forward_1)
            print('Pass')
