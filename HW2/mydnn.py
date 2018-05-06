import time
import numpy as np
import math


#################
### CONSTANTS ###
#################

INPUT = "input"
OUTPUT = "output"
NON_LINEAR = "non_linear"
REGULARIZATION = "regularization"

NON_LINEAR_OPTIONS = ["relu", "sigmoid", "softmax", "none"]
REGULARIZATION_OPTIONS = ["l1", "l2"]
LOSS_OPTIONS = ["MSE", "cross-entropy"]


######################
### CORE DNN CLASS ###
######################





class MyDNN:
    def __init__(self, architecture, loss, weight_decay=0):
        """
        :param arch: A list of dictionaries used to initialize Layer objects
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
            new_layer = Layer(layer_input, layer_output, non_linearity, regularization)
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

    def forward(self):
        current_input =  # something
        for layer in layers:
            current_input = layer.forward(current_input)


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

class ReluLayer(Layer):

    def __init__(self, layer_input):
        super(ReluLayer).__init__()
        self.layer_input = layer_input
        self.func_forward = lambda x: np.maximum(0, x)
        self.func_back =  # something to do

    def forward(self):
        Addition.forward()
        Multiplication.forward()
        nonlinaera.forward()

    def backward(self):
        before_relu = np.lf.layer_input.T @ self.weights + self.bias
        self.layer_input = self.func_back(self.layer_output)
        pass


class Addition(Node):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.forward_result = None
        self.backward_result = None

    def forward(self):
        self.forward_result = self.a + self.b
        return self.forward_result

    def backward(self, incoming_gradient):
        self.backward_result = incoming_gradient
        return self.backward_result


class Multiplication(Node):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return a * b

    def backward(self):
        return


class




