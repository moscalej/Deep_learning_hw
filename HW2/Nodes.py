
import numpy as np
import math
from abc import abstractmethod


class Node:
    """
    This class is for the nodes from a single input to a single output
    this can not be a layer given that a layer has also Weights and Bias
    to memorize and different methods to update the values
    This nodes has Only Forward and Backwards
    """
    def __init__(self):
        """
        Saves the values for the backward part of the proccess "Working point"
        The forward part is only execute the lambda function , and probably will be
        the backward
        """
        self.value = None
        self.func_forward = None

    def forward(self, forward_in):
        self.value = forward_in  # on forward we save x
        return self.func_forward(forward_in)

    @abstractmethod
    def backward(self, back_received):
        pass


class Gate():
    """
    This class is for the nodes from a Two input to a single output
    this can not be a layer given that a layer has also Weights and Bias
    to memorize and different methods to update the values
    This nodes has Only Forward and Backwards
    """
    def __init__(self):
        self.value = None
        self.func_forward = None

    def forward(self, X, W):
        self.value = [X, W]
        return self.func_forward(X, W)

    @abstractmethod
    def backward(self, back_received):
        pass


class Relu(Node):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x: np.maximum(x, 0)

    def backward(self, back_received):
        self.value[self.value >= 0] = 1
        self.value[self.value < 0] = 0
        return self.value * np.sum(back_received)  # todo be sure about bacward


class Sigmoid(Node):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x: 1 / (1 + np.exp(-1 * x))
        self.func_backward = lambda x: self.func_forward(x)(1 - self.func_forward(x))  # sigmoid '

    def backward(self, back_received):
        derivative_input = self.func_backward(self.value)
        return derivative_input * np.sum(back_received)


class SoftMax(Node):  # todo just copy past from adove
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x: np.exp(x) / np.sum(np.exp(x))
        self.func_backward = lambda x: self.func_forward(x)(1 - self.func_forward(x))

    def backward(self, back_received):
        """
        Need to be done
        :param back_received:
        :return:
        """
        pass


class NoneNode(Node):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x: x
        self.func_backward = lambda x: x

    def backward(self, back_received):
        super().backward(back_received)


class Multiplication(Gate):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x, y: x.T @ y

    def backward(self, back_received):
        val_1 = np.ones(self.value[0].shape) * np.mean(back_received)
        val_2 = np.ones(self.value[1].shape) * np.mean(back_received)
        return val_1, val_2


class Add_node(Gate):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x, y: x + y

    def backward(self, back_received):
        pass


def node_factory(node_name):
    nodes = dict(
        multi=Multiplication,
        add=Add_node,
        relu=Relu,
        sigmoid=Sigmoid,
        softmax=SoftMax
    )
    return nodes[node_name]()
