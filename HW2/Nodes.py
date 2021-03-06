import numpy as np
from abc import abstractmethod

import unittest
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


class Gate:
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
        return self.value * back_received


class TanH(Node):
    def __init__(self):
        super(TanH, self).__init__()
        self.func_forward = lambda x: (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

    def backward(self, back_received):
        result = back_received * ((-4 * np.exp(-2 * self.value)) / (1 - np.exp(-2 * self.value) ** 2))
        return result


class Sigmoid(Node):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x: 1 / (1 + np.exp(-1 * x))
        self.func_backward = lambda x: self.func_forward(x) * (1 - self.func_forward(x))  # sigmoid '

    def backward(self, back_received):
        derivative_input = self.func_backward(self.value)
        return derivative_input * back_received


class NoneNode(Node):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x: x
        self.func_backward = lambda x: x

    def backward(self, back_received):
        return back_received


class Multiplication(Gate):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda X, W: W @ X

    def backward(self, back_received):
        gx = self.value[1].T @ back_received  # W.T dot DL/Dmul
        gw = back_received @ self.value[0].T  # DL/Dmul dot X.T
        return gx, gw


class Add_node(Gate):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x, y: x + y

    def backward(self, back_received):
        return back_received, back_received


class SoftMax(Node):

    def __init__(self):
        super().__init__()
        self.func_forward = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
        self.jacobian = lambda S: np.diag(S) - np.outer(S, S)

    def backward(self, back_received):
        S = self.func_forward(self.value).T
        gradiant = back_received.T
        out = np.zeros(self.value.shape).T
        for index, sample in enumerate(S):
            jac = self.jacobian(sample)
            out[index] = gradiant[index] @ jac

        return out.T


class Loss:
    def __init__(self):
        self.input_size_inv = None
        self.error = None
        self.gradient = None
        # self.value = None

    @abstractmethod
    def forward(self, y_hat, y, num_samples):
        pass

    def backward(self):
        return self.gradient

    def get_loss(self):
        return self.error


class MSE(Loss):
    def __init__(self):
        super().__init__()
        self.norm = lambda x, y: np.sum(np.square(x - y))

    def forward(self, y_hat, y, num_samples):
        self.input_size_inv = 1 / num_samples
        sq_norm = self.norm(y_hat, y)
        self.error = (0.5 * sq_norm * self.input_size_inv)
        self.gradient = (y_hat - y) * self.input_size_inv


class Entropy(Loss):
    def __init__(self):
        super().__init__()
        self.func = lambda y, y_hat: - np.sum(y * np.log(y_hat))

    def forward(self, y_hat, y, num_samples):
        self.input_size_inv = 1 / num_samples
        self.error = self.func(y, y_hat) * self.input_size_inv
        inv = - 1 / y_hat
        self.gradient = y * inv * self.input_size_inv


def node_factory(node_name):
    nodes = dict(
        multi=Multiplication,
        add=Add_node,
        relu=Relu,
        tanh=TanH,
        sigmoid=Sigmoid,
        softmax=SoftMax,
        none=NoneNode,
        MSE=MSE,
    )
    nodes['cross-entropy'] = Entropy
    return nodes[node_name]()

