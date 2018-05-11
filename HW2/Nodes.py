
import numpy as np
import math
from abc import abstractmethod


# todo strong check dimentions

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


class Multiplication(Gate):  # its for a matrix and a vector
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x, W: W @ x

    def backward(self, back_received):
        gx = self.value[1].T @ back_received
        gw = back_received @ self.value[0].T
        return gx, gw


class Add_node(Gate):
    def __init__(self):
        super().__init__()
        self.func_forward = lambda x, y: x + y

    def backward(self, back_received):
        return back_received, back_received


class SoftMax(Sigmoid):

    def __init__(self):
        super().__init__()
        self.func_forward = lambda x: np.exp(x) / np.sum(np.exp(x))
        self.func_backward = lambda x: self.func_forward(x) * (1 - self.func_forward(x))


class Loos_l2:
    def __init__(self, imput_size):
        self.imput_size_inv = 1 / imput_size
        self.error = []
        self.gradiand = []
        self.value = None
        self.norm = lambda x, y: ((x - y).T @ (x - y))
    def forward(self, y, y_hat):
        y = np
        sq_norm = self.norm(y, y_hat)
        self.error.append(0.5 * sq_norm * self.imput_size_inv)
        self.gradiand.append(np.sqrt(sq_norm) * self.imput_size_inv)

    def backward(self):
        mean_gradiand = np.mean(self.gradiand)
        self.gradiand = []
        return mean_gradiand


class Loos_l1:  # Todo need to check this function
    def __init__(self, imput_size):
        self.imput_size_inv = 1 / imput_size
        self.error = []
        self.gradiand = []
        self.value = None
        self.norm = lambda x, y: np.sum(np.abs(x - y))

    def forward(self, y, y_hat):
        sq_norm = self.norm(y, y_hat)
        self.error.append(sq_norm * self.imput_size_inv)
        self.gradiand.append(np.sqrt(sq_norm) * self.imput_size_inv)

    def backward(self):
        mean_gradiand = np.mean(self.gradiand)
        self.gradiand = []
        return mean_gradiand


def node_factory(node_name):
    nodes = dict(
        multi=Multiplication,
        add=Add_node,
        relu=Relu,
        sigmoid=Sigmoid,
        softmax=SoftMax,
        l2=Loos_l2,
        l1=Loos_l1
    )
    return nodes[node_name]()

if __name__ == '__main__':
    np.random.seed(4)
    x = np.array([1, 2, 3, 4])
    w = np.random.rand(20).reshape([5, 4])
    b = np.random.rand(5)
    m = Multiplication()
    add = Add_node()
    sig = Sigmoid()
    total = sig.forward(add.forward(m.forward(x, w), b))
    ones = np.array([1, 1, 1, 1, 1])
    y = np.random.randint(0, 4, 5)
    y_hat = np.random.randint(0, 4, 5)
    # l = Loos(5)
    # l.forward(y, y_hat)
