
import numpy as np
import Macros
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
        # todo need to re do for batch
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
        self.func_forward = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
        self.func_backward = lambda x: self.func_forward(x) * (1 - self.func_forward(x))


class Loss:
    def __init__(self, num_bacht):
        self.input_size_inv = 1 / num_bacht
        self.error = None
        self.gradiant = None
        self.value = None

    @abstractmethod
    def forward(self, y_hat, y):
        pass

    def backward(self):
        return self.gradiant

    def get_loss(self):
        return self.error


class MSE(Loss):
    def __init__(self, num_bacht):
        super().__init__(num_bacht=num_bacht)

        self.norm = lambda x, y: np.sum(np.square(x - y))

    def forward(self, y_hat, y):
        sq_norm = self.norm(y_hat, y)
        self.error = (0.5 * sq_norm * self.input_size_inv)
        self.gradiant = np.sqrt(sq_norm) * self.input_size_inv


class Entropy(Loss):
    def __init__(self, num_bacht):
        super().__init__(num_bacht=num_bacht)
        self.func = lambda y, y_hat: - np.sum(y * np.log(y_hat))

    def forward(self, y_hat, y):
        self.error = self.func(y, y_hat) * self.input_size_inv
        self.gradiant = (y - y_hat) * self.input_size_inv


class Normal_l1:  # Todo need to check this function
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
        l1=Normal_l1
    )
    return nodes[node_name]()

if __name__ == '__main__':
    np.random.seed(4)
    x = np.random.randn(16).reshape([4, 4])
    w = np.random.randn(20).reshape([5, 4])
    b = np.random.randn(5).reshape([5, 1])
    m = Multiplication()
    add = Add_node()
    sig = Sigmoid()
    soft = SoftMax()
    relu = Relu()
    mse = MSE(4)
    ent = Entropy(4)
    total = soft.forward(add.forward(m.forward(x, w), b))
    ent.forward(total, np.ones([5, 4]))
    out = soft.backward(ent.gradiand)
    b_d, a = add.backward(out)
    xm, wm = m.backward(b_d)
