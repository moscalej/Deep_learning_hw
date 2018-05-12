import pickle
import gzip
import numpy as np
import urllib.request
from sklearn.preprocessing import scale
from mydnn import MyDNN
import Macros


def generate_layer(input_dims, output_dims, non_linearity, regularization, learning_rate):
    return {
        Macros.INPUT: input_dims,
        Macros.OUTPUT: output_dims,
        Macros.NON_LINEAR: non_linearity,
        Macros.REGULARIZATION: regularization,
        Macros.LEARNING_RATE: learning_rate
    }








if __name__ == "__main__":
    data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    training_samples = scale(train_set[0], axis=0, with_std=False)
    validation_samples = scale(valid_set[0], axis=0, with_std=False)

    num_samples, num_pixels = training_samples.shape

    layers = [generate_layer(num_pixels, 100, "relu", "l2", 0.2)]
    layers.append(generate_layer(100, 50, "relu", "l2", 0.2))
    layers.append(generate_layer(50, 9, "softmax", "l2", 0.2))

    net = MyDNN(layers, "MSE")  # MSE
    net.fit(training_samples, train_set[1], 300, 1, 0.02, validation_samples, valid_set[1])
