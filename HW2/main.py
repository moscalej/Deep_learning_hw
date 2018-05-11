import pickle
import gzip
import numpy as np
import urllib.request
from sklearn.preprocessing import scale
import mydnn
import Macros


def generate_layer(input_dims, output_dims, non_linearity, regularization, learning_rate):
    return {
        Macros.INPUT: input_dims,
        Macros.OUTPUT: output_dims,
        Macros.NON_LINEAR: non_linearity,
        Macros.REGULARIZATION: regularization,
        Macros.LEARNING_RATE: learning_rate
    }


def main():
    data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        training_samples = scale(train_set[0], axis=0, with_std=False)
        validation_samples = scale(valid_set[0], axis=0, with_std=False)

        num_samples, num_pixels = training_samples.shape

        layers = [generate_layer(num_pixels, 100, "relu", "l2", 0.2)]
        layers.append(generate_layer(100, 50, "relu", "l2", 0.2))
        layers.append(generate_layer(50, 1, "softmax", "l2", 0.2))

        mydnn.MyDNN(layers, Macros.LOSS_OPTIONS[0]) # MSE


        results = []
        # A list of tuples. Each entry in the tuples is a vector.
        # These represent y and y_hat respectively.
        for i in range(len(num_samples)):
            results.append((training_samples[1], valid_set[i]))


if __name__ == "__main__":
    main()
