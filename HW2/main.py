import pickle
import gzip
import urllib.request
from sklearn.preprocessing import scale
from mydnn import MyDNN
from Macros import generate_layer


if __name__ == "__main__":
    # data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    # urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    training_samples = scale(train_set[0], axis=0, with_std=False)
    validation_samples = scale(valid_set[0], axis=0, with_std=False)
    training_classifications = train_set[1]
    validation_classifications = valid_set[1]
    num_samples, num_pixels = training_samples.shape

    layers = [generate_layer(num_pixels, 100, "relu", "l2", 0.2)]
    layers.append(generate_layer(100, 50, "relu", "l2", 0.2))
    layers.append(generate_layer(50, 10, "softmax", "l2", 0.2))

    net = MyDNN(layers, "cross-entropy")  # MSE
    log = {}
    net.fit(training_samples, training_classifications, 200, 512, 0.4, validation_samples, validation_classifications)
