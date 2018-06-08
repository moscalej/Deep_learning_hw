import numpy as np
import pickle
import keras
from keras.datasets import cifar10, cifar100
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.initializers import Constant
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras import regularizers

#################
### CONSTANTS ###
#################
batch_size = 32
num_classes = 10
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3

######################
### PRE PROCESSING ###
######################
def preproces_cfar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # X dimensions should be:
    #
    # (<number of samples * number of input channels>,
    #  <input channel height>,
    #  <input channel width>,
    #  <1>)

    x_train = K.cast_to_floatx(x_train) / 255
    x_train = x_train.reshape(-1, 32, 32, 3)

    x_test = K.cast_to_floatx(x_test) / 255
    x_test = x_test.reshape(-1, 32, 32, 3)

    #     num_train_samples = (x_train.shape[0], 'train samples')
    #     num_test_samples = (x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    def normalize(X_train, X_test):
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    x_train, x_test = normalize(x_train, x_test)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test

# TODO: fix this
#     x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=0.3,
#                                           random_state=42,stratify=y_train)
