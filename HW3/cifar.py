import pickle
import keras
from keras.datasets import cifar10, cifar100


#################
### CONSTANTS ###
#################
batch_size = 32
num_classes = 10
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)

    num_train_samples = (x_train.shape[0], 'train samples')
    num_test_samples = (x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
