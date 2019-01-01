"""

#           #           #           #
# Authors:  Alejandro Moscoso <alejandro.moscoso@intel.com>
#
#
"""
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import cifar10
from keras.layers import *

from keras.models import Model
from keras.layers import Input, Dense
import keras

num_classes = 10
batch_size = 1024
epochs = 30
input_shape = [32, 32, 3]
weight_decay = 4e-5


def block_1(x, size):
    layer_0 = Conv2D(size, (3, 3), padding='same')(x)
    layer_2 = BatchNormalization()(layer_0)
    layer_3 = Activation('relu')(layer_2)
    return (layer_3)


def block_2(x, size):
    layer_0 = Conv2D(size, (1, 1), padding='same')(x)
    layer_2 = BatchNormalization()(layer_0)
    layer_3 = Activation('relu')(layer_2)
    return (layer_3)


input_net = Input(input_shape)
layer_0 = block_1(input_net, 5)
layer_0 = block_1(layer_0, 8)
layer_0 = block_1(layer_0, 8)

layer_1 = MaxPool2D((2, 2))(layer_0)
layer_1 = block_1(layer_1, 16)
layer_1 = block_1(layer_1, 16)
layer_2 = MaxPool2D((2, 2))(layer_1)
layer_2 = block_1(layer_2, 16)
layer_3 = MaxPool2D((2, 2))(layer_2)
layer_3 = block_1(layer_3, 32)
layer_3 = block_2(layer_3, 32)
layer_4 = MaxPool2D()(layer_3)
layer_4 = block_1(layer_4, 60)
layer_5 = MaxPool2D()(layer_4)
flat = Flatten()(layer_5)
enbede = Dense(256)(flat)
enbede = BatchNormalization()(enbede)
enbede = LeakyReLU()(enbede)
soft_max = Dense(10, activation='softmax')(enbede)
model = Model(inputs=input_net, outputs=soft_max)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
model.summary()


def run():
    Tf_log = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW3\TF'
    Model_save_p =r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW3\saved_models\finkmos'
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = K.cast_to_floatx(x_train) / 255
    x_train = x_train.reshape(-1, 32, 32, 3)

    x_test = K.cast_to_floatx(x_test) / 255
    x_test = x_test.reshape(-1, 32, 32, 3)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    def normalize(X_train, X_test):
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    x_train, x_test = normalize(x_train, x_test)


    tbCallBack = keras.callbacks.TensorBoard(log_dir=Tf_log,
                                             histogram_freq=0,
                                             batch_size=32,
                                             write_graph=True,
                                             write_grads=True,
                                             write_images=True,
                                             embeddings_freq=0,
                                             embeddings_layer_names=None,
                                             embeddings_metadata=None)
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.5,
                                  patience=2,
                                  min_lr=0.000001,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)

    history_fully = model.fit(x_train, y_train,
                              epochs=50, batch_size=1024,
                              validation_data=(x_test, y_test), callbacks=[tbCallBack, reduce_lr])
    model.save(Model_save_p)
    return history_fully

