"""

#           #           #           #
# Authors:  Alejandro Moscoso <alejandro.moscoso@intel.com>
#
#
"""
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
    layer_0 = Conv2D(size, (3, 3),padding='same')(x)
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
layer_4 = block_1(layer_4,60)
layer_5 = MaxPool2D()(layer_4)
flat = Flatten()(layer_5)
enbede = Dense(256)(flat)
enbede = BatchNormalization()(enbede)
enbede = LeakyReLU()(enbede)
soft_max = Dense(10,activation='softmax')(enbede)
model = Model(inputs=input_net, outputs=soft_max)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
model.summary()

# tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/', histogram_freq=0, batch_size=32, write_graph=True,
#
# history_fully = model.fit(x_train, y_train, epochs=200, batch_size=1024, validation_data=(x_test, y_test))
