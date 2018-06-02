# Here  we will build a inception Network with squeeze and exite

from keras.layers import *

from keras.models import Model
from keras.layers import Input, Dense
import keras

num_classes = 10
batch_size = 1024
epochs = 30
input_shape = [32, 32, 3]
weight_decay = 5e-5


def fire_module(x, squeeze=8, expand=20, name=''):
    def convolution(input, kernel_size):
        # One Convolution 3*3
        if kernel_size == 1:
            out_layer = Conv2D(expand, (1, 1), padding='same',
                               kernel_regularizer=keras.regularizers.l1(weight_decay),
                               name=f'Inc_{name}_cov{kernel_size}_expand')(input)
            out_layer = BatchNormalization(name=f'Inc_{name}_cov{kernel_size}_BN_2')(out_layer)
            out_layer = Activation('relu', name=f'Inc_{name}_cov{kernel_size}_Active2')(out_layer)
            return out_layer

        out_layer = Conv2D(squeeze, (1, 1), padding='same',
                           kernel_regularizer=keras.regularizers.l1(weight_decay),
                           name=f'Inc_{name}_cov{kernel_size}_Sque')(input)
        out_layer = BatchNormalization(name=f'Inc_{name}_cov{kernel_size}_BN_1__Sque')(out_layer)
        out_layer = Activation('relu', name=f'Inc_{name}_cov{kernel_size}_Active1__Sque')(out_layer)
        out_layer = Conv2D(expand, (kernel_size, kernel_size),
                           kernel_regularizer=keras.regularizers.l1(weight_decay),
                           padding='same',
                           name=f'Inc_{name}_cov{kernel_size}_expand')(out_layer)
        out_layer = BatchNormalization(name=f'Inc_{name}_cov{kernel_size}_BN_2')(out_layer)
        out_layer = Activation('relu', name=f'Inc_{name}_cov{kernel_size}_Active2')(out_layer)
        return out_layer

    # One Convolution 1*1
    left_1 = convolution(x, 1)
    left_2 = convolution(x, 3)

    # Avrage Pulling
    left_3 = AveragePooling2D(pool_size=(2, 2), padding='same', strides=(1, 1)
                              , name=f'Inc_{name}_Pulling')(x)
    left_3 = Conv2D(expand, (1, 1), padding='same', name=f'Inc_{name}_Pulling_conv')(left_3)
    left_3 = Activation('relu', name=f'Inc_{name}_Pulling_Active')(left_3)

    # One Convolution 3*3
    left_4 = convolution(x, 5)

    x = concatenate([left_1, left_2, left_3, left_4], axis=3)
    return x


def squeeze_and_exite(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch // ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return multiply([in_block, x])


input = Input(input_shape)
conv0 = Activation('tanh')(Conv2D(5, (7, 7))(input))
fire_1 = fire_module(conv0, name='first', squeeze=4, expand=8)
pull_1 = MaxPool2D()(squeeze_and_exite(fire_1, 32, ratio=9))
fire_2 = fire_module(pull_1, name='second', squeeze=9, expand=15)
pull_2 = MaxPool2D()(squeeze_and_exite(fire_2, 60, ratio=9))
fire_3 = fire_module(pull_2, name='fird', squeeze=10, expand=25)
pull_3 = MaxPool2D()(squeeze_and_exite(fire_3, 100, ratio=9))

fire_4 = fire_module(pull_3, name='four', squeeze=10, expand=25)
pull_4 = AveragePooling2D((3, 3))(squeeze_and_exite(fire_4, 100, ratio=9))

FL1 = Dropout(0.4)(Dense(64, kernel_regularizer=keras.regularizers.l2(weight_decay))(Flatten()(pull_4)))

out = Activation('softmax')(Dense(10)(FL1))

model = Model(inputs=input, outputs=out)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
model.summary()
