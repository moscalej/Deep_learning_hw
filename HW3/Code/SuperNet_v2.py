# Here  we will build a inception Network with squeeze and exite
import os
from keras.layers import *

from keras.models import Model, load_model
from keras.layers import Input, Dense
import keras

num_classes = 10
batch_size = 1024
epochs = 30
input_shape = [32, 32, 3]
weight_decay = 2e-5


def fire_module(x, squeeze=8, expand=20):
    def convolution(input, kernel_size):
        # One Convolution 3*3
        if kernel_size == 1:
            out_layer = Conv2D(expand, (1, 1), padding='same',
                               kernel_regularizer=keras.regularizers.l2(weight_decay))(input)
            out_layer = BatchNormalization()(out_layer)
            out_layer = LeakyReLU()(out_layer)
            return out_layer

        out_layer = Conv2D(squeeze, (1, 1), padding='same',
                           kernel_regularizer=keras.regularizers.l2(weight_decay), )(input)
        out_layer = BatchNormalization()(out_layer)
        out_layer = LeakyReLU()(out_layer)

        out_layer = Conv2D(expand, (kernel_size, 1),
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           padding='same', )(out_layer)
        out_layer = BatchNormalization()(out_layer)
        out_layer = LeakyReLU()(out_layer)
        out_layer = Conv2D(expand, (1, kernel_size),
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           padding='same', )(out_layer)
        out_layer = BatchNormalization()(out_layer)
        out_layer = LeakyReLU()(out_layer)
        return out_layer

    # One Convolution 1*1
    left_1 = convolution(x, 1)
    left_2 = convolution(x, 3)

    # Avrage Pulling
    left_3 = AveragePooling2D(pool_size=(2, 2), padding='same', strides=(1, 1))(x)
    left_3 = Conv2D(expand, (1, 1), padding='same')(left_3)
    left_3 = LeakyReLU()(left_3)

    # One Convolution 5*5
    left_4 = convolution(x, 5)

    x = concatenate([left_1, left_2, left_3, left_4], axis=3)
    return x


def squeeze_and_exite(in_block, ch=5, ratio=7):
    x = GlobalAveragePooling2D()(in_block)
    x = LeakyReLU()(Dense(ratio)(x))
    x = Dropout(0.2)(Dense(ch, activation='sigmoid')(x))
    return multiply([in_block, x])


def mul_fire(x, res=None, squeeze=10, expand=10):
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    if res is not None:
        x = concatenate([x, res], axis=3)
    x = Conv2D(expand * 4, (1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def multi_nice(conv0, squeeze=3, expand=10, out=20, ratio=4, plus=True):
    multi_1 = mul_fire(conv0, squeeze=squeeze, expand=expand)
    multi_2 = mul_fire(multi_1, res=conv0, squeeze=squeeze, expand=expand)
    multi_4 = add([multi_1, multi_2]) if (plus) else multi_2
    squese = squeeze_and_exite(multi_4, out, ratio=ratio)

    return squese


def Start_block(input):
    conv0 = Activation('tanh')(Conv2D(15, (5, 5),
                                      kernel_regularizer=keras.regularizers.l2(1.35 * weight_decay))(input))
    conv1 = LeakyReLU()(BatchNormalization()((Conv2D(20, (3, 3), strides=(1, 1),
                                                     kernel_regularizer=keras.regularizers.l2(weight_decay))(conv0))))
    return mul_fire(conv1, res=conv1, squeeze=7, expand=10)


def smart_bottleneck(input, out, ratio=7, k_size=(2, 2)):
    left = MaxPool2D()(input)
    right = Activation('relu')(Conv2D(out // 2, k_size, strides=(2, 2))(input))
    center = Activation('relu')(Conv2D(out, (1, 1))(concatenate([left, right], axis=3)))
    squese = squeeze_and_exite(center, out, ratio=ratio)
    return squese


input = Input(input_shape)

conv0 = Start_block(input)
pull_1 = MaxPool2D()(conv0)
super_2 = mul_fire(pull_1, res=pull_1, squeeze=5, expand=12)
pull_2 = MaxPool2D()(super_2)
super_3 = mul_fire(pull_2, res=pull_2, squeeze=7, expand=15)
pull_3 = MaxPool2D()(super_3)
# pull_3=smart_bottleneck(super_3,80,ratio=7,k_size=(3,3))


FL1 = Dropout(0.4)(LeakyReLU()(Dense(16, kernel_regularizer=keras.regularizers.l2(weight_decay))(Flatten()(pull_3))))

out = Activation('softmax')(Dense(10)(FL1))

model = Model(inputs=input, outputs=out)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
model.summary()

# #
# tbCallBack = keras.callbacks.TensorBoard(log_dir='.Code/logs/Multi_net_v2_run4/', histogram_freq=0, batch_size=32, write_graph=True,
#                                          write_grads=True, write_images=True, embeddings_freq=0,
#                                          embeddings_layer_names=None, embeddings_metadata=None)
#
# history_fully = model.fit(x_train, y_train, epochs=1, batch_size=1024, validation_data=(x_test, y_test),callbacks=[tbCallBack])
#
model.save('superNet_v2.h5')
