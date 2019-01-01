"""

#           #           #           #
# Authors:  Alejandro Moscoso <alejandro.moscoso@intel.com>
#
#
"""
# Here  we will build a inception Network with squeeze and exite
import os

from keras.callbacks import ReduceLROnPlateau
from keras.datasets import cifar10
from keras.layers import *

from keras.models import Model, load_model
from keras.layers import Input, Dense
import keras
from keras_preprocessing.image import ImageDataGenerator

num_classes = 10
batch_size = 1024
epochs = 30
input_shape = [32, 32, 3]
weight_decay = 1e-5


def fire_module(x, squeeze=10, expand=20):
    def convolution(input, kernel_size):
        # One Convolution 3*3
        if kernel_size == 1:
            out_layer = Conv2D(expand, (1, 1), padding='same',
                               kernel_regularizer=keras.regularizers.l2(weight_decay))(input)
            out_layer = BatchNormalization()(out_layer)
            out_layer = LeakyReLU()(out_layer)
            return out_layer

        # out_layer = Conv2D(squeeze, (1, 1), padding='same',
        #                    kernel_regularizer=keras.regularizers.l2(weight_decay))(input)
        # out_layer = BatchNormalization()(out_layer)
        # out_layer = LeakyReLU()(out_layer)

        out_layer = Conv2D(expand, (kernel_size, 1), # todo try to reduce 1x1
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           padding='same', )(x)
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


def mul_fire(x, res=None, squeeze=10, expand=10, dim_out=40):
    x_1 = fire_module(x, squeeze=squeeze, expand=expand)
    x_1 = fire_module(x_1, squeeze=squeeze, expand=expand)
    # out = multiply([x_1,x])
    # left_3 = Conv2D(dim_out, (1, 1), padding='same')(out)
    # left_3 = BatchNormalization()(left_3)
    # left_3 = LeakyReLU()(left_3)

    return x_1


def multi_nice(conv0, squeeze=10, expand=10,dim_out=40):
    multi_1 = mul_fire(conv0, squeeze=squeeze, expand=expand,dim_out=dim_out)
    multi_2 = mul_fire(multi_1, squeeze=squeeze-2, expand=expand,dim_out=dim_out)

    return multi_2



def Start_block(input):
    conv0 = Activation('tanh')(Conv2D(8, (5, 5),
                                      kernel_regularizer=keras.regularizers.l2(1.35 * weight_decay))(input))
    conv1 = LeakyReLU()(BatchNormalization()((Conv2D(16, (3, 3), strides=(1, 1),
                                                     kernel_regularizer=keras.regularizers.l2(weight_decay))(conv0))))
    return mul_fire(conv1, res=conv1, squeeze=7, expand=5)




input = Input(input_shape)

conv0 = Start_block(input)
pull_1 = MaxPool2D()(conv0)
super_2 = multi_nice(pull_1, squeeze=8, expand=6,dim_out=16)
pull_2 = MaxPool2D()(super_2)
super_3 = multi_nice(pull_2, squeeze=16, expand=12,dim_out=32)
pull_3 = MaxPool2D()(super_3)
# super_4 =Conv2D(64,(3,3),activation='relu')(pull_3)

FL1 = LeakyReLU()(Dense(32, kernel_regularizer=keras.regularizers.l2(weight_decay))(Flatten()(pull_3)))

out = Activation('softmax')(Dense(10)(FL1))

model = Model(inputs=input, outputs=out)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
model.summary()


def run():
    Tf_log = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW3\TF\Super_optimize_4'
    Model_save_p = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW3\saved_models\Super_optimize_4.h5'
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

    img = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             rotation_range=15,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             brightness_range=None,
                             shear_range=0.0,
                             zoom_range=0.1,
                             channel_shift_range=0.0,
                             fill_mode='nearest',
                             cval=0.0,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rescale=None,
                             preprocessing_function=None,
                             data_format=None,
                             validation_split=0.)

    img.fit(x_test)

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
                                  factor=0.6,
                                  patience=4,
                                  min_lr=0.00000001,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)

    history_fully = model.fit_generator(img.flow(x_train, y_train, batch_size=1024), steps_per_epoch=48,
                                        shuffle=True,
                                        epochs=1,
                                        initial_epoch=0,
                                        validation_data=(x_test, y_test), callbacks=[tbCallBack, reduce_lr])
    model.save(Model_save_p)
    return history_fully
