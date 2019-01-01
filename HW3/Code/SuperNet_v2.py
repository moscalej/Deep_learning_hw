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


def fire_module(x, squeeze=8, expand=20):
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

        out_layer = Conv2D(expand, (kernel_size, 1),
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


def squeeze_and_exite(in_block, ch=5, ratio=7):
    x = GlobalAveragePooling2D()(in_block)
    x = LeakyReLU()(Dense(ratio)(x))
    x = Dropout(0.2)(Dense(ch, activation='sigmoid')(x))
    return multiply([in_block, x])


def mul_fire(x, res=None, squeeze=10, expand=10):
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    # if res is not None:
    #     x = concatenate([x, res], axis=3)
    # x = Conv2D(expand * 4, (1, 1))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    return x


def multi_nice(conv0, squeeze=3, expand=10, out=20, ratio=4, plus=True):
    multi_1 = mul_fire(conv0, squeeze=squeeze, expand=expand)
    multi_2 = mul_fire(multi_1, res=conv0, squeeze=squeeze, expand=expand)
    multi_4 = add([multi_1, multi_2]) if (plus) else multi_2
    # squese = squeeze_and_exite(multi_4, out, ratio=ratio)

    return multi_4


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
    return center


input = Input(input_shape)

conv0 = Start_block(input)
pull_1 = MaxPool2D()(conv0)
super_2 = mul_fire(pull_1, res=pull_1, squeeze=7, expand=10)
pull_2 = MaxPool2D()(super_2)
super_3 = mul_fire(pull_2, res=pull_2, squeeze=7, expand=9)
pull_3 = AveragePooling2D()(super_3)
# pull_3=smart_bottleneck(super_3,80,ratio=7,k_size=(3,3))


FL1 = LeakyReLU()(Dense(61, kernel_regularizer=keras.regularizers.l2(weight_decay))(Flatten()(pull_3)))

out = Activation('softmax')(Dense(10)(FL1))

model = Model(inputs=input, outputs=out)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
model.summary()


def run():
    Tf_log = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW3\TF\Super_net_o_v7'
    Model_save_p = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW3\saved_models\Super_net_o_v7.h5'
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
                             zca_whitening=False,
                             zca_epsilon=1e-06,
                             rotation_range=0.15,
                             width_shift_range=0.15,
                             height_shift_range=0.15,
                             brightness_range=None,
                             shear_range=0.0,
                             zoom_range=0.1,
                             channel_shift_range=0.0,
                             fill_mode='nearest',
                             cval=0.0,
                             horizontal_flip=True,
                             vertical_flip=True,
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
                                  patience=2,
                                  min_lr=0.00000001,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)

    history_fully = model.fit_generator(img.flow(x_train, y_train, batch_size=1024), steps_per_epoch=48,
                                        shuffle=True,
                                        epochs=150,
                                        initial_epoch=0,
                                        validation_data=(x_test, y_test), callbacks=[tbCallBack, reduce_lr])
    model.save(Model_save_p)
    return history_fully
