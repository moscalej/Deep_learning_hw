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
call_num_123 = 1
def model_S(weight_decay=1e-5 ,epochs=30,factor=0.6 , patience=2, s_1=7,ex_1=10,s_2=10,ex_2=20,use_redux=False,
          drop = 0.3, plus= True, dense_num= 62):
    global call_num_123
    print(call_num_123)
    call_num_123 += 1
    try:
        def fire_module(x, squeeze=8, expand=20):
            def convolution(input, kernel_size):
                # One Convolution 3*3
                if kernel_size == 1:
                    out_layer = Conv2D(expand, (1, 1), padding='same',
                                       kernel_regularizer=keras.regularizers.l2(weight_decay))(input)
                    out_layer = BatchNormalization()(out_layer)
                    out_layer = LeakyReLU()(out_layer)
                    return out_layer

                if use_redux:
                    out_layer = Conv2D(squeeze, (1, 1), padding='same',
                                       kernel_regularizer=keras.regularizers.l2(weight_decay))(input)
                    out_layer = BatchNormalization()(out_layer)
                    out_layer = LeakyReLU()(out_layer)
                else:
                    out_layer = input

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


        def mul_fire(x, res=None, squeeze=10, expand=10,plus=False):
            x = fire_module(x, squeeze=squeeze, expand=expand)
            x = fire_module(x, squeeze=squeeze, expand=expand)
            multi_4 = add([x, conv0]) if plus else x

            return multi_4


        def multi_nice(conv0, squeeze=3, expand=10, out=20, ratio=4, plus=plus):
            multi_1 = mul_fire(conv0, squeeze=squeeze, expand=expand)
            multi_2 = mul_fire(multi_1, res=conv0, squeeze=squeeze, expand=expand)
            multi_4 = add([multi_1, conv0]) if (plus) else multi_2
            # squese = squeeze_and_exite(multi_4, out, ratio=ratio)

            return multi_4


        def Start_block(input):
            conv0 = Activation('tanh')(Conv2D(15, (5, 5),
                                              kernel_regularizer=keras.regularizers.l2(weight_decay))(input))
            conv1 = LeakyReLU()(BatchNormalization()((Conv2D(20, (3, 3), strides=(1, 1),
                                                             kernel_regularizer=keras.regularizers.l2(weight_decay))(conv0))))
            return mul_fire(conv1, res=conv1, squeeze=7, expand=10)


        input = Input(input_shape)
        conv0 = Start_block(input)
        pull_1 = MaxPool2D()(conv0)
        super_2 = mul_fire(pull_1, res=pull_1, squeeze=s_1, expand=ex_1, plus=plus)
        pull_2 = MaxPool2D()(super_2)
        super_3 = mul_fire(pull_2, res=pull_2, squeeze=s_2, expand=ex_2,plus=plus)
        pull_3 = AveragePooling2D()(super_3)

        FL1 = LeakyReLU()(Dense(dense_num, kernel_regularizer=keras.regularizers.l2(weight_decay))(Flatten()(pull_3)))

        out = Activation('softmax')(Dense(10)(FL1))


        model = Model(inputs=input, outputs=out)
        model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
        print("hola")
    except :

        return 0

    config = "_".join([weight_decay ,epochs,factor , patience, s_1,ex_1,s_2,ex_2,use_redux,
          drop, plus, dense_num])
    def run(epochs=epochs,factor=factor , patience=patience,config=config):

        Tf_log = r"C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW3\TF\{config}".format(config=config)
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
                                      factor=factor,
                                      patience=patience,
                                      min_lr=0.00000001,
                                      embeddings_layer_names=None,
                                      embeddings_metadata=None)

        history_fully = model.fit_generator(img.flow(x_train, y_train, batch_size=1024), steps_per_epoch=48,
                                            shuffle=True,
                                            epochs=epochs,
                                            initial_epoch=0,
                                            validation_data=(x_test, y_test), callbacks=[ reduce_lr,tbCallBack])

        return history_fully.history['val_acc'][0]



    if int(np.sum([K.count_params(p) for p in set(model._collected_trainable_weights)])) > 50_000:
        print("Falil" )
        return 0

    history_fully = run(epochs,factor=factor , patience=patience,config=config)
    print(history_fully)
    print()
    return history_fully