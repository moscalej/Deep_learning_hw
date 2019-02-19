import keras.layers as kl
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
import keras
from keras.regularizers import l2
import numpy as np


def fire_disc(weight_decay: float = 5e-5, input_size: int = 32) -> keras.Model:
    def fire_module(x, squeeze=10, expand=20):
        def convolution(input, kernel_size):
            # One Convolution 3*3
            if kernel_size == 1:
                out_layer = kl.Conv2D(expand, (1, 1), padding='same', kernel_regularizer=l2(weight_decay))(input)
                out_layer = kl.BatchNormalization()(out_layer)
                out_layer = kl.LeakyReLU()(out_layer)
                return out_layer

            out_layer = kl.Conv2D(squeeze, (1, 1), padding='same',
                                  kernel_regularizer=l2(weight_decay))(input)
            out_layer = kl.BatchNormalization()(out_layer)
            out_layer = kl.LeakyReLU()(out_layer)

            out_layer = kl.Conv2D(expand, (kernel_size, 1),  # todo try to reduce 1x1
                                  kernel_regularizer=l2(weight_decay),
                                  padding='same', )(out_layer)
            out_layer = kl.BatchNormalization()(out_layer)
            out_layer = kl.LeakyReLU()(out_layer)

            out_layer = kl.Conv2D(expand, (1, kernel_size),
                                  kernel_regularizer=l2(weight_decay),
                                  padding='same', )(out_layer)
            out_layer = kl.BatchNormalization()(out_layer)
            out_layer = kl.LeakyReLU()(out_layer)
            return out_layer

        # One Convolution 1*1
        left_1 = convolution(x, 1)
        left_2 = convolution(x, 3)

        # Avrage Pulling
        left_3 = kl.MaxPool2D(pool_size=(2, 2), padding='same', strides=(1, 1))(x)
        left_3 = kl.Conv2D(expand, (1, 1), padding='same')(left_3)
        left_3 = kl.LeakyReLU()(left_3)

        # One Convolution 5*5
        left_4 = convolution(x, 5)

        x = kl.concatenate([left_1, left_2, left_3, left_4], axis=3)
        return x

    def mul_fire(x, squeeze=10, expand=10, dim_out=40):
        x_1 = fire_module(x, squeeze=squeeze, expand=expand)
        x_1 = fire_module(x_1, squeeze=squeeze, expand=expand)
        out = kl.add([x_1, x])
        left_3 = kl.Conv2D(dim_out, (1, 1), padding='same')(out)
        left_3 = kl.BatchNormalization()(left_3)
        left_3 = kl.LeakyReLU()(left_3)

        return left_3

    def multi_nice(conv0, squeeze=10, expand=10, dim_out=40):
        multi_1 = mul_fire(conv0, squeeze=squeeze, expand=expand, dim_out=dim_out)
        multi_2 = mul_fire(multi_1, squeeze=squeeze, expand=expand, dim_out=dim_out)

        return multi_2

    def Start_block(input):
        conv0 = kl.Activation('tanh')(kl.Conv2D(12, (3, 3), kernel_regularizer=l2(1.35 * weight_decay))(input))
        conv1 = kl.LeakyReLU()(kl.BatchNormalization()((kl.Conv2D(32, (3, 3), strides=(1, 1),
                                                                  kernel_regularizer=l2(weight_decay))(
            conv0))))
        return mul_fire(conv1, squeeze=8, expand=8, dim_out=32)

    input_1 = kl.Input(shape=[input_size, input_size, 1])
    input_2 = kl.Input(shape=[input_size, input_size, 1])

    conv0_0 = Start_block(input_1)
    conv0_1 = Start_block(input_2)
    x = kl.concatenate([conv0_0, conv0_1], axis=3)
    pull_1 = kl.MaxPool2D()(x)
    super_2 = multi_nice(pull_1, squeeze=16, expand=16, dim_out=64)
    pull_2 = kl.MaxPool2D()(super_2)
    super_3 = multi_nice(pull_2, squeeze=16, expand=16, dim_out=64)
    pull_3 = kl.MaxPool2D()(super_3)
    # super_4 =Conv2D(64,(3,3),activation='relu')(pull_3)

    FL1 = kl.LeakyReLU()(kl.Dense(128, kernel_regularizer=l2(weight_decay))(kl.Flatten()(pull_3)))

    out = kl.Activation('softmax')(kl.Dense(5)(FL1))

    model = Model(inputs=[input_1, input_2], outputs=out)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    model.summary()
    return model


def create_arch_discrimitor(ud_lr: str = "up", weight_decay: float = 5e-5, input_size: int = 32,depth:int =32) -> keras.Model:
    def start_block(up_down):
        encoder = kl.Conv2D(depth, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(up_down)
        encoder = kl.BatchNormalization()(encoder)
        encoder_f = kl.Activation('tanh')(encoder)
        encoder = kl.Conv2D(depth, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(encoder_f)
        encoder = kl.BatchNormalization()(encoder)
        encoder = kl.LeakyReLU()(encoder)
        encoder_add = kl.add([encoder_f, encoder])

        return encoder_add

    def res_part(up_down, size: int =48):
        encoder = kl.Conv2D(size, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(up_down)
        encoder = kl.BatchNormalization()(encoder)
        encoder = kl.LeakyReLU()(encoder)
        # 4
        encoder = kl.Conv2D(size, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(encoder)
        encoder = kl.BatchNormalization()(encoder)
        encoder = kl.LeakyReLU()(encoder)

        return kl.add([up_down, encoder])

    input_2 = kl.Input(shape=[input_size, input_size, 1])
    # input_3 = kl.Input(shape=[input_size, input_size, 1])
    if ud_lr == 'up':
        input_1 = kl.Input(shape=[input_size, input_size * 2, 1])
        up_down = input_1
    if ud_lr == 'grad':
        input_1 = kl.Input(shape=[input_size, input_size * 2, 3])
        up_down = input_1

    else:
        input_1 = kl.Input(shape=[input_size, input_size, 1])

        up_down = kl.concatenate([start_block(input_1), start_block(input_2)], axis=3)
    encoder = kl.MaxPool2D()(up_down)
    encoder = kl.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = res_part(encoder,64)

    encoder = kl.MaxPool2D()(encoder)
    encoder = kl.Conv2D(96, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = res_part(encoder,96)

    encoder = kl.MaxPool2D()(encoder)
    encoder = kl.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = res_part(encoder,128)


    encoder = kl.MaxPool2D()(encoder)
    encoder = kl.Conv2D(196, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder)  # 2

    encoder = kl.Flatten()(encoder)
    encoder = kl.Dense(254, kernel_regularizer=l2(weight_decay))(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Dropout(0.2)(encoder)
    if ud_lr in ['grad', 'up']:
        out = kl.Dense(2, activation='sigmoid', kernel_regularizer=l2(weight_decay))(encoder)
        model = Model(inputs=[input_1], outputs=out)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])

    else:
        out = kl.Dense(5, activation='softmax', kernel_regularizer=l2(weight_decay))(encoder)
        model = Model(inputs=[input_1, input_2], outputs=out)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])
    model.summary()
    return model


def border_compare(weight_decay: float = 5e-5, input_size: int = 32, optimizer: str = 'adam') -> keras.Model:
    real = kl.Input([input_size, 4])
    flat = kl.Flatten()(real)
    encoder = kl.Dense(160, kernel_regularizer=l2(weight_decay))(flat)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Dropout(0.1)(encoder)
    encoder = kl.Dense(160, kernel_regularizer=l2(weight_decay))(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Dropout(0.1)(encoder)
    encoder = kl.Dense(160, kernel_regularizer=l2(weight_decay))(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Dropout(0.1)(encoder)
    encoder = kl.Dense(2, activation='softmax', kernel_regularizer=l2(weight_decay))(encoder)
    model = Model(inputs=[real], outputs=encoder)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model


def create_call_backs(path_tf: str, path_w: str, monitor: str = 'val_loss') -> tuple:
    tf_log = path_tf
    checkpoint_path = path_w
    tb_call_back = TensorBoard(log_dir=tf_log,
                               histogram_freq=0,
                               batch_size=32,
                               write_graph=True,
                               write_grads=True,
                               write_images=True,
                               embeddings_freq=0,
                               embeddings_layer_names=None,
                               embeddings_metadata=None)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor=monitor, verbose=1, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=0.8,
                                  patience=3,
                                  min_lr=0.000001,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)
    return tb_call_back, checkpoint, reduce_lr
