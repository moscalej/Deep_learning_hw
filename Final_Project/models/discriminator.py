import keras.layers as kl
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
import keras


def create_arch_discrimitor(ud_lr: str = "up", weight_decay: float = 5e-5) -> keras.Model:
    weight_decay = weight_decay
    input_1 = kl.Input(shape=[32, 32, 1])
    input_2 = kl.Input(shape=[32, 32, 1])
    if ud_lr == 'up':
        up_down = kl.concatenate([input_1, input_2], axis=1)
    else:
        up_down = kl.concatenate([input_1, input_2], axis=3)

    encoder = kl.Conv2D(32, (3, 3),
                        padding='same',
                        kernel_regularizer=keras.regularizers.l2(weight_decay)
                        )(up_down)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Conv2D(32, (3, 3),
                        padding='same',
                        kernel_regularizer=keras.regularizers.l2(weight_decay)
                        )(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder)  # 16
    encoder = kl.Conv2D(64, (3, 3),
                        padding='same',
                        kernel_regularizer=keras.regularizers.l2(weight_decay)
                        )(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder)  # 8
    encoder = kl.Conv2D(96, (3, 3),
                        padding='same',
                        kernel_regularizer=keras.regularizers.l2(weight_decay)
                        )(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder)  # 4
    encoder = kl.Conv2D(128, (3, 3),
                        padding='same',
                        kernel_regularizer=keras.regularizers.l2(weight_decay)
                        )(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Conv2D(128, (3, 3),
                        padding='same',
                        kernel_regularizer=keras.regularizers.l2(weight_decay)
                        )(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.MaxPool2D()(encoder)  # 2
    encoder = kl.Flatten()(encoder)
    encoder = kl.Dense(128,
                       kernel_regularizer=keras.regularizers.l2(weight_decay)
                       )(encoder)
    encoder = kl.BatchNormalization()(encoder)
    encoder = kl.LeakyReLU()(encoder)
    encoder = kl.Dropout(0.4)(encoder)
    if ud_lr in ['up']:
        out = kl.Dense(2, activation='softmax', kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)
    else:
        out = kl.Dense(5, activation='softmax', kernel_regularizer=keras.regularizers.l2(weight_decay))(encoder)

    model = Model(inputs=[input_1, input_2], outputs=out)

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999),
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
