from keras import layers
from keras import models
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import cifar10
from keras.layers import *
import keras
#
# image dimensions
#
# from tensorflow.python.estimator import keras
from keras_preprocessing.image import ImageDataGenerator

img_height = 32
img_width = 32
img_channels = 3
num_classes = 10
batch_size = 1024
epochs = 30
input_shape = [32, 32, 3]
weight_decay = 4e-5


#
# network params
#

cardinality = 8


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.

    """

    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # conv1
    x = layers.Conv2D(94, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 8, 16, _project_shortcut=project_shortcut)

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 8, 16, _strides=strides)

    # conv4
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 16, 32, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 32, 64, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    out = Activation('softmax')(Dense(10)(x))

    return out


image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = residual_network(image_tensor)

model = models.Model(inputs=[image_tensor], outputs=[network_output])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
model.summary()
def run():
    Tf_log = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW3\TF\Alex_net_v2'
    Model_save_p =r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\HW3\saved_models\Alex_net_v2.h5'
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
                                  factor=0.5,
                                  patience=2,
                                  min_lr=0.000001,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)

    history_fully = model.fit_generator(img.flow(x_train, y_train, batch_size=1024),
                                        steps_per_epoch=48,
                                        shuffle=True,
                                        epochs=150,
                                        initial_epoch=0,
                                        validation_data=(x_test, y_test), callbacks=[tbCallBack, reduce_lr])
    model.save(Model_save_p)
    return history_fully