import keras
from keras.layers import *
from keras.models import Model


def benchmark_model(number_lstm=25, state_size=1024, dense_size=1024, vgg_trainable=False, optimizer='adam'):
    """
    Create a DNN witch will be use to predict the sequence
    :param number_lstm: Lenght of the sequence
    :param state_size:
    :param dense_size:
    :param vgg_trainable:
    :param optimizer:
    :return:
    """
    #  Load vgg architecture
    model_vgg = keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(224, 224, 3), pooling=None, classes=1000)
    model_vgg.trainable = vgg_trainable

    #  Set Input
    inputs = Input(shape=(224, 224, 1))

    #  VGG flow feed thru vgg convolutions and then use a single dense
    #  before passing to the RNN part of the network
    vgg_flow = Conv2D(3, (7, 7), padding='same')(inputs)
    vgg_flow = model_vgg(vgg_flow)
    vgg_flow = Flatten()(vgg_flow)
    vgg_flow = Dense(dense_size)(vgg_flow)
    vgg_flow = BatchNormalization()(vgg_flow)
    vgg_flow = LeakyReLU()(vgg_flow)

    #  RNN consist of two layers of by directional LSTM
    rnn_flow = RepeatVector(number_lstm)(vgg_flow)
    rnn_flow = Bidirectional(LSTM(state_size, dropout=0.4, recurrent_dropout=0.2, return_sequences=True))(rnn_flow)
    rnn_flow = Bidirectional(LSTM(state_size, dropout=0.4, recurrent_dropout=0.2, return_sequences=True))(rnn_flow)
    out_final = TimeDistributed(Dense(number_lstm, activation='softmax'))(rnn_flow)

    #  Model Creation and compilation
    model = Model(inputs=inputs, outputs=out_final)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    return model