######################
# CORE DNN CLASS ###
######################


import time
import numpy as np
import math
from Macros import *
from Layer import Layer
from Nodes import node_factory


class MyDNN:
    def __init__(self, architecture, loss, weight_decay=0.):
        """
        :param architecture: A list of dictionaries used to initialize Layer objects
        :param loss: a string denoting the type of loss function we should use for this network
        :param weight_decay: a float denoting the weight decay within this network. Default is 0.
        """

        # Assertions

        assert isinstance(architecture, list)
        assert loss in LOSS_OPTIONS, (loss + " is not a valid loss function")
        # assert isinstance(weight_decay, float), \
        #     ("weight_decay should be a float, not a " + str(type(weight_decay)))

        # Attribute setting

        self.layers = []

        for layer in architecture:
            layer_input = layer[INPUT]
            layer_output = layer[OUTPUT]
            non_linearity = layer[NON_LINEAR]
            regularization = layer[REGULARIZATION]
            learning_rate = layer[LEARNING_RATE]
            new_layer = Layer(layer_input, layer_output, non_linearity, regularization, learning_rate)
            self.layers.append(new_layer)

        self.loss = node_factory(loss)
        self.weight_decay = weight_decay

    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, x_val=None, y_val=None):
        """
        Description:

        The function will run SGD for a user-defined number of epochs, with the
        defined batch size. On every epoch the data will be reshuffled (make sure you
        shuffle the x’s and y’s together).
        For every batch the data should be passed forward and gradients pass backward.
        After gradients are computed, weights update will be performed using
        the learning rate.
        After every epoch the following line will be printed on screen with the values
        of the last epoch.

        :param x_train: a Numpy nd-array where each row is a sample
        :param y_train: a 2d array, the labels of X in one-hot representation for classification
                        or a value for each sample for regression.
        :param epochs: number of epochs to run
        :param batch_size: batch size for SGD
        :param learning_rate:  float, a fixed learning rate that will be used in SGD.
        :param x_val: the validation x data (same structure as train data) default is None.
                      When validation data is given, evaluation over this data will be made at the end of every epoch.
        :param y_val: the corresponding validation y data (labels) whose structure is identical to y_train.
        :return: history - intermediate optimization results, which is a list of dictionaries,
                such that each epoch has a corresponding dictionary containing all relevant
                results. These dictionaries do not contain formatting information (you will
                later use the history to print various things including plots of learning and
                convergence curves for your networks).



        """

        history = []
        Data = x_train.copy()
        sumple_num = Data.shape[0]
        Label = self._one_shot(y_train.copy())
        time.clock()
        t_current = 0
        for episode in range(1, epochs + 1):
            loss, acc = self._train_epochs(Data, Label, sumple_num, batch_size)
            val_loss, val_acc = self._test(Data, Label)

            print(f'Epoch {episode} / {epochs + 1} - {t_current} seconds - loss: {loss} -'
                  f' acc: {acc} - val_loss: {val_loss} - val_acc: {val_acc}')
            history.append(dict(
                episode=episode,
                time=t_current,
                loss=loss,
                acc=acc,
                val_loss=val_loss,
                val_acc=val_acc,

            ))

        return history

    def prdict(self, data):

        return

    def _train_epochs(self, Data, Label, sumple_num, batch_size):

        data_e, label_e = self._shuffle(Data, Label)
        current = 0
        acc = 0.
        loss = []
        for batch in range(round(sumple_num / batch_size) - 1):

            for sample in range(batch):
                print(current)
                y_hat = self._forward(data_e[current])
                self.loss.forward(y_hat, label_e[current])
                if np.argmax(y_hat) == np.argmax(Label[sample]):
                    acc += 1
                current += 1
            loss.append(self.loss.get_loss())
            self._backward(self.loss.backward())

        return sum(loss), (acc / sumple_num)

    def _test(self, Data, Label):
        acc = 0.
        loss = []
        sumple_num = Data.shape[0]
        for sample in range(sumple_num):
            y_hat = self._forward(Data[sample])
            self.loss.forward(y_hat, Label[sample])
            if np.argmax(y_hat) == np.argmax(Label[sample]):
                acc += 1
        acc = acc / sumple_num
        loss = self.loss.get_loss()
        return loss, acc

    def _forward(self, in_put):
        current_input = in_put.reshape([len(in_put), 1])  # something
        for layer in self.layers:
            current_input = layer.forward(current_input)

        return current_input

    def _backward(self, gradiand):
        current_gradiant = gradiand
        for layer in self.layers:
            current_gradiant = layer.backward(current_gradiant)

    def _one_shot(self, labels):
        range = np.max(labels) - np.min(labels) + 1
        Labels_one = np.zeros([len(labels), range])
        Labels_one[np.arange(len(labels)), labels.reshape(-1)] = 1
        return Labels_one

    def _shuffle(self, Data, Labels):
        index_s = np.arange(Data.shape[0])
        np.random.shuffle(index_s)
        return Data.copy()[index_s, :], Labels.copy()[index_s, :]
