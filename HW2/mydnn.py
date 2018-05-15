import time
import numpy as np
from Macros import *
from Layer import Layer
from Nodes import node_factory
from Nodes import MSE
from Nodes import Entropy


######################
# CORE DNN CLASS ###
######################

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
        assert isinstance(weight_decay, float), \
            ("weight_decay should be a float, not a " + str(type(weight_decay)))

        # Attribute setting

        self.layers = []

        # Generate a layer object for each layer dictionary in the architecture list
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
        Label = self._one_hot(y_train.copy())

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

    def _train_epochs(self, Data, Label, sample_num, batch_size):
        """

        :param Data: An n x m matrix where n represents the number of samples and
        m represents the number of features per sample.
        :param Label: An n x k matrix where n represents the number of samples, and
        k represents the number dimension of the output of the network (i.e., the
        number of possible classifications we can have for a given input).
        :param sample_num: An integer. The number of samples we are currently working with.
        Going by the notation above, this is the number n.
        :param batch_size: the size of any particular batch (perhaps except for the last one).
        An integer
        :return: A tuple consisting of

                                    (Loss, Accuracy)

        Both Loss and Accuracy are numbers.
        """

        shuffled_data, shuffled_labels = self._shuffle(Data, Label)
        shuffled_data = shuffled_data.T
        shuffled_labels = shuffled_labels.T

        acc = []
        error =[]
        for batch in range(0, sample_num, batch_size):

            # y_hat is a t x b matrix where t is the output dimension of the layer
            # before the classifier and b is the batch size
            y_hat = self._forward(shuffled_data[:, batch: batch_size + batch])

            weights_norm_sum = 0
            for layer in self.layers:
                weights_norm_sum += layer.weights_norm

            self.loss.forward(y_hat, shuffled_labels[:, batch: batch_size + batch], sample_num)

            self._backward(self.loss.gradiant)
            error.append(self.loss.error + (weights_norm_sum * self.weight_decay))
            diff = sum(np.argmax(y_hat, axis=0) == np.argmax(Label[:, batch: batch_size + batch], axis=0))
            acc.append(diff / y_hat.shape[1])

        return np.mean(acc), np.mean(error)

    def _test(self, Data, Label):
        Data = Data.T
        Label = Label.T
        y_hat = self._forward(Data)
        weights_norm_sum = 0
        for layer in self.layers:
            weights_norm_sum += layer.weights_norm
        self.loss.forward(y_hat, Label, Data.shape[1])
        acc = sum(np.argmax(y_hat, axis=0) == np.argmax(Label, axis=0))
        loss = self.loss.get_loss() + (weights_norm_sum * self.weight_decay)
        return loss, acc

    def _forward(self, batch_inputs):
        """

        :param batch_inputs: A matrix of dimensions m x b, where m is the
        number of features, and b is the batch size (represents a particular
        subset of our initial samples).
        :return: A matrix of dimensions t x b, where t is the dimension of the output
        before the classifier
        """

        current_output = batch_inputs

        for layer in self.layers:
            current_input = current_output
            current_output = layer.forward(current_input)

        return current_output

    def _backward(self, gradiant):
        """

        :param gradiant:
        :return:
        """
        current_gradiant = gradiant

        for index in range(len(self.layers) - 1, -1, -1):
            current_gradiant = self.layers[index].backward(current_gradiant)

    def _one_hot(self, labels):
        """

        :param labels: An array of integers (labels).
        The length of this array is the number of samples.
        :return:
        """

        range = np.max(labels) - np.min(labels) + 1
        Labels_one = np.zeros([len(labels), range])
        Labels_one[np.arange(len(labels)), labels.reshape(-1)] = 1
        return Labels_one

    def _shuffle(self, Data, Labels):
        """

        :param Data: A matrix of size n x m, where n is the number of samples,
        and m is the number of features per sample.
        :param Labels: a vector of length n, where n is the number of samples.
        :return: A tuple composed of the following:

                                    (A, B)

        Where A is a shuffled copy of the Data matrix (i.e., rows are shuffled)

        and

        Where B is a shuffled copy of the Labels vector (i.e., rows are shuffled)
        """
        index_s = np.arange(Data.shape[0])
        np.random.shuffle(index_s)
        return Data.copy()[index_s, :], Labels.copy()[index_s, :]
