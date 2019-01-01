#################
### CONSTANTS ###
#################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT = "input"
OUTPUT = "output"
NON_LINEAR = "non_linear"
REGULARIZATION = "regularization"
LEARNING_RATE = "learning_rate"
LEARNING_RATE_RATE = "learning_rate_decay"
DECAY_RATE = "decay_rate"
LR_MIN = "lr_min"

NON_LINEAR_OPTIONS = ["relu", "sigmoid", "softmax", 'tanh', "none"]
REGULARIZATION_OPTIONS = ["l1", "l2"]
LOSS_OPTIONS = ["MSE", "cross-entropy"]


def generate_layer(input_dims, output_dims, non_linearity, regularization, learning_rate=0.2, learning_rate_decay=0.6,
                   decay_rate=15, lr_min=0.0001):
    return {
        INPUT: input_dims,
        OUTPUT: output_dims,
        NON_LINEAR: non_linearity,
        REGULARIZATION: regularization,
        LEARNING_RATE: learning_rate,
        LEARNING_RATE_RATE: learning_rate_decay,
        DECAY_RATE: decay_rate,
        LR_MIN: lr_min
    }


def plot_graphs(log, title=""):
    log = pd.DataFrame(log)

    fig, ax = plt.subplots(1, 2, figsize=[20, 10])

    ax[0].plot(log['acc'], label='Acc test')
    ax[0].plot(log['val_acc'], label='Acc Val')
    ax[0].set_title(f'{title} :Acuarecy')
    ax[0].legend()
    ax[1].plot(log['loss'], label='Loss test')
    ax[1].plot(log['val_loss'], label='Loss Val')
    ax[1].set_title(f'{title}Loss')
    ax[1].legend()
    plt.show()


def one_hot(labels):
    """

    :param labels: An array of integers (labels).
    The length of this array is the number of samples.
    :return: An n x k matrix where n is the number of samples
    and k is the number of possible classifications of this network
    """

    range = np.max(labels) - np.min(labels) + 1
    Labels_one = np.zeros([len(labels), range])
    Labels_one[np.arange(len(labels)), labels.reshape(-1)] = 1
    return Labels_one
