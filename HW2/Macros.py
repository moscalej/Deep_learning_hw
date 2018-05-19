#################
### CONSTANTS ###
#################

import pandas as pd
import matplotlib.pyplot as plt

INPUT = "input"
OUTPUT = "output"
NON_LINEAR = "non_linear"
REGULARIZATION = "regularization"
LEARNING_RATE = "learning_rate"

NON_LINEAR_OPTIONS = ["relu", "sigmoid", "softmax", "none"]
REGULARIZATION_OPTIONS = ["l1", "l2"]
LOSS_OPTIONS = ["MSE", "cross-entropy"]



def generate_layer(input_dims, output_dims, non_linearity, regularization, learning_rate):
    return {
        INPUT: input_dims,
        OUTPUT: output_dims,
        NON_LINEAR: non_linearity,
        REGULARIZATION: regularization,
        LEARNING_RATE: learning_rate
    }


def plot_graphs(log):
    log = pd.DataFrame(log)

    fig, ax = plt.subplots(1, 2, figsize=[20, 10])

    ax[0].plot(log['acc'], label='Acc test')
    ax[0].plot(log['val_acc'], label='Acc Val')
    ax[0].set_title('Acuarecy')
    ax[0].legend()
    ax[1].plot(log['loss'], label='Loss test')
    ax[1].plot(log['val_loss'], label='Loss Val')
    ax[1].set_title('Loss')
    ax[1].legend()
    plt.show()
