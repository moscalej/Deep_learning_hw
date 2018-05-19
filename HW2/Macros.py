#################
### CONSTANTS ###
#################


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
