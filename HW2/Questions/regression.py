import numpy as np
import mydnn
from Macros import generate_layer
from Macros import plot_graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_points(m):
    """

    :param m: the number of samples we will have
    :return: a list of m randomly (generated via uniform distribution) pairs, (x_1, x_2)
    """

    return np.random.uniform(-2, 2, [m, 2])


def func(x1, x2):
    """

    :param x1: a float between 0 and 2
    :param x2: a float between 0 and 2
    :return: a float equal to
                        x1 * e ^ ( - (x1 ^ 2) - (x2 ^ 2))

    """
    return x1 * np.exp(- (x1 ** 2) - (x2 ** 2))


def create_data_sets():
    small = make_points(100)
    small_vals = []
    for x in small:
        small_vals.append(func(x[0], x[1]))
    small_vals = np.array(small_vals).reshape([100, 1])

    big = make_points(1000)
    big_vals = []
    for x in big:
        big_vals.append(func(x[0], x[1]))
    big_vals = np.array(big_vals).reshape([1000, 1])

    test_set = np.linspace(-2, 2, 100)
    test = []
    test_vals = []
    for x in range(len(test_set)):
        for y in range(len(test_set)):
            test.append([test_set[x], test_set[y]])
            test_vals.append(func(test_set[x], test_set[y]))
    test = np.array(test)
    test_vals = np.array(test_vals).reshape([len(test_vals), 1])

    return [small, small_vals, big, big_vals, test, test_vals]


def surface_plot(matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)



if __name__ == "__main__":

    small, small_vals, big, big_vals, test, test_vals = create_data_sets()
    small_layers = [generate_layer(2, 25, "relu", "l2", 0.4),
                    generate_layer(25, 1, "relu", "l2", 0.4)]


    small_net = mydnn.MyDNN(small_layers, "MSE", 5e-5)
    log_s = small_net.fit(small, small_vals, 1_800, 20, 0.2)
    plot_graphs(log_s)

    big_layers = [generate_layer(2, 50, "relu", "l2"),
                  generate_layer(50, 25, "relu", "l2"),
                  generate_layer(25, 1, "relu", "l2")]
    big_net = mydnn.MyDNN(big_layers, "MSE", 5e-5)
    log_b = big_net.fit(big, big_vals, 1_800, 512, 0.4)
    plot_graphs(log_b)

    small_results = small_net.evaluate(test, test_vals)
    big_results = big_net.evaluate(test, test_vals)

    y_hat_s = small_net.predict(test).reshape([100, 100])
    fig, ax, surf = surface_plot(y_hat_s)
    ax.set_title('Prediction of Small net f(x1,x2)')
    plt.show()
    y_hat_b = big_net.predict(test).reshape([100, 100])
    fig, ax, surf = surface_plot(y_hat_b)
    ax.set_title('Prediction of Big net  f(x1,x2)')
    plt.show()
    y_real = test_vals.reshape([100, 100])
    fig, ax, surf = surface_plot(y_real)
    ax.set_title('Real Values f(x1,x2)')
    plt.show()