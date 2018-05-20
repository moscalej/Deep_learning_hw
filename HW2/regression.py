import numpy as np


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


if __name__ == "__main__":
    small = make_points(100)
    small_vals = []
    for x in small:
        small_vals.append(func(x[0], x[1]))

    big = make_points(100)
    big_vals = []
    for x in big:
        big_vals.append(func(x[0], x[1]))

    test = np.linspace(-2, 2, 1000)
    test_set = []
    test_vals = []
    for x in range(1000):
        for y in range(1000):
            test_set.append(np.array([test[x], test[y]]))
            test_vals.append(func(test[x], test[y]))
