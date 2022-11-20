# This code segment calculates the best exponential function exponent for the function defined by points described
# by x and y arrays

import numpy as np


def norm(x, y, exp):
    new_y = x ** exp
    new_y = new_y / new_y[-1] * y[-1]
    return np.sum(np.square(y - new_y))


def approximate(results, smallest_power_of_2, largest_power_of_2):

    x = np.array(range(smallest_power_of_2, largest_power_of_2 + 1))
    x = 2 ** x
    y = np.array(results)

    best_norm = np.inf
    best_exp = 0

    precision = 10 ** 5

    for i in range(precision):
        exp = 1 + i / precision * 4
        new_norm = norm(x, y, exp)
        if new_norm < best_norm:
            best_norm = new_norm
            best_exp = exp

    return best_exp
