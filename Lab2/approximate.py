# This code segment calculates the best exponential function exponent for the function defined by points described
# by x and y arrays

import numpy as np


def norm(x, y, exp):
    new_y = x ** exp
    new_y = new_y / new_y[-1] * y[-1]
    return np.sum(np.square(y - new_y))


def approximate(sizes, results, min_exp=1, max_exp=5):

    x = np.array(sizes)
    y = np.array(results)

    best_norm = np.inf
    best_exp = 0

    precision = 10 ** 5

    for i in range(precision):
        exp = min_exp + i / precision * (max_exp - 1)
        new_norm = norm(x, y, exp)
        if new_norm < best_norm:
            best_norm = new_norm
            best_exp = exp

    return best_exp
