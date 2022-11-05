# This code segment calculates the best exponential function exponent for the function defined by points described
# by x and y arrays

import numpy as np


def norm(x, y, exp):
    new_y = x ** exp
    new_y = new_y / new_y[-1] * y[-1]
    return np.sum(np.square(y - new_y))


# Binet
# Operation count
y = np.array([112, 960, 7936, 64512, 520192, 4177920, 33488896, 268173312])
# Time
# y = np.array([0.0, 0.001, 0.0279, 0.0708, 0.5765, 4.1834, 32.8724, 263.2043])
# Strassen
# Operation count
# y = np.array([247, 2017, 15271, 111505, 798967, 5666497, 39960391, 280902385])
# Time
# y = np.array([0.001, 0.001, 0.008, 0.0788, 0.4219, 3.027, 20.7699, 146.1128])

x = np.array(range(0, y.shape[0]))
x = 2 ** x

best_norm = np.inf
best_exp = 0

precision = 10 ** 5

for i in range(precision):
    exp = 1 + i / precision * 3
    new_norm = norm(x, y, exp)
    if new_norm < best_norm:
        best_norm = new_norm
        best_exp = exp

print(best_exp)
