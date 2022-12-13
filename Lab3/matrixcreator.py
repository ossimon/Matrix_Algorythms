import numpy as np


def create_random_matrix(shape, filled_ratio):
    if filled_ratio > 1:
        filled_ratio /= 100

    values_to_zero = int(shape[0] * shape[1] * (1 - filled_ratio))
    result = np.random.rand(shape[0], shape[1])

    for i in range(values_to_zero):
        x = np.random.randint(shape[1])
        y = np.random.randint(shape[0])

        while result[x, y] == 0:
            x = np.random.randint(shape[1])
            y = np.random.randint(shape[0])

        result[x, y] = 0

    return result
