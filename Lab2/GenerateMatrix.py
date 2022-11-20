import numpy as np


def generate_matrix(size):
    result = np.zeros((size, size))
    while np.linalg.det(result) == 0:
        result = np.random.random((size, size))

    return result
