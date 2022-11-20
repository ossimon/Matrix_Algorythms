# Recursive determinant calculation

from lufact import lufact
import numpy as np


def determinant(matrix):
    l, u, cost = lufact(matrix)
    size = l.shape[0]
    indexes = np.eye(size, dtype='bool')
    return np.prod(l, where=indexes), cost + size
