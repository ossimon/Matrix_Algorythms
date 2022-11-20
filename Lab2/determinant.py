# Recursive determinant calculation

from lufact import lufact
import numpy as np


def determinant(matrix):
    l, u = lufact(matrix)
    size = l.shape[0]
    indexes = [(i, i) for i in range(size)]
    return np.multiply(l[indexes])
