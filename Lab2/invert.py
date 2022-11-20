# Recursive matrix inversion

import numpy as np
from StrassensMatrixMultiplication import strassens_multiply


def invert(matrix):
    size = matrix.shape[0]
    half = size // 2

    if size == 1:
        return 1 / matrix

    a11 = matrix[0:half, 0:half]
    a12 = matrix[0:half, half:size]
    a21 = matrix[half:size, 0:half]
    a22 = matrix[half:size, half:size]

    ia11 = invert(a11)
    s22 = a22 - strassens_multiply(strassens_multiply(a21, ia11)[0], a12)[0]
    is22 = invert(s22)
    b11 = strassens_multiply(ia11, (np.eye(half) +
                                    strassens_multiply(strassens_multiply(strassens_multiply(a12, is22)[0], a21)[0], ia11)[0]))[0]
    b12 = strassens_multiply(strassens_multiply(-ia11, a12)[0], is22)[0]
    b21 = strassens_multiply(strassens_multiply(-is22, a21)[0], ia11)[0]
    b22 = is22

    b1 = np.concatenate((b11, b12), axis=1)
    b2 = np.concatenate((b21, b22), axis=1)

    return np.concatenate((b1, b2), axis=0)
