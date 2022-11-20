# Recursive matrix inversion

import numpy as np
from StrassensMatrixMultiplication import strassens_multiply


def invert(matrix):
    size = matrix.shape[0]
    half = size // 2

    if size == 1:
        return 1 / matrix, 1

    a11 = matrix[0:half, 0:half]
    a12 = matrix[0:half, half:size]
    a21 = matrix[half:size, 0:half]
    a22 = matrix[half:size, half:size]

    cost = [0 for _ in range(14)]

    ia11, cost[0] = invert(a11)
    a21ia11, cost[1] = strassens_multiply(a21, ia11)
    a21ia11a12, cost[2] = strassens_multiply(a21ia11, a12)
    s22 = a22 - a21ia11a12
    cost[3] = s22.shape[0] ** 2
    is22, cost[4] = invert(s22)
    a12is22, cost[5] = strassens_multiply(a12, is22)
    a12is22a21, cost[6] = strassens_multiply(a12is22, a21)
    a12is22a21ia11, cost[7] = strassens_multiply(a12is22a21, ia11)
    a12is22a21ia11plusi = np.eye(half) + a12is22a21ia11
    cost[8] = a12is22a21ia11plusi.shape[0] ** 2
    b11, cost[9] = strassens_multiply(ia11, a12is22a21ia11plusi)
    mia11a12, cost[10] = strassens_multiply(-ia11, a12)
    b12, cost[11] = strassens_multiply(mia11a12, is22)
    mis22a21, cost[12] = strassens_multiply(-is22, a21)
    b21, cost[13] = strassens_multiply(mis22a21, ia11)
    b22 = is22

    b1 = np.concatenate((b11, b12), axis=1)
    b2 = np.concatenate((b21, b22), axis=1)

    return np.concatenate((b1, b2), axis=0), np.sum(cost)
