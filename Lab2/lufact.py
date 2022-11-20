# Recursive LU factorization

from invert import invert
from StrassensMatrixMultiplication import strassens_multiply as mul
import numpy as np


def lufact(matrix):
    size = matrix.shape[0]
    half = size // 2

    if size == 1:
        return matrix, np.ones((1, 1)), 0

    a11 = matrix[0:half, 0:half]
    a12 = matrix[0:half, half:size]
    a21 = matrix[half:size, 0:half]
    a22 = matrix[half:size, half:size]

    cost = [0 for _ in range(11)]
    l11, u11, cost[1] = lufact(a11)
    iu11, cost[2] = invert(u11)
    l21, cost[3] = mul(a21, iu11)
    il11, cost[4] = invert(l11)
    u12, cost[5] = mul(il11, a12)
    a21iu11, cost[6] = mul(a21, iu11)
    a21iu11il11, cost[7] = mul(a21iu11, il11)
    a21iu11il11a12, cost[8] = mul(a21iu11il11, a12)
    s = a22 - a21iu11il11a12
    cost[9] = s.shape[0] ** 2
    l22, u22, cost[10] = lufact(s)

    l = np.zeros(shape=(size, size))
    l[0:half, 0:half] = l11
    l[half:size, 0:half] = l21
    l[half:size, half:size] = l22

    u = np.zeros(shape=(size, size))
    u[0:half, 0:half] = u11
    u[0:half, half:size] = u12
    u[half:size, half:size] = u22

    return l, u, np.sum(cost)
