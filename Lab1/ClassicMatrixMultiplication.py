# Implementation of classic matrix multiplication algorithm (also knows as Binét's method)

import numpy as np


def classic_multiply(matrix1, matrix2):
    # Breaking matrices into four pieces:
    # matrix1: a b
    #          c d
    # matrix2: e f
    #          g h
    # result: r1 r2  |  ae+bg af+bh
    #         r3 r4  |  ce+dg cf+dh
    # print('matrix1:', matrix1, sep = '\n')
    # print('matrix2:', matrix2, sep = '\n')
    # print()

    size = matrix1.shape[0]
    half = size // 2

    if size == 1:
        return matrix1 * matrix2

    a = matrix1[0:half, 0:half]
    b = matrix1[0:half, half:size]
    c = matrix1[half:size, 0:half]
    d = matrix1[half:size, half:size]

    e = matrix2[0:half, 0:half]
    f = matrix2[0:half, half:size]
    g = matrix2[half:size, 0:half]
    h = matrix2[half:size, half:size]

    r1 = classic_multiply(a, e) + classic_multiply(b, g)
    r2 = classic_multiply(a, f) + classic_multiply(b, h)
    r3 = classic_multiply(c, e) + classic_multiply(d, g)
    r4 = classic_multiply(c, f) + classic_multiply(d, h)

    r12 = np.concatenate((r1, r2), axis=1)
    r34 = np.concatenate((r3, r4), axis=1)

    return np.concatenate((r12, r34), axis=0)
