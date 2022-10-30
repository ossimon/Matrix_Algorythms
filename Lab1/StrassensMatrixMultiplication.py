# Implementation of Strassen's matrix multiplication algorithm

import numpy as np


def strassens_multiply(matrix1, matrix2):
    # Breaking matrices into four pieces:
    # matrix1: a11 a12
    #          a21 a22
    # matrix2: b11 b12
    #          b21 b22
    # result: r1 r2  |  p1+p4-p5+p7 p3+p5
    #         r3 r4  |  p2+p4       p1-p2+p3+p6

    size = matrix1.shape[0]
    half = size // 2

    if size == 1:
        return matrix1 * matrix2

    a11 = matrix1[0:half, 0:half]
    a12 = matrix1[0:half, half:size]
    a21 = matrix1[half:size, 0:half]
    a22 = matrix1[half:size, half:size]

    b11 = matrix2[0:half, 0:half]
    b12 = matrix2[0:half, half:size]
    b21 = matrix2[half:size, 0:half]
    b22 = matrix2[half:size, half:size]

    p1 = strassens_multiply(a11 + a22, b11 + b22)
    p2 = strassens_multiply(a21 + a22, b11)
    p3 = strassens_multiply(a11, b12 - b22)
    p4 = strassens_multiply(a22, b21 - b11)
    p5 = strassens_multiply(a11 + a12, b22)
    p6 = strassens_multiply(a21 - a11, b11 + b12)
    p7 = strassens_multiply(a12 - a22, b21 + b22)

    r1 = p1 + p4 - p5 + p7
    r2 = p3 + p5
    r3 = p2 + p4
    r4 = p1 - p2 + p3 + p6

    r12 = np.concatenate((r1, r2), axis=1)
    r34 = np.concatenate((r3, r4), axis=1)

    return np.concatenate((r12, r34), axis=0)
