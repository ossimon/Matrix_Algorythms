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

    size = matrix1.shape[0]
    half = size // 2

    if size == 1:
        return matrix1 * matrix2, 1

    a = matrix1[0:half, 0:half]
    b = matrix1[0:half, half:size]
    c = matrix1[half:size, 0:half]
    d = matrix1[half:size, half:size]

    e = matrix2[0:half, 0:half]
    f = matrix2[0:half, half:size]
    g = matrix2[half:size, 0:half]
    h = matrix2[half:size, half:size]

    ae, ae_cost = classic_multiply(a, e)
    bg, bg_cost = classic_multiply(b, g)
    af, af_cost = classic_multiply(a, f)
    bh, bh_cost = classic_multiply(b, h)
    ce, ce_cost = classic_multiply(c, e)
    dg, dg_cost = classic_multiply(d, g)
    cf, cf_cost = classic_multiply(c, f)
    dh, dh_cost = classic_multiply(d, h)

    r1 = ae + bg
    r2 = af + bh
    r3 = ce + dg
    r4 = cf + dh

    r12 = np.concatenate((r1, r2), axis=1)
    r34 = np.concatenate((r3, r4), axis=1)

    # The total cost at each recursion step is equal to eight block multiplications plus four block additions
    number_of_operations = 8 * ae_cost + 4 * (ae.shape[0] ** 2)

    return np.concatenate((r12, r34), axis=0), number_of_operations
