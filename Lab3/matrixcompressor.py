import treecreator as tc
import numpy as np


def is_zeros(matrix):
    return not np.any(matrix)


def compress_matrix(matrix, u, v, s, tmin, tmax, smin, smax, r):
    shape = (tmax - tmin, smax - smin)
    if is_zeros(matrix[tmin:tmax, smin:smax]):
        return tc.HierarchyTreeNode(rank=0, shape=shape)

    return tc.HierarchyTreeNode(u, v, r, shape)
