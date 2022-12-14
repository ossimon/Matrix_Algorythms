from sklearn.decomposition import TruncatedSVD
import matrixcompressor as mc
import numpy as np


class HierarchyTreeNode:
    def __init__(self, u=None, v=None, rank=None, shape=None):
        self.nodes = []
        self.u = u
        self.v = v
        self.rank = rank
        self.shape = shape

    def add_node(self, node):
        self.nodes.append(node)

    def recreate(self):
        # print(self.shape, self.rank)
        if self.rank is not None:
            if self.rank == 0:
                return np.zeros(self.shape)
            return self.u @ self.v

        # for i in range(4):
        #     print(self.nodes[i].recreate())

        upper = np.concatenate((self.nodes[0].recreate(), self.nodes[1].recreate()), axis=1)
        lower = np.concatenate((self.nodes[2].recreate(), self.nodes[3].recreate()), axis=1)
        return np.concatenate((upper, lower), axis=0)

    def printable(self):
        if self.rank is not None:
            result = np.ones(self.shape, dtype='int32')
            result[self.rank:, self.rank:] = 0
            return result
        else:
            upper = np.concatenate((self.nodes[0].printable(), self.nodes[1].printable()), axis=1)
            lower = np.concatenate((self.nodes[2].printable(), self.nodes[3].printable()), axis=1)
            return np.concatenate((upper, lower), axis=0)


def create_tree(matrix, t_min, t_max, s_min, s_max, r, epsilon):

    shape = (t_max - t_min, s_max - s_min)
    block = matrix[t_min:t_max, s_min:s_max]
    # print(shape)

    if shape[0] <= r:
        return HierarchyTreeNode(block, np.eye(np.max(shape)), rank=shape[0], shape=shape)

    svd = TruncatedSVD(n_components=shape[0], n_iter=7, random_state=42)
    u = svd.fit_transform(block)
    v = svd.components_
    s = svd.singular_values_
    # print(s)
    # print(block)
    # print(np.dot(u, v))
    # print()
    if s.shape[0] <= r:
        return HierarchyTreeNode(block, np.eye(np.max(shape)), rank=s.shape[0], shape=shape)

    if abs(s[r]) < epsilon:
        return mc.compress_matrix(matrix, u, v, s, t_min, t_max, s_min, s_max, r)

    node = HierarchyTreeNode(shape=shape)

    t_mid = t_min + shape[0] // 2
    s_mid = s_min + shape[1] // 2

    node.add_node(create_tree(matrix, t_min, t_mid, s_min, s_mid, r, epsilon))
    node.add_node(create_tree(matrix, t_min, t_mid, s_mid, s_max, r, epsilon))
    node.add_node(create_tree(matrix, t_mid, t_max, s_min, s_mid, r, epsilon))
    node.add_node(create_tree(matrix, t_mid, t_max, s_mid, s_max, r, epsilon))

    return node
