# Main file, where we're running our implementations of algorithms

import numpy as np

import ClassicMatrixMultiplication as cmm
import StrassensMatrixMultiplication as smm
import AIGeneratedMatrixMultiplication as aimm

# m1 = np.array([[1, 2], [3, 4]])
# m2 = np.ones(shape=(2, 2))
m3 = np.ones(shape=(256, 256))
# print(cmm.classic_multiply(m2, m1))
# print(smm.strassens_multiply(m2, m1))

for k in range(0, 11):
    m = np.zeros(shape=(2 ** k, 2 ** k))
    # print(cmm.classic_multiply(m, m)[1])
    print(smm.strassens_multiply(m, m)[1])

# t0 = np.array(range(0, 10))
# t0 = 2 ** t0
# t = np.array([1, 12, 112, 960, 7936, 64512, 520192, 4177920, 33488896, 268173312])
# print(t0)
# print(t)
#
# for i in range(1, 100):
#     exp = i / 100 * 4
#     print(512 ** exp)
