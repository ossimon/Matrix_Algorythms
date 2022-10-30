# Main file, where we're running our implementations of algorythms

import numpy as np

import ClassicMatrixMultiplication as cmm
import StrassensMatrixMultiplication as smm
import AIGeneratedMatrixMultiplication as aimm

m1 = np.array([[1, 2], [3, 4]])
m2 = np.ones(shape=(2, 2))
print(cmm.classic_multiply(m2, m1))
