# Main function
import numpy as np

from lufact import lufact as lu
from invert import invert as inv
from determinant import determinant as det
from GenerateMatrix import generate_matrix

# matrix = generate_matrix(4)
# print(matrix)
# l, u, cost = lu(matrix)
# print(l)
# print(u)
# print(l @ u)
# print(det(matrix))
# print(np.linalg.det(matrix))

matrix = np.array([[1., 1., 1., 2.], [1., 1., 2., 1.], [1., 2., 1., 1.], [2., 1., 1., 1.]])
print(det(matrix))
