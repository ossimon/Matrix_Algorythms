# Main function
import numpy as np

import lufact as lu
import invert as inv
import determinant as det

from GenerateMatrix import generate_matrix
from invert import invert

matrix = generate_matrix(4)
print(matrix)
print(invert(matrix))
print(np.linalg.inv(matrix))
