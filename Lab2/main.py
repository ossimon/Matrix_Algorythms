# Main function
import numpy as np
from time import time
import matplotlib.pyplot as plt

from GenerateMatrix import generate_matrix
from approximate import approximate
from invert import invert
from lufact import lufact as lu
from determinant import determinant as det

# Simple test for small matrix
test_matrix = np.array([[0.92252088, 0.64621687, 0.57633497, 0.67388787],
                        [0.4469564, 0.96570353, 0.10687379, 0.58400998],
                        [0.67534764, 0.62991871, 0.32299196, 0.26900836],
                        [0.32309314, 0.09388855, 0.1462494, 0.43539867]])

inv_matrix = invert(test_matrix)[0]
l = lu(test_matrix)[0]
u = lu(test_matrix)[1]
det_matrix = det(test_matrix)[0]
print('Inversion\n', inv_matrix, '\n')
print(np.linalg.inv(test_matrix))
print('LU factorization\n')
print('L = \n', l, '\n')
print('U = \n', u, '\n')
print(l @ u)
print('Determinant\n', det_matrix, '\n')
print(np.linalg.det(test_matrix))

# Norm for our results and numpy results
print('\nNORM:\n')
print('Inversion\n', (inv_matrix - np.linalg.inv(test_matrix)) ** 2, '\n')
print('LU factorization\n', (l @ u - test_matrix) ** 2, '\n')
print('Determinant\n', (det_matrix - np.linalg.det(test_matrix)) ** 2, '\n')

# Calculating time and number of operations for matrix 2^k x 2^k

times_invert = []
times_lu = []
times_det = []

operations_invert = []
operations_lu = []
operations_det = []

for k in range(2, 7):
    size = 2 ** k
    matrix = generate_matrix(size)

    start = time()
    operations_invert.append(invert(matrix)[1])
    end = time()
    times_invert.append(round(end - start, 4))

    start = time()
    operations_lu.append(lu(matrix)[2])
    end = time()
    times_lu.append(round(end - start, 4))

    start = time()
    operations_det.append(det(matrix)[1])
    end = time()
    times_det.append(round(end - start, 4))

print("\nINVERSION:\n")
print("Time:\n", times_invert)
print("Floating point operations:\n", operations_invert)
print("\nLU FACTORIZATION:\n")
print("Time:\n", times_lu)
print("Floating point operations:\n", operations_lu)
print("\nDETERMINANT:\n")
print("Time:\n", times_det)
print("Floating point operations:\n", operations_det)

# Overriding lists with already calculated values
times_invert = [0.0, 0.003, 0.017, 0.1366, 0.9335, 6.2299, 42.7913, 304.1721, 2234.2181]
times_lu = [0.001, 0.002, 0.023, 0.1242, 0.7844, 5.665, 37.9695, 276.4866, 2015.5023]
times_det = [0.0, 0.002, 0.0339, 0.1192, 0.7844, 5.64, 37.8737, 279.6936, 2017.9579]
operations_invert = [286, 3074, 26446, 206114, 1529326, 11056514, 78810766, 557356514]
operations_lu = [173, 2169, 20635, 170773, 1312323, 9682229, 69826355, 497141733]
operations_det = [177, 2177, 20651, 170805, 1312387, 9682357, 69826611, 497142245]

# Calculation of complexity
print('\nCOMPLEXITY:')
print("\nINVERSION:\n")
print("Time:\n", round(approximate(times_invert, 2, 10), 4))
print("Floating point operations:\n", round(approximate(operations_invert, 2, 9), 4))
print("\nLU FACTORIZATION:\n")
print("Time:\n", round(approximate(times_lu, 2, 10), 4))
print("Floating point operations:\n", round(approximate(operations_lu, 2, 9), 4))
print("\nDETERMINANT:\n")
print("Time:\n", round(approximate(times_det, 2, 10), 4))
print("Floating point operations:\n", round(approximate(operations_det, 2, 9), 4))

# Plots for time and floating point operations

plt.plot(range(2, len(times_invert) + 2), times_invert)
plt.xlabel('Size of matrix [2^k x 2^k]')
plt.ylabel('Time [s]')
# plt.title('Wykres 1.1. Czas działania algorytmu odwracania macierzy')
plt.show()

plt.plot(range(2, len(operations_invert) + 2), operations_invert)
plt.xlabel('Size of matrix [2^k x 2^k]')
plt.ylabel('Floating point operations')
# plt.title('Wykres 1.2. Ilość operacji zmiennoprzecinkowych algorytmu odwracania macierzy')
plt.show()

plt.plot(range(2, len(times_lu) + 2), times_lu)
plt.xlabel('Size of matrix [2^k x 2^k]')
plt.ylabel('Time [s]')
# plt.title('Wykres 2.1. Czas działania algorytmu LU faktoryzacji')
plt.show()

plt.plot(range(2, len(operations_lu) + 2), operations_lu)
plt.xlabel('Size of matrix [2^k x 2^k]')
plt.ylabel('Floating point operations')
# plt.title('Wykres 2.2. Ilość operacji zmiennoprzecinkowych algorytmu LU faktoryzacji')
plt.show()

plt.plot(range(2, len(times_lu) + 2), times_det)
plt.xlabel('Size of matrix [2^k x 2^k]')
plt.ylabel('Time [s]')
# plt.title('Wykres 3.1. Czas działania algorytmu obliczania wyznacznika')
plt.show()

plt.plot(range(2, len(operations_lu) + 2), operations_det)
plt.xlabel('Size of matrix [2^k x 2^k]')
plt.ylabel('Floating point operations')
# plt.title('Wykres 3.2. Ilość operacji zmiennoprzecinkowych algorytmu obliczania wyznacznika')
plt.show()
