# Main file, where we're running our implementations of algorithms

import numpy as np
from time import time
import matplotlib.pyplot as plt

import ClassicMatrixMultiplication as cmm
import StrassensMatrixMultiplication as smm
from GenerateMatrix import generate_matrix

# Simple test for small matrix
mA = np.array([[0.05638678, 0.3609324,  0.74906327, 0.05357696],
 [0.80698438, 0.5203182,  0.9071571,  0.53237322],
 [0.18277115, 0.16197957, 0.71735672, 0.58922524],
 [0.18017283, 0.58594149, 0.01307891, 0.83503161]])

mB = np.array([[0.07809653, 0.79444128, 0.23680782, 0.97269623],
 [0.52989394, 0.46961841, 0.90966219, 0.99167667],
 [0.04228784, 0.82503773, 0.63286437, 0.98541714],
 [0.06724143, 0.82957049, 0.78835744, 0.83225435]])

print('Classic solution\n', cmm.classic_multiply(mA, mB)[0], '\n')
print('Strassen solution\n', smm.strassens_multiply(mA, mB)[0], '\n')

# Calculating time and number of operations for matrix 2^k x 2^k

times_classic = []
times_strassen = []
times_numpy = []

operations_classic = []
operations_strassen = []

for k in range(2, 7):
    size = 2 ** k
    m1 = generate_matrix(size)
    m2 = generate_matrix(size)

    start = time()
    operations_classic.append(cmm.classic_multiply(m2, m1)[1])
    end = time()
    times_classic.append(round(end - start, 4))

    start = time()
    operations_strassen.append(smm.strassens_multiply(m2, m1)[1])
    end = time()
    times_strassen.append(round(end - start, 4))

print("\nCLASSIC:\n")
print("Time:\n", times_classic)
print("Floating point operations:\n", operations_classic)
print("\nSTRASSEN:\n")
print("Time:\n", times_strassen)
print("Floating point operations:\n", operations_strassen)

# Overriding lists with already calculated values
times_classic = [0.0, 0.001, 0.0279, 0.0708, 0.5765, 4.1834, 32.8724, 263.2043, 2062.088]
times_strassen = [0.001, 0.001, 0.008, 0.0788, 0.4219, 3.027, 20.7699, 146.1128, 1018.5908]
operations_classic = [112, 960, 7936, 64512, 520192, 4177920, 33488896, 268173312, 2146435072]
operations_strassen = [247, 2017, 15271, 111505, 798967, 5666497, 39960391, 280902385, 1971035287]


# Plots for time and floating point operations

fig, ax = plt.subplots()
ax.plot(times_classic, color='green', label='Binet')
ax.plot(times_strassen, color='red', label='Strassen')
ax.legend(loc='upper left')
x = range(2, len(times_classic) + 2)
default_x_ticks = range(len(x))
plt.xticks(default_x_ticks, x)
plt.xlabel('Size of matrix [2^k x 2^k]')
plt.ylabel('Time [s]')

plt.show()

fig, ax = plt.subplots()
ax.plot(operations_classic, color='green', label='Binet')
ax.plot(operations_strassen, color='red', label='Strassen')
ax.legend(loc='upper left')
x = range(2, len(operations_classic) + 2)
default_x_ticks = range(len(x))
plt.xticks(default_x_ticks, x)
plt.xlabel('Size of matrix [2^k x 2^k]')
plt.ylabel('Floating point operations')

plt.show()
