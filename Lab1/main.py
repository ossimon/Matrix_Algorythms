# Main file, where we're running our implementations of algorithms

import numpy as np
from time import time
import matplotlib.pyplot as plt

import ClassicMatrixMultiplication as cmm
import StrassensMatrixMultiplication as smm
import AIGeneratedMatrixMultiplication as aimm
from GenerateMatrix import generate_matrix

# Simple test for small matrix
# m3 = np.ones(shape=(256, 256))
# print(cmm.classic_multiply(m3, m3)[1])
# print(smm.strassens_multiply(m3, m3)[1])

#

times_classic = []
times_strassen = []

for k in range(2, 3):
    size = 2 ** k
    m1 = generate_matrix(size)
    m2 = generate_matrix(size)

    start = time()
    cmm.classic_multiply(m2, m1)
    end = time()
    times_classic.append(round(end - start, 4))

    start = time()
    smm.strassens_multiply(m2, m1)
    end = time()
    times_strassen.append(round(end - start, 4))

print("\nCLASSIC:\n")
print("Time:\n", times_classic)
print("\nSTRASSEN:\n")
print("Time:\n", times_strassen)


times_classic = [0.0, 0.001, 0.008, 0.0628, 0.5321, 4.0438, 31.8409, 256.0709]
times_strassen = [0.0, 0.001, 0.008, 0.0708, 0.4104, 2.9003, 20.2428, 145.5252]


fig, ax = plt.subplots()
ax.plot(times_classic, color='green', label='Binet')
ax.plot(times_strassen, color='red', label='Strassen')
ax.legend(loc='upper left')
x = range(2, len(times_classic) + 2)
default_x_ticks = range(len(x))
plt.xticks(default_x_ticks, x)
plt.show()
