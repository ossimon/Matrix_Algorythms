import treecreator as tc
import matrixcreator as mc
import numpy as np
import rysowacz as rys
import time
import approximation as apr


def print_img(img):
    print('[', end='')
    for i in range(img.shape[0]):
        print('[', end='')
        for j in range(img.shape[1]):
            print(img[i, j], end='')
            if j != img.shape[1] - 1:
                print(', ', end='')
        print(']', end='')
        if i != img.shape[0] - 1:
            print(', ')
    print('],\n')


r = 3
epsilon = 10 ** -8
min_exp = 1
max_exp = 10
min_fill = 1
max_fill = 6
times = np.zeros(shape=(5, max_exp - 1))
sizes = 2 ** np.arange(start=min_exp, stop=max_exp, step=1)

for i in range(min_fill, max_fill):
    filled_percentage = i * 10
    print(filled_percentage)
    for j in range(min_exp, max_exp):

        size = 2 ** j
        a = mc.create_random_matrix((size, size), filled_percentage)
        tmin = 0
        tmax = a.shape[0]
        smin = 0
        smax = a.shape[1]

        start = time.time()
        root = tc.create_tree(a, tmin, tmax, smin, smax, r, epsilon)
        stop = time.time()

        total_time = stop - start
        times[i - 1, j - 1] = total_time
        print(total_time)
        # img = root.printable()
        # rys.draw_matrix(img, 'r1_good_compressed_matrix' + str(filled_percentage) + '.png')
        # print(np.round(a, 3))
        # recreated_a = root.recreate()
        # print(np.round(img, 3))
        # print(np.round(recreated_a, 3))
        # print(np.round(recreated_a - a, 3))
        # print(np.sum(np.square(a - recreated_a)), end=',\n')

    print()

for i, results in enumerate(times):
    print('approximation for fill percent equal to:', 10 * (i + 1))
    print(apr.approximate(sizes, results, 1, 3))
