import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List

def rmse(x, y) -> float:
    if len(x) != len(y):
        raise Exception("Lists of different length: ", len(x), len(y))
    return sum(map(lambda a: a**2, [y[i] - x[i] for i in range(len(x))]))


def jacobi(a: np.matrix, b: np.matrix, iterations) -> np.matrix:
    iter = 0
    a = np.asarray(a)
    b = np.asarray(b)
    N = a.shape[0]
    x1 = np.zeros(N)
    x0 = np.asarray(x1)
    while iter < iterations:
        for i in range(N):

            x1[i] = b[i][0]
            for j in range(N):
                if j != i:
                    x1[i] -= a[i][j] * x0[j]
            x1[i] /= a[i][i]

        x0 = np.asarray(x1)
        iter += 1

    return np.asmatrix(x1)


def gauss_seidel_solve(a: np.matrix, b: np.matrix, iterations) -> np.matrix:
    iter = 0
    a = np.asarray(a)
    b = np.asarray(b)
    N = a.shape[0]
    x1 = np.zeros(N)
    x0 = x1.copy()
    while iter < iterations:
        for i in range(N):

            x1[i] = b[i][0]
            for j in range(i):
                x1[i] -= a[i][j] * x1[j]
            for j in range(i + 1, N):
                x1[i] -= a[i][j] * x0[j]
            x1[i] /= a[i][i]

        x0 = x1.copy()
        iter += 1
    return np.asmatrix(x1)


def sor_solve(a: np.matrix, b: np.matrix, w, iterations) -> np.matrix:
    iter = 0
    a = np.asarray(a)
    b = np.asarray(b)
    N = a.shape[0]
    x1 = np.zeros(N)
    x0 = x1.copy()
    while iter < iterations:
        for i in range(N):

            x1[i] = b[i][0]
            for j in range(i):
                x1[i] -= a[i][j] * x1[j]
            for j in range(i + 1, N):
                x1[i] -= a[i][j] * x0[j]
            x1[i] = (1-w) * x0[i] + x1[i] * w / a[i][i]

        x0 = x1.copy()
        iter += 1

    return np.asmatrix(x1)



A = np.matrix([[10, -5.0300, 0, 0],
               [2.2660, 1.9950,  1.2120, 0],
               [0, 5.6810,  4.5520, 1.3020],
               [0, -0,  2.9080, 3.9700]])

b = np.matrix([9.5740, 7.2190, 5.7300, 6.2910]).transpose()
res = np.linalg.solve(A, b)
my_res = gauss_seidel_solve(A, b, 50).T

print(np.allclose(res, my_res))

A = np.matrix([[12, 3, 4],
               [5 , 9, 1],
               [1 , 4, 8]])

b = np.matrix([30, 26, 33]).transpose()
my_res = sor_solve(A, b, 1.3, 50).T
res = np.linalg.solve(A, b)

iter = 20
j = [rmse(np.asarray(res.T)[0], np.asarray(jacobi(A, b, i+1))[0]) for i in range(iter)]
gs = [rmse(np.asarray(res.T)[0], np.asarray(gauss_seidel_solve(A, b, i+1))[0]) for i in range(iter)]
sor = [rmse(np.asarray(res.T)[0], np.asarray(sor_solve(A, b, 0.7, i+1))[0]) for i in range(iter)]

print(j)
print(gs)
print(sor)
plt.plot(list(range(1, iter + 1)), j, "go")
plt.plot(list(range(1, iter + 1)), gs, "bo")
plt.plot(list(range(1, iter + 1)), sor, "ro")

plt.show()