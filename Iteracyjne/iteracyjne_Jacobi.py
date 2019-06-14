import numpy as np
import matplotlib.pyplot as plt
import random

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



A = np.matrix([[10, 2],
                [3, 24]])
b = np.matrix([4, 12.53]).transpose()

my_res = jacobi(A, b, 100).T
res = np.linalg.solve(A, b)
print(np.allclose(res, my_res))

A = np.matrix([[32, 5, 3],
               [0, 43, 0.001],
               [3, 4, 33]])
b = np.matrix([2, 53, 0]).T

my_res = jacobi(A, b, 100).T
res = np.linalg.solve(A, b)
print(np.allclose(res, my_res))

A = np.matrix([[10, -5.0300, 0, 0],
               [2.2660, 1.9950,  1.2120, 0],
               [0, 5.6810,  4.5520, 1.3020],
               [0, -0,  2.9080, 3.9700]])

b = np.matrix([9.5740, 7.2190, 5.7300, 6.2910]).transpose()
my_res = jacobi(A, b, 100).T
res = np.linalg.solve(A, b)
print(np.allclose(res, my_res))

# A = [[random.randint(100, 200)/(abs(i-j)+1) for j in range(100)] for i in range(100)]
#
# b = np.matrix([random.randint(-100, 100) for i in range(100)]).transpose()

error = []

for i in range(100):
    my_res = jacobi(A, b, i).T
    res = np.linalg.solve(A, b)
    error.append(sum(abs(np.asarray(res-my_res))))
plt.plot(error)
plt.ylabel('error')


A = [[random.randint(100, 200)/(abs(i-j)+1) for j in range(10)] for i in range(10)]

b = np.matrix([random.randint(-100, 100) for i in range(10)]).transpose()
error = []

for i in range(100):
    my_res = jacobi(A, b, i).T
    res = np.linalg.solve(A, b)
    error.append(sum(abs(np.asarray(res-my_res))))
plt.plot(error)
plt.show()


