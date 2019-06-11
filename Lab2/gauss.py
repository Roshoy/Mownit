from functools import reduce
from typing import Optional, Tuple
import numpy as np


def agh_superfast_matrix_multiply(a: np.matrix, b: np.matrix) -> np.matrix:
    """Perform totally ordinary multiplication of matrices.

    :param a: matrix with dimensions 4(X) by 4(Y)
    :param b: matrix with dimensions 1 by 4(Y)
    :return:  matrix with dimensions 1 by 4(Y)
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape[1] != b.shape[0]:
        raise Exception("Zle wymiary macierzy")
    res = np.zeros((a.shape[0], b.shape[1]))

    for x in range(b.shape[1]):
        for y in range(a.shape[0]):
            for i in range(a.shape[1]):
                res[y][x] += a[y][i]*b[i][x]
    return np.asmatrix(res)


m1 = np.matrix([[1, 2],
                [3, 4],
                [4, 5],
                [5, 1]])

m2 = np.matrix([[1, 2, 3],
                [4, 5, 6]])


res = agh_superfast_matrix_multiply(m1, m2)
assert np.allclose(res, m1 * m2), "Wrong multiplication result"
print(np.allclose(res, m1 * m2))



def gauss_elim(A, B):
    a = np.asarray(A, dtype=float)
    b = np.asarray(B, dtype=float)
    # print("-------info B -----------")
    # print(repr(B))
    # print("-------info b------------")
    # print(repr(b))
    #
    # print("---------------changed----------")
    # print(B)
    Y = a.shape[0]
    X = a.shape[1]
    if X > Y:
        raise Exception("No solution")
    res = np.zeros((Y, X+b.shape[1]))
    for y in range(0, Y):
        for x in range(0, res.shape[1]):
            if x < X:
                res[y][x] = a[y][x]
            else:
                res[y][x] = b[y][x-X]
    a = res
    for x in range(Y):
        a[x] /= a[x][x]
        for y in range(x + 1, Y):
             a[y] = np.subtract(a[y], a[x]*a[y][x])


    for x in range(min(X, Y)-1, 0, -1):
        for y in range(0, x):
            a[y] = np.subtract(a[y], a[x] * a[y][x])

    # for r in range(Y):
    #     b[r][0] = a[r][X]
    res = np.zeros((Y,1))
    for r in range(Y):
        res[r][0] = a[r][X]
    # print("----------No pivot b--------")
    # print(b)
    # print("----------B-----------------")
    # print(B)
    # print("-----------res--------------")
    # print(res)
    # print("\n-----------------------------\n")
    return b


def gauss_elim_pivot(A, B):
    a = np.asarray(A, dtype=float)
    b = np.asarray(B, dtype=float)
    # print("---------------changed----------")
    # print(B)
    Y = a.shape[0]
    X = a.shape[1]
    if X > Y:
        raise Exception("No solution")
    res = np.zeros((Y, X+b.shape[1]))
    for y in range(0, Y):
        for x in range(0, res.shape[1]):
            if x < X:
                res[y][x] = a[y][x]
            else:
                res[y][x] = b[y][x-X]
    a = res
    # print(a)
    # print("---------------changed 2----------")
    # print(B)
    for x in range(Y):

        x_max = reduce(lambda aa, b: aa if abs(aa[1]) > abs(b[1]) else b, list(enumerate([i[x] for i in a[x:]])))[0]+x
        tmp = [i for i in a[x]]
        for i in range(len(a[x_max])):
            a[x][i] = a[x_max][i]
        for i in range(len(a[x_max])):
            a[x_max][i] = tmp[i]
        # print(a[x])
        # print(a[x_max])
        a[x] /= a[x][x]
        for y in range(x + 1, Y):
             a[y] = np.subtract(a[y], a[x]*a[y][x])
        #print(a)

    for x in range(min(X, Y)-1, 0, -1):
        for y in range(0, x):
            a[y] = np.subtract(a[y], a[x] * a[y][x])
        #print(a)
    # print(a)
    # print("---------------changed 3----------")
    # print(B)
    # res = np.zeros(Y)
    # for r in range(Y):
    #     res[r] = a[r][X]
    #     #print(b[r])
    # #print(b)
    # print("---------------changed 4----------")
    # print(B)
    # print(res)
    #
    # print(a)
    for r in range(Y):
        b[r][0] = a[r][X]
    # print("----------Pivot--------")
    # print(b)
    return b

A = np.matrix([[0.0001, -5.0300, 5.8090, 7.8320],
               [2.2660, 1.9950,  1.2120, 8.0080],
               [8.8500, 5.6810,  4.5520, 1.3020],
               [6.7750, -2.253,  2.9080, 3.9700]])

A = np.asarray(A)
b = np.matrix([9.5740, 7.2190, 5.7300, 6.2910]).transpose()
x = np.linalg.solve(A, b)
print(x)
res = gauss_elim_pivot(A, b)

res_bez_piv = gauss_elim(A, b)

print("-------Pivot res-------")
print(res)
print("-------bez pivot-------")
print(res_bez_piv)
print(np.allclose(x, res_bez_piv))
print(np.allclose(x, res))
print(np.allclose(res_bez_piv,res))
