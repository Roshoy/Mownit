import numpy as np
import math as m


def is_positively_defined(A) -> bool:
    A = np.asarray(A)
    N = A.shape[0]
    for i in range(1, N):
        B = np.zeros((i, i))
        for x in range(i):
            for y in range(i):
                B[x][y] = A[x][y]
        if 0 > np.linalg.det(B):
            return False
    return True


def Cholesky(A) -> np.ndarray:
    A = np.asarray(A)
    if not is_positively_defined(A):
        raise Exception("Matrix is not positively defined")
    N = A.shape[0]
    L = np.zeros((N, N))
    for s in range(N):
        for i in range(s, N):
            L[i][s] = A[i][s]
            for j in range(s):
                L[i][s] -= L[i][j] * L[s][j]
            if s == i:
                L[i][s] = m.sqrt(L[i][s])
            else:
                L[i][s] = L[i][s]/L[s][s]
    return L


A = np.matrix([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]])
L = Cholesky(A)
print(L)
xL = np.linalg.cholesky(A)
B = L @ L.T
print(np.allclose(L, xL))