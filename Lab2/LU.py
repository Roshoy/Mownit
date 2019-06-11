from functools import reduce
from typing import Optional, Tuple
import numpy as np

def LU(A):
    A = np.asarray(A)
    N = A.shape[0]
    L = np.zeros((N, N))
    U = np.zeros((N, N))
    for i in range(N):
        L[i][i] = 1
    for j in range(N):
        for i in range(N):
            if i <= j:
                U[i][j] = A[i][j]
                for k in range(i-1):
                    U[i][j] += L[i][k] * U[k][j]
            else:
                L[i][j] = A[i][j]
                for k in range(j-1):
                    L[i][j] -= L[i][k] * U[k][j]
                L[i][j] /= U[j][j]
    print(L)
    print(U)

A = np.matrix([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

LU(A)
