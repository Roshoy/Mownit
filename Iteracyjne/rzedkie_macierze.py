import numpy as np

# przykładowe
NNZ = 10    # liczba niezerowych
M = N = 5   # wymiary macierzy N x M

A_OG = np.matrix ([[1, 0, 0],
                   [0, 4, 1],
                   [0, 0, 3],
                   [0, 0, 2]])

### Współrzedne

A1 = [{'x': 0, 'y': 0, 'v': 0}]
A2 = {(0, 0): 0}
A3 = np.zeros((3, NNZ))
# A3[0] - x
# A3[1] - y
# A3[2] - v

def to_coords(A) -> np.ndarray:
    A = np.asarray(A)
    N, M = A.shape
    A3 = []
    for y in range(M):
        for x in range(N):
            if A[x][y] !=0:
                A3.append([x, y, A[x][y]])
    A3 = np.asarray(A3)
    return A3

A_C = to_coords(A_OG)
print(A_C)

### CSC
AV = np.zeros(NNZ)  # wartości wpisywane kolumnami
AI = np.zeros(NNZ)  # numery wierszy odpowiadające wartościom w AV
AJ = np.zeros(M)  # w którym miejscu kończy się kolumna

def to_csc(A):
    A = np.asarray(A)
    N, M = A.shape

    AV = []  # wartości wpisywane kolumnami
    AI = []  # numery wierszy odpowiadające wartościom w AV
    AJ = np.zeros(M)  # w którym miejscu kończy się kolumna
    i = 0

    for y in range(M):
        for x in range(N):
            if A[x][y] != 0:
                AV.append(A[x][y])
                AI.append(x)
                i += 1
        AJ[y] = i-1
    AV = np.asarray(AV)
    AI = np.asarray(AI)
    return (AV, AI, AJ)

A_CSC = to_csc(A_OG)
print(A_CSC[0])
print(A_CSC[1])
print(A_CSC[2])

class SparseVector:
    def __init__(self, A=((),)):
        A = np.asarray(A)
        if len(A.shape) > 2:
            raise Exception("To many dimensions")
        elif len(A.shape) == 2:
            if A.shape[0] != 1 and A.shape[1] != 1:
                raise Exception("To many dimensions")
            else:
                A = A.flatten()

        self.v = []
        self.x = []
        for x in range(A.shape[0]):
            if A[x] != 0:
                self.x.append(x)
                self.v.append(A[x])

    def __len__(self):
        return len(self.v)

    def __iter__(self):
        for x in range(len(self)):
            yield (self.x[x], self.v[x])

    def __getitem__(self, item):
        return self.x[item], self.v[item]

    def get_value(self, key):
        for i, x in enumerate(self.x):
            if x == key:
                return self.v[i]
        raise KeyError(key)

    def __truediv__(self, other):
        nv = SparseVector()
        nv.v = [v/other for v in self.v]
        nv.x = [x for x in self.x]
        return nv

    def __mul__(self, other):
        nv = SparseVector()
        nv.v = [v * other for v in self.v]
        nv.x = [x for x in self.x]

        return nv

    def __add__(self, other):
        nv = SparseVector()
        for x1 in range(len(self)):
            found = False
            for x2 in range(len(other)):
                if self.x[x1] == other.x[x2]:
                    nv.v.append(self.v[x1] + other.v[x2])
                    nv.x.append(self.x[x1])
                    found = True
                    break
            if not found:
                nv.v.append(self.v[x1])
                nv.x.append(self.x[x1])
        for i, x in enumerate(other.x):
            if x not in nv.x:
                nv.v.append(other.v[i])
                nv.x.append(x)
        return nv

    def dot(self, other):
        nv = SparseVector()
        for x1 in range(len(self)):
            for x2 in range(len(other)):
                if self.x[x1] == other.x[x2]:
                    nv.v.append(self.v[x1] * other.v[x2])
                    nv.x.append(self.x[x1])
                    break
        return nv

class SparseMatrix:
    def __init__(self, A=((),)):
        A = np.asarray(A)
        self.shape = A.shape
        if len(A.shape) > 2:
            raise Exception("To many dimensions")

        self.v = []
        self.x = []
        self.y = []
        for x in range(A.shape[0]):
            for y in range(A.shape[1]):
                if A[x][y] != 0:
                    self.x.append(x)
                    self.v.append(A[x][y])
                    self.y.append(y)

    def __len__(self):
        return len(self.v)

    def __iter__(self):
        for x in range(len(self)):
            yield self.x[x], self.y[x], self.v[x]

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.v[item]

    def get_value(self, key):
        for i in range(len(self.x)):
            if self.x[i] == key[0] and self.y[i] == key[1]:
                return self.v[i]
        raise KeyError(key)

    def __truediv__(self, other):
        nv = SparseMatrix()
        nv.v = [v/other for v in self.v]
        nv.x = [x for x in self.x]
        nv.y = [y for y in self.y]
        return nv

    def __mul__(self, other):
        nv = SparseMatrix()
        nv.v = [v * other for v in self.v]
        nv.x = [x for x in self.x]
        nv.y = [y for y in self.y]
        return nv

    def __add__(self, other):
        if self.shape != other.shape:
            raise Exception("Wrong matrix dimensions")
        nv = SparseMatrix()
        nv.shape = self.shape
        for x1 in range(len(self)):
            found = False
            for x2 in range(len(other)):
                if self.x[x1] == other.x[x2] and self.y[x1] == other.y[x2]:
                    nv.v.append(self.v[x1] + other.v[x2])
                    nv.x.append(self.x[x1])
                    nv.y.append(self.y[x1])
                    found = True
                    break
            if not found:
                nv.v.append(self.v[x1])
                nv.x.append(self.x[x1])
                nv.y.append(self.y[x1])

        for i, x in enumerate(other.x):
            exist = False
            for j in range(len(self)):

                if x == self.x[j] and other.y[i] == self.y[j]:
                    exist = True
                    break
            if not exist:
                nv.v.append(other.v[i])
                nv.x.append(x)
                nv.y.append(other.y[i])
        return nv

    def dot(self, other):
        if self.shape[1] != other.shape[0]:
            raise Exception("Wrong matrix dimensions")

        nv = SparseMatrix()
        nv.shape = (self.shape[1], other.shape[0])
        for k1, v1 in enumerate(self.v):
            for k2, v2 in enumerate(other.v):
                if self.y[k1] == other.x[k2]:
                    exist = False
                    for i in range(len(nv)):
                        if nv.x[i] == self.x[k1] and nv.y[i] == other.y[k2]:
                            exist = True
                            nv.v[i] += v1 * v2
                            break
                    if not exist:
                        nv.v.append(v1 * v2)
                        nv.x.append(self.x[k1])
                        nv.y.append(other.y[k2])

        return nv

    def __str__(self):
        res = ""
        for i in range(len(self)):
            res += str((self.x[i], self.y[i], self.v[i])) + '\n'
        return res

A = np.matrix([[1], [2], [3]])
print(A)
A = np.asarray(A)
A = A.flatten()
print(A)

SM = SparseMatrix(A_OG)
print(SM)

A_OG2 = np.matrix ([[1, 0, 0],
                   [0, 4, 1],
                   [0, 0, 3],
                   [0, 10, 2]])
A_OG3 = np.matrix ([[1, 0, 0],
                   [0, 4, 1],
                   [0, 10, 2]])

SM3 = SparseMatrix(A_OG3)
SM2 = SparseMatrix(A_OG2)
SM_Add = SM + SM2
print(SM_Add)
SM_Dot = SM.dot(SM3)
print(SM_Dot)