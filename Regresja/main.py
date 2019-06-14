import numpy as np
from typing import List, Tuple
import statistics
from matplotlib import pyplot as plt
import pandas as pd
import csv

def rmse(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise Exception("Lists of different length")
    return sum(map(lambda a: a**2, [y[i] - x[i] for i in range(len(x))]))


print(rmse([0,-1, 2,-3], [1, -2, 3, -4]))


def lin_reg(data: List[Tuple[float, float]]) -> Tuple[float, float]:
    mx = statistics.mean([i[0] for i in data])
    my = statistics.mean([i[1] for i in data])
    multi = sum([(data[i][0] - mx)*(data[i][1] - my) for i in range(len(data))])

    B1 = multi / (sum(map(lambda a: (a - mx)**2, [i[0] for i in data])))

    B0 = my - B1 * mx
    return B0, B1


A = [(1, 1),
     (2, 3),
     (4, 3),
     (3, 2),
     (5, 5)]

B0, B1 = lin_reg(A)
print(B0, B1)

from typing import List, Tuple, Optional


class LinearRegressor():
    def __init__(self):
        self._coeffs = None  # type: Optional[Tuple[float, float]]

    def fit(self, x: List[float], y: List[float]) -> None:
        mx = statistics.mean(x)
        my = statistics.mean(y)
        multi = sum([(x[i] - mx) * (y[i] - my) for i in range(len(x))])
        B1 = multi / sum(map(lambda a: (a - mx) ** 2, x))
        B0 = my - B1 * mx
        self._coeffs = (B0, B1)

    def predict(self, x: List[float]) -> List[float]:
        return [self.coeffs[0] + i * self.coeffs[1] for i in x]

    @property
    def coeffs(self) -> Tuple[float, float]:
        if self._coeffs is None:
            raise Exception('You need to call `fit` on the model first.')

        return self._coeffs

x = [1, 2, 4, 3, 5]
y = [1, 3, 3, 2, 5]


def plot_data(x: List[float], y: List[float]) -> None:
    plt.plot(x, y, "bo")
    lr = LinearRegressor()
    lr.fit(x, y)
    y_new = lr.predict(x)
    plt.plot(x, y_new, "r-")
    plt.show()
    print("Błąd: ", rmse(y, y_new))


data = [[], []]

with open('data.csv') as c:
    for row in c:
        row = row.split()
        data[0].append(float(row[0]))
        data[1].append(float(row[1]))
    plot_data(data[0], data[1])

