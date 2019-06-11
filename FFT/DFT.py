import numpy as np
import math
import csv
from scipy import fftpack
import matplotlib.pyplot as plt
import pandas


def dft(x):
    x = np.asarray(x, dtype=float) # make array from list
    N = x.shape[0] # take size
    n = np.arange(N)    # make part of skeleton for output
    k = n.reshape((N, 1))   # kind of transpose, make it vertical not horizontal
    M = np.exp(-2j * np.pi * k * n / N)   # here's some magic of dft formula
    return np.dot(M, x)


def fft(data):
    x = np.asarray(data, dtype=float)
    N = x.shape[0]
    if N % 2 != 0:
        raise Exception("Should be power of 2")
    if N <= 8:
        return dft(data)
    else:
        res_even = fft(data[::2])
        res_odd = fft(data[1::2])
        factor = np.exp(-2j * math.pi * np.arange(N) / N)
        return np.concatenate([res_even + factor[:N / 2] * res_odd,
                               res_even + factor[N / 2:] * res_odd])


def DFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N < 8:
        return DFT(x)
    if N % 2 != 0:
        raise Exception("Not divisible by 2")
    res_even = FFT(x[::2])
    res_odd = FFT(x[1::2])
    factor = np.exp(-2j*np.pi*np.arange(N/2)/N)
    return np.concatenate([res_even + factor * res_odd,
                       res_even - factor * res_odd])

csvfile = open("new_data.csv")
readCSV = csv.reader(csvfile, delimiter=",")
tab = [row for row in readCSV]
tab = np.asarray(tab, dtype=int)
#tab = pandas.DataFrame.resample()
x = [i[0] for i in tab]
y = [i[1] for i in tab]

ffy = FFT(y)
print(np.allclose(ffy, np.fft.fft(y)))

y = FFT(y)
#y[0] = 0
x = np.fft.fftfreq(len(y))
fig, ax = plt.subplots()
ax.stem(x, np.abs(y))
ax.set_xlabel('Frequency in things/us')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_ylim(-1, 7000)
plt.show()
