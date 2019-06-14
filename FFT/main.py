import pandas as pd
import numpy as np
import scipy as sp
import datetime
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt


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
    if N <= 32:
        return DFT(x)
    # if N % 2 != 0:
    #     raise Exception("Not divisible by 2")
    res_even = FFT(x[::2])
    res_odd = FFT(x[1::2])
    factor = np.exp(-2j*np.pi*np.arange(N)/N)
    if N//2 != res_even.shape[0]:
        print("hej")
        print(N//2, res_even.shape[0])
        print(N)
    return np.concatenate([res_even + factor[:N//2] * res_odd,
                       res_even + factor[N//2:] * res_odd])


df = pd.read_csv('new_data.csv', parse_dates=['Datetime'])

# df.plot(x='Datetime', y='AEP_MW')

# plt.show()

df['Datetime'] = df['Datetime'].apply(lambda x: x.timestamp())

n = df.shape[0]

freqs = fftfreq(n)

df_fft = FFT(df['AEP_MW'])

plt.plot(freqs[1:n//2], np.abs(df_fft)[1:n//2])
plt.xlabel('Frequency /hour')
plt.ylabel('Frequency Domain (MW)')

plt.show()
