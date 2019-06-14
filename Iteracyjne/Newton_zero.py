import numpy as np


def d(fun, x, h=10e-5):
    return (fun(x+h) - fun(x-h))/(2*h)
    #return 2*x-3


def zero_of_fun(f, x1=0, epsilon=10e-6):

    x0 = x1+2*epsilon+1
    while abs(f(x1)) >= epsilon and abs(x1 - x0) >= epsilon:
        x0 = x1
        df = d(f, x0)
        if df == 0:
            df = d(f, x0, 10e-3)
        print("----iter----")
        print(df)
        x1 = x0 - f(x0) / df
        print(x1)


    return x1 if abs(f(x1)) < abs(f(x0)) else x0


f = lambda a: a**2 - 3 * a + 2

print(zero_of_fun(f))
