import scipy.integrate.quadrature as qt
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import scipy
#Zad 1
def Legandre_polys(k):
    if k == 0:
        return lambda x: 1.
    elif k == 1:
        return lambda x: x
    else:
        return lambda x: (2 * k - 1) / k * x * Legandre_polys(k-1)(x) - (k - 1) / (k) * Legandre_polys(k-2)(x)


lx = np.arange(-1, 1.0001, 0.0001)
lyk = [[Legandre_polys(k)(i) for i in lx] for k in range(6)]
#lyk = [Legandre_polys(i, 2) for i in lx]

for i in lyk:
    plt.plot(lx, i)
plt.show()
#
for i in range(2, 5):
    print("Stopień:", i)
    for j in range(i):
        r = 2 / i * j
        print(scipy.optimize.bisect(Legandre_polys(i), -1 + r, -1 + r + 2 / i))
    print("Biblioteczne:")
    print(np.polynomial.legendre.leggauss(i)[0])
##########
# Zad 2

def gauss_integr(k, f):
    gauss_x, w = np.polynomial.legendre.leggauss(k)
    res = 0
    for i in range(k):
        res += w[i] * f(gauss_x[i])
    return res

def integ_test(f):
    print("Obliczone:  ", gauss_integr(4, f))
    print("Rzeczywiste:", scipy.integrate.quad(f, -1 , 1)[0])

integ_test(lambda x: x + 4)
integ_test(lambda x: 10*x**3 + 2*x**2 - 0.456 * x ** 1 - 3)
integ_test(lambda x: 12/(x**2 + 3) - x + 2)
print("Sprawdzanie dokładności")
for i in range(1,5):
    integ_test(lambda x: x**(i*2))

#Zad 3
print("Z przedzialami:")

def gauss_integr_ab(k, f, a, b):
    return gauss_integr(k, lambda x: f((b+a)/2 + x * (b-a)/2)*(b-a)/2)

def integ_test_ab(f, a, b):
    print("Obliczone:  ", gauss_integr_ab(4, f, a, b))
    print("Rzeczywiste:", scipy.integrate.quad(f, a, b)[0])

integ_test_ab(lambda x: x + 4, 0, 5)
integ_test_ab(lambda x: 10*x**3 + 2*x**2 - 0.456 * x ** 1 - 3, -2, 7)
integ_test_ab(lambda x: 12/(x**2 + 3) - x + 2, 0, 2)


#Zad 5
print("Metoda prostokatow:")
def rect_center(k, f, a, b):
    if k == 0:
        return 0
    res = 0
    r = (b-a)/k
    for i in range(k):
        res += r * f(a + i * r + 0.5 * r)
    return res

def prostok_test_ab(k, f, a, b, show=True):
    q = rect_center(k, f, a, b)
    w = scipy.integrate.quad(f, a, b)[0]
    if show:
        print("Obliczone:  ", rect_center(k, f, a, b))
        print("Rzeczywiste:", scipy.integrate.quad(f, a, b)[0])
    return abs(q-w)

prostok_test_ab(5,lambda x: x + 4, 0, 5)
prostok_test_ab(5,lambda x: 10*x**3 + 2*x**2 - 0.456 * x ** 1 - 3, -2, 7)
prostok_test_ab(5,lambda x: 12/(x**2 + 3) - x + 2, 0, 2)

print("Prostokatow dokladnosc")

diffs = []

for i in range(100):
    diffs.append(prostok_test_ab(i, lambda x: 10*x**3 + 2*x**2 - 0.456 * x ** 1 - 3, -2, 7, show=False))

plt.plot(diffs)
plt.show()