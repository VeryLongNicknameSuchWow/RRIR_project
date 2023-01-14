import operator
from math import sin
import numpy as np
import matplotlib.pyplot as plt

X0 = 0
X1 = 2
N = 20

I_RANGE = range(1, N + 1)  # pominięcie 0 wynika z warunku Dirichleta
STEP = (X1 - X0) / N

INTEGRAL_STEPS = 1000
INTEGRAL_STEPS_RANGE = range(INTEGRAL_STEPS)
INTEGRAL_WIDTH = (X1 - X0) / INTEGRAL_STEPS


def integral(f1, f2):
    def f(x):
        return f1(x) * f2(x)

    # lewa suma Riemanna
    return sum([f(X0 + i * INTEGRAL_WIDTH) * INTEGRAL_WIDTH for i in INTEGRAL_STEPS_RANGE])


def e(i: int):
    start = max(X0 + (i - 1) * STEP, X0)  # x(i-1)
    mid = X0 + i * STEP  # x(i)
    end = min(X0 + (i + 1) * STEP, X1)  # x(i+1)

    def f(x: float):
        if x == mid:
            return 1
        if start < x < mid:
            return (x - start) / (mid - start)
        if mid < x < end:
            return (end - x) / (end - mid)
        return 0

    return f


def de(i):
    start = max(X0 + (i - 1) * STEP, X0)  # x(i-1)
    mid = X0 + i * STEP  # x(i)
    end = min(X0 + (i + 1) * STEP, X1)  # x(i+1)

    def f(x):
        if start < x < mid:
            return 1 / (mid - start)
        if mid < x < end:
            return 1 / (mid - end)
        return 0

    return f


def b(i, j):
    def u(x): return e(i)(x)

    def v(x): return e(j)(x)

    def du(x): return de(i)(x)

    def dv(x): return de(j)(x)

    return -u(2) * v(2) + integral(du, dv) - integral(u, v)


def l(j):
    def v(x): return e(j)(x)

    return integral(sin, v)


if __name__ == '__main__':
    bs = np.array([[b(i, j) for i in I_RANGE] for j in I_RANGE])
    ls = np.array([l(j) for j in I_RANGE])

    # Macierzowo bs * us = ls
    us = np.linalg.solve(bs, ls)


    def u(x):
        es = [e(i)(x) for i in I_RANGE]
        # u(x) = suma po i ( u(i) * e(i)(x) )
        return sum(map(operator.mul, us, es))


    domain = np.linspace(X0, X1, 1000)
    values = np.array([u(x) for x in domain])
    plt.plot(domain, values)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid()
    plt.show()
