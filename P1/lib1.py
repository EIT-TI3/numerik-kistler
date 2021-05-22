#!/usr/bin/env python3
# (c) 2021 Martin Kistler


from numpy import e, pi
from math import copysign


def get_x(a, b, n):
    h = (b - a) / n
    for i in range(n):
        yield a + i * h


def Rh(f, a, b, n):
    h = (b - a) / n
    return h * sum(f(x + h / 2) for x in get_x(a, b, n))


def Th(f, a, b, n):
    h = (b - a) / n
    return h * (((f(a) + f(b)) / 2) + sum(f(x) for x in get_x(a, b, n)))


def Sh(f, a, b, n):
    h = (b - a) / n
    return (h / 6) * (f(a) + f(b)
                      + 2 * sum(f(x) for x in get_x(a, b, n))
                      + 4 * sum(f(x + h / 2) for x in get_x(a, b, n)))


def R(f, a, b):
    return f((a + b) / 2) * (b - a)


def T(f, a, b):
    return (f(a) + f(b)) / 2 * (b - a)


def S(f, a, b):
    return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))


def phi(x):
    return e ** (-x ** 2 / 2) / (2 * pi) ** 0.5


def gauss_simpson_error(a, b, h):
    return (h ** 4 / 2880) * (b - a) * 3 / (2 * pi) ** 0.5


def Gauß(a, b, xabsmax=100):
    a = copysign(1, a) * xabsmax if abs(a) > xabsmax else a
    b = copysign(1, b) * xabsmax if abs(b) > xabsmax else b

    error = 1
    h = 1
    while abs(error) > 1e-11:
        error = gauss_simpson_error(a, b, h)
        h /= 2

    n = int(abs(a - b) / h) + 1

    return Sh(phi, a, b, n)
