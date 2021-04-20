#!/usr/bin/env python3
# Copyright (c) 2021 Martin Kistler

from numpy import e, pi
from math import copysign


def Rh(f, a, b, n):
    h = (b - a) / n
    x = tuple(a + i * h for i in range(n + 1))
    return h * sum(f(x[i] + h / 2) for i in range(n))


def Th(f, a, b, n):
    h = (b - a) / n
    x = tuple(a + i * h for i in range(n + 1))
    return h * (((f(a) + f(b)) / 2) + sum(f(x[i]) for i in range(1, n)))


def Sh(f, a, b, n):
    h = (b - a) / n
    x = tuple(a + i * h for i in range(n + 1))
    return (h / 6) * (f(a) + f(b) + 2 * sum(f(x[i]) for i in range(1, n)) + 4 * sum(f(x[i] + h / 2) for i in range(n)))


def R(f, a, b):
    return f((a + b) / 2) * (b - a)


def T(f, a, b):
    return (f(a) + f(b)) / 2 * (b - a)


def S(f, a, b):
    return (b - a) / 6 * (f(a) + 4*f((a + b) / 2) + f(b))


def phi(x):
    return e**(-x**2/2)/(2*pi)**0.5


def gauss_simpson_error(a, b, h):
    return (h**4/2880) * (b - a) * 3/(2*pi)**0.5


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
