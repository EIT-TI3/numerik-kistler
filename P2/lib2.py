#!/usr/bin/env python3
# (c) 2021 Martin Kistler


import numpy as np
import random as rnd


def Bisektion(f, a, b, n):
    result = [a, b]
    for _ in range(n - 2):
        m = (a + b) / 2
        result.append(m)

        if f(m) == 0 or b == a:
            break
        elif f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return result


def Regula_falsi(f, a, b, n):
    result = [a, b]
    for _ in range(n - 2):
        m = a - ((b - a) / (f(b) - f(a))) * f(a)
        result.append(m)

        if f(m) == 0 or b == a:
            break
        elif f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return result


def Sekantenverfahren(f, a, b, delta, n):
    result = [a, b]
    for _ in range(n - 2):
        if f(a) == f(b) or abs(a - b) <= delta:
            break
        else:
            tmp = b
            b = b - ((b - a) / (f(b) - f(a))) * f(b)
            a = tmp
            result.append(b)
    return result


def Newton_Verfahren(f, df, a, delta, n):
    result = [a]
    for _ in range(n - 1):
        if df(a) == 0:
            break

        a = a - f(a) / df(a)
        result.append(a)

        if abs(a - result[-2]) <= delta:
            break

    return result


def Hornerschema(a, x0):
    n = len(a)
    result = np.zeros(n, dtype=complex)
    z = np.zeros(n + 1, dtype=complex)

    for i in range(n):
        result[i] = a[i] + z[i]
        z[i + 1] = result[i] * x0

    return result


def Polynom_Funktion(a):
    n = len(a)

    def polynom(x):
        return sum(coeff * x ** (n - i - 1) for i, coeff in enumerate(a))

    return polynom


def Diff_Polynom(a):
    return [i * coeff for i, coeff in enumerate(a[::-1]) if i != 0][::-1]


def Nullstellen_Polynom(a, delta, k):
    n = len(a) - 1

    for _ in range(n):
        f = Polynom_Funktion(a)
        df = Polynom_Funktion(Diff_Polynom(a))

        while True:
            x = rnd.random()*1j + rnd.random()
            N = Newton_Verfahren(f, df, x, delta, k)
            if abs(N[-1] - N[-2]) <= delta:
                z = N[-1]
                break
        yield z

        a = Hornerschema(a, z)[:-1]
