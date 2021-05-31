#!/usr/bin/env python3
# (c) 2021 Martin Kistler

import matplotlib.pyplot as plt
import numpy as np

from vorgabe3 import plot_Richtungsfeld, Lösung_AWP


def Euler(f, a, xa, b, n):
    result = [xa]
    h = (b-a)/n

    for i in range(n):
        xi = result[-1]
        t = a + i*h

        result.append(xi + h*f(t, xi))

    return result


def Mittelpunktsregel(f, a, xa, b, n):
    result = [xa]
    h = (b - a) / n

    for i in range(n):
        xi = result[-1]
        t = a + i * h

        k1 = f(t, xi)
        k2 = f(t + h / 2, xi + (h / 2) * k1)

        result.append(xi + h*k2)

    return result


def Heun(f, a, xa, b, n):
    result = [xa]
    h = (b - a) / n
    for i in range(n):
        xi = result[-1]
        t = a + i * h

        k1 = f(t, xi)
        k2 = f(t + h, xi + h * k1)

        result.append(xi + h*(k1+k2)/2)

    return result


def Runge_Kutta(f, a, xa, b, n):
    result = [xa]
    h = (b - a) / n
    for i in range(n):
        xi = result[-1]
        t = a + i * h

        k1 = f(t, xi)
        k2 = f(t + h / 2, xi + (h / 2) * k1)
        k3 = f(t + h / 2, xi + (h / 2) * k2)
        k4 = f(t + h, xi + h * k3)

        result.append(xi + h*(k1 + 2*k2 + 2*k3 + k4)/6)
    return result


def plot(f, awp, datai):
    a, xa, b, n = datai
    x = np.linspace(a, b, n)
    plt.plot(x, Lösung_AWP(awp, a, xa)(x), label='exakt')

    for method in Euler, Mittelpunktsregel, Heun, Runge_Kutta:
        plt.plot(x, method(f, a, xa, b, n)[:-1], label=method.__name__)
    plot_Richtungsfeld(f)
    plt.legend(loc="lower right")
    plt.show()

    for method in Euler, Mittelpunktsregel, Heun, Runge_Kutta:
        exact = Lösung_AWP(awp, a, xa)(b)
        ns = np.arange(n, 4*n)
        error = [abs(exact - method(f, a, xa, b, n)[-1]) for n in ns]
        plt.plot(ns, error, label=method.__name__)
    plt.legend(loc="lower right")
    plt.show()


def plot_2D(f, datai, solver):
    a, xa, b, n = datai
    x = np.linspace(a, b, n)
    data = solver(f, a, xa, b, n)

    plt.plot(x, data[:-1])
    plt.show()
    plt.plot([e[0] for e in data], [e[1] for e in data])
    plt.show()
