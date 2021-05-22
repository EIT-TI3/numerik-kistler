#!/usr/bin/env python3
# (c) 2021 Martin Kistler

import numpy as np
import matplotlib.pyplot as plt
from lib1 import Sh


def f1(x):
    return x**3


def f2(x):
    return x**4 - 2*x**3 + 3


def f3(x):
    return 3*x**0.2


def f4(x):
    return np.sin(x)


# contains f: (a, b, Integral(f, a, b)) as Key-Value pairs
functions = {f1: (-1, 2, 3.75), f2: (0, 2, 4.4), f3: (0, 32, 160), f4: (0, 7*np.pi, 2)}


def plot_functions(functions):
    x = np.linspace(-10, 10, 1000)
    for f in functions:
        plt.plot(x, f(x))
        plt.show()


def true_result(f):
    return functions[f][2]


Anzahlen = []
for f in functions:
    e = 1
    c = 0
    while abs(e) > 0.1:
        c += 1
        e = functions[f][2] - Sh(f, functions[f][0], functions[f][1], c)
    Anzahlen.append(c)
