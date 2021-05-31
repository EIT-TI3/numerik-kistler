#!/usr/bin/env python3
# (c) 2021 Martin Kistler

import numpy as np

from lib3 import plot, Runge_Kutta, plot_2D
from vorgabe3 import data


def f1(t, x):
    return t**2


def f2(t, x):
    return 2*x


def f3(t, x):
    return -2*t*x


def f4(t, x):
    return np.sin(t)*np.exp(x)


def awp1(t, a, xa):
    c = xa - a**3/3
    return c + t**3/3


def awp2(t, a, xa):
    c = xa / np.e**(2*a)
    return c*np.e**(2*t)


def awp3(t, a, xa):
    c = xa / np.e**(-a**2)
    return c * np.e**(-t**2)


def awp4(t, a, xa):
    c = np.e**(-xa) - np.cos(a)
    return -np.log(c + np.cos(t))


data[1][3] = 5
plot(f1, awp1, data[1])

data[2][3] = 10
plot(f2, awp2, data[2])

data[3][3] = 10
plot(f3, awp3, data[3])

data[4][3] = 20
plot(f4, awp4, data[4])


def f5(t, x):
    assert len(t) == 2 and len(x) == 2
    c = np.log(2)/(2*np.pi)
    return np.array([-c*x[0] - x[1], x[0] - c*x[1]])


plot_2D(f5, (np.array([0, 0]), np.array([4, 0]), 8*np.pi, 100), Runge_Kutta)
Endpunkt = (0.25, 0)


def f6(t, x):
    assert len(t) == 2 and len(x) == 2
    return np.array([x[1], -np.sin(x[0])])


for deg in 5, 30, 177:
    rad = deg*2*np.pi/360
    plot_2D(f6, (np.array([0, 0]), np.array([rad, 0]), 8*np.pi, 1000), Runge_Kutta)

Perioden = (6.286, 6.392, 20.119)
