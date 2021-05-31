#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Startpunkt für Numerikpraktikum P3

"""
@author: Nils Rosehr
"""

data = {1: [-2,   -2, 1.5, 2],   # a, xa, b, n   für f1
        2: [ 0,   -1,   3, 2],   # a, xa, b, n   für f2
        3: [-2, 0.01,   2, 2],   # a, xa, b, n   für f3
        4: [ 0,   -2,  30, 5]}   # a, xa, b, n   für f4

import numpy as np
import matplotlib.pyplot as plt

def Lösung_AWP(awp, a, xa):
    def x(t):
        return awp(t, a, xa)
    return np.vectorize(x)   # wie return x

def plot_Richtungsfeld(f, t_linspace=None, x_linspace=None, n=15):
    if t_linspace == None:
        t_linspace = (*plt.xlim(), n)
    if x_linspace == None:
        x_linspace = (*plt.ylim(), n)
    t, x = np.meshgrid(np.linspace(*t_linspace), np.linspace(*x_linspace))
    u, v = 0*t+1, f(t, x)
    plt.quiver(t, x, u, v, angles='xy')

def plot_Richtungsfeld_2D(f, t=0, x0_linspace=None, x1_linspace=None, n=15):
    if x0_linspace == None:
        x0_linspace = (*plt.xlim(), n)
    if x1_linspace == None:
        x1_linspace = (*plt.ylim(), n)
    x = x0, x1 = np.meshgrid(np.linspace(*x0_linspace), np.linspace(*x1_linspace))
    u, v = f(t, x)
    norm = np.sqrt(u**2 + v**2)
    plt.quiver(x0, x1, u/norm, v/norm, angles='xy', pivot='mid', width=0.004, color='lightgrey')
