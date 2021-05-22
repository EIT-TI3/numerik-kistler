# -*- coding: utf-8 -*-
"""
@author: Nils Rosehr
"""

import numpy as np
from lib2 import Bisektion, Regula_falsi, Sekantenverfahren, Newton_Verfahren


### Definition der Funktionen und Parameter

data = {}   # leeres Dictionary

def f1(x):
    return x**2 - 2
def df1(x):
    return 2*x
data[1] = f1, df1, 1, 10, 20, np.sqrt(2)   # f, df, a, b, n, true_result

def f2(x):
    return np.cos(x)
def df2(x):
    return -np.sin(x)
data[2] = f2, df2, 1, 2, 20, np.pi/2   # f, df, a, b, n, true_result

def f3(x):       # damit man auch den Namen 'f3' von f3=f2 mit f3.__name__ abrufen kann
    return f2(x)
data[3] = f3, df2, 0.01, 10, 20, 2.5 * np.pi   # f, df, a, b, n, true_result

def f4(x):
    return np.sign(x) * np.sqrt(abs(x))
def df4(x):
    return 0.5 / np.sqrt(abs(x))
data[4] = f4, df4, -1/4, 1/2, 20, 0   # f, df, a, b, n, true_result

def f5(x):
    return (x - 2)**5 * (x**2 + 1)
def df5(x):
    return 5*(x - 2)**4 * (x**2 + 1) + (x - 2)**5 * 2*x
data[5] = f5, df5, 1, 3, 20, 2   # f, df, a, b, n, true_result


## Methoden zur Ausgabe

def Tabelle(f, df, a, b, n, delta, ordnung=1):
    T = np.empty((n, 4)) * np.nan
    x = Bisektion(f, a, b, n)
    y = Regula_falsi(f, a, b, n)
    u = Sekantenverfahren(f, a, b, delta, n)
    if ordnung == 1:
        v = Newton_Verfahren(f, df, a, delta, n)
    else:
        v = Newton_Verfahren(f, df, a, delta, n, ordnung=ordnung)
    for i in range(n):
        T[i,:] = [a[i] if i < len(a) else np.nan for a in (x, y, u, v)]
    return T

def print_Tabelle(T, Spaltennamen=('Bisektion', 'regula falsi', 'Sekantenverfahren', 'Newton-Verfahren')):
    Anzahl = len(Spaltennamen)
    z, s = T.shape
    assert s == Anzahl, 'Die Matrix T muss {} Spalten haben.'.format(Anzahl)
    print('  ', (Anzahl * ' {:20}').format(*Spaltennamen))
    for i in range(z):
        row = T[i,:]
        print(('{:2}' + Anzahl * ' {:20.17f}').format(i, *row))
    print()
