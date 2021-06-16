#!/usr/bin/env python3
# (c) 2021 Martin Kistler

import numpy as np


def Rückwärtsauflösen(Ab: np.array) -> np.array:
    n = Ab.shape[0]
    result = np.zeros(n, dtype=complex)
    right = Ab[:, -1]
    left = Ab[:, :-1]

    for i in range(n - 1, -1, -1):
        result[i] = (right[i] - sum(left[i][j] * result[j] for j in range(n))) / left[i][i]

    return result


def Tausche(Ab: np.array, i: int, j: int) -> None:
    Ab[[i, j]] = Ab[[j, i]]


def Addiere(Ab: np.array, a: float, i: int, j: int) -> None:
    Ab[j] += a * Ab[i]


def Skaliere(Ab: np.array, a: float, i: int) -> None:
    Ab[i] *= a


def Gauß_Elimination(Ab: np.array, Pivotisierung: bool = False, Äquilibrierung: bool = False):
    n = Ab.shape[0]
    det = 1
    Ab = Ab * 1.0

    if Äquilibrierung:
        for k in range(n):
            f = np.sum(np.abs(Ab[k]))
            if f == 0:
                return None, 0
            else:
                Skaliere(Ab, f**-1, k)
                det *= f

    for i in range(n):
        if Pivotisierung:
            ajis = [abs(Ab[j][i]) for j in range(i, n)]
            j = i + ajis.index(max(ajis))
            if i != j:
                Tausche(Ab, i, j)
                det *= -1
            if Ab[i, i] == 0:
                return None, 0

        else:
            if Ab[i][i] == 0:
                for j in range(i + 1, n):
                    if Ab[j][i] != 0:
                        Tausche(Ab, i, j)
                        det *= -1
                        break
                if Ab[i, i] == 0:
                    return None, 0

        for j in range(i + 1, n):
            Ab[j] = Ab[j] - (Ab[j][i] / Ab[i][i]) * Ab[i]

    for x in range(n):
        det *= Ab[x, x]

    return Rückwärtsauflösen(Ab), det


def LikeMatrix(links):
    N = len(links)
    L = np.zeros((N, N), dtype='float64')

    for i in range(N):
        L[links[i], i] = 1
        L[links[i], i] /= len(links[i])

    return L


def PageRank(L, d=0.85):
    N = L.shape[0]
    b = np.array([(1 - d)/N] * N)
    A = np.identity(N, dtype='float64') - d*L
    Ab = np.c_[A, b]
    return Gauß_Elimination(Ab)[0]
