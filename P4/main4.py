#!/usr/bin/env python3
# (c) 2021 Martin Kistler

import pickle
import numpy as np

from routes import routes_from_to, airports, routes_to_from
from lib4 import Gauß_Elimination, PageRank, LikeMatrix

with open('Ab_x_list.data', 'rb') as file:
    Ab_x_list = pickle.load(file)

results = np.zeros((len(Ab_x_list), 4))

for i, (Ab, xi) in enumerate(Ab_x_list):
    for j, (piv, eq) in enumerate(((a, b) for a in [False, True] for b in [False, True])):
        x = Gauß_Elimination(Ab, Pivotisierung=piv, Äquilibrierung=eq)[0]
        results[i][j] = np.sum(np.abs(x - xi)**2)

with np.printoptions(precision=3):
    print(results)


targets = PageRank(LikeMatrix(routes_from_to))
starts = PageRank(LikeMatrix(routes_to_from))

BesterAbflug = sorted(list(zip(targets, airports)), key=lambda item: item[0], reverse=True)[:15]
BesteAnkunft = sorted(list(zip(starts, airports)), key=lambda item: item[0], reverse=True)[:15]
