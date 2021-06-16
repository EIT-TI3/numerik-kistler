# -*- coding: utf-8 -*-

# Nils Rosehr, 2019-05-30

import pickle
Ab_x_list = pickle.load(open('Ab_x_list.data', 'rb'))

print('Anzahl Matrizen und Ergebnisvektoren:', len(Ab_x_list))
print('Format der Matrizen:', Ab_x_list[0][0].shape)
print('Erste Matrix:')
print(Ab_x_list[0][0])
print('Erster Ergebnisvektor:')
print(Ab_x_list[0][1])

Tiere = ['Katze', 'Hund', 'Maus']
Maße = [2, 3, 1]

print()
print('Zwei Listen:')
print('Tiere =', Tiere)
print('Maße =', Maße)
print('Einfaches Sortieren:')
print(sorted(Tiere))
print(sorted(Maße))
print('Sortieren der Tiere nach Maß:')
print(sorted(zip(Maße, Tiere)))
print('Sortieren der Tiere nach absteigendem Maß:')
print(sorted(zip(Maße, Tiere), reverse=True))

from routes import airports, routes_from_to, routes_to_from

print()
print('Einige Flughäfen:')
print(airports[600:605])
print('Einige Flug-Verbindungen:')
print(routes_from_to[600:605])
print(airports[50])
print(routes_to_from[50])
print(airports[773])
for i in routes_from_to[773]:
    print('   --> {}'.format(airports[i]))
