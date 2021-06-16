#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:11:18 2018

@author: Nils Rosehr
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import unittest
import pickle
import pprint
#import os
import sys
import re
import hashlib
import base64
import collections
import contextlib


P_number = 4
exec('import main{}'.format(P_number))
exec('main = main{}'.format(P_number))
exec('import lib{}'.format(P_number))
exec('lib = lib{}'.format(P_number))
test_file = 'test{}.py'.format(P_number)
vorgabe_file = 'vorgabe{}.py'.format(P_number)
data_file = 'test{}.data'.format(P_number)
lib_file = 'lib{}.py'.format(P_number)
info_file = 'info{}.md'.format(P_number)
_subtest_msg_sentinel = object() 


class ERROR(Exception):
    pass


class Test_Numerik(unittest.TestCase):
    
    def assertLessEqualMultiple(self, result, true_value, multiple=2):
        self.assertLessEqual(result, true_value * multiple)
    
    def assertBothIterableOfSameLength(self, result, true_value):
        if not hasattr(result, '__iter__'):
            self.fail('"result" should be iterable: result={}; true_value={}'.format(result, true_value))
        if not hasattr(true_value, '__iter__'):
            self.fail('"true value" should be iterable: result={}; true_value={}'.format(result, true_value))
        if not hasattr(result, '__len__'):
            self.fail('"result" does not have a length. This should not happen: result={}; true_value={}'.format(result, true_value))
        if not hasattr(true_value, '__len__'):
            self.fail('"True value" does not have a length. This should not happen: result={}; true_value={}'.format(result, true_value))
        self.assertEqual(len(result), len(true_value), msg='Lengths of "result" and "true value" are not equal: result={}; true_value={}'.format(result, true_value))
    
    def assertBothIterableOrBothNotIterable(self, result, true_value):
        """Check if both arguments (named result and true_value) are iterable
        of if both are not iterable. Otherwise raise assertion error.
        Return True if both arguments are iterable and False otherwise."""
        if hasattr(true_value, '__iter__'):
            self.assertBothIterableOfSameLength(result, true_value)
            return True
        else:
            if hasattr(result, '__iter__'):
                self.fail('"result" is iterable", but "true value" is not.')
            return False
        
    def assertAlmostEqualPlaces(self, result, true_value, places=7):
        # check if both arguments are iterable or if both are not iterable (otherwise raise assertion error)
        if self.assertBothIterableOrBothNotIterable(result, true_value):
            for r, t in zip(result, true_value):
                self.assertAlmostEqualPlaces(r, t, places=places)
        else:
            self.assertAlmostEqual(result, true_value, places=places)
        
    def assertAlmostEqualRelative(self, result, true_value, relative=10**-7, **kwargs):
        # check if both arguments are iterable or if both are not iterable (otherwise raise assertion error)
        if self.assertBothIterableOrBothNotIterable(result, true_value):
            for result, true_value in zip(result, true_value):
                self.assertAlmostEqualRelative(result, true_value, relative=relative, **kwargs)
        else:
            if result == None or true_value == None:
                self.assertEqual(true_value, result, **kwargs)
            else:
                delta = max(abs(true_value), abs(result)) * relative
                self.assertAlmostEqual(result, true_value, delta=delta, **kwargs)
        
    def assertNormAlmostZero(self, result, delta=10**-7):
        self.assertAlmostEqual(np.sqrt(np.sum(np.abs(result)**2)), 0.0, delta=delta)
        
    def assertAlmostEqualRelativeAbs(self, result, true_value, relative=10**-7, **kwargs):
        # check if both arguments are iterable or if both are not iterable (otherwise raise assertion error)
        if self.assertBothIterableOrBothNotIterable(result, true_value):
            for result, true_value in zip(result, true_value):
                self.assertAlmostEqualRelativeAbs(result, true_value, relative=relative, **kwargs)
        else:
            delta = max(abs(true_value) + abs(result), 1) * relative
            self.assertAlmostEqual(result, true_value, delta=delta, **kwargs)
        
    def assertAlmostEqualRelativePadding(self, result, true_value, relative=10**-7):
        if not hasattr(result, '__iter__'):
            self.fail('"result" should be iterable.')
        if not hasattr(true_value, '__iter__'):
            self.fail('"true value should be iterable."')
        for l in (result, true_value):
            while len(l) > 0 and (np.isnan(l[-1]) or np.isinf(l[-1])):
                l.pop()
        while len(result) != len(true_value):
            if len(result) < len(true_value):
                result.append(result[-1])
            else:
                true_value.append(true_value[-1])
        self.assertAlmostEqualRelative(result, true_value, relative=relative)
#        for result, true_value in zip(result, true_value):
#            delta = (abs(result) + abs(true_value)) * relative
#            self.assertAlmostEqual(result, true_value, delta=delta)
        
    def assertAlmostEqualUnorderedListRelative(self, result, true_value, relative=10**-7):
        self.assertBothIterableOfSameLength(result, true_value)
        for (a, b) in ((result, true_value), (true_value, result)):
            for r in a:
                closest, dist = None, np.inf
                for t in b:
                    d = abs(r - t)
                    if d < dist:
                        closest, dist = t, d
                delta = (abs(r) + abs(closest)) * relative
                self.assertAlmostEqual(r, closest, delta=delta)
        
    def assertAlmostEqualSquareSum(self, result, true_value, delta=10**-7):
        self.assertBothIterableOfSameLength(result, true_value)
        error = sum(np.abs(np.array(result) - np.array(true_value))**2)
        self.assertLessEqual(error, delta)

    def runner(self, method, args, assertMethod,
               post=lambda x: x, true_value=None, marker=None, *vargs, **kwargs):
        if marker:
            print(marker, 'Vor Aufruf, Argumente:', args)
        result = post(method(*args))
            
#        with warnings.catch_warnings():
#            warnings.filterwarnings('error')
#            try:
#                result = post(method(*args))
#            except:
#                if results.collecting:
#                    print('WARNUNG: Ersatzwert 17..17 genutzt wegen Fehler:', sys.exc_info())
#                result = int(17 * '17')

        if results.collecting:
            if not true_value is None:
                true_val = true_value
            else:
                true_val = result
            results.set(true_val)
        else:
            true_val = results.get()
        if marker:
            print(marker, 'Nach Aufruf, Ergebnisse:', result, true_val)
        assertMethod(result, true_val, *vargs, **kwargs)
        
    def setUp(self):
        results.start_new_test(self._testMethodName)
        
    def tearDown(self):
        results.finish_test()

    def subTest_orig(self, *args, **kwargs):
        return unittest.TestCase.subTest(self, *args, **kwargs)

    # Redefine subTest:
    # Code is copied from unittest.case.py python version 3.6
    # and ammended by code executed before and after block (2 lines added).
    # In three places "unittest.case." had to be inserted.
    @contextlib.contextmanager
    def subTest(self, msg=_subtest_msg_sentinel, name='', **params):
        """Return a context manager that will return the enclosed block
        of code in a subtest identified by the optional message and
        keyword parameters.  A failure in the subtest marks the test
        case as failed but resumes execution at the end of the enclosed
        block, allowing further test code to be executed.
        """
        results.start_new_subtest(name)   # executed before block
        if not self._outcome.result_supports_subtests:
            yield
            return
        parent = self._subtest
        if parent is None:
            params_map = collections.ChainMap(params)
        else:
            params_map = parent.params.new_child(params)
        self._subtest = unittest.case._SubTest(self, msg, params_map)
        try:
            with self._outcome.testPartExecutor(self._subtest, isTest=True):
                yield
            if not self._outcome.success:
                result = self._outcome.result
                if result is not None and result.failfast:
                    raise unittest.case._ShouldStop
            elif self._outcome.expectedFailure:
                # If the test is expecting a failure, we really want to
                # stop now and register the expected failure.
                raise unittest.case._ShouldStop
        finally:
            results.finish_subtest()   # executed after block
            self._subtest = parent
    
    ### set up registration of object names        
    object_names_dict_by_id = dict()
    objects_dict_by_name = dict()
        
    @classmethod
    def set_name(cls, main_object, name, use_main_object_itself=False):
        if use_main_object_itself:
            obj = main_object
        else:
            obj = getattr(main_object, name)
        cls.object_names_dict_by_id[id(obj)] = name
        assert not name in cls.objects_dict_by_name, 'The name is not unique.'
        cls.objects_dict_by_name[name] = obj
        
    @classmethod
    def get_name(cls, obj):
        assert id(obj) in cls.object_names_dict_by_id, 'Object name not stored.'
        return cls.object_names_dict_by_id[id(obj)]
    
    @classmethod
    def get_object(cls, name):
        assert name in cls.objects_dict_by_name, 'Name does not exist.'
        return cls.objects_dict_by_name[name]
        

class Results:
    
    def __init__(self, data_file, collect_results=False):
        self.filename = data_file
        self.collecting = collect_results
        self.test = tuple()
        if self.collecting:
            self.data = {}   # dict of tests and subtests
            self.index = None   # indeces are not used in collecting mode: we can append
        else:
            self.data = pickle.load(open(self.filename, 'rb'))
            # we keep current indeces for all subtest, so that we can resume a subtest with the same name
            self.index = {k: 0 for k in self.data}   # set indeces for all subtests to 0
            
    def testname(self):
        return ':'.join(self.test)
            
    def __str__(self, concise=True):
        if concise:
            return str({k: self.data[k][:3] + ['...'] for k in self.data})
        else:
            return str(self.data)
    
    def set(self, result):
        """Store new result value."""
        assert self.collecting == True, 'Can only set value in collecting mode.'
        self.data[self.test].append(result)
    
    def get(self):
        """Get next result value."""
        assert self.collecting == False, 'Cannot get value in collecting mode.'
#        value = self.data[self.testname()][self.j]
#        try:
#            values = self.data[self.testname()]
#        except:
#            raise Error("Could not access (sub)-test '{}'.".format(self.testname()))
        Error_if(not self.test in self.data,
                 "Could not access (sub)-test '{}'.".format(self.testname()))
        values = self.data[self.test]
#        try:
#            value = values[self.j]
#        except:
#            raise Error("Could not get index {} of (sub)-test '{}'.".format(self.j, self.testname()),
#                        "The (sub)-test has {} values:".format(len(values)),
#                        values)
        Error_if(self.index[self.test] >= len(values),
                 "Could not get index {} of (sub)-test '{}'.".format(self.index[self.test], self.testname()),
                 "The (sub)-test has {} values:".format(len(values)),
                 values)
        value = values[self.index[self.test]]
        self.index[self.test] += 1
        return value
        
    def start_new_test(self, name):
        self.finish_test()
        if self.collecting:
            Error_if(name in self.data,
                     "Test name '{}' is not unique.".format(name))
        self.start_new_subtest(name)
            
    def start_new_subtest(self, name):
#        assert not ':' in name, "Character ':' not allowed in (sub)-test name."
        self.test += (name, ) 
        if self.collecting:
            if not self.test in self.data:
                self.data[self.test] = []
        else:
            Error_if(not self.test in self.data,
                     "(Sub)-Test name '{}' does not exist in results.".format(self.testname()))
            
    def finish_test(self):
        self.test = tuple()
            
    def finish_subtest(self):
        self.test = self.test[:-1]
            
    def dump(self):
        """Dump string which can be evaluated to store found results
        from collecting phase into object instance 'results'."""
        if self.collecting:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.data, f)
            with open(self.filename + '.pprint', 'w') as f:
                self.show(stream=f, concise=False)
            print()
            print("Results written to file '{}'.".format(self.filename))
            print()
            print('Results in concise form:')
            self.show()
            
    def show(self, stream=None, concise=True):
        joined_keys_dict = d = {'-'.join(k): self.data[k] for k in sorted(self.data)}
        if concise:
            shortened = {k: d[k] if len(d[k]) <= 5 else d[k][:5] + ['...'] for k in d}
            for k in shortened:
                for i, l in enumerate(shortened[k]):
                    try:
                        if len(l) > 5:
                            shortened[k][i] = l[:5] + ['...']
                    except:
                        pass
            ppstr = pprint.pformat(shortened, compact=True, depth=3)
            print(ppstr.replace("'...'", "..."), file=stream)
        else:
            pprint.pprint(joined_keys_dict, stream)
 



class Test_Praktikum(Test_Numerik):
    
    try:
        true_values_data = main.true_values
    except (NameError, AttributeError):
        true_values_data = dict()
        
    def true_values(self, key, default='no_value_provided_by_main'):
        return self.true_values_data.get(key, default)
    
    @classmethod
    def random_matrix(cls, n, m, domain='reell'):
        if domain == 'reell':
            return np.random.random(n*m).reshape(n,m)
        else:
            return cls.random_matrix(n, m) + cls.random_matrix(n, m) * 1j

    Ab_list = [np.array([1, 5, 2,  49,  
                         0, 1, 7, 121,  
                         1, 2, 9, 162]).reshape(3, 4),
               np.array([1, 0, 1,  
                         0, 0, 1]).reshape(2, 3),
               np.array([0, 2, 1, 3, 7,
                         3, 1, 7, 1, 5,
                         3, -3, 5, 2, -2,
                         6, 0, 18, 0, -6]).reshape(4, 5),
               np.array([0.00031, 1, -3,
                         1, 1, -7]).reshape(2,3),
               np.array([1+1j,    5, 2-3j,  52+44j,
                         0, 1+5j,    7, 121+17j,
                         1-7j, 2+8j,    9, 162+10j]).reshape(3, 4)]
    
    Ab_x_list = pickle.load(open('Ab_x_list.data', 'rb'))
    links1 = [[1, 2], [0, 2, 3], [0, 1, 3], [0, 2]]
    links2 = [[4, 3], [3, 4], [3, 0], [2], [2, 3, 0]]
    h, d = 1/2, 1/3
    L1 = np.array([[0, d, d, h],
                   [h, 0, d, 0],
                   [h, d, 0, h],
                   [0, d, d, 0]])
    L2 = np.array([[0, 0, h, 0, d],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 1, d],
                   [h, h, h, 0, d],
                   [h, h, 0, 0, 0]]) 

    @classmethod
    def setUpClass(cls):
        ### register names for objects
        ### in most cases cls.get_name(x) will yield the same name as x.__name__
        ### for examples in the most common case when a method is defined with def x(...).
        ### examples where x.__name__ does not work are lambda definitions or assignments
        ### x = y, where y is a previously defined method, and, of course, all the simple
        ### variables like x = 17, where x.__name__ is not even defined.
        cls.Abi = ['Ab{}'.format(i+1) for i in range(len(cls.Ab_list))]
        for i, Ab in enumerate(cls.Abi):
            cls.set_name(cls.Ab_list[i], Ab, use_main_object_itself=True)
#        cls.Antworten = ['Antwort1', 'Antwort2', 'Antwort3']
#        cls.Antworten_lower = [200, 100, 200]
#        for Antwort in cls.Antworten:
#            cls.set_name(main, Antwort)
#        cls.BesteSeiten = 'BesteSeiten'
#        cls.set_name(main, cls.BesteSeiten)

    def test_Rückwärtsauflösen(self):
        np.random.seed(311933)
        for domain in 'reell', 'komplex':
            for n in (3, 7, 20, 60):
                A = np.triu(self.random_matrix(n, n, domain))
                for i in range(n):
                    A[i, i] += 1.0
                x = np.random.random(n)
                if domain == 'komplex':
                    x = x + np.random.random(n) * 1j
                b = A.dot(x.reshape(-1, 1))
                Ab = np.hstack((A, b))
                if n <= 10:
                    message = 'Rückwärtsauflösen nicht korrekt für {}e Matrix Ab=\n{}'.format(domain, Ab)
                else:
                    message = 'Rückwärtsauflösen nicht korrekt für {}e Matrix der Dimension {}.'.format(domain, n)
                with self.subTest(msg=message):
                    sol = lib.Rückwärtsauflösen(Ab).reshape(n)
                    self.runner(lambda s: s, (sol, ), self.assertAlmostEqualRelative, true_value=x, msg='Lösung zu ungenau.', relative=10**-10)
                    # self.assertNormAlmostZero(lib.Rückwärtsauflösen(Ab).reshape(-1, 1) - x)

    def test_Gauß_Elimination(self):
        ### test some explicit matrices
        for name in self.Abi:
            Ab = self.get_object(name)
            with self.subTest(msg='Problem bei folgender Matrix:\n{}'.format(Ab), name=name):
                self.runner(lib.Gauß_Elimination, (Ab, ), self.assertAlmostEqualRelative)
        ### produce some matrices with approximately two thirds of zeros
        Ab_x_list_thin = []
        np.random.seed(71234567)
        for domain in 'reell', 'komplex':
            for n in (3, 7, 10, 20):
                A = self.random_matrix(n, n, domain)
                x = np.random.random(n)   # x is column vector
                if domain == 'komplex':
                    x = x + np.random.random(n) * 1j
                A = A * (self.random_matrix(n, n, domain) <= 1/3)   # set two thirds to zero
                b = A.dot(x.reshape(-1, 1))
                Ab = np.hstack((A, b))
                Ab_x_list_thin.append((Ab, x))
        ### test Gauß
        for Piv in ['X', 'Piv']:
            for Äq in ['X', 'Äq']:
                name = Piv + '-' + Äq
                err = {'X-X': 10**-6, 'X-Äq': 10**-6, 'Piv-X': 10**-9, 'Piv-Äq': 10**-10}
                with self.subTest(name=name):
                    for i, (Ab, x) in enumerate(self.Ab_x_list + Ab_x_list_thin):
                        with self.subTest(msg='Problem bei Verfahren "{}" mit {}-ter Matrix mit Einträgen:\n{}'.format(name, i, Ab), name=str(i+1)):
                            # self.runner(lib.Gauß_Elimination, (Ab, Piv=='Piv', Äq=='Äq'), self.assertAlmostEqualRelative, relative=10**-10)
                            sol, det = lib.Gauß_Elimination(Ab, Piv=='Piv', Äq=='Äq')
                            d = np.linalg.det(Ab[:, :-1])
                            if sol is None or det == 0:   # correct x, d if solution is not unique
                                x, d = None, 0
                            self.runner(lambda s: s, (sol, ), self.assertAlmostEqualRelative, true_value=x, msg='Lösung zu ungenau.', relative=err[name])
                            self.runner(lambda d: d, (det, ), self.assertAlmostEqualRelative, true_value=d, msg='Determinante zu ungenau.', relative=10**-9)
                            # if d != 0:
                            #     self.assertNotEqual(det, d, msg='Bitte nicht "numpy.linalg" benutzen.')

    def test_LikeMatrix(self):
        for links in self.links1, self.links2:
            with self.subTest(msg="Die LikeMatrix zu zu den Links {} ist nicht korrekt.".format(links)):
                self.runner(lib.LikeMatrix, (links, ), self.assertAlmostEqualRelative)

    def test_PageRank(self):
        for L in self.L1, self.L2:
            with self.subTest(msg="Der PageRank zur LikeMatrix {} ist nicht korrekt.".format(L)):
                self.runner(lib.PageRank, (L, ), self.assertAlmostEqualRelative)
      
    def test_BesteFlughäfen(self):
        Länge = 25
        BesterAbflug = main.BesterAbflug
        BesteAnkunft = main.BesteAnkunft

        with self.subTest(msg="Liste BesterAbflug hat nicht die richtige Länge.", name='Anzahl'):
            self.runner(lambda x: x, (len(BesterAbflug), ), self.assertEqual)
        for i, (rank, page) in enumerate(BesterAbflug[:Länge]):
            with self.subTest(msg='Der {}-te Flughafen ist nicht korrekt.'.format(i+1), name='p{}'.format(i)):
                self.runner(lambda x: x, (page, ), self.assertEqual, post=scramble)
            with self.subTest(msg='Der PageRank des {}-ten Flughafens ist nicht korrekt.'.format(i+1), name='r{}'.format(i)):
                self.runner(lambda x: x, (rank, ), self.assertEqual, post=scramble_float())

        with self.subTest(msg="Liste BesteAnkunft hat nicht die richtige Länge.", name='Anzahl'):
            self.runner(lambda x: x, (len(BesteAnkunft), ), self.assertEqual)
        for i, (rank, page) in enumerate(BesteAnkunft[:Länge]):
            with self.subTest(msg='Der {}-te Flughafen ist nicht korrekt.'.format(i+1), name='p{}'.format(i)):
                self.runner(lambda x: x, (page, ), self.assertEqual, post=scramble)
            with self.subTest(msg='Der PageRank des {}-ten Flughafens ist nicht korrekt.'.format(i+1), name='r{}'.format(i)):
                self.runner(lambda x: x, (rank, ), self.assertEqual, post=scramble_float())
    
    def test_Dateikonsistenz(self):
        for file in [test_file, vorgabe_file, 'Ab_x_list.data', 'routes.py']:
            with self.subTest(msg='Datei {} ist verändert worden oder nicht lesbar.'.format(file), name=file):
                self.runner(Hash_file, (file, ), self.assertEqual)
            
    def test_info(self):
        output = '''
        
   Die Datei "{}" muss korrekt angegeben werden.

   FEHLER:

   {{}}
'''.format(info_file)
        # Info file must be readable
        try:
            with open(info_file, 'rb') as f:
                info_byte = f.read()
        except:
#            self.fail(output.format('Info-Datei "{}" konnte nicht gelesen werden.'.format(info_file)))
            Error(output.format('Info-Datei "{}" konnte nicht gelesen werden.'.format(info_file)))
        try:
            f.close()   # On some systems the above with statement does not close file
        except:
            pass
        # Info file must be utf-8
        try:
            info = str(info_byte, 'utf-8')
        except:
            Error(output.format('''Die Info-Datei ist nicht "UTF-8"-kodiert.
Editieren Sie die vorgegebene Datei in "Spyder" und
überprüfen Sie dort, dass rechts unten "Encoding: UTF-8" steht.
'''))
        # Info file must be splittable into headings and data
        try:
            info_list = re.split('\s*##+\s*', info)
            info_dict = {}
            for entry in info_list:
                data = re.split('\n+', entry, maxsplit=1) + ['']
                info_dict[data[0]] = data[1]
        except:
            Error(output.format('Die Info-Datei hat falsches Format, Überschriften können nicht zugeordnet werden.'))
        # info file must contain correct headings
        for key in ('Team', 'Teilnehmer 1', 'Teilnehmer 2', 'Email 1', 'Email 2', 'Quellen', 'Bemerkungen', 'Bestätigung'):
            if not key in info_dict:
                Error(output.format('Die Überschrift "## {}" ist nicht korrekt angegeben.'.format(key)))
        if re.fullmatch('\s*Teamname\s*', info_dict['Team']):
            Error(output.format('"Teamname" kann nicht als Teamname gewählt werden, wählen Sie einen eigenen.'))
        if not re.fullmatch('\s*[^\n]+\s*', info_dict['Team']):
            Error(output.format('Das Format für Teamname ist nicht korrekt.'))
        if info_dict['Teilnehmer 1'].count(',') != 1:
            Error(output.format('Format "Name, Vorname" für "Teilnehmer 1" ist nicht korrekt.'))
        if info_dict['Teilnehmer 2'].count(',') != 1 and not 'KEIN' in info_dict['Teilnehmer 2']:
            Error(output.format('Format "Name, Vorname" bzw. "KEINER" für "Teilnehmer 2" ist nicht korrekt.'))
        if not re.fullmatch('\s*[^\s<>"\',;]+@[^\s<>"\',;]+\.[^\s<>"\',;]+\s*', info_dict['Email 1']) and  not 'KEIN' in info_dict['Email 2']:
            self.fail(output.format('Email Format für "Teilnehmer 1" ist nicht korrekt.'))
        if not re.fullmatch('\s*[^\s<>"\',;]+@[^\s<>"\',;]+\.[^\s<>"\',;]+\s*', info_dict['Email 2']) and  not 'KEIN' in info_dict['Email 2']:
            self.fail(output.format('Email Format für "Teilnehmer 2" ist nicht korrekt.'))
        if not re.fullmatch('\s*(Ich|Wir|Ich\s*.\s*wir)\s*bestätigen\s*.?\s*dass\s*wir\s*nur\s*die\s*angegebenen\s*Quellen\s*benutzt\s*haben?.?\s*', info_dict['Bestätigung'], flags=re.I+re.S):
            Error(output.format('Die Bestätigung ist nicht korrekt.'))

        
def scramble(object):
    if not type(object) == bytes:
        object = bytes(str(object), encoding='utf8')
    try:
        return str(base64.b64encode(hashlib.sha1(object).digest())[:10], encoding='ascii')
    except:
        return 'Hashwert nicht berechenbar !!!'

def scramble_float(precision=5):
    def scrmble(x):
        try:
            x = float(x)
        except:
            Error('Der angegebene Wert ist kein "float".')
        return scramble('{:.{p}e}'.format(x, p=precision))
    return scrmble

def Hash_file(file_name):
    try:
        with open(file_name, 'rb') as f:
            content = f.read()
        return scramble(content)
    except:
        return 'Datei nicht lesbar !!!'
    
def Error(*args):
    raise ERROR('\n'.join(map(str, args)))
        
def Error_if(condition, *args):
    if condition:
        Error(*args)


if __name__ == '__main__':
    
    collect_results = False
    try:
        Results(data_file, collect_results)
    except:
        print()
        print('The data file {} cannot be read.'.format(data_file))
        print('Trying to produce a new one.')
        try:
            import test_config
            if test_config.collect_results_automatically:
                collect_results = True
                print('Module test_config found.')
                print('Automatic results collection is switched on.')
        except:
            print('No module test_config.')
            print('No automatic results collection.')
        print()
    try:
        if 'collect' == sys.argv[1]:
            collect_results = True
    except:
        pass
    results = Results(data_file, collect_results)
    unittest.main(argv=[sys.argv[0]])
    results.dump()
