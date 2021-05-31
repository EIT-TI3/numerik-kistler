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


P_number = 3
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
            self.fail('"result" should be iterable.')
        if not hasattr(true_value, '__iter__'):
            self.fail('"true value should be iterable."')
        if not hasattr(result, '__len__'):
            self.fail('"result" does not have a length. This should not happen.')
        if not hasattr(true_value, '__len__'):
            self.fail('"True value" does not have a length. This should not happen.')
        self.assertEqual(len(result), len(true_value), msg='Lengths of "result" and "true value" are not equal.')
    
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
            if true_value:
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
    
    @classmethod
    def save(cls, eval_string, name=None):
        if not name:
            name = eval_string
        if results.collecting:
            obj = eval(eval_string)
            results.set_global(name, [obj])  # save singleton of obj as all values are lists
            return obj
        else:
            return results.get_global(name)[0]  # get singleton and retrieve element (all values must be stored as lists)
        
        

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
    
    def set_global(self, key, value):
        """Store value."""
        if self.collecting:   # do not do anything if not collecting
            assert not key.startswith('test_'), 'The key must not start with "test_".'
            self.data[(key,)] = value
    
    def get_global(self, key):
        """Get stored value."""
        assert not key.startswith('test_'), 'The key must not start with "test_".'
        return self.data[(key,)]
        
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
        
    def store_data(self, key, value):
        if self.collecting:
            self.data[('stored', key)] = value
            
    def read_data(self, key):
        return self.data[('stored', key)]
        
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
 


f_data_awp_list = [(main.f1, main.data[1], main.awp1),
                   (main.f2, main.data[2], main.awp2),
                   (main.f3, main.data[3], main.awp3),
                   (main.f4, main.data[4], main.awp4)]



class Test_Praktikum(Test_Numerik):
    
    try:
        true_values_data = main.true_values
    except (NameError, AttributeError):
        true_values_data = dict()
        
    def true_values(self, key, default='no_value_provided_by_main'):
        return self.true_values_data.get(key, default)
        
    @classmethod
    def setUpClass(cls):
        ### register names for objects
        ### in most cases cls.get_name(x) will yield the same name as x.__name__
        ### for examples in the most common case when a method is defined with def x(...).
        ### examples where x.__name__ does not work are lambda definitions or assignments
        ### x = y, where y is a previously defined method, and, of course, all the simple
        ### variables like x = 17, where x.__name__ is not even defined.
        for i in range(1, 5):
            cls.set_name(main, 'f{}'.format(i))
        for Verfahren in ['Euler', 'Mittelpunktsregel' , 'Heun', 'Runge_Kutta']:
            cls.set_name(lib, Verfahren)
        cls.Winkel = cls.save('main.Winkel')
        cls.AnzahlPerioden = cls.save('len(main.Perioden)')

    def test_data(self):
        for f, data, awp in f_data_awp_list:
            name = self.get_name(f)
            with self.subTest(msg="Daten a, xa, b für {} nicht korrekt.".format(name), name=name + '-a-xa-b'):
                self.runner(lambda x: x[:3], (data, ), self.assertEqual)
#            lower, upper = values.get(f, (None, None))
            lower, upper = self.true_values(f, default=(None, None))
            with self.subTest(msg="n für {} zu klein.".format(name), name=name):
                self.runner(lambda x: x[3], (data, ), self.assertGreaterEqual, true_value=lower)
            with self.subTest(msg="n für {} zu groß.".format(name), name=name):
                self.runner(lambda x: x[3], (data, ), self.assertLessEqual, true_value=upper)

    def test_fi(self):
        for f, data, awp in f_data_awp_list:
            name = self.get_name(f)
            with self.subTest(msg="Differentialgleichung {} nicht korrekt definiert.".format(name), name=name):
                for t in np.linspace(-10, 10, 11):
                    for x in np.linspace(-10, 10, 11):
                        self.runner(f, (t, x), self.assertAlmostEqualRelative, msg='t={}, x={}'.format(t, x))

    def test_awp(self):
        for f, data, awp in f_data_awp_list:
            name = self.get_name(f)
            with self.subTest(msg="Lösung AWP zu {} nicht korrekt.".format(name), name=name):
                for t in np.linspace(-10, 10, 6):
                    for a in np.linspace(-10, 10, 6):
                        for xa in np.linspace(-10, -1, 6):
                            self.runner(awp, (t, a, xa), self.assertAlmostEqualRelative, msg='t={}, a={}, xa={}'.format(t, a, xa))    

    def test_Verfahren(self):
        for Verfahren in (lib.Euler, lib.Mittelpunktsregel, lib.Heun, lib.Runge_Kutta):
            Vname = self.get_name(Verfahren)
            with self.subTest(name=Vname):
                for f, data, awp in f_data_awp_list:
                    fname = self.get_name(f)
                    with self.subTest(msg="Verfahren {} nicht korrekt für {}.".format(Vname, fname), name=fname):
                        a, xa, b, n = data
                        for k in range(10):
                            self.runner(Verfahren, (f, a, xa, b, 2**k), self.assertAlmostEqualRelative, msg='n={}'.format(2**k))

    def test_Runge_Kutta_2D(self):
        def f6(t, x):
            c = np.log(2) / (2*np.pi)
            x0, x1 = x[0], x[1]
            return np.array([-x1 - c*x0, x0 - c*x1])
        data6 = 0, np.array([1, 0]), 10*np.pi, 1000
        lib.plot_2D(f6, data6, lib.Runge_Kutta)
        self.runner(lib.Runge_Kutta, (f6, *data6), self.assertAlmostEqualRelative)
                        
    def test_Endpunkt(self):
        with self.subTest(msg="Endpunkt nicht korrekt."):
            for y in main.Endpunkt:
                self.runner(lambda x: x, (y, ), self.assertEqual, post=scramble_float())                

    def test_Perioden(self):
        Perioden = main.Perioden
        AnzahlPerioden = self.AnzahlPerioden
        Winkel = self.Winkel
        with self.subTest(msg="Liste der Perioden für Winkel {}° zu kurz.".format(Winkel), name='Anzahl'):
            self.runner(lambda x: x, (len(Perioden), ), self.assertEqual)
        if len(Perioden) < AnzahlPerioden:
            Perioden = tuple(Perioden) + (AnzahlPerioden - len(Perioden)) * (-1,)
        for periode, winkel in zip(Perioden, Winkel):
            with self.subTest(msg="Periode für Winkel {}° zu ungenau.".format(winkel), name=str(winkel)):
                self.runner(lambda x: x, (periode, ), self.assertEqual, post=scramble_float(2))
                
    def test_Dateikonsistenz(self):
        for file in [test_file, vorgabe_file]:
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
                data = re.split('[\n\r]+', entry, maxsplit=1) + ['']
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
