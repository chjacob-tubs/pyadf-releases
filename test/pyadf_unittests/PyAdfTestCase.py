# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2022 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Maria Chekmeneva,
# Thomas Dresselhaus, Kevin Focke, Andre S. P. Gomes, Andreas Goetz, 
# Michal Handzlik, Karin Kiewisch, Moritz Klammler, Lars Ridder, 
# Jetze Sikkema, Lucas Visscher, Johannes Vornweg and Mario Wolter.
#
#    PyADF is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyADF is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyADF.  If not, see <https://www.gnu.org/licenses/>.
import os
import unittest
import time
import math
import numpy


class PyAdfTestCase(unittest.TestCase):

    @staticmethod
    def testDuration():
        return 'unittest'

    def shortName(self):
        name = self.shortDescription()
        if name is None:
            name = str(self)
        return name

    def assertAlmostEqualVectors(self, first, second, places=7, msg=None):
        for i, j in zip(first, second):
            self.assertAlmostEqual(i, j, places, msg)

    # noinspection PyMethodMayBeStatic
    def assertAlmostEqualNumpy(self, first, second, places=7, msg=None):
        numpy.testing.assert_allclose(first, second, rtol=0.5 * 10**(-places),
                                      atol=0.5 * 10**(-places), err_msg=msg)

    def assertAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        if isinstance(first, numpy.ndarray):
            self.assertAlmostEqualNumpy(first, second, places, msg)
        elif isinstance(first, list):
            self.assertAlmostEqualVectors(first, second, places, msg)
        else:
            super().assertAlmostEqual(first, second, places, msg, delta)

    def assertAlmostEqualMolecules(self, first, second, places=3, msg=None):

        def build_atsyms_dict(mol):
            atsyms_dict = {}
            for ii, atom in enumerate(mol.get_atom_symbols(prefix_ghosts=True)):
                if atom not in atsyms_dict:
                    atsyms_dict[atom] = []
                atsyms_dict[atom].append(ii)
            return atsyms_dict

        # build two dictionaries with mapping between atom symbols and atom indices
        atsyms_first = build_atsyms_dict(first)
        atsyms_second = build_atsyms_dict(second)

        # and get the Cartesian coordinates
        coords_first = first.get_coordinates()
        coords_second = second.get_coordinates()

        # check that the number of different atomic symbols matches
        if not (len(atsyms_first) == len(atsyms_second)):
            raise self.failureException(msg or "Molecules have different number of atom types")

        for at, indices in atsyms_first.items():
            if not (at in atsyms_second):
                raise self.failureException(msg or
                                            f"Atom symbol {at} not found in second molecule")
            if not (len(indices) == len(atsyms_second[at])):
                raise self.failureException(msg or
                                            f"Molecules have different number of {at} atoms")

            indices_second = atsyms_second[at]

            for i in indices:
                found_index = -1
                for j in indices_second:
                    almost_equal = True
                    # noinspection PyBroadException
                    try:
                        for k in range(3):
                            self.assertAlmostEqual(coords_first[i][k], coords_second[j][k], places)
                    except self.failureException:
                        almost_equal = False

                    if almost_equal:
                        found_index = j
                        break

                if found_index > -1:
                    indices_second = [j for j in indices_second if not (j == found_index)]
                else:
                    raise self.failureException(msg or
                                                f"Coordinates not equal for {at} atoms within "
                                                f"{places:d} places")


def _time2str(s):
    minutes = math.floor(s / 60)
    seconds = s - minutes * 60

    if minutes > 0:
        timestr = f"{minutes:d}m{seconds:04.1f}s"
    else:
        timestr = f"{seconds:4.1f}s"

    return timestr


class PyAdfTextTestResult(unittest.TextTestResult):

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.showAll = verbosity > 1

        self._start_time = 0.0
        self.test_timings = []

        self._first_doctest = True
        self._first_unittest = True

        self._linebuffer = None

    def startTest(self, test):
        save_showAll = self.showAll
        self.showAll = False
        super().startTest(test)
        self.showAll = save_showAll

        if not isinstance(test, PyAdfTestCase):
            if self._first_doctest:
                self.stream.write('PyADF Doctests ')
                self._first_doctest = False
            self.showAll = False
            self.dots = True
        elif str(test).startswith('test'):
            if self._first_unittest:
                self._first_unittest = False
                self.stream.write('\n')
                self.stream.write('PyADF Unittests ')
            self.showAll = False
            self.dots = True
        else:
            if self.dots:
                self.stream.write('\n')
            self.showAll = True
            self.dots = False

        self._start_time = time.time()

        if self.showAll:
            if 'PYADF_TEST_LINEBUFFERING' in os.environ:
                self._linebuffer = self.getDescription(test) + " ... "
            else:
                self.stream.write(self.getDescription(test))
                self.stream.write(" ... ")
                if 'PYADF_TEST_LINEBREAKS' in os.environ:
                    self.stream.write('\n')
                self.stream.flush()

    def addSuccess(self, test):
        elapsed = time.time() - self._start_time
        if isinstance(test, PyAdfTestCase):
            self.test_timings.append((test.shortName(), test.testDuration(), elapsed))
        else:
            self.test_timings.append((str(test), 'doctest', elapsed))

        if self.showAll:
            self.showAll = False
            super().addSuccess(test)

            if self._linebuffer is not None:
                self.stream.write(self._linebuffer)
                self._linebuffer = None
            self.stream.writeln(f"ok   ( {_time2str(elapsed):s} )")

            self.showAll = True
        else:
            super().addSuccess(test)


class PyAdfTextTestRunner(unittest.TextTestRunner):

    def __init__(self, print_timing_report=False, *args, **kwargs):
        super().__init__(verbosity=2, descriptions=False, resultclass=PyAdfTextTestResult, *args, **kwargs)
        self._print_timing_report = print_timing_report

    def run(self, test):
        result = super().run(test)

        if self._print_timing_report:
            # noinspection PyUnresolvedReferences
            all_timings = result.test_timings

            print()
            print(" Timing Report ")
            print(" ============= \n")

            doctest_timings = [t[2] for t in all_timings if t[1] == 'doctest']
            if len(doctest_timings) > 0:
                doctests_total = sum(doctest_timings)
                print(f" Doctests: total time {_time2str(doctests_total):s}")
                print()

            unittest_timings = [t[2] for t in all_timings if t[1] == 'unittest']
            if len(unittest_timings) > 0:
                unittests_total = sum(unittest_timings)
                print(f" Unittests: total time {_time2str(unittests_total):s}")
                print()

            for testset in ['short', 'medium', 'long']:
                timings = [(t[0], t[2]) for t in all_timings if t[1] == testset]
                if len(timings) > 0:
                    timings.sort(key=lambda t: t[1])
                    times = numpy.array([t[1] for t in timings])
                    print(f" Input tests ({testset.upper():s} testset): total time {_time2str(numpy.sum(times)):s}")
                    print(f"   avg {_time2str(numpy.mean(times)):s}"
                          f"   min {_time2str(numpy.min(times)):s}"
                          f"   max {_time2str(numpy.max(times)):s}")
                    print()
                    print(f"   Slowest {testset:s} tests: ")
                    for i in range(1, min(3, len(timings))+1):
                        print(f"      {timings[-i][0]:s}  ( {_time2str(timings[-i][1]):s} )")
                    print()
                    print(f"   Fastest {testset:s} tests: ")
                    for i in range(min(3, len(timings))):
                        print(f"      {timings[i][0]:s}  ( {_time2str(timings[i][1]):s} )")
                    print()

            timings = [(t[0], t[2]) for t in all_timings if (t[1] in ['all', 'unkonwn'])]
            if len(timings) > 0:
                print(" Uncategorized tests: ")
                for t in timings:
                    print(f"      {t[0]:s}  ( {_time2str(t[1]):s} )")
                print()

        return result
