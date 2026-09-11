# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2026 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Maria Chekmeneva,
# Thomas Dresselhaus, Kevin Focke, Andre S. P. Gomes, Andreas Goetz,
# Michal Handzlik, Karin Kiewisch, Moritz Klammler, Toni M. Maier, 
# Malgorzata Olejniczak, Lars Ridder, Jetze Sikkema, Erik Tsvetaev, 
# Lucas Visscher, Johannes Vornweg, Michael Welzel, and Mario Wolter.
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
import numpy as np

from .CollectExceptions import decorate_asserts


@decorate_asserts
class PyAdfTestCase(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.do_not_catch_errors = False
        self._errors = []

    @staticmethod
    def testDuration():
        return 'unittest'

    def shortName(self):
        name = self.shortDescription()
        if name is None:
            name = str(self)
        return name

    def tearDown(self):
        self.checkAndClearExceptions()  # usually does nothing, but fails
        # if assertionErrors have been collected up to this point
        super().tearDown()  # keeps the old behavior in other cases

    def append_error(self, error):
        self._errors.append(error)

    @property
    def errors(self):
        return self._errors

    def checkAndClearExceptions(self):
        """
        This function does nothing when no errors were collected.
        Errors are only caught and collected if the environment variable
        PyADF_COLLECT_ASSERTS is set in some way.
        The errors are re-raised with a combination of all error messages
        as generated in the CollectExceptions.py file.

        For the case when this error is expected, as in the TestTesting
        test, the testobj._errors attribute is re-set to avoid re-raising
        an expected and caught error.
        """
        if self._errors:
            errors = [str(err) for err in self._errors]
            emsg = 'Encountered following assertion errors:\n'
            emsg += "\n".join(errors)
            self._errors = []  # Clear exceptions
            self.fail(emsg)
        self._errors = []  # Clear exceptions

    def assertAlmostEqualVectors(self, first, second, places=7, msg=''):
        for i, j in zip(first, second):
            self.assertAlmostEqual(i, j, places, msg)

    # noinspection PyMethodMayBeStatic
    def assertAlmostEqualNumpy(self, first, second, places=7, msg=''):
        np.testing.assert_allclose(first, second, rtol=0.5 * 10**(-places),
                                   atol=0.5 * 10**(-places), err_msg=msg)

    def assertAlmostEqual(self, first, second, places=7, msg='', delta=None):
        if isinstance(first, np.ndarray):
            self.assertAlmostEqualNumpy(first, second, places, msg)
        elif isinstance(first, list):
            self.assertAlmostEqualVectors(first, second, places, msg)
        else:
            super().assertAlmostEqual(first, second, places, msg, delta)

    def assertAlmostEqualMolecules(self, first, second, places=3, msg=''):

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
                            """
                            If we would use self.assertAlmostEqual here, we would not know
                            where this was caught as we would use wrapped versions of the
                            different assert functions. The exception on the lower level
                            is caught and recorded, but the upper level never fails.

                            Since the exception would be recorded, the test would correctly
                            be marked as failed. Therefore this does not constitute an example
                            where the wrapped version leads to "tests passing wrongly" which
                            should be impossible.
                            It would simply make it much harder to find the site of the
                            assertion that raised the exception.

                            This issue was caught through the TestTesting test.
                            """
                            unittest.TestCase().assertAlmostEqual(
                                coords_first[i][k], coords_second[j][k], places)
                    except self.failureException:
                        almost_equal = False

                    if almost_equal:
                        found_index = j
                        break

                if found_index > -1:
                    indices_second = [
                        j for j in indices_second if not (j == found_index)]
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
        self._newline = False

    def startTest(self, test):
        save_showAll = self.showAll
        self.showAll = False
        super().startTest(test)
        self.showAll = save_showAll
        self._newline = False

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
        self.stream = None
        super().__init__(verbosity=2, descriptions=False, resultclass=PyAdfTextTestResult, *args, **kwargs)
        self._print_timing_report = print_timing_report

    def run(self, test):
        result = super().run(test)

        if self._print_timing_report:
            # noinspection PyUnresolvedReferences
            all_timings = result.test_timings

            self.stream.writeln()
            self.stream.writeln(" Timing Report ")
            self.stream.writeln(" ============= \n")

            doctest_timings = [t[2] for t in all_timings if t[1] == 'doctest']
            if len(doctest_timings) > 0:
                doctests_total = sum(doctest_timings)
                self.stream.writeln(f" Doctests: total time {_time2str(doctests_total):s}")
                self.stream.writeln()

            unittest_timings = [t[2] for t in all_timings if t[1] == 'unittest']
            if len(unittest_timings) > 0:
                unittests_total = sum(unittest_timings)
                self.stream.writeln(f" Unittests: total time {_time2str(unittests_total):s}")
                self.stream.writeln()

            for testset in ['short', 'medium', 'long']:
                timings = [(t[0], t[2]) for t in all_timings if t[1] == testset]
                if len(timings) > 0:
                    timings.sort(key=lambda t: t[1])
                    times = np.array([t[1] for t in timings])
                    self.stream.writeln(f" Input tests ({testset.upper():s} testset):"
                                        f" total time {_time2str(np.sum(times)):s}")
                    self.stream.writeln(f"   avg {_time2str(np.mean(times)):s}"
                                        f"   min {_time2str(np.min(times)):s}"
                                        f"   max {_time2str(np.max(times)):s}")
                    self.stream.writeln()
                    self.stream.writeln(f"   Slowest {testset:s} tests: ")
                    for i in range(1, min(3, len(timings))+1):
                        self.stream.writeln(f"      {timings[-i][0]:s}  ( {_time2str(timings[-i][1]):s} )")
                    self.stream.writeln()
                    self.stream.writeln(f"   Fastest {testset:s} tests: ")
                    for i in range(min(3, len(timings))):
                        self.stream.writeln(f"      {timings[i][0]:s}  ( {_time2str(timings[i][1]):s} )")
                    self.stream.writeln()

            timings = [(t[0], t[2]) for t in all_timings if (t[1] in ['all', 'unkonwn'])]
            if len(timings) > 0:
                self.stream.writeln(" Uncategorized tests: ")
                for t in timings:
                    self.stream.writeln(f"      {t[0]:s}  ( {_time2str(t[1]):s} )")
                self.stream.writeln()
            self.stream.flush()

        return result
