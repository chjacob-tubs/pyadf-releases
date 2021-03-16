# -*- coding: utf-8 -*-

# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2021 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Thomas Dresselhaus,
# Andre S. P. Gomes, Andreas Goetz, Michal Handzlik, Karin Kiewisch,
# Moritz Klammler, Lars Ridder, Jetze Sikkema, Lucas Visscher, and
# Mario Wolter.
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
#    along with PyADF.  If not, see <http://www.gnu.org/licenses/>.

import unittest
from pyadf_unittests.PyAdfTestCase import PyAdfTestCase

try:
    import cProfile as profile
except ImportError:
    profile = None

import os
import shutil
import glob


class PyADFInputTestCase(PyAdfTestCase):
    # pylint: disable=R0904

    def __init__(self, methodName='runTest', testname='bla', keep=False,
                 prof=False, molclass="openbabel", jobrunnerconf=None):

        unittest.TestCase.__init__(self, methodName)
        self._testname = testname
        self._keep = keep
        self._profile = prof
        self._molclass = molclass
        self._jobrunnerconf = jobrunnerconf
        self._cwd = os.getcwd()

    def __str__(self):
        return "PyADF test input file: " + self._testname

    def setUp(self):

        os.chdir(self._cwd)

        self._pyadfpath = os.path.join(os.environ['PYADFHOME'], 'src', 'scripts')
        if 'pyadf' not in os.listdir(self._pyadfpath):
            self.fail('pyadf not found')

        # make a temporary directory
        self._tempdirname = os.path.join(self._cwd, "pyadf_testdir_" + self._testname)
        if os.path.exists(self._tempdirname):
            shutil.rmtree(self._tempdirname)
        os.mkdir(self._tempdirname)

        # test directory
        testdir = os.path.join(os.environ['PYADFHOME'], 'test', 'testinputs', self._testname)

        # copy test input to temporary directory
        shutil.copy(os.path.join(testdir, self._testname + ".pyadf"), self._tempdirname)

        # copy needed coordinates files to temporary directory
        coordfiles = glob.glob(os.path.join(testdir, 'coordinates', '*'))

        for f in coordfiles:
            if os.path.isdir(f):
                shutil.copytree(f, os.path.join(self._tempdirname, os.path.basename(f)))
            else:
                shutil.copy(f, self._tempdirname)

        os.chdir(self._tempdirname)

    def tearDown(self):

        os.chdir(self._cwd)

        if not self._keep:
            shutil.rmtree(self._tempdirname, True)

        import gc
        gc.collect()

    def runTest(self):

        globs = {'pyadfinput': self._testname + ".pyadf", 'testobj': self,
                 'testing_molclass': self._molclass, 'testing_jobrunnerconf': self._jobrunnerconf}

        if not self._profile:
            execfile(os.path.join(self._pyadfpath, 'pyadf'), globs, {})
        else:
            profile.runctx("execfile (pyadf_path)", globs,
                           {'pyadf_path': os.path.join(self._pyadfpath, 'pyadf')}, "pyadf_profile")

        del globs


def make_testinputs_suite(tests="all", testnames=None, dalton=True, dirac=True, nwchem=True,
                          espresso=True, molcas=True, turbomole=True, orca=True, openbabel=True,
                          keep=False, prof=False, molclass="openbabel", jobrunnerconf=None):

    testsetorder = ['short', 'medium', 'long', 'all']

    suite = unittest.TestSuite()

    # test directory
    testdir = os.path.join(os.environ['PYADFHOME'], 'test', 'testinputs')

    if testnames is None:
        alltests = os.listdir(testdir)
    else:
        alltests = testnames

    alltests = [d for d in alltests if os.path.isdir(os.path.join(testdir, d))]
    alltests.sort()

    for testname in alltests:

        if os.path.exists(os.path.join(testdir, testname, testname + '.pyadf')):

            if os.path.exists(os.path.join(testdir, testname, 'short')):
                testset = 'short'
            elif os.path.exists(os.path.join(testdir, testname, 'medium')):
                testset = 'medium'
            elif os.path.exists(os.path.join(testdir, testname, 'long')):
                testset = 'long'

            use_test = False
            if tests == 'dalton':
                use_test = ('dalton' in testname.lower())
            elif tests == 'dirac':
                use_test = ('dirac' in testname.lower())
            elif tests == 'nwchem':
                use_test = ('nwchem' in testname.lower())
            elif tests == 'espresso':
                use_test = ('espresso' in testname.lower())
            elif tests == 'molcas':
                use_test = ('molcas' in testname.lower())
            elif tests == 'turbomole':
                use_test = (('turbomole' in testname.lower()) or ('snf' in testname.lower()))
            elif tests == 'orca':
                use_test = ('orca' in testname.lower())
            elif tests == 'openbabel':
                use_test = (('openbabel' in testname.lower()) or ('3fde' in testname.lower()))
            elif testsetorder.index(tests) >= testsetorder.index(testset):
                use_test = True
                if ('dalton' in testname.lower()) and not dalton:
                    use_test = False
                if ('dirac' in testname.lower()) and not dirac:
                    use_test = False
                if ('nwchem' in testname.lower()) and not nwchem:
                    use_test = False
                if ('espresso' in testname.lower()) and not espresso:
                    use_test = False
                if ('molcas' in testname.lower()) and not molcas:
                    use_test = False
                if (('turbomole' in testname.lower())
                        or ('snf' in testname.lower())) and not turbomole:
                    use_test = False
                if ('orca' in testname.lower()) and not orca:
                    use_test = False
                if (('openbabel' in testname.lower())
                        or ('3fde' in testname.lower())) and not openbabel:
                    use_test = False

            if use_test:
                suite.addTest(PyADFInputTestCase(testname=testname, keep=keep, prof=prof,
                                                 molclass=molclass, jobrunnerconf=jobrunnerconf))

    return suite
