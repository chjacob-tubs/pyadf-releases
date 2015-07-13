# This file is part of 
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2011 by Christoph R. Jacob, S. Maya Beyhan,
# Rosa E. Bulo, Andre S. P. Gomes, Andreas Goetz, Karin Kiewisch,
# Jetze Sikkema, and Lucas Visscher 
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
try:
    import profile
except ImportError:
    pass

import os, shutil, glob

class PyADFInputTestCase (unittest.TestCase) :

    def __init__ (self, methodName='runTest', testname='bla', keep=False, prof=False) :
        
        unittest.TestCase.__init__ (self, methodName)
        self._testname = testname
        self._keep     = keep
        self._profile  = prof
        self._cwd      = os.getcwd()

    def shortDescription (self) :
        return "PyADF test input file: "+self._testname

    def assertAlmostEqualVectors (self, first, second, places=7, msg=None):
        for i, j in zip(first, second) :
            self.assertAlmostEqual(i, j, places, msg)

    def assertAlmostEqualMolecules (self, first, second, places=3, msg=None):

        def build_atsyms_dict(mol):
            atsyms_dict = {}
            for i, at in enumerate(mol.get_atom_symbols(prefix_ghosts=True)) :
                if not at in atsyms_dict :
                    atsyms_dict[at] = []
                atsyms_dict[at].append(i)
            return atsyms_dict

        # build two dictionaries with mapping between atom symbols and atom indices
        atsyms_first  = build_atsyms_dict(first)
        atsyms_second = build_atsyms_dict(second)

        # and get the Cartesian coordinates
        coords_first  = first.get_coordinates ()
        coords_second = second.get_coordinates ()
        
        # check that the number of different atomic symbols matches
        if not (len(atsyms_first) == len(atsyms_second)) :
            raise self.failureException(msg or "Molecules have different number of atom types")
        
        for at, indices in atsyms_first.iteritems() :
            if not (at in atsyms_second) :
                raise self.failureException(msg or 
                                            "Atom symbol %s not found in second molecule"%at)
            if not (len(indices) == len(atsyms_second[at])) :
                raise self.failureException(msg or 
                                            "Molecules have different number of %s atoms"%at)

            indices_second = atsyms_second[at]

            for i in indices :
                found_index = -1
                for j in indices_second :
                    almost_equal = True
                    try:
                        for k in range(3) :
                            self.assertAlmostEqual (coords_first[i][k], coords_second[j][k], places)
                    except self.failureException :
                        almost_equal = False

                    if almost_equal :
                        found_index = j
                        break
                
                if found_index > -1 :
                    indices_second = [j for j in indices_second if not (j==found_index)]
                else:
                    raise self.failureException(msg or 
                                                "Coordinates not equal for %s atoms within " 
                                                "%i places" % (at, places))                    

    def setUp (self) :

        os.chdir(self._cwd)

        self._pyadfpath = os.path.join(os.environ['PYADFHOME'], 'src', 'scripts')
        if not 'pyadf' in os.listdir(self._pyadfpath) :
            self.fail('pyadf not found')

        # make a temporary directory
        self._tempdirname = os.path.join(self._cwd, "pyadf_testdir_" + self._testname)
        if os.path.exists(self._tempdirname) :
            shutil.rmtree(self._tempdirname)
        os.mkdir (self._tempdirname)

        # test directory
        testdir = os.path.join(os.environ['PYADFHOME'], 'test', 'testinputs', self._testname)

        # copy test input to temporary directory
        shutil.copy (os.path.join(testdir, self._testname+".pyadf"), self._tempdirname)

        # copy needed coordinates files to temporary directory
        coordfiles  = glob.glob(os.path.join(testdir, 'coordinates', '*'))

        for f in coordfiles :
            if os.path.isdir(f):
                shutil.copytree(f, os.path.join(self._tempdirname, os.path.basename(f)))
            else:
                shutil.copy (f, self._tempdirname)

        os.chdir(self._tempdirname)
        
    def tearDown (self) :

        os.chdir(self._cwd)
    
        if (self._keep==False) :
            shutil.rmtree(self._tempdirname, True)

        import gc
        gc.collect()

    def runTest (self) :
        
        globs = {'pyadfinput':self._testname+".pyadf", 'testobj':self}

        if not self._profile :
            execfile (os.path.join(self._pyadfpath, 'pyadf'), globs, {})
        else:
            profile.runctx("execfile (pyadf_path)", globs, 
                           {'pyadf_path':os.path.join(self._pyadfpath, 'pyadf')}, "pyadf_profile")
            
        del globs
        

def make_testinputs_suite(tests="all", testnames=None, dalton=True, dirac=True, nwchem=True, keep=False, prof=False) : 

    testsetorder = ['short', 'medium', 'long', 'all']

    suite = unittest.TestSuite()

    # test directory
    testdir = os.path.join(os.environ['PYADFHOME'], 'test', 'testinputs')

    if testnames == None :
        alltests = os.listdir(testdir)
    else :
        alltests = testnames

    alltests = [d for d in alltests if os.path.isdir(os.path.join(testdir, d))]
    alltests.sort()

    for testname in alltests :

        if os.path.exists(os.path.join(testdir, testname, testname+'.pyadf')) :

            if (os.path.exists(os.path.join(testdir, testname, 'short'))) :
                testset = 'short'
            elif (os.path.exists(os.path.join(testdir, testname, 'medium'))) :
                testset = 'medium'
            elif (os.path.exists(os.path.join(testdir, testname, 'long'))) :
                testset = 'long'

            use_test = False
            if tests == 'dalton': 
                use_test = ('dalton' in testname.lower())
            elif tests == 'dirac' :
                use_test = ('dirac' in testname.lower())
            elif tests == 'nwchem' :
                use_test = ('nwchem' in testname.lower())
            elif testsetorder.index(tests) >= testsetorder.index(testset) :
                use_test = True
                if ('dalton' in testname.lower()) and not dalton :
                    use_test = False
                if ('dirac' in testname.lower()) and not dirac :
                    use_test = False
                if ('nwchem' in testname.lower()) and not nwchem :
                    use_test = False
                  
            if use_test :
                suite.addTest(PyADFInputTestCase(testname=testname, keep=keep, prof=prof))

    return suite
