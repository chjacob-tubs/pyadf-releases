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

import doctest
import unittest
import re


class AlmostEqualChecker (doctest.OutputChecker) :
    """
    Derived OutputChecker to deal with floating point numbers in doctests.

    The comparison is done by searching for all floating point numbers
    in both the original string and the obtained output and replacing
    them with a normalized form of the number using 8 decimals.
    """

    def normalize (self, string) :
        """
        Some clever regular explession stuff for normalizing the numbers
        """
        return re.sub(r'(\d+\.\d*|\d*\.\d+)', 
                      lambda m: "%.8f" % float(m.group(0)), 
                      string)

    def check_output (self, want, got, optionflags) :
        return doctest.OutputChecker.check_output (self, 
                                                   self.normalize(want), 
                                                   self.normalize(got), 
                                                   optionflags)

    def output_difference (self, example, got, optionflags) :
        return doctest.OutputChecker.output_difference (self, example, 
                                                        self.normalize(got), 
                                                        optionflags)


def make_doctest_suite() : 

    import pyadf.Molecule

    molecule_doctests = \
        doctest.DocTestSuite(pyadf.Molecule, 
                             setUp       = pyadf.Molecule._setUp_doctest, 
                             tearDown    = pyadf.Molecule._tearDown_doctest, 
                             optionflags = doctest.NORMALIZE_WHITESPACE, 
                             checker     = AlmostEqualChecker())

    return molecule_doctests

def test () :
    suite = make_doctest_suite()
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    test()
