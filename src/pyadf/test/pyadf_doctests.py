# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2024 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Maria Chekmeneva,
# Thomas Dresselhaus, Kevin Focke, Andre S. P. Gomes, Andreas Goetz,
# Michal Handzlik, Karin Kiewisch, Moritz Klammler, Lars Ridder,
# Jetze Sikkema, Lucas Visscher, Johannes Vornweg, Michael Welzel,
# and Mario Wolter.
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

import doctest
import unittest
import re


class AlmostEqualChecker(doctest.OutputChecker):
    """
    Derived OutputChecker to deal with floating point numbers in doctests.

    The comparison is done by searching for all floating point numbers
    in both the original string and the obtained output and replacing
    them with a normalized form of the number using 8 decimals.
    """

    def __init__(self):
        pass

    @staticmethod
    def normalize(string):
        """
        Some clever regular explession stuff for normalizing the numbers
        """
        return re.sub(r'(\d+\.\d*|\d*\.\d+)',
                      lambda m: f"{float(m.group(0)):.8f}",
                      string)

    def check_output(self, want, got, optionflags):
        return doctest.OutputChecker.check_output(self,
                                                  self.normalize(want),
                                                  self.normalize(got),
                                                  optionflags)

    def output_difference(self, example, got, optionflags):
        return doctest.OutputChecker.output_difference(self, example,
                                                       self.normalize(got),
                                                       optionflags)


def make_doctest_suite(molclass="openbabel"):
    try:
        if molclass == "openbabel":
            import pyadf.Molecule.OBMolecule as Molecule
        elif molclass == "rdkit":
            import pyadf.Molecule.RDMolecule as Molecule
        elif molclass == "obfree":
            import pyadf.Molecule.OBFreeMolecule as Molecule
        else:
            import pyadf.Molecule.OBFreeMolecule as Molecule

        if molclass == "obfree" or molclass is None:
            molecule_doctests = None
        else:
            # noinspection PyProtectedMember
            molecule_doctests = \
                doctest.DocTestSuite(Molecule,
                                     setUp=Molecule._setUp_doctest,
                                     tearDown=Molecule._tearDown_doctest,
                                     optionflags=doctest.NORMALIZE_WHITESPACE,
                                     checker=AlmostEqualChecker())
    except ImportError:
        print('The requested testsuite is empty, please choose the appropriate molecule class for your environment')
        raise

    return molecule_doctests


def test():
    suite = make_doctest_suite()
    if suite is not None:
        unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite)


if __name__ == '__main__':
    test()
