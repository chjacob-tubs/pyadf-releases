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
import numpy


class PyAdfTestCase(unittest.TestCase):

    def assertAlmostEqualVectors(self, first, second, places=7, msg=None):
        for i, j in zip(first, second):
            self.assertAlmostEqual(i, j, places, msg)

    def assertAlmostEqualNumpy(self, first, second, places=7, msg=None):
        numpy.testing.assert_allclose(first, second, rtol=0.5 * 10 ** (-places),
                                      atol=0.5 * 10 ** (-places), err_msg=msg)

    def assertAlmostEqual(self, first, second, places=7, msg=None):
        if isinstance(first, numpy.ndarray):
            self.assertAlmostEqualNumpy(first, second, places, msg)
        elif isinstance(first, list):
            self.assertAlmostEqualVectors(first, second, places, msg)
        else:
            super(PyAdfTestCase, self).assertAlmostEqual(first, second, places, msg)

    def assertAlmostEqualMolecules(self, first, second, places=3, msg=None):

        def build_atsyms_dict(mol):
            atsyms_dict = {}
            for i, at in enumerate(mol.get_atom_symbols(prefix_ghosts=True)):
                if not at in atsyms_dict:
                    atsyms_dict[at] = []
                atsyms_dict[at].append(i)
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

        for at, indices in atsyms_first.iteritems():
            if not (at in atsyms_second):
                raise self.failureException(msg or
                                            "Atom symbol %s not found in second molecule" % at)
            if not (len(indices) == len(atsyms_second[at])):
                raise self.failureException(msg or
                                            "Molecules have different number of %s atoms" % at)

            indices_second = atsyms_second[at]

            for i in indices:
                found_index = -1
                for j in indices_second:
                    almost_equal = True
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
                                                "Coordinates not equal for %s atoms within "
                                                "%i places" % (at, places))
