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

import unittest
from .PyAdfTestCase import PyAdfTestCase

import os
import numpy
from pyadf.Plot.FileReaders import GridReader
from pyadf.Plot.FileReaders import GridFunctionReader


TESTDATA_DIRNAME = os.path.join(os.path.dirname(__file__), 'data_PlotFileReaders')


class TestGridReader(PyAdfTestCase):

    def test_read_xvzw(self):
        testgrid_filename = os.path.join(TESTDATA_DIRNAME, 'grid.xyzw')
        grid = GridReader.read_xyzw(testgrid_filename)

        self.assertAlmostEqual(grid.npoints, 10)
        saved_weights = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xyzv-weights.npy'))
        self.assertAlmostEqualNumpy(grid.weights, saved_weights, 8)
        saved_coords = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xyzv-coords.npy'))
        self.assertAlmostEqualNumpy(grid.get_coordinates(), saved_coords, 8)

    def test_read_xml(self):
        testgrid_filename = os.path.join(TESTDATA_DIRNAME, 'diracexp.xml')
        grid = GridReader.read_xml(testgrid_filename)

        self.assertAlmostEqual(grid.npoints, 15)
        saved_weights = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xml-weights.npy'))
        self.assertAlmostEqualNumpy(grid.weights, saved_weights, 8)
        saved_coords = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xml-coords.npy'))
        self.assertAlmostEqualNumpy(grid.get_coordinates(), saved_coords, 8)


class TestGridFunctionReader(PyAdfTestCase):

    def test_read_xvzwv(self):
        testgf_filename = os.path.join(TESTDATA_DIRNAME, 'gridfunction.xyzwv')
        gf = GridFunctionReader.read_xyzwv(testgf_filename)

        self.assertAlmostEqual(gf.grid.npoints, 10)
        saved_weights = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xyzv-weights.npy'))
        self.assertAlmostEqualNumpy(gf.grid.weights, saved_weights, 8)
        saved_coords = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xyzv-coords.npy'))
        self.assertAlmostEqualNumpy(gf.grid.get_coordinates(), saved_coords, 8)
        saved_values = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xyzv-values.npy'))
        self.assertAlmostEqualNumpy(gf.get_values(), saved_values, 8)

    def test_read_density_elpot_xyzwv(self):
        testdens_filename = os.path.join(TESTDATA_DIRNAME, 'density.diracimp')
        grid, pot_gf, rho_gf = GridFunctionReader.read_density_elpot_xyzwv(testdens_filename)

        self.assertAlmostEqual(grid.npoints, 10)
        saved_weights = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xyzv-weights.npy'))
        self.assertAlmostEqualNumpy(grid.weights, saved_weights, 8)
        saved_coords = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xyzv-coords.npy'))
        self.assertAlmostEqualNumpy(grid.get_coordinates(), saved_coords, 8)

        saved_pot = numpy.load(os.path.join(TESTDATA_DIRNAME, 'density-pot.npy'))
        self.assertAlmostEqualNumpy(pot_gf.get_values(), saved_pot, 8)
        saved_rho = numpy.load(os.path.join(TESTDATA_DIRNAME, 'density-rho.npy'))
        self.assertAlmostEqualNumpy(rho_gf[0].get_values(), saved_rho, 8)
        saved_rhod = numpy.load(os.path.join(TESTDATA_DIRNAME, 'density-rhod.npy'))
        self.assertAlmostEqualNumpy(rho_gf[1].get_values(), saved_rhod, 8)

    def test_read_density_elpot_xml(self):
        testdens_filename = os.path.join(TESTDATA_DIRNAME, 'diracexp.xml')
        grid, elpot_gf, nucpot_gf, rho_gf = GridFunctionReader.read_density_elpot_xml(testdens_filename)

        self.assertAlmostEqual(grid.npoints, 15)
        saved_weights = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xml-weights.npy'))
        self.assertAlmostEqualNumpy(grid.weights, saved_weights, 8)
        saved_coords = numpy.load(os.path.join(TESTDATA_DIRNAME, 'grid-xml-coords.npy'))
        self.assertAlmostEqualNumpy(grid.get_coordinates(), saved_coords, 8)

        saved_pot = numpy.load(os.path.join(TESTDATA_DIRNAME, 'density-xml-elpot.npy'))
        self.assertAlmostEqualNumpy(elpot_gf.get_values(), saved_pot, 8)
        saved_pot = numpy.load(os.path.join(TESTDATA_DIRNAME, 'density-xml-nucpot.npy'))
        self.assertAlmostEqualNumpy(nucpot_gf.get_values(), saved_pot, 8)

        saved_rho = numpy.load(os.path.join(TESTDATA_DIRNAME, 'density-xml-rho.npy'))
        self.assertAlmostEqualNumpy(rho_gf[0].get_values(), saved_rho, 8)
        saved_rhod = numpy.load(os.path.join(TESTDATA_DIRNAME, 'density-xml-rhod.npy'))
        self.assertAlmostEqualNumpy(rho_gf[1].get_values(), saved_rhod, 8)
        saved_rhodd = numpy.load(os.path.join(TESTDATA_DIRNAME, 'density-xml-rhodd.npy'))
        self.assertAlmostEqualNumpy(rho_gf[2].get_values(), saved_rhodd, 8)


if __name__ == '__main__':
    unittest.main()
