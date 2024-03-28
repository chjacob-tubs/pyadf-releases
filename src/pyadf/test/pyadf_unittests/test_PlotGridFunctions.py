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

import unittest
from .PyAdfTestCase import PyAdfTestCase

import numpy as np
from pyadf.PyEmbed.Plot.GridFunctions import GridFunction
from pyadf.PyEmbed.Plot.GridFunctions import GridFunction1D
from pyadf.PyEmbed.Plot.GridFunctions import GridFunction2D
from pyadf.PyEmbed.Plot.GridFunctions import GridFunctionDensity
from pyadf.PyEmbed.Plot.GridFunctions import GridFunctionPotential

from pyadf.PyEmbed.Plot.GridFunctions import GridFunctionFactory
from pyadf.PyEmbed.Plot.GridFunctions import GridFunctionContainer


class MockGrid:
    npoints = 3
    shape = (1, 3, 1)

    checksum = "MockGrid"

    @staticmethod
    def coorditer():
        return [[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]]

    weights = np.array([1.0, 0.5, 0.5])

    def weightiter(self):
        return self.weights.__iter__()

    @staticmethod
    def voronoiiter():
        return [1, 1, 2]


class TestGridFunction(PyAdfTestCase):

    def setUp(self):
        grid = MockGrid()

        values1 = np.array([[[1., 2.], [3., 4.]],
                               [[5., 6.], [7., 8.]],
                               [[2., 2.], [3., 3.]]])
        self.gf1 = GridFunction(grid, values1)

        values2 = np.array([[[2., 2.], [2., 2.]],
                               [[1., 1.], [1., 1.]],
                               [[1., 2.], [3., 4.]]])
        self.gf2 = GridFunction(grid, values2)

    def test_add(self):
        ref_values = np.array([[[3., 4.], [5., 6.]],
                                  [[6., 7.], [8., 9.]],
                                  [[3., 4.], [6., 7.]]])

        summ = self.gf1 + self.gf2
        self.assertAlmostEqual(summ.values, ref_values)

    def test_sub(self):
        ref_values = np.array([[[-1., 0.], [1., 2.]],
                                  [[4., 5.], [6., 7.]],
                                  [[1., 0.], [0., -1.]]])
        diff = self.gf1 - self.gf2
        self.assertAlmostEqual(diff.values, ref_values)

    def test_mul_scalar(self):
        ref_values = np.array([[[2., 4.], [6., 8.]],
                                  [[10., 12.], [14., 16.]],
                                  [[4., 4.], [6., 6.]]])
        prod = self.gf1 * 2.0
        self.assertAlmostEqual(prod.values, ref_values)

    def test_mul_scalar_rev(self):
        ref_values = np.array([[[2., 4.], [6., 8.]],
                                  [[10., 12.], [14., 16.]],
                                  [[4., 4.], [6., 6.]]])
        prod = 2.0 * self.gf1
        self.assertAlmostEqual(prod.values, ref_values)

    def test_negative(self):
        ref_values = np.array([[[-1., -2.], [-3., -4.]],
                                  [[-5., -6.], [-7., -8.]],
                                  [[-2., -2.], [-3., -3.]]])
        res = -self.gf1
        self.assertAlmostEqual(res.values, ref_values)

    def test_div_scalar(self):
        ref_values = np.array([[[0.5, 1.], [1.5, 2.]],
                                  [[2.5, 3.], [3.5, 4.]],
                                  [[1., 1.], [1.5, 1.5]]])
        qout = self.gf1 / 2.0
        self.assertAlmostEqual(qout.values, ref_values)

    def test_apply_function(self):
        ref_values = np.array([11., 83., 12.])
        res = self.gf1.apply_function(lambda x: np.dot(x[0], x[1]))
        self.assertAlmostEqual(res.values, ref_values)

        ref_values = np.array([[[1., 4.], [9., 16.]],
                                  [[25., 36.], [49., 64.]],
                                  [[4., 4.], [9., 9.]]])
        res = self.gf1.apply_function(lambda x: x**2)
        self.assertAlmostEqual(res.values, ref_values)

    def test_filter_volume(self):
        ref_values = np.array([[[1., 2.], [3., 4.]],
                                  [[0., 0.], [0., 0.]],
                                  [[0., 0.], [0., 0.]]])
        res = self.gf1.filter_volume(lambda x: x[0] + x[1] + x[2] < 2.0)
        self.assertAlmostEqual(res.values, ref_values)

    def test_get_values(self):
        vals = self.gf1.get_values()
        self.assertEqual(vals.shape, (1, 3, 1, 2, 2))


class TestGridFunction1D(PyAdfTestCase):

    def setUp(self):
        grid = MockGrid()

        values1 = np.array([1., 2., 3.])
        self.gf1 = GridFunction1D(grid, values1)

        values2 = np.array([2., 2., 6.])
        self.gf2 = GridFunction1D(grid, values2)

        values3 = np.array([-1., 0., 1.])
        self.gf3 = GridFunction1D(grid, values3)

        values4 = np.array([-0.001, 0.001, 0.1])
        self.gf4 = GridFunction1D(grid, values4)

    def test_add(self):
        ref_values = np.array([3., 4., 9.])
        summ = self.gf1 + self.gf2
        self.assertAlmostEqual(summ.values, ref_values)

    def test_sub(self):
        ref_values = np.array([1., 0., 3.])
        diff = self.gf2 - self.gf1
        self.assertAlmostEqual(diff.values, ref_values)

    def test_add_constant(self):
        ref_values = np.array([4., 5., 6.])
        summ = self.gf1 + 3.0
        self.assertAlmostEqual(summ.values, ref_values)

    def test_add_constant_rev(self):
        ref_values = np.array([4., 5., 6.])
        summ = 3.0 + self.gf1
        self.assertAlmostEqual(summ.values, ref_values)

    def test_sub_constant(self):
        ref_values = np.array([1., 1., 5.])
        diff = self.gf2 - 1.0
        self.assertAlmostEqual(diff.values, ref_values)

    def test_sub_constant_rev(self):
        ref_values = np.array([-1., -1., -5.])
        diff = 1.0 - self.gf2
        self.assertAlmostEqual(diff.values, ref_values)

    def test_mul_scalar(self):
        ref_values = np.array([2., 4., 6.])
        prod = self.gf1 * 2.0
        self.assertAlmostEqual(prod.values, ref_values)

    def test_mul_scalar_rev(self):
        ref_values = np.array([2., 4., 6.])
        prod = 2.0 * self.gf1
        self.assertAlmostEqual(prod.values, ref_values)

    def test_mul_pointwise(self):
        ref_values = np.array([2., 4., 18.])
        prod = self.gf1 * self.gf2
        self.assertAlmostEqual(prod.values, ref_values)

    def test_pow(self):
        ref_values = np.array([1., 4., 9.])
        power = self.gf1**2.
        self.assertAlmostEqual(power.values, ref_values)

    def test_div_scalar_rev(self):
        ref_values = np.array([1., 1., 1. / 3.])
        qout = 2.0 / self.gf2
        self.assertAlmostEqual(qout.values, ref_values)

    def test_filter_volume(self):
        ref_values = np.array([1., 0., 0.])
        res = self.gf1.filter_volume(lambda x: x[0] + x[1] + x[2] < 2.0)
        self.assertAlmostEqual(res.values, ref_values)

    def test_filter_positive(self):
        ref_values = np.array([0., 0., 1.])
        res = self.gf3.filter_positive()
        self.assertAlmostEqual(res.values, ref_values)

        ref_values = np.array([2., 2., 3.])
        res = self.gf1.filter_positive(thresh=2.0)
        self.assertAlmostEqual(res.values, ref_values)

    def test_filter_negative(self):
        ref_values = np.array([-1., 0., 0.])
        res = self.gf3.filter_negative()
        self.assertAlmostEqual(res.values, ref_values)

        ref_values = np.array([1., 2., 2.])
        res = self.gf1.filter_negative(thresh=2.0)
        self.assertAlmostEqual(res.values, ref_values)

    def test_filter_zeros(self):
        ref_values = np.array([-0.01, 0.01, 0.1])
        res = self.gf4.filter_zeros(thresh=0.01)
        self.assertAlmostEqual(res.values, ref_values)

    def test_integral(self):
        integral = self.gf1.integral()
        self.assertAlmostEqual(integral, 3.5)

        integral = self.gf1.integral(func=lambda x: x * x)
        self.assertAlmostEqual(integral, 7.5)

        integral = self.gf1.integral(ignore=np.array([False, False, True]))
        self.assertAlmostEqual(integral, 2.0)

        integral = self.gf1.integral(func=lambda x: x * x, ignore=np.array([True, False, False]))
        self.assertAlmostEqual(integral, 6.5)

    def test_integral_voronoi(self):
        integral = self.gf1.integral_voronoi(atoms=[1, 2])
        self.assertAlmostEqual(integral, [2., 1.5])

        integral = self.gf1.integral_voronoi(atoms=[1, 2], func=lambda x: x * x)
        self.assertAlmostEqual(integral, [3., 4.5])

    def test_get_values(self):
        vals = self.gf1.get_values()
        self.assertEqual(vals.shape, (1, 3, 1))


class TestGridFunction2D(PyAdfTestCase):

    def setUp(self):
        grid = MockGrid()

        values1 = np.array([[1., 2., 3.], [1., 1., 1.], [4., 5., 6.]])
        self.gf1 = GridFunction2D(grid, values1)

    def test_abssquare(self):
        ref_values = np.array([14., 3., 77.])
        abssqr = self.gf1.abssquare()
        self.assertAlmostEqual(abssqr.values, ref_values)

    def test_filter_volume(self):
        ref_values = np.array([[1., 2., 3.], [0., 0., 0.], [0., 0., 0.]])
        res = self.gf1.filter_volume(lambda x: x[0] + x[1] + x[2] < 2.0)
        self.assertAlmostEqual(res.values, ref_values)

    def test_get_values(self):
        vals = self.gf1.get_values()
        self.assertEqual(vals.shape, (1, 3, 1, 3))


class TestGridFunctionFactory(PyAdfTestCase):

    def setUp(self):
        self.grid = MockGrid()

        self.values_1d = np.array([1., 2., 3.])
        self.values_2d = np.array([[1., 2., 3.], [1., 1., 1.], [4., 5., 6.]])
        self.values_3d = np.array([[[1., 2.], [3., 4.]],
                                      [[5., 6.], [7., 8.]],
                                      [[2., 2.], [3., 3.]]])

    def test_newGridFunction1D(self):
        gf = GridFunctionFactory.newGridFunction(self.grid, self.values_1d)
        self.assertIsInstance(gf, GridFunction1D)
        self.assertNotIsInstance(gf, GridFunctionDensity)
        self.assertNotIsInstance(gf, GridFunctionPotential)

    def test_newGridFunctionPotential(self):
        gf = GridFunctionFactory.newGridFunction(self.grid, self.values_1d, gf_type='potential')
        self.assertIsInstance(gf, GridFunctionPotential)

    def test_newGridFunctionDensity(self):
        gf = GridFunctionFactory.newGridFunction(self.grid, self.values_1d, gf_type='density')
        self.assertIsInstance(gf, GridFunctionDensity)

    def test_newGridFunction2D(self):
        gf = GridFunctionFactory.newGridFunction(self.grid, self.values_2d)
        self.assertIsInstance(gf, GridFunction2D)

    def test_newGridFunction(self):
        gf = GridFunctionFactory.newGridFunction(self.grid, self.values_3d)
        self.assertIsInstance(gf, GridFunction)
        self.assertNotIsInstance(gf, GridFunction1D)
        self.assertNotIsInstance(gf, GridFunction2D)


class TestGridFunctionOperatorsResultTypes(PyAdfTestCase):

    def setUp(self):
        grid = MockGrid()

        values_1d = np.array([1., 2., 3.])
        values_2d = np.array([[1., 2., 3.], [1., 1., 1.], [4., 5., 6.]])
        values_3d = np.array([[[1., 2.], [3., 4.]],
                                 [[5., 6.], [7., 8.]],
                                 [[2., 2.], [3., 3.]]])

        self.gf = GridFunction(grid, values_3d)
        self.gf_1d = GridFunction1D(grid, values_1d)
        self.gf_2d = GridFunction2D(grid, values_2d)
        self.gf_pot = GridFunctionPotential(grid, values_1d)
        self.gf_dens = GridFunctionDensity(grid, values_1d)

    def test_resultTypeGf(self):
        res = self.gf + self.gf
        self.assertIsInstance(res, GridFunction)
        self.assertNotIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunction2D)

        res = self.gf * 2.0
        self.assertIsInstance(res, GridFunction)
        self.assertNotIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunction2D)

    def test_resultType1D(self):
        res = self.gf_1d + self.gf_1d
        self.assertIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunctionDensity)
        self.assertNotIsInstance(res, GridFunctionPotential)

        res = self.gf_1d + 1.0
        self.assertIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunctionDensity)
        self.assertNotIsInstance(res, GridFunctionPotential)

    def test_resultType2D(self):
        res = self.gf_2d + self.gf_2d
        self.assertIsInstance(res, GridFunction2D)

        res = self.gf_2d * 2.0
        self.assertIsInstance(res, GridFunction2D)

        res = self.gf_2d.abssquare()
        self.assertIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunctionDensity)
        self.assertNotIsInstance(res, GridFunctionPotential)

    def test_resultTypePot(self):
        res = self.gf_pot + self.gf_pot
        self.assertIsInstance(res, GridFunctionPotential)

        res = self.gf_pot + 1.0
        self.assertIsInstance(res, GridFunctionPotential)

        res = self.gf_pot + self.gf_1d
        self.assertIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunctionDensity)
        self.assertNotIsInstance(res, GridFunctionPotential)

        res = self.gf_1d + self.gf_pot
        self.assertIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunctionDensity)
        self.assertNotIsInstance(res, GridFunctionPotential)

        res = self.gf_pot + self.gf_dens
        self.assertIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunctionDensity)
        self.assertNotIsInstance(res, GridFunctionPotential)

    def test_resultTypeDens(self):
        res = self.gf_dens + self.gf_dens
        self.assertIsInstance(res, GridFunctionDensity)

        res = self.gf_dens + 1.0
        self.assertIsInstance(res, GridFunctionDensity)

        res = self.gf_dens + self.gf_1d
        self.assertIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunctionDensity)
        self.assertNotIsInstance(res, GridFunctionPotential)

        res = self.gf_1d + self.gf_dens
        self.assertIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunctionDensity)
        self.assertNotIsInstance(res, GridFunctionPotential)

        res = self.gf_dens + self.gf_pot
        self.assertIsInstance(res, GridFunction1D)
        self.assertNotIsInstance(res, GridFunctionDensity)
        self.assertNotIsInstance(res, GridFunctionPotential)


# noinspection PyTypeChecker,PyUnresolvedReferences
class TestGridFunctionContainer(PyAdfTestCase):

    def setUp(self):
        grid = MockGrid()

        values_1a = np.array([1., 2., 3.])
        self.gf1a = GridFunction1D(grid, values_1a)
        values_1b = np.array([4., 5., 6.])
        self.gf1b = GridFunction1D(grid, values_1b)

        self.gf_container_1 = GridFunctionContainer([self.gf1a, self.gf1b])

        values_2a = np.array([2., 2., 2.])
        self.gf2a = GridFunction1D(grid, values_2a)
        values_2b = np.array([3., 4., -3.])
        self.gf2b = GridFunction1D(grid, values_2b)

        self.gf_container_2 = GridFunctionContainer([self.gf2a, self.gf2b])

    def test_init_getitem(self):
        ref_values_1a = np.array([1., 2., 3.])
        ref_values_1b = np.array([4., 5., 6.])

        self.assertAlmostEqual(self.gf_container_1[0].values, ref_values_1a)
        self.assertAlmostEqual(self.gf_container_1[1].values, ref_values_1b)

    def test_add_constant(self):
        res_container = self.gf_container_1 + 1.0

        ref_values_a = np.array([2., 3., 4.])
        ref_values_b = np.array([5., 6., 7.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_add_constant_rev(self):
        res_container = 1.0 + self.gf_container_1

        ref_values_a = np.array([2., 3., 4.])
        ref_values_b = np.array([5., 6., 7.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_add(self):
        res_container = self.gf_container_1 + self.gf_container_2

        ref_values_a = np.array([3., 4., 5.])
        ref_values_b = np.array([7., 9., 3.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_sub_constant(self):
        res_container = self.gf_container_1 - 1.0

        ref_values_a = np.array([0., 1., 2.])
        ref_values_b = np.array([3., 4., 5.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_sub_constant_rev(self):
        res_container = 1.0 - self.gf_container_1

        ref_values_a = np.array([0., -1., -2.])
        ref_values_b = np.array([-3., -4., -5.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_sub(self):
        res_container = self.gf_container_1 - self.gf_container_2

        ref_values_a = np.array([-1., 0., 1.])
        ref_values_b = np.array([1., 1., 9.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_mul_scalar(self):
        res_container = self.gf_container_1 * 2.0

        ref_values_a = np.array([2., 4., 6.])
        ref_values_b = np.array([8., 10., 12.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_mul_scalar_rev(self):
        res_container = 2.0 * self.gf_container_1

        ref_values_a = np.array([2., 4., 6.])
        ref_values_b = np.array([8., 10., 12.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_mul(self):
        res_container = self.gf_container_1 * self.gf_container_2

        ref_values_a = np.array([2., 4., 6.])
        ref_values_b = np.array([12., 20., -18.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_div(self):
        res_container = self.gf_container_1 / self.gf_container_2

        ref_values_a = np.array([0.5, 1., 1.5])
        ref_values_b = np.array([4. / 3., 1.25, -2.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_div_scalar(self):
        res_container = self.gf_container_1 / 4.0

        ref_values_a = np.array([0.25, .5, 0.75])
        ref_values_b = np.array([1., 1.25, 1.5])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_div_scalar_rev(self):
        res_container = 2.0 / self.gf_container_2

        ref_values_a = np.array([1., 1., 1.])
        ref_values_b = np.array([2. / 3., 0.5, -2. / 3.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_pow(self):
        res_container = self.gf_container_1**2.

        ref_values_a = np.array([1., 4., 9.])
        ref_values_b = np.array([16., 25., 36.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_apply_function(self):
        res_container = self.gf_container_1.apply_function(lambda x: x * x)

        ref_values_a = np.array([1., 4., 9.])
        ref_values_b = np.array([16., 25., 36.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_filter_positive(self):
        res_container = self.gf_container_2.filter_positive()

        ref_values_a = np.array([2., 2., 2.])
        ref_values_b = np.array([3., 4., 0.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_filter_negative(self):
        res_container = self.gf_container_2.filter_negative()

        ref_values_a = np.array([0., 0., 0.])
        ref_values_b = np.array([0., 0., -3.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_filter_zeros(self):
        res_container = self.gf_container_2.filter_zeros(thresh=3.5)

        ref_values_a = np.array([3.5, 3.5, 3.5])
        ref_values_b = np.array([3.5, 4., -3.5])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_add_gf(self):
        res_container = self.gf_container_1 + self.gf2b

        ref_values_a = np.array([4., 6., 0.])
        ref_values_b = np.array([7., 9., 3.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_add_gf_rev(self):
        res_container = self.gf2b + self.gf_container_1

        ref_values_a = np.array([4., 6., 0.])
        ref_values_b = np.array([7., 9., 3.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_sub_gf(self):
        res_container = self.gf_container_1 - self.gf2b

        ref_values_a = np.array([-2., -2., 6.])
        ref_values_b = np.array([1., 1., 9.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)

    def test_sub_gf_rev(self):
        res_container = self.gf2b - self.gf_container_1

        ref_values_a = np.array([2., 2., -6.])
        ref_values_b = np.array([-1., -1., -9.])
        self.assertAlmostEqual(res_container[0].values, ref_values_a)
        self.assertAlmostEqual(res_container[1].values, ref_values_b)


if __name__ == '__main__':
    unittest.main()
