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
"""
Defines classes for functions defined on grids.
"""

import numbers
import numpy as np

from pyadf.Errors import PyAdfError

from . import Grids


class GridFunction:
    """
    Basic class for grid functions consisting of a grid and values on this grid.
    """

    def __init__(self, grid, values, checksum=None):
        """
        Initialize a gridfunction.

        A gridfunction consists of a grid and the corresponding values, which
        are stored in a Numpy array. The first dimension contains the grid points,
        and there can be an arbitrary number of additional dimensions here.

        @param grid: the grid
        @type grid: subclass of C{Grids.grid}

        @param values: the values, as described above. The values are read-only
            and should not (and cannot) be changed. (This is done to ensure
            consistency of the data and checksum).
        @type values: Numpy array

        @param checksum:
            checksum of the data, ideally generated from the input used to generate
            the data. If not given, a checksum is generated from the Numpy data.
            Using a checksum generated from the input is preferred, because otherwise
            the checksum will depend on numerical noise in the calculations.
        @type checksum: str
        """
        self.grid = grid
        self._values = values
        self._checksum = checksum

        self.prop = None
        self.type = None

        # FIXME: remove
        self.nspin = 1

        # consistency check: first dimension of Numpy array values must match
        # the number of grid points

        if not (self._values.shape[0] == self.grid.npoints):
            raise PyAdfError("Grid and Values not consistent")

    @property
    def checksum(self):
        """
        Checksum of the gridfunction.

        Read-only, because it should either be set when constructing a
        gridfunction, or calculated automatically from the Numpy data.
        """
        if self._checksum is None:
            import hashlib
            m = hashlib.md5()
            m.update(self.grid.checksum.encode('utf-8'))
            m.update(self._values.data)

            self._checksum = m.hexdigest()

        return self._checksum

    def get_values(self):
        """
        Returns the values of the density on the grid.

        Depending on the type of the grid, the values are given
        in a one-dimensional or multi-dimensional array (for cubegrid).

        @rtype: Numpy array of float
        """
        new_shape = self.grid.shape + self._values.shape[1:]
        return self._values.reshape(new_shape)

    @property
    def values(self):
        """
        The values of the gridfunction. Read-only.

        In contrast to get_values, the values are given in a
        Numpy array with the first dimension matching the
        number of grid points.
        """
        return self._values

    def valueiter(self):
        """
        Returns an iterator over the values of the gridfunction in the grid points.

        The order of these values is the same as in the
        grid point iterators of L{Grids.grid} and its subclasses,
        see L{Grids.grid.coorditer} and L{Grids.grid.weightiter}.

        @exampleuse:

            Print coordinates of all grid points where the density is larger than 1.0

            >>> # 'gf' is an instance of GridFunction
            >>>
            >>> for coord, val in zip(gf.grid.coorditer(), gf.valueiter()) :
            ...     if val > 1.0 :
            ...         print "Grid point: ", coord

        @rtype: iterator
        """
        return self._values.__iter__()

    def _result_type_for_operators(self, other):
        return None

    def _result_gridfunction_for_operators(self, new_values, checksum, other=None):
        if other is None:
            gf_type = self.type
        else:
            gf_type = self._result_type_for_operators(other)
        return GridFunctionFactory.newGridFunction(self.grid, new_values, checksum, gf_type)

    def _check_match_for_operators(self, other):

        # first check that the grid is the same
        if not ((self.grid is other.grid) or (self.grid.checksum == other.grid.checksum)):
            raise PyAdfError('grids have to be the same for binary operation on gridfunctions')

        if not (self.values.shape == other.values.shape):
            raise PyAdfError('gridfunctions must have the same dimensions for binary operation')

    def _add_constant(self, const):

        new_values = self.values + const

        # calculate checksum for added density
        import hashlib
        m = hashlib.md5()
        m.update(b"Grid function obtained by adding :\n")
        m.update(self.checksum.encode('utf-8'))
        m.update(f"Constant shift {const:18.10f}".encode('utf-8'))

        return self._result_gridfunction_for_operators(new_values, m.hexdigest())

    def _add_with_factor(self, other, fact=1.0):

        self._check_match_for_operators(other)

        new_values = self.values + fact * other.values

        # calculate checksum for added density
        import hashlib
        m = hashlib.md5()
        m.update(b"Grid function obtained by adding :\n")
        m.update(self.checksum.encode('utf-8'))
        m.update(f"with factor {fact:18.10f} times \n".encode('utf-8'))
        m.update(other.checksum.encode('utf-8'))

        return self._result_gridfunction_for_operators(new_values, m.hexdigest(), other)

    def __add__(self, other):
        """
        Addition of two grid functions.

        For two instances of C{GridFunction}, an addition
        is defined. The result of such an addition is another
        instance, which contains the sum of the two gridfunctions.

        The two instances must use the same grid.

        @exampleuse:

            Addition of grid functions.

            >>> # 'dens1' is a GridFunction associated with the density of fragment 1
            >>> # 'dens2' is a GridFunction associated with the density of fragment 2
            >>>
            >>> dens_tot = dens1 + dens2
            >>> dens_tot.get_cubfile('total_density.cub')

        """
        if isinstance(other, numbers.Complex):
            return self._add_constant(other)
        elif isinstance(other, GridFunction):
            return self._add_with_factor(other)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction of two grid functions.

        For two instances of C{GridFunction}, a subtraction
        is defined. The result of such a subtraction is another
        instance, which contains the difference of the two gridfunctions.

        The two instances must use the same grid.

        @exampleuse:

            Calculation of a difference density.

            >>> # 'dens1' is a GridFunction1D associated with the exact density
            >>> # 'dens2' is a GridFunction1D associated with an approximate density (e.g. FDE)
            >>>
            >>> diffdens = dens1 - dens2
            >>> diffdens.get_cubfile('difference_density.cub')
            >>> print "RMS density deviation: ", math.sqrt(diffdens.integral(lambda x: x*x))

        """
        if isinstance(other, numbers.Complex):
            return self._add_constant(-other)
        elif isinstance(other, GridFunction):
            return self._add_with_factor(other, fact=-1.0)
        else:
            return NotImplemented

    def __rsub__(self, other):
        """
        Calculate other - self.
        """
        return -self.__sub__(other)

    def _mul_with_constant(self, fact):
        """
        Multiply gridfunction with a constant factor/

        @fact: factor
        @type fact: L{Number}
        """
        new_values = self.values * fact

        # calculate checksum for added density
        import hashlib
        m = hashlib.md5()
        m.update(b"Grid function obtained by multiplying with factor :\n")
        m.update(self.checksum.encode('utf-8'))
        m.update(str(fact).encode('utf-8'))

        return self._result_gridfunction_for_operators(new_values, m.hexdigest())

    def __mul__(self, other):
        """
        Multiplication of gridfunctions.

        For multiplication with a single float, all values are scaled
        accordingly.

        For two instances of C{GridFunction1D}, a multiplication
        is defined pointwise. The two instances must use the same grid.
        """
        if isinstance(other, numbers.Complex):
            return self._mul_with_constant(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self._mul_with_constant(-1.0)

    def __truediv__(self, other):
        """
        Division of gridfunctions.

        For multiplication with a single float, all values are scaled
        accordingly.

        For two instances of C{GridFunction1D}, a division
        is defined pointwise. The two instances must use the same grid.
        """
        if isinstance(other, numbers.Complex):
            return self._mul_with_constant(1.0 / other)
        else:
            return NotImplemented

    def __pow__(self, exp):

        new_values = self.values**exp

        import hashlib
        m = hashlib.md5()
        m.update(b"Density obtained by taking\n")
        m.update(self.checksum.encode('utf-8'))
        m.update(b"to the power of \n")
        m.update(str(exp).encode('utf-8'))

        return self._result_gridfunction_for_operators(new_values, m.hexdigest())

    def apply_function(self, func):
        """
        Apply a function to the values on each grid point.
        """

        # test shape resulting from the function
        elem = func(self.values[0, ...])

        if isinstance(elem, np.ndarray):
            new_values = np.empty((self.values.shape[0],) + elem.shape)
        else:
            new_values = np.empty((self.values.shape[0],))

        # now acually apply the function for each grid point
        for i in range(self.values.shape[0]):
            new_values[i, ...] = func(self.values[i, ...])

        # calculate checksum for result gridfunction
        import hashlib
        m = hashlib.md5()
        m.update(b"Grid function obtained by applying function :\n")
        m.update(self.checksum.encode('utf-8'))

        if isinstance(func, np.ufunc):
            m.update(func.__name__.encode('utf-8'))
        else:
            # func.__code__.co_code contains the bytecode of the function
            m.update(func.__code__.co_code)

        return self._result_gridfunction_for_operators(new_values, m.hexdigest())

    def filter_volume(self, involume):
        """
        Set all values outside a given volume to zero.

        @param involume:
            a function of coordinates defining a box or volume, for which values are
            written to the cubfile. Returns a boolean. For example:
            >>> def withinradius(coord) :
            ...     geom_center = [2.0, 4.0, 3.5]
            ...     radius = coord-geom_center
            ...     rad_abs = math.sqrt(radius[0]**2 + radius[1]**2 + radius[2]**2)
            ...     coord_within_radius = rad_abs < 5.0
            ...     return coord_within_radius

        @type involume : function
        """
        new_values = self.values.copy()
        for i, c in enumerate(self.grid.coorditer()):
            if not involume(c):
                new_values[i, ...] = 0.0

        # calculate checksum for result gridfunction
        import hashlib
        m = hashlib.md5()
        m.update(b"Grid function obtained by applying function :\n")
        m.update(self.checksum.encode('utf-8'))

        if isinstance(involume, np.ufunc):
            m.update(involume.__name__.encode('utf-8'))
        else:
            # func.__code__.co_code contains the bytecode of the function
            m.update(involume.__code__.co_code)

        return self._result_gridfunction_for_operators(new_values, m.hexdigest())


class GridFunction1D(GridFunction):
    """
    Class for a 1D gridfunction, i.e., one value per grid point.
    """

    def __init__(self, grid, values, checksum=None):
        if not (len(values.shape) == 1):
            raise PyAdfError("Wrong shape of values for GridFunction1D")

        super().__init__(grid, values, checksum)

    def filter_negative(self, thresh=0.0):

        new_values = np.minimum(self.values, [thresh])

        # calculate checksum for negative density
        import hashlib
        m = hashlib.md5()
        m.update(b"Grid function keeping only negative values\n")
        m.update(f"thresh {thresh:18.8f} :\n".encode('utf-8'))
        m.update(self.checksum.encode('utf-8'))

        return self._result_gridfunction_for_operators(new_values, m.hexdigest())

    def filter_positive(self, thresh=0.0):

        new_values = np.maximum(self.values, [thresh])

        # calculate checksum for negative density
        import hashlib
        m = hashlib.md5()
        m.update(b"Grid function keeping only positve values \n")
        m.update(f"Thresh {thresh:18.8f} :\n".encode('utf-8'))
        m.update(self.checksum.encode('utf-8'))

        return self._result_gridfunction_for_operators(new_values, m.hexdigest())

    def filter_zeros(self, thresh=1e-4):

        new_values = np.where(np.logical_and(self.values >= 0.0, self.values < thresh),
                                 +thresh, self._values)

        new_values = np.where(np.logical_and(new_values < 0.0, new_values > -thresh),
                                 -thresh, new_values)

        # calculate checksum for negative density
        import hashlib
        m = hashlib.md5()
        m.update(b"Grid function obtained from filter_zeros \n")
        m.update(f"Thresh {thresh:18.8f} :\n".encode('utf-8'))
        m.update(self.checksum.encode('utf-8'))

        return self._result_gridfunction_for_operators(new_values, m.hexdigest())

    def _mul(self, other):
        """
        Pointwise multiplication of grid functions.
        """
        self._check_match_for_operators(other)

        new_values = self.values * other.values

        # calculate checksum for product
        import hashlib
        m = hashlib.md5()
        m.update(b"Grid function obtained by multiplying :\n")
        m.update(self.checksum.encode('utf-8'))
        m.update(other.checksum.encode('utf-8'))

        return self._result_gridfunction_for_operators(new_values, m.hexdigest(), other)

    def __mul__(self, other):
        if isinstance(other, numbers.Complex):
            return self._mul_with_constant(other)
        elif isinstance(other, GridFunction):
            return self._mul(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Complex):
            return self._mul_with_constant(1.0 / other)
        elif isinstance(other, GridFunction):
            return self._mul(other**(-1.0))
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """
        Division other / self.
        """
        return other * (self**(-1.0))

    def integral(self, func=None, ignore=None, involume=None):
        """
        Returns the integral of (a function of) the density.

        This calculates the integral S{integral} f(S{rho}(r)) dr,
        where f is the function given as argument.

        @exampleuse:

            Calculate integral of the square of the density

            >>> # 'dens' is an instance of densfresults
            >>>
            >>> ii = dens.integral(f=lambda x: x*x, ignore=(dens.get_values() < 0))
            >>> print "Integral of the squared density: ", ii

        @param func:
            A function of one variable that is applied to the density
            before the integration. If None, the density is integrated
            directly, i.e., the identity function is used.
        @type  func: function

        @param ignore:
            An array of booleans to exclude certain points
        @type ignore:
            np.ndarray

        @param involume:
            a function of coordinates defining a box or volume over which to integrate,
            returns a boolean, e.g.
            >>> def withinradius(coord) :
            ...     geom_center = [2.0, 4.0, 3.5]
            ...     radius = coord-geom_center
            ...     rad_abs = math.sqrt(radius[0]**2 + radius[1]**2 + radius[2]**2)
            ...     coord_within_radius = rad_abs < 5.0
            ...     return coord_within_radius
        @type involume : function

        @returns: integral of (a function of) the density
        @rtype: float
        """
        if involume:
            filtered_gf = self.filter_volume(involume)
            ii = filtered_gf.integral(func=func, ignore=ignore)
        else:
            if ignore is not None:
                ig = ignore.reshape((ignore.size,))
                w = np.where(ig, 0.0, self.grid.weights)
            else:
                w = self.grid.weights

            if func is not None:
                vfunc = np.vectorize(func)
                v = vfunc(self.values)
            else:
                v = self.values

            ii = np.dot(w, v)

        return ii

    def integral_voronoi(self, atoms, func=None):
        """
        Returns the integral of (a function of) the density over the Voronoi cells
        of a given list of atoms. Intended for analysis of the (difference) density
        per Voronoi cell.

        This calculates the integral S{integral} f(S{rho}(r)) dr,
        where f is the function given as argument.

        @param func:
            A function of one variable that is applied to the density
            before the integration. If None, the density is integrated
            directly, i.e., the identity function is used.
        @type  func: function

        @param atoms: for which atoms (more precisely their Voronoi cells) to perform the
                      analysis (counting starts at 1)
        @type atoms: list of ints

        @returns: integral over density per Voronoi cell
        @rtype: list of floats
        """
        if func is None:
            func = lambda x: x

        vor_int = np.zeros((len(atoms),))

        for w, val, v in zip(self.grid.weightiter(), self.valueiter(), self.grid.voronoiiter()):

            for iatom in range(len(atoms)):
                if atoms[iatom] == v:
                    vor_int[iatom] += w * func(val)

        return vor_int

    # FIXME: This should be moved to Writer
    def get_tape41(self, filename, section, variable):
        from ...kf import kf

        f = kf.kffile(filename)
        self.grid.write_grid_to_t41(f)

        f.writereals(section, variable, self.values)
        f.close()

    def get_cubfile(self, filename, involume=None):
        """
        Obtain a Gaussian-type cube file of the associated density.

        @param filename: The filename of the cube file to be written.
        @type  filename: str

        @param involume:
            a function of coordinates defining a box or volume, for which values are
            written to the cubfile. Returns a boolean. For example:
            >>> def withinradius(coord) :
            ...     geom_center = [2.0, 4.0, 3.5]
            ...     radius = coord-geom_center
            ...     rad_abs = math.sqrt(radius[0]**2 + radius[1]**2 + radius[2]**2)
            ...     coord_within_radius = rad_abs < 5.0
            ...     return coord_within_radius

        @type involume : function

        """
        if involume:
            filtered_gf = self.filter_volume(involume)
            filtered_gf.get_cubfile(filename)
        else:
            from .FileWriters import GridFunctionWriter
            GridFunctionWriter.write_cube(self, filename)

    def get_xyzwvfile(self, filename, bohr=True, endmarker=False, add_comment=True):
        """
        Obtain an XYZWV file of the associated density.

        See L{GridFunctionWriter.write_xyzwv} for details on file format.
        """
        from .FileWriters import GridFunctionWriter
        GridFunctionWriter.write_xyzwv(self, filename, bohr,
                                       endmarker=endmarker, add_comment=add_comment)

    def get_xyzvfile(self, filename, bohr=False, endmarker=False, add_comment=True):
        """
        Obtain an XYZV file of the associated density.

        See L{GridFunctionWriter.write_xyzv} for details on file format.
        """
        from .FileWriters import GridFunctionWriter
        GridFunctionWriter.write_xyzv(self, filename, bohr,
                                      endmarker=endmarker, add_comment=add_comment)

    def interpolate(self, int_grid):
        """
        Convert the 1D gridfunction to another grid using interpolation.

        This method returns the same density, but on another grid.
        For the interpolation, the IMLS algorithm is used (see
        L{interpolation} for details).

        The main purpose of this routine is to obtain a density/potential on
        a L{cubegrid} that is suitable for visualization from one that is
        available only on an L{adfgrid}.

        @param int_grid: the grid to use for the interpolated density
        @type  int_grid: subclass of L{grid}

        @return: the interpolated gridfunction
        @rtype:  L{GridFiunction1D}
        """
        new_values = np.empty((int_grid.npoints,))

        interp = Grids.interpolation(self)
        for i, point in enumerate(int_grid.coorditer()):
            if i % 500 == 0:
                print(f"Interpolating point {i:d} of {int_grid.npoints:d} ")
            new_values[i] = interp.get_value_at_point(point)

        import hashlib
        m = hashlib.md5()
        m.update(b"Interpolated from :\n")
        m.update(self.checksum.encode('utf-8'))
        m.update(b"on grid :\n")
        m.update(int_grid.get_grid_block(True).encode('utf-8'))

        return GridFunction1D(int_grid, new_values, m.hexdigest())

    def get_value_at_point(self, point):
        """
        Get value at one point by interpolation.

        @param point: the point for which the interpolated gridfunction is needed.
        """
        interp = Grids.interpolation(self)
        return interp.get_value_at_point(point)


class GridFunction2D(GridFunction):
    """
    Class for a 2D gridfunction, i.e., one vector per grid point.
    """

    def __init__(self, grid, values, checksum=None):
        if not (len(values.shape) == 2):
            raise PyAdfError("Wrong shape of values for GridFunction1D")

        super().__init__(grid, values, checksum)

    def abssquare(self):
        """
        Return a 1D-grid containing the absolute value squared of the vector at each grid point.
        """
        new_values = np.empty((self.values.shape[0],))
        for i in range(self.values.shape[0]):
            new_values[i] = np.dot(self.values[i, :], self.values[i, :])

        import hashlib
        m = hashlib.md5()
        m.update(b"Density obtained by taking absulte value squared of\n")
        m.update(self.checksum.encode('utf-8'))

        return self._result_gridfunction_for_operators(new_values, m.hexdigest())


class GridFunctionDensity(GridFunction1D):
    """
    Class for densities as special case of 1D grid functions.
    """

    def __init__(self, grid, values, checksum=None):
        super().__init__(grid, values, checksum)
        self.type = 'density'

    def _result_type_for_operators(self, other):
        if self.type == other.type:
            gf_type = self.type
        else:
            gf_type = None
        return gf_type

    def get_xsffile(self, filename):
        """
        Obtain a XSF file of the associated density.

        @attention: Density values in the file are in atomic units!

        @param filename: The filename of the xsf file to be written.
        @type  filename: str
        """
        from .FileWriters import GridFunctionWriter
        GridFunctionWriter.write_xsf(self, filename)

    def get_electronic_dipole_moment_grid(self, involume=None):
        """
        Gets a (local) electronic dipole moment in atomic units by calculating the integral
        S{integral} S{rho}(r)*(x,y,z) dr
        for a given volume. Nuclear contribution in class molecule.
        Caution: inaccuracies introduced by numerical integration introduce an origin
        dependence of the total dipole moment (and a mismatch with the ADF results).
        One possible solution is to use the integral of the
        density to renormalize the dipole moments.

        @param involume:
            a function of coordinates defining a box or volume over which to integrate,
            returns a boolean
        @type involume : function

        @returns: electronic dipole moment (complete molecule or part of it)
        @rtype: np.array of floats
        """
        if involume:
            filtered_gf = self.filter_volume(involume)
            dipole = filtered_gf.get_electronic_dipole_moment_grid()
        else:
            ii_x = ii_y = ii_z = 0.0
            for w, val, c in zip(self.grid.weightiter(), self.valueiter(),
                                 self.grid.coorditer(bohr=True)):
                ii_x += -w * val * c[0]
                ii_y += -w * val * c[1]
                ii_z += -w * val * c[2]

            dipole = np.array([ii_x, ii_y, ii_z])

        return dipole

    def get_electronic_dipole_voronoi(self, atoms):
        """
        Gets a (local) electronic dipole moment in atomic units for Voronoi cells.

        This is done by calculating the integral S{integral} S{rho}(r)*(x,y,z) dr
        for the Voronoi cell around a given atom (list). Nuclear contribution in class molecule.
        Caution: inaccuracies introduced by numerical integration introduce an origin
        dependence of the total dipole moment (and a mismatch with the ADF results).
        One possible solution is to use the integral of the total
        density to renormalize the dipole moments.

        @param atoms:
            atom numbers
        @type atoms:
            list of integers
        @returns: electronic dipole moment (per Voronoi cell)
        @rtype: list of np.array of floats
        """
        ii_x = ii_y = ii_z = 0.0

        voronoidip = []
        for iatom in range(len(atoms)):
            voronoidip.append([ii_x, ii_y, ii_z])

        for w, val, c, v in zip(self.grid.weightiter(), self.valueiter(),
                                self.grid.coorditer(bohr=True), self.grid.voronoiiter()):

            for iatom in range(len(atoms)):
                if atoms[iatom] == v:
                    voronoidip[iatom][0] += -w * val * c[0]
                    voronoidip[iatom][1] += -w * val * c[1]
                    voronoidip[iatom][2] += -w * val * c[2]

        lvoronoidip = []
        for iatom in range(len(atoms)):
            lvoronoidip.append(np.array(voronoidip[iatom]))

        return lvoronoidip

    def get_efield_in_point_grid(self, pointcoord):
        """
        Calculates the electronic contribution to the electric field in a point in
        atomic units. Nuclear contribution in class molecule.
        This subroutine employs the density on the grid, preferably the potential
        should be used to calculate the electronic contribution.

        @param pointcoord:
        @type pointcoord: array of float, in Angstrom coordinates

        @returns: the electronic contribution to the electric field in atomic units
        @rtype: np.array of float

        """
        e_x = e_y = e_z = 0.0
        for w, val, c in zip(self.grid.weightiter(), self.valueiter(), self.grid.coorditer(bohr=True)):
            dist = np.sqrt((c[0] - pointcoord[0])**2
                              + (c[1] - pointcoord[1])**2
                              + (c[2] - pointcoord[2])**2)
            e_x += - w * val * (c[0] - pointcoord[0]) / dist**3
            e_y += - w * val * (c[1] - pointcoord[1]) / dist**3
            e_z += - w * val * (c[2] - pointcoord[2]) / dist**3

        return np.array([e_x, e_y, e_z])


class GridFunctionPotential(GridFunction1D):
    """
    Class for potentials as special case of 1D grid functions.
    """

    def __init__(self, grid, values, checksum=None):
        super().__init__(grid, values, checksum)
        self.type = 'potential'

    def _result_type_for_operators(self, other):
        if self.type == other.type:
            gf_type = self.type
        else:
            gf_type = None
        return gf_type


class _GridFunctionContainerMetaclass(type):
    """
    Meta class to handle the delegation in GridFunctionContainer.
    """

    @staticmethod
    def delegate_unary(methodname):
        def _delegate_unary(self, *args, **kwargs):
            res = []
            for gf in self.wrapped:
                method = getattr(gf, methodname)
                res.append(method(*args, **kwargs))
            return self.__class__(res)

        return _delegate_unary

    @staticmethod
    def delegate_binary(methodname):
        def _delegate_binary(self, other, *args, **kwargs):
            res = []
            if isinstance(other, GridFunctionContainer):
                for gf1, gf2 in zip(self.wrapped, other.wrapped):
                    method = getattr(gf1, methodname)
                    rr = method(gf2, *args, **kwargs)
                    if rr is NotImplemented:
                        return NotImplemented
                    else:
                        res.append(rr)
            else:
                for gf in self.wrapped:
                    method = getattr(gf, methodname)
                    rr = method(other, *args, **kwargs)
                    if rr is NotImplemented:
                        return NotImplemented
                    else:
                        res.append(rr)
            return self.__class__(res)

        return _delegate_binary

    def __new__(mcs, name, bases, attrs):

        unary_delegations = ['__pow__', 'apply_function', 'filter_positive',
                             'filter_negative', 'filter_zeros', 'abssquare']
        for methodname in unary_delegations:
            attrs[methodname] = mcs.delegate_unary(methodname)

        binary_delegations = ['__add__', '__radd__', '__sub__', '__rsub__',
                              '__mul__', '__rmul__', '__truediv__', '__rtruediv__']
        for methodname in binary_delegations:
            attrs[methodname] = mcs.delegate_binary(methodname)

        return type(name, bases, attrs)


class GridFunctionContainer(metaclass=_GridFunctionContainerMetaclass):
    """
    GridFunctionContainers wrap several grid functions on the same grid into one.
    """

    def __init__(self, wrapped_gfs, checksum=None, gf_type=None):
        f"""
        @param wrapped_gfs: a list of the gridfunctions to wrap into this container
        """
        self.grid = wrapped_gfs[0].grid
        self.wrapped = wrapped_gfs

        for gf in wrapped_gfs:
            if not (isinstance(gf, GridFunction) or isinstance(gf, GridFunctionContainer)):
                raise PyAdfError('Only GridFunctions can be wrapped into GradFunctionContainer')
            if gf.grid is not wrapped_gfs[0].grid:
                raise PyAdfError('GridFunctions in container must use the same grid')

        self._checksum = checksum
        self.type = gf_type

    def __getitem__(self, key):
        return self.wrapped[key]

    @property
    def checksum(self):
        if self._checksum is None:
            import hashlib
            m = hashlib.md5()
            for gf in self.wrapped:
                m.update(gf.checksum.encode('utf-8'))
            self._checksum = m.hexdigest()

        return self._checksum


class GridFunctionUnrestricted(GridFunctionContainer):
    """
    Unrestricted gridfunctions contain both alpha and beta values.
    """

    def __init__(self, wrapped_gfs, checksum=None, gf_type=None):
        if not len(wrapped_gfs) == 2:
            raise PyAdfError('Exactly two grid functions needed for GridFunctionUnrestricted')
        super().__init__(wrapped_gfs, checksum, gf_type)

        # FIXME: remove
        self.nspin = 2

    def __getitem__(self, key):
        if key == 'alpha':
            return self.wrapped[0]
        elif key == 'beta':
            return self.wrapped[1]
        elif key == 'tot':
            return self.wrapped[0] + self.wrapped[1]
        elif key == 'spin':
            return self.wrapped[0] - self.wrapped[1]
        else:
            raise KeyError

    @property
    def values(self):
        vals = np.empty(self.wrapped[0].values.shape + (2,))
        vals[..., 0] = self.wrapped[0].values
        vals[..., 1] = self.wrapped[1].values

        return vals


class GridFunctionDensityWithDerivatives(GridFunctionContainer):
    """
    Container for density, density gradient, and density Laplacian.
    """

    def __init__(self, wrapped_gfs, checksum=None):
        if len(wrapped_gfs) <= 1:
            raise PyAdfError("No density derivatives provided")

        self.order = len(wrapped_gfs)

        super().__init__(wrapped_gfs, checksum, gf_type="density")


class GridFunctionFactory:

    @classmethod
    def newGridFunction(cls, grid, values, checksum=None, gf_type=None):
        if gf_type == 'potential':
            if not len(values.shape) == 1:
                raise PyAdfError('Values for GridFunction with type potential must be 1D')
            new_gf = GridFunctionPotential(grid, values, checksum)
        elif gf_type == 'density':
            if not len(values.shape) == 1:
                raise PyAdfError('Values for GridFunction with type density must be 1D')
            new_gf = GridFunctionDensity(grid, values, checksum)
        elif len(values.shape) == 1:
            new_gf = GridFunction1D(grid, values, checksum)
        elif len(values.shape) == 2:
            new_gf = GridFunction2D(grid, values, checksum)
        else:
            new_gf = GridFunction(grid, values, checksum)
        return new_gf

    @classmethod
    def newGridFunctionUnrestricted(cls, grid, values_alpha, values_beta,
                                    checksum=None, gf_type=None):
        if checksum is not None:
            import hashlib
            ma = hashlib.md5()
            ma.update(b"Alpha grid function:")
            ma.update(checksum.encode('utf-8'))
            checksum_a = ma.hexdigest()

            mb = hashlib.md5()
            mb.update(b"Beta grid function:")
            mb.update(checksum.encode('utf-8'))
            checksum_b = ma.hexdigest()
        else:
            checksum_a = checksum_b = None

        gf_a = GridFunctionFactory.newGridFunction(grid, values_alpha, checksum_a, gf_type)
        gf_b = GridFunctionFactory.newGridFunction(grid, values_beta, checksum_b, gf_type)

        new_gf = GridFunctionUnrestricted([gf_a, gf_b], checksum=checksum, gf_type=gf_type)

        return new_gf

    @classmethod
    def newGridFunctionFromFile(cls, filename, file_format=None, gf_type=None):
        import os.path
        from .FileReaders import GridFunctionReader

        if file_format is None:
            fmt = os.path.splitext(filename)[1]
        else:
            fmt = file_format

        if fmt == 'xyzwv':
            return GridFunctionReader.read_xyzwv(filename, gf_type=gf_type)
        else:
            raise PyAdfError('Unknown file type for reading gridfunction')
