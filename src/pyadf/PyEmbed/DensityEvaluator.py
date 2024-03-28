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
 Functionality for evaluating densities and potentials on Grids.

 @author: Kevin Focke
"""

from abc import ABCMeta, abstractmethod
import functools

from pyadf.BaseJob import results


# this method is a decorator for the following get_* functions in DensityEvaluatorInterface
def use_default_grid(func):
    """
    Decorator to ensure that get_* class method use the proper default grid.
    """

    @functools.wraps(func)
    def _wrapper(self, grid=None, spacing=0.5, *args, **kwargs):
        if grid is None:
            grid = self.get_default_grid(spacing)
        return func(self, grid=grid, *args, **kwargs)

    return _wrapper


class DensityEvaluatorInterface(results, metaclass=ABCMeta):
    """
    This class defines a minimal interface for results classes that give
    access to densities and potentials, as needed for PyEmbed.
    """

    def __init__(self, j=None):
        """
        Constructor for results class.

        @param j: L{job} object of the corresponding job
        """
        super().__init__(j)
        self._grid = None
        self._default_grids = {}

    @property
    def grid(self):
        return self._grid

    def get_default_grid(self, spacing=0.5):
        from .Plot.Grids import cubegrid

        if spacing not in self._default_grids:
            self._default_grids[spacing] = cubegrid(self.get_molecule(), spacing)
        return self._default_grids[spacing]

    @use_default_grid
    def get_density(self, grid=None, order=None, *args, **kwargs):
        """
        Create a density grid function.
        Takes either a raw grid or spacing or nothing at all.

        @param grid:  The grid to use. Optional. For details see L{Plot.Grids}.
        @type grid:   subclass of L{grid}
        @param order: order of derivates of the density to calculate (1 and 2 possible)
        @type order:  int
        @returns:     A density grid function.
        @rtype:       L{GridFunctionDensity}
        """
        from .Plot.GridFunctions import GridFunctionDensityWithDerivatives

        if (order is None) or (order == 0):
            return self._get_density(grid, *args, **kwargs)
        elif (order == 1) or (order == 2):
            gfs = [self._get_density(grid, *args, **kwargs)]

            # density gradient
            if order >= 1:
                dg = self.get_densgradient(grid, *args, **kwargs)
                gfs.append(dg)

            # density Hessian
            if order >= 2:
                dh = self.get_density_hessian(grid, *args, **kwargs)
                gfs.append(dh)

            return GridFunctionDensityWithDerivatives(gfs)

        else:
            raise NotImplementedError

    @abstractmethod
    @use_default_grid
    def _get_density(self, grid=None):
        """
        Returns the electron density (without derivatives).

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunctionDensity}
        """
        raise NotImplementedError

    @abstractmethod
    @use_default_grid
    def get_densgradient(self, grid=None):
        """
        Returns the gradient of the electron density.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunction2D}
        """
        raise NotImplementedError

    @use_default_grid
    def get_sqrgradient(self, grid=None, *args, **kwargs):
        """
        Returns the squared gradient of the electron density.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunction1D}
        """
        densgrad = self.get_densgradient(grid, *args, **kwargs)
        return densgrad.abssquare()

    @abstractmethod
    @use_default_grid
    def get_density_hessian(self, grid=None):
        """
        Returns the Hessian of the electron density.

        For details on the processing of the Laplacian,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunction2D}
        """
        raise NotImplementedError

    @use_default_grid
    def get_laplacian(self, grid=None, *args, **kwargs):
        """
        Returns the Laplacian of the electron density.

        For details on the processing of the Laplacian,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunction1D}
        """
        from .Plot.GridFunctions import GridFunctionFactory

        denshess = self.get_density_hessian(grid, *args, **kwargs)

        lapl_values = denshess.values[:, 0] + denshess.values[:, 3] + denshess.values[:, 5]

        import hashlib
        m = hashlib.md5()
        m.update(b"Laplacian obtained from density Hessian: \n")
        m.update(denshess.checksum.encode('utf-8'))

        return GridFunctionFactory.newGridFunction(grid, lapl_values, checksum=m.hexdigest(), gf_type='density')

    @use_default_grid
    def get_potential(self, grid=None, pot=None):
        """
        Returns the total potential or one of its components.

        For PyEmbed, the nuclear and Coulomb potentials are required,
        other types of potentials might also be implemented.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param pot: Which potential to calculate. Options include:
                    - 'nuc' for the nuclear potential
                    - 'coul' for the Coulomb potential
                    - 'elstat' for the total electrostatic potential, i.e., sum of nuc and coul
                    - 'xc' for the XC potential (optional, not needed for PyEmbed)
                    - 'total' for the total KS potential, i.e, sum of nuc, coul, and xc
                      (optional, not needed for PyEmbed)
        @type  pot: str

        @rtype: L{GridFunctionPotential}
        """
        if pot == 'nuc':
            return self._get_nuclear_potential(grid)
        elif pot == 'coul':
            return self._get_coulomb_potential(grid)
        elif pot == 'elstat':
            return self._get_nuclear_potential(grid) + self._get_coulomb_potential(grid)
        else:
            raise NotImplementedError('Potential ' + pot + 'not implemented')

    @abstractmethod
    @use_default_grid
    def _get_nuclear_potential(self, grid=None):
        """
        Returns the nuclear potential.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunctionPotential}
        """
        raise NotImplementedError

    @abstractmethod
    @use_default_grid
    def _get_coulomb_potential(self, grid=None):
        """
        Returns the nuclear potential.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunctionPotential}
        """
        raise NotImplementedError

    @use_default_grid
    def get_nonfrozen_density(self, grid=None, *args, **kwargs):
        """
        Return the electron density of the nonfrozen fragments.

        @rtype: L{GridFunctionDensity}
        """
        return self.get_density(grid=grid, *args, **kwargs)

    @use_default_grid
    def get_nonfrozen_potential(self, grid=None, *args, **kwargs):
        """
        Return the potential of the nonfrozen fragments.

        @rtype: L{GridFunctionPotential}
        """
        return self.get_potential(grid=grid, *args, **kwargs)


class GTODensityEvaluatorMixin(DensityEvaluatorInterface):
    """
    This class provides an implementation of the L{DensityEvaluatorInterface}
    for GTO-based quantum-chemical calculations.

    The implementation makes use of the Molden/PySCF interface.
    """

    def __init__(self, j=None):
        """
        Constructor for results class.

        @param j: L{job} object of the corresponding job
        """
        super().__init__(j)
        self._grid = None
        self._default_grids = {}

        self._pyscf_object = None

    @abstractmethod
    def read_molden_file(self):
        """
        Returns Molden results file as a string.
        """
        raise NotImplementedError

    @property
    def pyscf_interface(self):
        """
        Create a pyscf_object for the result object.
        """
        from .DensityEvaluator_PySCF import PyScfInterface

        if self._pyscf_object is None:
            self._pyscf_object = PyScfInterface(self.read_molden_file())

        return self._pyscf_object

    @use_default_grid
    def get_density(self, grid=None, order=None, *args, **kwargs):
        """
        Create a density grid function.
        Takes either a raw grid or spacing or nothing at all.

        @param grid:  The grid to use. Optional. For details see L{Plot.Grids}.
        @type grid:   subclass of L{grid}
        @param order: order of derivates of the density to calculate (1 and 2 possible)
        @type order:  int
        @returns:     A density grid function.
        @rtype:       L{GridFunctionDensity}
        """
        gridpoints = grid.get_coordinates(bohr=True)

        if order is None:
            deriv = 0
        else:
            deriv = order

        self.pyscf_interface.calc_density_values(gridpoints, deriv=deriv, cache=True)
        dens = super().get_density(grid, order=order, *args, **kwargs)
        self.pyscf_interface.reset_cache()

        return dens

    @use_default_grid
    def _get_density(self, grid=None):
        """
        Returns the electron density (without derivatives).

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunctionDensity}
        """
        from .Plot.GridFunctions import GridFunctionFactory

        gridpoints = grid.get_coordinates(bohr=True)
        dens_values = self.pyscf_interface.density(gridpoints)

        import hashlib
        m = hashlib.md5()
        m.update(b"Electron density calculated for job: \n")
        m.update(self.checksum.encode('utf-8'))
        m.update(b"and grid: \n")
        m.update(grid.checksum.encode('utf-8'))

        dens_gf = GridFunctionFactory.newGridFunction(grid, dens_values, gf_type='density',
                                                      checksum=m.hexdigest())
        return dens_gf

    @use_default_grid
    def get_densgradient(self, grid=None):
        """
        Returns the gradient of the electron density.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunction2D}
        """
        from .Plot.GridFunctions import GridFunctionFactory

        gridpoints = grid.get_coordinates(bohr=True)
        mixed_dens_values = self.pyscf_interface.density(gridpoints, deriv=1)

        import hashlib
        m = hashlib.md5()
        m.update(b"Electron density gradient calculated for job: \n")
        m.update(self.checksum.encode('utf-8'))
        m.update(b"and grid: \n")
        m.update(grid.checksum.encode('utf-8'))

        return GridFunctionFactory.newGridFunction(grid, mixed_dens_values[1:4].transpose(),
                                                   checksum=m.hexdigest())

    @use_default_grid
    def get_density_hessian(self, grid=None):
        """
        Returns the Hessian of the electron density.

        For details on the processing of the Laplacian,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunction2D}
        """
        from .Plot.GridFunctions import GridFunctionFactory

        gridpoints = grid.get_coordinates(bohr=True)
        mixed_dens_values = self.pyscf_interface.density(gridpoints, deriv=2)

        import hashlib
        m = hashlib.md5()
        m.update(b"Electron density Hessian calculated for job: \n")
        m.update(self.checksum.encode('utf-8'))
        m.update(b"and grid: \n")
        m.update(grid.checksum.encode('utf-8'))

        return GridFunctionFactory.newGridFunction(grid, mixed_dens_values[4:10].transpose(),
                                                   checksum=m.hexdigest())

    @use_default_grid
    def get_laplacian(self, grid=None):
        """
        Returns the Laplacian of the electron density.

        For details on the processing of the Laplacian,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunction1D}
        """
        from .Plot.GridFunctions import GridFunctionFactory

        gridpoints = grid.get_coordinates(bohr=True)
        lapl_values = self.pyscf_interface.laplacian(gridpoints)
        lapl_gf = GridFunctionFactory.newGridFunction(grid, lapl_values)

        return lapl_gf

    @use_default_grid
    def _get_nuclear_potential(self, grid=None):
        """
        Returns the nuclear potential.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunctionPotential}
        """
        from .Plot.GridFunctions import GridFunctionFactory

        gridpoints = grid.get_coordinates(bohr=True)
        pot_values = self.pyscf_interface.nuclear_potential(gridpoints)

        import hashlib
        m = hashlib.md5()
        m.update(b"Nuclear potential calculated for job: \n")
        m.update(self.checksum.encode('utf-8'))
        m.update(b"and grid: \n")
        m.update(grid.checksum.encode('utf-8'))

        pot_gf = GridFunctionFactory.newGridFunction(grid, pot_values, gf_type='potential',
                                                     checksum=m.hexdigest())

        return pot_gf

    @use_default_grid
    def _get_coulomb_potential(self, grid=None):
        """
        Returns the nuclear potential.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @rtype: L{GridFunctionPotential}
        """
        from .Plot.GridFunctions import GridFunctionFactory

        gridpoints = grid.get_coordinates(bohr=True)
        pot_values = self.pyscf_interface.coulomb_potential(gridpoints)

        import hashlib
        m = hashlib.md5()
        m.update(b"Coulomb potential calculated for job: \n")
        m.update(self.checksum.encode('utf-8'))
        m.update(b"and grid: \n")
        m.update(grid.checksum.encode('utf-8'))

        pot_gf = GridFunctionFactory.newGridFunction(grid, pot_values, gf_type='potential',
                                                     checksum=m.hexdigest())

        return pot_gf
