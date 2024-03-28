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
 Defines grid classes for plotting of densities and potentials.

 See documentation of module L{Plot} for more details.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Grid:
     grid, cubegrid, adfgrid, pyscfgrid, customgrid
"""

from pyadf.Utils import Units
from .FileWriters import GridWriter

import numpy as np
import math


class grid:
    """
    Abstract base class representing a grid.

    A grid is an array of grid points and possibly
    associated weights that can be used for plotting
    and integrating densities and related quantities.
    Most important use of such grids in L{densfjob}

    Since different implementations of such grids
    are possible, this abstract base class defines
    a common interface. This abstract class does
    not do anything itself, it only defines the methods
    that have to be overridden in the derived classes.

    @group Initialization:
        __init__
    @group Access to grid properties:
        get_number_of_points, coorditer, weightiter
    @group Helper routines for densfjob/densfresults:
        get_grid_block, before_densf_run, after_densf_run

    @undocumented:
        __copy__, __deepcopy__
    """

    def __init__(self):
        """
        Constructor for grid.
        """
        self._mol = None
        self._checksum = None
        self._weights = None

        # cache the coordinates (in Angstrom) to avoid recalculation
        self._coords_cache = None

    def __copy__(self):
        return self

    # pylint: disable=W0613
    def __deepcopy__(self, memo):
        return self

    @property
    def npoints(self):
        """
        Number of grid points
        """
        return self.get_number_of_points()

    @property
    def shape(self):
        """
        The shape of the grid, i.e., the shape of the array holding the grid points.
        """
        return self.npoints,

    @property
    def mol(self):
        """
        The molecule this grid has been constructed for.
        """
        return self._mol

    def get_number_of_points(self):
        """
        Return the total number of grid points.

        @rtype: int
        """
        raise NotImplementedError

    def get_grid_block(self, checksumonly):
        """
        Get the GRID input block for densf to use this grid.
        """
        raise NotImplementedError

    @property
    def checksum(self):
        if self._checksum is None:
            import hashlib
            m = hashlib.md5()
            m.update(self.get_grid_block(True).encode('utf-8'))
            self._checksum = m.hexdigest()

        return self._checksum

    def before_densf_run(self):
        """
        Template method that is called before a densf run in which this grid is used.
        """
        pass

    def after_densf_run(self):
        """
        Template method that is called after a densf run in which this grid is used.
        """
        pass

    def write_grid_to_t41(self, iu):
        """
        Write the grid to a TAPE41 file.
        """
        iu.writeints('Grid', 'total nr of points', self.npoints)

        iu.writelogicals('Grid', 'unrestricted', False)
        iu.writeints('Grid', 'nr of symmetries', 1)
        iu.writechars('Grid', 'labels', 'A')

        # write molecule information
        iu.writeints('Geometry', 'nnuc', self.mol.get_number_of_atoms())

        iu.writechars('Geometry', 'labels', self.mol.get_atom_symbols(prefix_ghosts=True))
        iu.writereals('Geometry', 'qtch', self.mol.get_atomic_numbers())
        coords = np.array(self.mol.get_coordinates())
        coords = coords * Units.conversion('angstrom', 'bohr')
        iu.writereals('Geometry', 'xyznuc', coords)

        iu.writereals('Geometry', 'unit of length', 1.0)

    def get_coordinates(self, bohr=False):
        """
        Returns an array with the grid point coordinates (default: Angstrom).
        """
        if self._coords_cache is None:
            self._coords_cache = np.zeros((self.npoints, 3))
            for i, c in enumerate(self.coorditer(bohr=False)):
                self._coords_cache[i, :] = c

        if bohr:
            return self._coords_cache * Units.conversion('angstrom', 'bohr')
        else:
            return self._coords_cache

    def _get_weights(self):
        """
        Read / calculate the weights.

        This is called if self._weights is not available yet.
        Dont call this function directly, but use the weights property.
        """
        pass

    @property
    def weights(self):
        """
        The weights of the gridpoints in a numpy array. Read-only.
        """
        if self._weights is None:
            self._weights = self._get_weights()
        return self._weights

    def coorditer(self, bohr=False):
        """
        Iterator over the coordinates of the grid points.

        The iteration is performed in the same order as for the
        weights in L{weightiter}
        The coordinates are give in Angstrom units.

        @exampleuse:

            Iteration over grid points and weights

            >>> for c, w in zip(grid.coorditer(), grid.weightiter()) :
            >>>     print "Coordinates of grid point: ", c
            >>>     print "Weight of grid point: ", w

        @rtype: iterator
        """
        raise NotImplementedError

    def weightiter(self):
        """
        Iteratator over the weights of the grid points.

        The iteration is performed in the same order as for the
        coordinates in L{coorditer}
        The coordinates are give in Angstrom units.

        @exampleuse:

            Iteration over grid points and weights

            >>> for c, w in zip(grid.coorditer(), grid.weightiter()) :
            >>>     print "Coordinates of grid point: ", c
            >>>     print "Weight of grid point: ", w

        @rtype: iterator
        """
        return self.weights.__iter__()


class cubegrid(grid):
    """
    Class to represent an evenly spaced cubic grid.
    """

    def __init__(self, mol, spacing=0.25, border=1.0):
        """
        Constructor for cubegrid.

        The grid points are inside a suitable box and are distributed
        evenly along the x, y, and z axis. All points have the same
        weight.

        Grids of this type are usually employed for visualization.
        They are not very accurate for integration.

        @param mol:
            The molecule the grid is for. This is used to determine the
            extend of the grid.
        @type  mol: L{molecule}
        @param spacing: The spacing between the grid points in Angstrom (default: 0.25).
        @type  spacing: float
        """
        super().__init__()

        self._spacing = spacing
        self._mol = mol

        coords = np.array(mol.get_coordinates())

        min_coords = np.array([math.floor(min(coords[:, 0]) / spacing) * spacing,
                                  math.floor(min(coords[:, 1]) / spacing) * spacing,
                                  math.floor(min(coords[:, 2]) / spacing) * spacing])
        max_coords = np.array([math.ceil(max(coords[:, 0]) / spacing) * spacing,
                                  math.ceil(max(coords[:, 1]) / spacing) * spacing,
                                  math.ceil(max(coords[:, 2]) / spacing) * spacing])

        self._startpoint = min_coords - border
        self._extend = max_coords - min_coords + 2 * border

        self._npoints = np.array([int(round(self._extend[0] / spacing)) + 1,
                                     int(round(self._extend[1] / spacing)) + 1,
                                     int(round(self._extend[2] / spacing)) + 1])
        # now adjust extend to make sure the grid is really even spaced
        self._extend = (self._npoints - 1) * spacing

    def set_dimensions(self, startpoint, npoints):
        """
        Modify dimensions of cube grid.

        @param startpoint: corner of the cube grid (in Angstrom)
        @type  startpoint: float[3]
        @param npoints: number of grid points in x,y,z directions
        @type  npoints: int[3]
        """
        self._startpoint = np.array(startpoint)
        self._npoints = np.array(npoints)
        self._extend = (self._npoints - 1) * self._spacing

    @property
    def shape(self):
        return tuple(self._npoints)

    def get_number_of_points(self):
        return self._npoints[0] * self._npoints[1] * self._npoints[2]

    def get_grid_block(self, checksumonly):
        block = " GRID\n"
        block += "  {:14.5f} {:14.5f} {:14.5f} \n".format(*self._startpoint)
        block += "  {:10d} {:10d} {:10d} \n".format(*self._npoints)
        block += f"  1.0  0.0  0.0  {self._extend[0]:14.5f} \n"
        block += f"  0.0  1.0  0.0  {self._extend[1]:14.5f} \n"
        block += f"  0.0  0.0  1.0  {self._extend[2]:14.5f} \n"
        block += " END \n\n"
        return block

    def write_grid_to_t41(self, iu):
        iu.writereals('Grid', 'Start_point',
                      np.array(self._startpoint) * Units.conversion('angstrom', 'bohr'))

        iu.writeints('Grid', 'nr of points x', self._npoints[0])
        iu.writeints('Grid', 'nr of points y', self._npoints[1])
        iu.writeints('Grid', 'nr of points z', self._npoints[2])

        iu.writereals('Grid', 'x-vector',
                      [self._spacing * Units.conversion('angstrom', 'bohr'), 0.0, 0.0])
        iu.writereals('Grid', 'y-vector',
                      [0.0, self._spacing * Units.conversion('angstrom', 'bohr'), 0.0])
        iu.writereals('Grid', 'z-vector',
                      [0.0, 0.0, self._spacing * Units.conversion('angstrom', 'bohr')])

        grid.write_grid_to_t41(self, iu)

    def get_cube_header(self):
        """
        Returns the grid header for a cube file.
        """

        header = ""

        # line 3: no of atoms, grid origin
        header += "{:5d}{:12.6f}{:12.6f}{:12.6f}\n" \
            .format(self._mol.get_number_of_atoms(),
                    self._startpoint[0] * Units.conversion('angstrom', 'bohr'),
                    self._startpoint[1] * Units.conversion('angstrom', 'bohr'),
                    self._startpoint[2] * Units.conversion('angstrom', 'bohr'))
        # lines 4-6: no grid points in x,y,z direction + unit vector
        header += "{:5d}{:12.6f}{:12.6f}{:12.6f}\n" \
            .format(self._npoints[0], self._spacing * Units.conversion('angstrom', 'bohr'),
                    0.0, 0.0)
        header += "{:5d}{:12.6f}{:12.6f}{:12.6f}\n" \
            .format(self._npoints[1], 0.0, self._spacing * Units.conversion('angstrom', 'bohr'),
                    0.0)
        header += "{:5d}{:12.6f}{:12.6f}{:12.6f}\n" \
            .format(self._npoints[2], 0.0, 0.0,
                    self._spacing * Units.conversion('angstrom', 'bohr'))

        header += self._mol.get_cube_header()

        return header

    def get_xsf_header(self):
        """
        Return the header for an xsf file.
        """

        molout = ''
        molout += f'{self._mol.get_number_of_atoms():d} 1 \n'
        molout += self._mol.print_coordinates(index=False)

        header = ""
        header += 'INFO\n'
        header += 'nunit   0   0   0 \n'
        header += 'unit  cell\n'
        header += 'celltype primcell\n'
        header += 'shape parapipedal\n'
        header += 'END_INFO\n'
        header += 'DIM-GROUP\n'
        header += '0 1 \n'
        header += 'PRIMVEC\n'
        header += '1.000000000000000  0.000000000000000  0.000000000000000 \n'
        header += '0.000000000000000  1.000000000000000  0.000000000000000 \n'
        header += '0.000000000000000  0.000000000000000  1.000000000000000 \n'
        header += 'CONVVEC\n'
        header += '1.000000000000000  0.000000000000000  0.000000000000000 \n'
        header += '0.000000000000000  1.000000000000000  0.000000000000000 \n'
        header += '0.000000000000000  0.000000000000000  1.000000000000000 \n'
        header += 'PRIMCOORD\n'
        header += molout
        header += 'CONVCOORD\n'
        header += molout
        header += 'RECIP-PRIMVEC\n'
        header += '1.0000000000    0.0000000000    0.0000000000 \n'
        header += '0.0000000000    1.0000000000    0.0000000000 \n'
        header += '0.0000000000    0.0000000000    1.0000000000 \n'
        header += 'RECIP-CONVEC\n'
        header += '1.0000000000    0.0000000000    0.0000000000 \n'
        header += '0.0000000000    1.0000000000    0.0000000000 \n'
        header += '0.0000000000    0.0000000000    1.0000000000 \n'
        header += 'ATOMS\n'
        header += molout

        header += 'BEGIN_BLOCK_DATAGRID3D\n'
        header += 'density_on_3d_data_grid\n'
        header += 'DATAGRID_3D_this_is_3Dgrid#1\n'
        header += f" {self._npoints[0]:5d} {self._npoints[1]:5d} {self._npoints[2]:5d} \n"
        header += " {:12.6f} {:12.6f} {:12.6f} \n" \
            .format(self._startpoint[0] * Units.conversion('angstrom', 'bohr'),
                    self._startpoint[1] * Units.conversion('angstrom', 'bohr'),
                    self._startpoint[2] * Units.conversion('angstrom', 'bohr'))
        header += " {:12.6f} {:12.6f} {:12.6f} \n" \
            .format(self._spacing * Units.conversion('angstrom', 'bohr'), 0.0, 0.0)
        header += " {:12.6f} {:12.6f} {:12.6f} \n" \
            .format(0.0, self._spacing * Units.conversion('angstrom', 'bohr'), 0.0)
        header += " {:12.6f} {:12.6f} {:12.6f} \n" \
            .format(0.0, 0.0, self._spacing * Units.conversion('angstrom', 'bohr'))

        return header

    # noinspection PyMethodMayBeStatic
    def get_xsf_footer(self):
        """
        Return the header for an xsf file.
        """
        footer = ''
        footer += 'END_DATAGRID_3D\n'
        footer += 'END_BLOCK_DATAGRID3D\n'
        return footer

    def coorditer(self, bohr=False):
        if bohr:
            conv = Units.conversion('angstrom', 'bohr')
        else:
            conv = 1.0

        for ix in range(self._npoints[0]):
            for iy in range(self._npoints[1]):
                for iz in range(self._npoints[2]):
                    coord = np.array([self._startpoint[0] + ix * self._spacing,
                                         self._startpoint[1] + iy * self._spacing,
                                         self._startpoint[2] + iz * self._spacing])
                    yield coord * conv
        return

    def _get_weights(self):
        return np.ones((self._npoints[0] * self._npoints[1] * self._npoints[2],)) * \
               (self._spacing * Units.conversion('angstrom', 'bohr'))**3


class adfgrid(grid):
    """
    Class to represent the integration grid as used by ADF.

    The grid is imported from a TAPE10 file produced by ADF.

    Grids of this type can be used to accurately integrate
    densities and related quantities.

    See https://onlinelibrary.wiley.com/doi/10.1002/jcc.23323
    further details on the grids generated by ADF/AMS.

    @group Access to grid properties:
        get_number_of_points, coorditer, weightiter, voronoiiter
    """

    def __init__(self, adfres):
        """
        Constructor for adfgrid.

        @param adfres: The results of the ADF job to use.
        @type  adfres: L{adfsinglepointresults} or subclass
        """
        super().__init__()
        self._adfres = adfres

        self._mol = adfres.get_molecule()

        self._nblocks = self._adfres.get_result_from_tape('Points', 'nblock', tape=10)
        self._npoints = self._adfres.get_result_from_tape('Points', 'lblock', tape=10)
        self._eqvblocks = self._adfres.get_result_from_tape('Points', 'Equivalent Blocks', tape=10)

    def get_number_of_points(self):
        return self._nblocks * self._npoints * self._eqvblocks

    @property
    def mol(self):
        return self._adfres.get_molecule()

    def get_grid_block(self, checksumonly):
        block = " GRID IMPORT TAPE10 \n"
        if checksumonly:
            block += self.checksum
        return block

    @property
    def checksum(self):
        if self._checksum is None:
            import hashlib
            m = hashlib.md5()
            m.update(b'ADF Grid from Job:')
            m.update(self._adfres.checksum.encode('utf-8'))
            self._checksum = m.hexdigest()

        return self._checksum

    def link_grid_tape10(self, name='TAPE10'):
        """
        Get a link to the grid TAPE10 file.
        """
        self._adfres.link_tape(10, name)

    def get_grid_tape10(self, name='TAPE10'):
        """
        Get a copy of the grid TAPE10 file.
        """
        self._adfres.copy_tape(10, name)

    def get_grid_tape10_filename(self):
        """
        Get the TAPE10 file name.
        """
        return self._adfres.get_tape_filename(tape=10)

    def before_densf_run(self):
        self.link_grid_tape10('TAPE10')

    def after_densf_run(self):
        import os
        os.remove('TAPE10')

    def coorditer(self, bohr=False):
        if bohr:
            conv = 1.0
        else:
            conv = Units.conversion('bohr', 'angstrom')

        points_data = self._adfres.get_result_from_tape('Points', 'Data', tape=10)

        eqv_points_data = None
        if self._eqvblocks > 1:
            eqv_points_data = self._adfres.get_result_from_tape('PointsEquiv', 'Data', tape=10)

        ipoint = 0
        ipointeqv = 0

        for iblock in range(1, self._nblocks + 1):

            coords = points_data[ipoint:ipoint + 3 * self._npoints]
            coords = coords.reshape((self._npoints, 3), order='F')
            coords = coords * conv

            ipoint += 4 * self._npoints

            yield from coords

            if self._eqvblocks > 1:
                for ieqv in range(1, self._eqvblocks):

                    coords = eqv_points_data[ipointeqv:ipointeqv + self._npoints * 3]
                    coords = coords.reshape((self._npoints, 3), order='F')
                    coords = coords * conv

                    ipointeqv += 3 * self._npoints

                    yield from coords
        return

    def _get_weights(self):
        weights = np.zeros((self._nblocks * self._npoints * self._eqvblocks,))

        points_data = self._adfres.get_result_from_tape('Points', 'Data', tape=10)

        ipoint2 = 0
        for iblock in range(1, self._nblocks + 1):
            ipoint = self._npoints * (iblock - 1)
            w = points_data[ipoint * 4 + 3 * self._npoints:ipoint * 4 + 4 * self._npoints] / self._eqvblocks

            for ieqv in range(self._eqvblocks):
                weights[ipoint2:ipoint2 + self._npoints] = w
                ipoint2 = ipoint2 + self._npoints

        return weights

    def voronoiiter(self):
        """
        Returns the atomnumber to which Voronoi cell the grid point belongs

        """
        # FIXME: take care of ghost or cap atoms

        # molecule information
        natoms = self.mol.get_number_of_atoms()
        coords = np.array(self.mol.get_coordinates())

        atomnumbers = []
        for i in range(1, natoms + 1):
            atomnumbers.append(i)

        for c in self.coorditer():
            dists = []
            atomindices = []
            for coord, atomnumber in zip(coords, atomnumbers):
                d = (coord[0] - c[0])**2 + (coord[1] - c[1])**2 + (coord[2] - c[2])**2
                dists.append(d)
                atomindices.append(atomnumber)
            dindex = dists.index(min(dists))
            vatom = atomindices[dindex]
            yield vatom
        return


class pyscfgrid(grid):
    """
    Class to represent a becke integration grid as generated by pyscf.
    The following is copied from the PySCF documentation. It lists
    other options that could be implemented for this interface:

    Attributes for Grids:
        level : int
            To control the number of radial and angular grids. Large number
            leads to large mesh grids. The default level 3 corresponds to
            (50,302) for H, He;
            (75,302) for second row;
            (80~105,434) for rest.

            Grids settings at other levels can be found in
            pyscf.dft.gen_grid.RAD_GRIDS and pyscf.dft.gen_grid.ANG_ORDER

        atomic_radii : 1D array
            | radi.BRAGG_RADII  (default)
            | radi.COVALENT_RADII
            | None : to switch off atomic radii adjustment

        radii_adjust : function(mol, atomic_radii) => (function(atom_id, atom_id, g) => array_like_g)
            Function to adjust atomic radii, can be one of
            | radi.treutler_atomic_radii_adjust
            | radi.becke_atomic_radii_adjust
            | None : to switch off atomic radii adjustment

        radi_method : function(n) => (rad_grids, rad_weights)
            scheme for radial grids, can be one of
            | radi.treutler  (default)
            | radi.delley
            | radi.mura_knowles
            | radi.gauss_chebyshev

        becke_scheme : function(v) => array_like_v
            weight partition function, can be one of
            | gen_grid.original_becke  (default)
            | gen_grid.stratmann

        prune : function(nuc, rad_grids, n_ang) => list_n_ang_for_each_rad_grid
            scheme to reduce number of grids, can be one of
            | gen_grid.nwchem_prune  (default)
            | gen_grid.sg1_prune
            | gen_grid.treutler_prune
            | None : to switch off grid pruning

        symmetry : bool
            whether to symmetrize mesh grids (TODO)

        atom_grid : dict
            Set (radial, angular) grids for particular atoms.
            Eg, grids.atom_grid = {'H': (20,110)} will generate 20 radial
            grids and 110 angular grids for H atom.
    """

    def __init__(self, mol, level=None):
        """
        Constructor for pyscfgrid.

        @param mol:
            The molecule the grid is for.
        @type  mol: L{molecule}
        @param level:
            the level of the point density of the pyscf grid that will be created
            default set by PySCF is 3
        """
        super().__init__()
        from pyscf import dft
        self._mol = mol
        self._pyscf_obj = mol.get_pyscf_obj()
        self._pyscf_grid_obj = dft.gen_grid.Grids(self._pyscf_obj)
        if level:
            self._level = level
            self._pyscf_grid_obj.level = level
        else:
            self._level = None
        self._pyscf_grid_obj = self._pyscf_grid_obj.run()
        self._coords = self._pyscf_grid_obj.coords
        self._weights = self._pyscf_grid_obj.weights

        self._npoints = self._coords.shape[0]

    @property
    def checksum(self):
        if self._checksum is None:
            import hashlib
            m = hashlib.md5()
            m.update(b'PySCF Grid for Mol:')
            m.update(self._mol.checksum.encode('utf-8'))
            if self._level:
                m.update(b'and level:')
                m.update(self._level.to_bytes(length=16, byteorder='big'))
            self._checksum = m.hexdigest()

        return self._checksum

    def get_number_of_points(self):
        return self._npoints

    def get_grid_block(self, checksumonly):
        block = " GRID IMPORT TAPE10 \n"
        if checksumonly:
            block += self.checksum
        return block

    def write_grid_tape10(self, name='TAPE10'):
        """
        Write a TAPE10 file for this grid.
        """
        GridWriter.write_tape10(self, filename=name)

    def before_densf_run(self):
        self.write_grid_tape10('TAPE10')

    def after_densf_run(self):
        import os
        os.remove('TAPE10')

    def coorditer(self, bohr=False):
        if bohr:
            conv = 1.0
        else:
            conv = Units.conversion('bohr', 'angstrom')
        for i in range(self._npoints):
            yield self._coords[i, :] * conv
        return


class customgrid(grid):
    """
    Class to represent a custom integration grid.
    """

    def __init__(self, mol, coords, weights=None, checksum=None):
        """
        Constructor for customgrid.

        @param mol:
            The molecule the grid is for. This is only used when exporting to files.
        @type  mol: L{molecule}
        @param coords: A numpy array with the coordinates of the grid points (in Bohr)
        @type  coords: Numpy array with dimension (npoints,3)
        @param weights: The weights of each grid point (optional)
        @type  weights: Numpy array with dimension (npoints)
        @param checksum:
            checksum of the grid data, ideally generated from the input used to generate
            the grid. If not given, a checksum is generated from the Numpy data. Using a
            checksum generated from the input is preferred, because otherwise the checksum
            will depend on numerical noise in the calculations.
        @type checksum: str

        """
        super().__init__()
        self._mol = mol
        self._coords = coords

        self._npoints = self._coords.shape[0]
        if weights is None:
            self._weights = np.ones((self._npoints,))
        else:
            self._weights = weights

        self._checksum = checksum

    def get_number_of_points(self):
        return self._npoints

    def get_grid_block(self, checksumonly):
        block = " GRID IMPORT TAPE10 \n"
        if checksumonly:
            block += self.checksum
        return block

    @property
    def checksum(self):
        if self._checksum is None:
            import hashlib
            m = hashlib.md5()
            m.update(b'Custom Grid with data:')
            m.update(self._coords.data)
            m.update(self._weights.data)
            self._checksum = m.hexdigest()

        return self._checksum

    def write_grid_tape10(self, name='TAPE10'):
        """
        Write a TAPE10 file for this grid.
        """
        GridWriter.write_tape10(self, filename=name)

    def before_densf_run(self):
        self.write_grid_tape10('TAPE10')

    def after_densf_run(self):
        import os
        os.remove('TAPE10')

    def coorditer(self, bohr=False):
        if bohr:
            conv = 1.0
        else:
            conv = Units.conversion('bohr', 'angstrom')
        for i in range(self._npoints):
            yield self._coords[i, :] * conv
        return


class interpolation:
    """
    Class for performing interpolation of values on a general 3d-grid.
    """

    def __init__(self, gf):
        """
        Constructor to initialize the interpolation.

        @param gf: The density/potential to be interpolated
        @type  gf: L{GridFunction1D}
        """
        self.v = gf.get_values()
        self.z = gf.grid.get_coordinates()
        self.mol = gf.grid.mol

    def get_value_at_point(self, point, npoints=200):
        """
        Calculated the interpolated value at one given point.

        The routine uses a modufied verson of the IMLS algorithm.
        See K. Marti, M. Reiher, J. Comput. Chem 2009, DOI: 10.1003/jcc.21201
        and references therein for a description.

        For the purpose here, two small modifications are introduced:
           - only the 200 closest points are used for the interpolation
           - only points that are nor further away than 0.8*(distance to closest atom)
             are included (since otherwise the cusp at the atom introduces problems.

        Here, we always use monomials of up to third order as basis functions
        (i.e., 20 basis functions)

        @param point: the point for which the interpolated value is needed
        @type  point: array[3]

        @param npoints: number of neighboring grid points to include in the interpolation
        @type npoints: int
        """

        # FIXME: there room for performance improvements, method should be carefully profiled

        # calculate weights
        p = np.empty_like(self.z)
        for i in range(3):
            p[:, i] = point[i]
        dist = np.sqrt(((self.z - p)**2).sum(axis=1))
        w = 1.0 / (dist**6 + 1e-7)

        # use only the closest points
        indices = w.argsort()[-npoints:]

        # find distance to nearest atom
        # and only use points that are closer
        indices = [i for i in indices if dist[i] < self.mol.distance_to_point(self.z[i, :])]

        # use at least 10 points for interpolation
        if len(indices) < 10:
            indices = w.argsort()[-10:]

        w = w[indices]
        z = self.z[indices]
        pot = self.v[indices]

        # FIXME: generation of basisfunktions is hardcoded now
        #        (using monomials up to order 3). This could
        #        be automated / generalized to higher orders

        # setup B matrix
        nbas = 20

        b = np.empty((w.shape[0], nbas))
        b[:, 0] = np.ones_like(w.shape[0])
        for i in range(3):
            b[:, i + 1] = z[:, i]
            b[:, i + 4] = z[:, i]**2
            b[:, i + 10] = z[:, i]**3
        b[:, 7] = z[:, 0] * z[:, 1]
        b[:, 8] = z[:, 1] * z[:, 2]
        b[:, 9] = z[:, 0] * z[:, 2]
        b[:, 13] = z[:, 0] * z[:, 1] * z[:, 2]

        b[:, 14] = z[:, 0] * z[:, 0] * z[:, 1]
        b[:, 15] = z[:, 0] * z[:, 0] * z[:, 2]
        b[:, 16] = z[:, 1] * z[:, 1] * z[:, 0]
        b[:, 17] = z[:, 1] * z[:, 1] * z[:, 2]
        b[:, 18] = z[:, 2] * z[:, 2] * z[:, 0]
        b[:, 19] = z[:, 2] * z[:, 2] * z[:, 1]

        bz = np.array([1.0, point[0], point[1], point[2],
                          point[0]**2, point[1]**2, point[2]**2,
                          point[0] * point[1], point[1] * point[2], point[0] * point[2],
                          point[0]**3, point[1]**3, point[2]**3,
                          point[0] * point[1] * point[2],
                          point[0] * point[0] * point[1], point[0] * point[0] * point[2],
                          point[1] * point[1] * point[0], point[1] * point[1] * point[2],
                          point[2] * point[2] * point[0], point[2] * point[2] * point[1]
                          ])

        wp = np.diag(np.sqrt(w))
        wb = np.dot(wp, b)
        rhs = np.dot(wp, pot)

        # perform singular value decomposition using Numpy routine
        u, s, v = np.linalg.svd(wb)

        s_inv = 1.0 / s
        s_inv = np.where(abs(s) > 1e-10, s_inv, np.zeros_like(s_inv))
        sigma_inv = np.zeros_like(b.transpose())
        sigma_inv[:min(wb.shape), :min(wb.shape)] = np.diag(s_inv)

        inv = np.dot(v.transpose(), np.dot(sigma_inv, u.transpose()))
        a = np.dot(inv, rhs)

        return np.dot(a, bz)
