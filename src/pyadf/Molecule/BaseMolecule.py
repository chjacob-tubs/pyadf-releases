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
"""
 Defines the L{BaseMolecule} class.
"""

class BaseMolecule(object):
    """
    Class for representing a base molecule class, that does not
    use openbabel

    @author:        Rosa Bulo
    @organization:  Vrije Universiteit Amsterdam
    @contact:       bulo@few.vu.nl
    """

    def __init__(self):
        """
        Create the base molecule object
        """
        super(BaseMolecule, self).__init__()

    def add_atoms(self, atoms, coords, atomicunits=False, ghosts=False):
        """
        Adds atoms to itself

        @param atoms:
            list of either a) atomic numbers or b) atomic symbols
            of the atoms to add
        @type atoms: list with same length as C{coords}

        @param coords:
            the coordinates of the atoms to add (by default in Angstrom)
        @type coords: n x 3 list of floats or Numeric/numpy array

        @param atomicunits:
            Whether the coordinates are given in atomic units.
            By default, they are in Angstrom.
        @type atomicunits: bool
        """
        pass

    def get_geovar_atoms_block(self, geovar):
        """
        Print the coordinates for use in the ATOMS block of ADF, using geovars.

        @param geovar: The atoms for which geovars should be used.
        @type geovar: list
        """
        pass

    def get_geovar_block(self, geovar):
        """
        Print the GEOVAR block of ADF using the coordinates of the molecule.

        @param geovar: The atoms for which geovars should be used.
        @type geovar: list
        """
        pass

    def set_symmetry(self, symmetry):
        """
        Set the symmetry of the molecule.

        @param symmetry:
            A string specifying the symmetry of the molecule.
            This string should have the same format as the
            symmetry group labels used in ADF, e.g., C(2V).
        @type symmetry: str

        @returns: nothing
        """
        pass

    def get_atom_symbols(self, atoms=None, ghosts=True, prefix_ghosts=True):
        """
        Give back an array with the atom symbols.

        @param atoms:
           A list of the numbers of the atoms to include.
           (The numbering of the atoms starts at 1).
           If C{None} (default), all atoms are included.
        @type  atoms: list of int

        @param ghosts:
            Whether to include ghost atoms or not.
        @type  ghosts: bool

        @param prefix_ghosts:
            Whether to prefix the names of ghost atoms with C{Gh.}
        @type  prefix_ghosts: bool
        """
        pass

    def get_coordinates(self, atoms=None, ghosts=True):
        """
        Give back an array with the coordinates.

        @param atoms:
           A list of the numbers of the atoms to include
           (atom numbering starts at 1).
           If C{None} (default), all atoms are included.
        @type  atoms: list of int

        @param ghosts:
            Whether to include ghost atoms or not.
        @type  ghosts: bool
        """
        pass

    def print_coordinates(self, atoms=None, index=True, suffix=""):
        """
        Returns a string for printing the atomic coordinates.

        This method returns a string representation of the
        atomic coordinates.
        This string can be used for printing, the method
        does not print anything itself.

        The (optional) arguments make it possible to select
        specific atoms for printing and to modify the output
        format.

        @param atoms:
        A list with the numbers of atoms that should be
        included. (The numbering of the atoms starts at 1).
        If C{None}, all atoms are included (default)
        @type atoms: list of int's

        @param index:
        If C{True}, the number of the atom is included
        (see example below)
        @type index: bool

        @param suffix:
        A suffix that is appended to each line
        (see example below)
        @type suffix: str
        """
        pass

    def get_spin(self):
        """
        Returns the total spin multiplicity
        """
        pass

    def get_charge(self):
        """
        Returns the charge of the system
        """
        pass


