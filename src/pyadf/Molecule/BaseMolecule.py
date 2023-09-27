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
"""
 Defines the L{BaseMolecule} class.
"""


class BaseMolecule:
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
        super().__init__()

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def get_number_of_atoms(self):
        """
        Return the number of atoms.

        @returns: number of atoms
        @rtype: int
        """
        raise NotImplementedError

    def add_atoms(self, atoms, coords, atomicunits=False, ghosts=False):
        """
        Adds atoms to itself

        @param atoms:
            list of either (a) atomic numbers or (b) atomic symbols
            of the atoms to add
        @type atoms: list with same length as C{coords}

        @param coords:
            the coordinates of the atoms to add (by default in Angstrom)
        @type coords: n x 3 list of floats or Numeric/numpy array

        @param atomicunits:
            Whether the coordinates are given in atomic units.
            By default, they are in Angstrom.
        @type atomicunits: bool

        @param ghosts: Whether to add the atoms as ghosts.
        @type ghosts: bool
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def get_atomic_numbers(self, atoms=None, ghosts=True):
        """
        Give back an array with the atomic numbers.

        @param atoms:
           A list of the numbers of the atoms to include.
           (The numbering of the atoms starts at 1).
           If C{None} (default), all atoms are included.
        @type  atoms: list of int

        @param ghosts:
            Whether to include ghost atoms or not. If included,
            ghosts will have an atomic number of 0.
        @type  ghosts: bool

        @returns:
            A list of the requested atomic numbers
        @rtype: list of int
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def get_spin(self):
        """
        Returns the total spin multiplicity
        """
        raise NotImplementedError

    def get_charge(self):
        """
        Returns the charge of the system
        """
        raise NotImplementedError

    def get_number_of_electrons(self):
        """
        Returns the number of electrons in the molecule.

        This is the total number of electrons for the molecule, and does not know
        about for frozen cores etc. in the quantum-chemical calculation.
        """
        electrons = sum(self.get_atomic_numbers(ghosts=False))
        electrons = electrons - self.get_charge()
        return electrons

    def get_tip3p_pointcharges(self):
        """
        Returns a list of coordinates and point charge values corresponding to the
        TIP3P water model.

        The molecule consist only of water molecules for this to work. Ca2+ and F-
        ions are also possible.

        For single OH- and H3O+ molecules, charges can also be assigned, but these
        have to be the only molecule. For clusters containing OH- or H3O+, these
        need to be split up into their fragment molecules and this method has to
        be called for each of them.
        """

        # TIP3P charges: O = -0.834 , H = 0.417
        charges_TIP3P = {'O': -0.834, 'H': +0.417, 'F': -1.0, 'Ca': +2.0}

        # charges for OH- from PCCP 2013, 15, 20303-20312
        charges_OH = {'O': -1.183, 'H': +0.183}

        # charges for H3O+ from JACS 1987, 109, 6, 1607–1614
        charges_H3O = {'O': -0.571, 'H': +0.524}

        coords = self.get_coordinates()
        atoms = self.get_atom_symbols()

        pc_list = []

        for i, atom in enumerate(atoms):
            pc_list.append([j for j in coords[i]])
            if self.get_number_of_atoms() == 2:
                pc_list[i].append(charges_OH[atom])
            elif self.get_number_of_atoms() == 4:
                pc_list[i].append(charges_H3O[atom])
            else:
                pc_list[i].append(charges_TIP3P[atom])

        return pc_list

    def get_dmso_pointcharges(self):
        """
        Returns a list of coordinates and point charge values corresponding
        to the DMSO molecules (charges from PCCP 2004, 6, 2136-2144).
        """

        # charges for DMSO from PCCP 2004, 6, 2136-2144
        charges_DMSO = {'C': -0.267, 'O': -0.545, 'H': +0.129, 'S': +0.305}

        coords = self.get_coordinates()
        atoms = self.get_atom_symbols()

        pc_list = []

        for i, atom in enumerate(atoms):
            pc_list.append([j for j in coords[i]])
            pc_list[i].append(charges_DMSO[atom])

        return pc_list

    def get_nuclear_potential(self, grid):
        import numpy as np
        from ..Utils import Bohr_in_Angstrom
        from ..Plot.GridFunctions import GridFunctionFactory

        grid_coords = grid.get_coordinates(bohr=True)
        nucpot = np.zeros(grid.shape)

        for nuc_coord, nuc_charge in zip(self.get_coordinates(), self.get_atomic_numbers()):
            nc = np.array(nuc_coord) / Bohr_in_Angstrom
            dist = grid_coords - nc
            dist = np.sqrt(np.sum(dist*dist, axis=-1))

            nucpot = nucpot - nuc_charge / dist

        import hashlib
        m = hashlib.md5()
        m.update(b"Nuclear potential for molecule:")
        m.update(self.print_coordinates().encode('utf-8'))
        checksum = m.hexdigest()

        gf = GridFunctionFactory.newGridFunction(grid, nucpot, checksum, gf_type='potential')

        return gf
