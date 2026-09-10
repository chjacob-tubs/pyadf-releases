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
 Defines the L{BaseMolecule} class.
"""
from typing import List

from ..Errors import PyAdfError


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
        pass

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

    def get_atom_symbols(self, atoms=None, ghosts=True, prefix_ghosts=True) -> List[str]:
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

        @rtype: list of str
        """
        raise NotImplementedError

    def get_coordinates(self, atoms=None, ghosts=True, unit='angstrom') -> List[List[float]]:
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

    def get_atomic_numbers(self, atoms=None, ghosts=True) -> List[int]:
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

    def print_coordinates(self, atoms=None, index=True, suffix='', ghosts=True,
                          prefix_ghosts=True, unit='angstrom', f_format=None):
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

        if f_format is not None:
            fleng, declen = f_format
        else:
            fleng, declen = 21, 14

        atoms = self.get_atoms(atoms, ghosts=ghosts)

        coords = self.get_coordinates(atoms, unit=unit)
        symbs = self.get_atom_symbols(atoms, prefix_ghosts=prefix_ghosts)

        lines = ""
        for i in range(len(atoms)):
            symb = symbs[i]
            c = coords[i]
            if index:
                line = f"  {int(atoms[i]):3d}) {symb:>8} " + \
                       f"{c[0]:{fleng}.{declen}f} {c[1]:{fleng}.{declen}f} {c[2]:{fleng}.{declen}f}"
            else:
                line = f"  {symb:>8} {c[0]:{fleng}.{declen}f} {c[1]:{fleng}.{declen}f} {c[2]:{fleng}.{declen}f}"
            line += "    " + suffix + "\n"
            lines += line

        return lines

    def get_atoms(self, atomlist, ghosts=True) -> List:
        """get_atoms.
        return type depends on the method used, either a list of
        actual Atom objects (OBFree) or a list of integers.
        """
        raise NotImplementedError

    def print_coordinates_for_ams_input(self, atoms=None, index=True, suffix=""):
        atoms_block = self.print_coordinates(atoms, index, suffix)

        block = ""
        for line in atoms_block.splitlines():
            atsym, coords = line.split(maxsplit=1)
            atsym_split = atsym.split('.', maxsplit=1)

            if (len(atsym_split) == 1) or (atsym_split[0] == 'Gh'):
                block += line + '\n'
            else:
                block += '      ' + atsym_split[0] + '       ' + coords + \
                         ' adf.type='+atsym_split[1] + ' \n'
        return block

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
        charges_TIP3P = {'O': -0.834, 'H': +0.417, 'F': -1.0, 'Ca': +2.0, 'Cl': -1.0}

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

    def _nuclear_potential(self, gridcoords, atoms=None):
        """_nuclear_potential:
        The method that actually calculates the nuclear potential for a given
        list of coordinates. Atoms can be chosen.

        Parameters
        ----------

        gridcoords : ndarray
            An array (n, 3) of floats representing the coordinates for which the
            potential is calculated.
        atoms: list
            A list of atom indices (integers starting with 1) or atom objects
            (for the OBFree-subclass)

        Returns
        -------

        nucpot : ndarray
            The array (n) of nuclear potential values, each index corresponding
            to the index in of the coordinates.
        """
        import numpy as np

        atoms = self.get_atoms(atoms)
        atnums = self.get_atomic_numbers(atoms)
        atcoords = np.array(self.get_coordinates(atoms, unit='bohr'))

        nucpot = np.zeros(gridcoords.shape[0])
        for i in range(len(atoms)):
            rp_dist = atcoords[i] - gridcoords
            nucpot += -atnums[i] / np.sqrt(np.einsum('xi,xi->x', rp_dist, rp_dist))
        return nucpot

    def get_nuclear_potential(self, grid, atoms=None):
        """get_nuclear_potential:
        Returns the nuclear potential as a grid function object.

        Parameters
        ----------

        grid : L{Plot.Grids}
            The grid to use. For details, see L{Plot.Grids}.
        :type  grid: subclass of L{grid}

        :return: The nuclear potential on the given grid
        :rtype: L{GridFunctionPotential}


        :param grid:
        :param atoms:
        """
        from pyadf.PyEmbed.Plot.GridFunctions import GridFunctionFactory

        grid_coords = grid.get_coordinates(bohr=True)
        nucpot = self._nuclear_potential(grid_coords, atoms=atoms)

        import hashlib
        m = hashlib.md5()
        m.update(b"Nuclear potential for molecule:")
        m.update(self.print_coordinates().encode('utf-8'))
        m.update(b"and grid:")
        m.update(grid.checksum.encode('utf-8'))
        checksum = m.hexdigest()

        gf = GridFunctionFactory.newGridFunction(
            grid, nucpot, checksum, gf_type='potential')

        return gf

    def get_nuclear_repulsion_energy(self, atoms=None):
        """get_nuclear_repulsion_energy.

        :param atoms:
        """
        import numpy as np
        from pyadf.Utils import Bohr_in_Angstrom

        atoms = self.get_atoms(atoms)
        atnums = self.get_atomic_numbers(atoms)
        coords = np.array(self.get_coordinates()) / Bohr_in_Angstrom

        natoms = len(atoms)
        E_nuc = 0
        for i in range(natoms):
            for j in range(i+1, natoms):
                Z_i = atnums[i]
                Z_j = atnums[j]
                r_ij = np.linalg.norm(coords[i] - coords[j])
                E_nuc += Z_i * Z_j / r_ij

        return E_nuc

    def get_nuclear_interaction_energy(self, other):
        """
        Return the electrostatic interaction energy between the nuclei of this and another molecule.
        """
        import numpy as np

        inten = 0.0
        for coord1, atomNum1 in zip(self.get_coordinates(unit='bohr'), self.get_atomic_numbers()):
            for coord2, atomNum2 in zip(other.get_coordinates(unit='bohr'), other.get_atomic_numbers()):
                dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)
                inten = inten + atomNum1 * atomNum2 / dist
        return inten

    def get_pyscf_obj(self, atoms=None, basis=None, verbosity=0):
        """get_pyscf_obj.

        :param atoms: list of atom objects or integers representing
                      the atom objects
        :type atoms: list of ints or OBFree-Atoms
        :param basis: pyscf needs a basis to create an object
        :type basis: str
        :param verbosity:
        :type verbosity: int
        """
        # for more information: https://pyscf.org/user/gto.html
        from pyscf import gto
        mol = gto.Mole()
        atom_string = ''
        if atoms:
            n_atoms = len(atoms)
        else:
            n_atoms = self.get_number_of_atoms()
            atoms = [i for i in range(1, n_atoms + 1)]
        symbs = self.get_atom_symbols(atoms)
        coords = self.get_coordinates(atoms)
        for i in range(n_atoms):
            atom_string += symbs[i]
            atom_string += ' '
            # Job Molecule Interface (internal)
            atom_string += ' '.join(f'{f:21.14f}' for f in coords[i])
            atom_string += '\n'
        mol.atom = atom_string
        mol.charge = self.get_charge()
        mol.spin = self.get_spin()
        mol.verbose = verbosity
        if basis:
            # default is currently sto-3g
            mol.basis = basis
        mol.build()
        return mol

    def get_xyz_file(self, atoms=None, index=False, suffix='', ghosts=True,
                     prefix_ghosts=False, unit='angstrom', f_format=None):
        lines = ''
        # add the first two lines needed for xyz files
        # 1. number of atoms
        n_atoms = str(self.get_number_of_atoms())
        lines += f'{n_atoms:>2}\n'
        # 2. optional comment
        lines += '\n'
        # add the actual info
        lines += self.print_coordinates(atoms=atoms, index=index, suffix=suffix, ghosts=ghosts,
                                        prefix_ghosts=prefix_ghosts, unit=unit, f_format=f_format)
        return lines

    def get_tmol_coord_file(self):
        atoms = self.get_atoms(None)
        coords = self.get_coordinates(atoms, unit='bohr')
        symbs = self.get_atom_symbols(atoms)

        lines = '$coord\n'
        for i in range(len(atoms)):
            symb = symbs[i]
            c = coords[i]
            # Job Mol Interface
            lines += f'{c[0]:21.14f} {c[1]:21.14f} {c[2]:21.14f} {symb.lower():<8} \n'
        lines += '$end\n'
        return lines

    def write(self, filename, outputformat='xyz'):
        """
        Write the molecule to a file.

        @param filename:
            File name of the file to be written

        @param outputformat:
            File format, given by three letter filename extension.
            See C{babel -H} for available formats.
            Examples of possible formats are xyz and pdb.
            Default is xyz format.
        @type outputformat: str

        @returns: nothing

        @raise PyAdfError:
            Raises exception L{PyAdfError}
            in case an error occures when writeing the molecule
            to file

        """
        if outputformat == 'xyz':
            with open(filename, 'w') as f:
                f.write(self.get_xyz_file())
        elif outputformat == 'tmol':
            with open(filename, 'w') as f:
                f.write(self.get_tmol_coord_file())
        else:
            raise PyAdfError(f'Unknown output format: {outputformat}')

    def get_dalton_molfile(self, basis):
        atomlist = self.get_atoms(None)
        coords = self.get_coordinates(atomlist, unit='angstrom')
        symbs = self.get_atom_symbols(atomlist)
        atnums = self.get_atomic_numbers(atomlist)

        lines = 'BASIS\n' + basis + '\nThis Dalton molecule file was generated by PyADF\n' + \
                ' Homepage: https://www.pyadf.org\n'
        types = list(dict.fromkeys(atnums))
        lines += f'Angstrom Nosymmetry Atomtypes={len(types):d}\n'
        for tp in types:
            atoms = []
            for i in range(len(atomlist)):
                if atnums[i] == tp:
                    atoms.append(i)

            lines += f'Charge={tp:.1f} Atoms={len(atoms):d}\n'
            for ii, i in enumerate(atoms):
                c = coords[i]
                # Job Mol Interface
                lines += f'{symbs[i] + str(ii + 1):<4} {c[0]:21.14f} {c[1]:21.14f} {c[2]:21.14f} \n'
        return lines

    def write_dalton_molfile(self, filename, basis):
        f = open(filename, 'w')
        f.write(self.get_dalton_molfile(basis))
        f.close()
