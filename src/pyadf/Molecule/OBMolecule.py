# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2014 by Christoph R. Jacob, S. Maya Beyhan,
# Rosa E. Bulo, Andre S. P. Gomes, Andreas Goetz, Michal Handzlik,
# Karin Kiewisch, Moritz Klammler, Jetze Sikkema, and Lucas Visscher
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
 Defines the L{OBMolecule} class.

 This module defines the class molecule, which
 is used by PyADF to represent molecules
 (i.e., atomic coordinates, charge, ...)

 This version heavily relies on OpenBabel

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""

import openbabel

from ..Errors import PyAdfError
from ..Utils import pse, Bohr_in_Angstrom

import copy
import math


class BaseMolecule (object):

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

    def has_spin_assigned(self):
        """
        Returns a boolean stating wether spin has been assigned by user
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


class OBMolecule (BaseMolecule):

    """
    Class for representing a molecule.

    This class is used for representing molecules
    in PyADF (i.e., atomic coordinates, charge, etc.).

    B{Basic usage}

    Molecules can be constructed by reading from a file in
    any file format openbabel can handle, the default is xyz
    (see L{__init__}).

    >>> mol = OBMolecule('h2o.xyz')

    They can be read and written in any format openbabel can handle
    using L{read} and L{write}, respectively.
    For printing, L{print_coordinates} can be used.

    Furthermore, L{get_residues} can be used to obtain individual
    residues from a pdb file.

    @Note:
        Under the hood, this class uses the openbabel OBMol
        class for reading, writing, and saving the molecule data.

    @Warning:
        Even though symmetry handling is implemented, it is
        only very basic.
        The specified symmetry is never checked or enforced that the
        molecule indeed has the specified symmetry.

    @group Initialization:
        __init__, copy
    @group File Input/Output methods:
        read, write
    @group Printing methods:
        __str__, print_coordinates, get_geovar_atoms_block, get_geovar_block,
        get_dalton_molfile, write_dalton_molfile, get_cube_header, get_xyz_file
    @group Inquiry methods:
        get_number_of_atoms, get_charge, get_spin, has_spin_assigned,
        get_coordinates, get_atom_symbols,
        get_atomic_numbers, get_symmetry, get_all_bonds, distance, distance_to_point,
        get_center_of_mass, get_alternate_locations,
        get_nuclear_dipole_moment, get_nuclear_efield_in_point
    @group Manipulation methods:
        add_atoms, set_charge, set_spin, set_symmetry, __add__, add_as_ghosts,
        displace_atom, translate, rotate, align, delete_atoms
    @group Molecular-fragment related methods:
        get_fragment, separate
    @group Residue inquiry and manipulation methods:
        get_residues, get_residue_numbers_of_atoms, set_residue, delete_residue,
        get_restype_resnums
    @group Hydrogen-related methods:
        add_hydrogens, find_adjacent_hydrogens

    @undocumented: __deepcopy__, set_OBMol
    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__

    """

    def __init__(self, filename=None, inputformat='xyz'):
        """
        Create a new molecule from a given file.

        @param filename:
            File to read (if None, construct an empty molecule)
        @type filename: str

        @param inputformat:
            File format, given by three letter filename extension,
            see C{babel -H} for available formats.
            Examples of possible formats are xyz and pdb.
            Default is xyz format.
        @type inputformat: str

        @raise PyAdfError:
            Raises exception L{PyAdfError}
            in case an error occurs when reading the molecule
            from file (typically if the file is not found)

        @Note:
            Symmetry is initialized to C{None} by default.
            This will lead to no symmetry information being used,
            which for ADF means that the symmetry is auto-detected.

            If the symmetry should be explicitly specified,
            use L{set_symmetry}.

        """
        BaseMolecule.__init__(self)
        self.mol = openbabel.OBMol()
        self.symmetry = None

        self.is_ghost = []

        # By default, the charge is initialized to zero.
        # This is necessary because openbabel does not provide a
        # HasTotalChargeAssigned() method
        self.set_charge(0)
        self._charge = None

        if filename != None:
            self.read(filename, inputformat)

    def copy(self, other):
        """
        Copy another molecule into this molecule, overwriting it.

        @param other: The molecule to be copied.
        @type  other: L{molecule}
        """
        self.mol = other.mol
        self.symmetry = other.symmetry
        self._charge = other._charge
        self.is_ghost = other.is_ghost

    def __deepcopy__(self, memo):
        """
        Deepcopy: also copy the OBMol molecule.
        """
        new = self.__class__()
        new.mol = openbabel.OBMol(self.mol)
        new.set_charge(self.get_charge())
        if self.mol.HasSpinMultiplicityAssigned():
            new.set_spin(self.get_spin())

        if self.mol.HasChainsPerceived():
            new.mol.SetChainsPerceived()

        new.symmetry = self.symmetry
        new.is_ghost = copy.deepcopy(self.is_ghost, memo)

        return new

    def __str__(self):
        """
        Conversion to a string (for printing).

        Gives back a string listing the atoms in this
        molecule along with their Cartesian coordinates

        @returns: string representation
        @rtype:   str

        @exampleuse:

            >>> mol = OBMolecule('h2o.xyz')
            >>> print mol
            Cartesian coordinates:
            1)     H       -0.21489        3.43542        2.17104
            2)     O       -0.89430        3.96159        2.68087
            3)     H       -0.43479        4.75018        3.07278

        """
        return "  Cartesian coordinates: \n" + self.print_coordinates(index=True)

    def __add__(self, other):
        """
        Defines an addition for molecules.

        Two molecules are added by creating a new molecule
        that contains the coordinates of the atoms in
        both the molecules.

        @param other: The molecule to add.
        @type  other: L{molecule}

        @returns: The 'sum' of the two molecules.
        @rtype:    L{molecule}

        @exampleuse:

        >>> h2o = OBMolecule('h2o.xyz')
        >>> print h2o
        Cartesian coordinates:
        1)     H       -0.21489        3.43542        2.17104
        2)     O       -0.89430        3.96159        2.68087
        3)     H       -0.43479        4.75018        3.07278
        >>> an = OBMolecule('an.xyz')
        >>> print an
        Cartesian coordinates:
        1)     C        2.40366        0.63303       -0.29209
        2)     C        1.77188        1.66625        0.53174
        3)     N        1.27005        2.49581        1.19175
        4)     H        2.29842       -0.34974        0.18696
        5)     H        1.92918        0.59583       -1.28199
        6)     H        3.47247        0.85113       -0.42037
        >>> mol =  an + h2o
        >>> print mol
        Cartesian coordinates:
        1)     C        2.40366        0.63303       -0.29209
        2)     C        1.77188        1.66625        0.53174
        3)     N        1.27005        2.49581        1.19175
        4)     H        2.29842       -0.34974        0.18696
        5)     H        1.92918        0.59583       -1.28199
        6)     H        3.47247        0.85113       -0.42037
        7)     H       -0.21489        3.43542        2.17104
        8)     O       -0.89430        3.96159        2.68087
        9)     H       -0.43479        4.75018        3.07278

        """

        m = copy.deepcopy(self)
        # FIXME: (Good) Symmetry handling is still missing
        m.set_symmetry(self.get_symmetry())
        m.add_atoms(other.get_atom_symbols(), other.get_coordinates())
        m.set_charge(self.get_charge() + other.get_charge())
        if self.mol.HasSpinMultiplicityAssigned() or other.mol.HasSpinMultiplicityAssigned():
            m.set_spin(self.get_spin() + other.get_spin())
        return m

    def add_as_ghosts(self, other):
        """
        Addition on molecules similar to L{__add__},
        but the atoms in C{other} will become ghost atoms.

        @param other: The molecule to add.
        @type  other: L{molecule}

        @returns: The 'sum' of the two molecules.
        @rtype:   L{molecule}

        @exampleuse:

        >>> an = OBMolecule('an.xyz')
        >>> h2o = OBMolecule('h2o.xyz')
        >>> mol = an.add_as_ghosts(h2o)
        >>> print mol
        Cartesian coordinates:
        1)     C        2.40366        0.63303       -0.29209
        2)     C        1.77188        1.66625        0.53174
        3)     N        1.27005        2.49581        1.19175
        4)     H        2.29842       -0.34974        0.18696
        5)     H        1.92918        0.59583       -1.28199
        6)     H        3.47247        0.85113       -0.42037
        7)  Gh.H       -0.21489        3.43542        2.17104
        8)  Gh.O       -0.89430        3.96159        2.68087
        9)  Gh.H       -0.43479        4.75018        3.07278

        """

        m = copy.deepcopy(self)
        # FIXME: (Good) Symmetry handling is still missing
        m.set_symmetry(self.get_symmetry())
        m.add_atoms(other.get_atom_symbols(), other.get_coordinates(), ghosts=True)
        m.set_charge(self.get_charge())
        if self.mol.HasSpinMultiplicityAssigned():
            m.set_spin(self.get_spin())
        return m

    def displace_atom(self, atom=None, coordinate=None, displacement=0.01, atomicunits=True):
        """
        Displacement of one atom
        Attention: will also set symmetry to NOSYM

        @param atom:
            The atom which shall be displaced
        @type  atom: int

        @param coordinate:
            The coordinate which shall be displaced (x, y or z)
        @type coordinate: str

        @param displacement:
            The displacement in Angstrom or Bohr (atomic units)
        @type displacement: float

        @param atomicunits:
            Whether the displacement is given in atomic units.
            By default, it is in bohr.
        @type atomicunits: bool

        @returns: The molecule with displaced atom
        @rtype:   L{molecule}

        @exampleuse:

        >>> h2o = OBMolecule('h2o.xyz')
        >>> mol = h2o.displace_atom(atom=1, coordinate='x', atomicunits=False)
        >>> print h2o
        Cartesian coordinates:
        1)  H       -0.21489        3.43542        2.17104
        2)  O       -0.89430        3.96159        2.68087
        3)  H       -0.43479        4.75018        3.07278
        >>> print mol
        Cartesian coordinates:
        1)  H       -0.20489        3.43542        2.17104
        2)  O       -0.89430        3.96159        2.68087
        3)  H       -0.43479        4.75018        3.07278

        """

        if atom == None:
            raise PyAdfError("atom number missing in OBMolecule.displace_atom")
        elif coordinate == None:
            raise PyAdfError("coordinate missing in OBMolecule.displace_atom")
        coordinate = coordinate.lower()

        if atomicunits:
            displacement = displacement * Bohr_in_Angstrom

        m = copy.deepcopy(self)
        m.set_symmetry('NOSYM')
        a = m.mol.GetAtom(atom)
        x = a.GetX()
        y = a.GetY()
        z = a.GetZ()
        if coordinate == 'x':
            x += displacement
        elif coordinate == 'y':
            y += displacement
        elif coordinate == 'z':
            z += displacement
        a.SetVector(x, y, z)

        return m

    def read(self, filename, inputformat='xyz', ghosts=False):
        """
        Read molecule from a file.

        The molecule is read from the specified inputfile
        and the atoms of the newly read molecule are appended
        to the already existing ones.

        @param filename:
            File name of the file to read from
        @type filename: str

        @param inputformat:
            File format, given by three letter filename extension.
            See C{babel -H} for available formats.
            Examples of possible formats are xyz and pdb.
            Default is xyz format.
        @type inputformat: str

        @param ghosts:
            If True, the read atoms are appended as ghost atoms.
        @type ghosts: bool

        @returns: nothing

        @raise PyAdfError:
            Raises exception L{PyAdfError}
            in case an error occures when reading the molecule
            from file (typically if the file is not found)

        """

        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats(inputformat, 'xyz')
        if not conv.ReadFile(self.mol, filename):
            raise PyAdfError("Error reading molecule")
        numread = self.mol.NumAtoms() - len(self.is_ghost)
        self.is_ghost += [ghosts] * numread

    def set_OBMol(self, mol):
        """
        Initialize the molecule with a openbabel OBMol.

        This will destroy and replace anything that is already there.

        @param mol: the molecule
        @type  mol: openbabel.OBMol
        """

        self.mol = mol
        self.is_ghost = [False] * mol.NumAtoms()

        return self

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
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats('xyz', outputformat)

        if not conv.WriteFile(self.mol, filename):
            raise PyAdfError("Error writing molecule")

    def set_symmetry(self, symmetry):
        """
        Set the symmetry of the molecule.

        By default, the symmetry is set to None, which will
        cause no symmetry information to be generated so that
        ADF will auto-detect the symmetry.
        This method can be used to override this behavior.

        @param symmetry:
            A string specifying the symmetry of the molecule.
            This string should have the same format as the
            symmetry group labels used in ADF, e.g., C(2V).
        @type symmetry: str

        @returns: nothing

        @warning:
            It is never checked or enforced that the molecule
            does actually have the specified symmetry.
        """

        # FIXME: Symmetry is not checked of enforced
        self.symmetry = symmetry

    def get_symmetry(self):
        """
        Returns the symmetry of the molecule.

        @returns:
            string specifying symmetry group of the molecule,
            or C{None} if no symmetry information is associated
            with the molecule.
        @rtype: str or None
        """
        return self.symmetry

    def set_charge(self, charge):
        """
        Set the total charge of the molecule.

        @param charge: total charge
        @type charge: float

        @returns: nothing
        """
        if isinstance(charge, int):
            self._charge = None
            self.mol.SetTotalCharge(charge)
        else:
            self._charge = charge
            self.mol.SetTotalCharge(int(round(charge)))

    def get_charge(self):
        """
        Returns the total charge of the molecule.

        @returns: total charge
        @rtype:   float
        """
        if self._charge is None:
            return self.mol.GetTotalCharge()
        else:
            return self._charge

    def set_spin(self, spin):
        """
        Set the total spin of the molecule.

        @param spin: total spin
        @type spin: float

        @returns: nothing
        """
        self.mol.SetTotalSpinMultiplicity(spin)

    def get_spin(self):
        """
        Returns the total spin of the molecule.

        @returns: total spin
        @rtype:   float
        """

        if self.mol.HasSpinMultiplicityAssigned():
            spin = self.mol.GetTotalSpinMultiplicity()
        else:
            spin = 0

        return spin

    def add_atoms(self, atoms, coords, atomicunits=False, ghosts=False):
        """
        Add atoms to the molecule.

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

        @note:
            The atomic symbols C{atoms} can be obtained from another molecule
            using L{get_atom_symbols}, the coordinates using L{get_coordinates}

        @returns: nothing

        @exampleuse:

            >>> mol = OBMolecule()
            >>> atoms = ['H', 'H', 'O']
            >>> coords = [[-0.21489, 3.43542, 2.17104],
            ...           [-0.89430, 3.96159, 2.68087],
            ...           [-0.43479, 4.75018, 3.07278]]
            >>> mol.add_atoms(atoms, coords)
            >>> print mol.print_coordinates()
            1)     H       -0.21489        3.43542        2.17104
            2)     H       -0.89430        3.96159        2.68087
            3)     O       -0.43479        4.75018        3.07278

        """
        if len(atoms) != len(coords):
            raise PyAdfError('length of atoms and coords not matching')

        if atomicunits:
            for i in range(len(atoms)):
                for j in range(3):
                    coords[i][j] = coords[i][j] * Bohr_in_Angstrom

        self.mol.BeginModify()
        for i in range(len(atoms)):
            a = self.mol.NewAtom()
            if type(atoms[i]) == str:
                anum = pse.get_atomic_number(atoms[i])
                a.SetAtomicNum(anum)
                a.SetType(atoms[i])
            else:
                anum = int(atoms[i])
                a.SetAtomicNum(anum)
                a.SetType(pse.get_symbol(anum))
            a.SetVector(coords[i][0], coords[i][1], coords[i][2])
            self.is_ghost.append(ghosts)
        self.mol.EndModify()

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

        @returns: the coordinates, in an nx3 array
        @rtype:   array of floats

        @exampleuse:

            >>> mol = OBMolecule('h2o.xyz')
            >>> mol.get_coordinates()
            [[-0.21489, 3.43542, 2.17104],
             [-0.89430, 3.96159, 2.68087],
             [-0.43479, 4.75018, 3.07278]]

        """

        if atoms == None:
            atoms = range(1, self.mol.NumAtoms() + 1)
            if ghosts == False:
                atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        coords = []
        for i in atoms:
            a = self.mol.GetAtom(i)
            coords.append([a.GetX(), a.GetY(), a.GetZ()])

        return coords

    def get_center_of_mass(self):
        """
        Return the coordinates of the center of mass.
        """
        import numpy
        center = numpy.zeros((3,))
        for atom in openbabel.OBMolAtomIter(self.mol):
            vec = numpy.array([atom.GetX(), atom.GetY(), atom.GetZ()])
            center += atom.GetAtomicMass() * vec
        center = center / self.mol.GetMolWt()
        return center

    def get_number_of_atoms(self):
        """
        Return the number of atoms.

        @returns: number of atoms
        @rtype: int
        """
        return self.mol.NumAtoms()

    def get_atom_symbols(self, atoms=None, ghosts=True, prefix_ghosts=False):
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

        @returns:
            A list of the requested atom symbols
        @rtype: list of str

        @exampleuse:
            Obtaining a list of all atomic symbols.

            >>> mol = OBMolecule('an.xyz')
            >>> mol.get_atom_symbols()
            ['C', 'C', 'N', 'H', 'H', 'H']

        @exampleuse:
            Restricting the list to certain atoms.

            >>> mol = OBMolecule('an.xyz')
            >>> mol.get_atom_symbols(atoms=[1,3,5])
            ['C', 'N', 'H']

        """

        if atoms == None:
            atoms = range(1, self.mol.NumAtoms() + 1)
            if ghosts == False:
                atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        symbols = []
        for i in atoms:
            a = self.mol.GetAtom(i)
            symb = pse.get_symbol(a.GetAtomicNum())
            if prefix_ghosts and self.is_ghost[i - 1]:
                symb = "Gh." + symb
            symbols.append(symb)

        return symbols

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

        if atoms == None:
            atoms = range(1, self.mol.NumAtoms() + 1)
            if ghosts == False:
                atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        nums = []
        for i in atoms:
            a = self.mol.GetAtom(i)
            num = a.GetAtomicNum()
            if self.is_ghost[i - 1]:
                num = 0
            nums.append(num)

        return nums

    def get_nuclear_dipole_moment(self, atoms=None):
        """
        Gets the nuclear contribution to the dipole moment in atomic units.
        If an atomlist is given, the nuclear contribution is given atomwise.
        This allows for an analysis per atom/Voronoi cell (the electronic contribution
        can be calculated from the density)

        @param atoms:
            A list of the numbers of the atoms to include (numbering starting at 1).
            If C{None} (default), all atoms are included.
        @type atoms:
            list of int

        @returns:
            the (total) nuclear contribution to the dipole moment or a
            list of nuclear dipole moment contributions per atom
        @rtype: list of float

        """
        import numpy
        voronoinucdip = []
        printsum = False

        if atoms is None:
            atoms = range(1, self.mol.NumAtoms() + 1)
            printsum = True

        for coord, atomNum in zip(self.get_coordinates(atoms=atoms),
                                  self.get_atomic_numbers(atoms=atoms)):
            dip_nuc_x = coord[0] * atomNum / Bohr_in_Angstrom
            dip_nuc_y = coord[1] * atomNum / Bohr_in_Angstrom
            dip_nuc_z = coord[2] * atomNum / Bohr_in_Angstrom
            voronoinucdip.append(numpy.array([dip_nuc_x, dip_nuc_y, dip_nuc_z]))

        if printsum:
            return sum(voronoinucdip)
        else:
            return voronoinucdip

    def get_nuclear_efield_in_point(self, pointcoord):
        """
        Returns the nuclear contribution to the electric field in a point in atomic units.

        @param pointcoord:
        @type pointcoord: array of float, in Angstrom coordinates

        """
        import numpy

        E_x = 0.0
        E_y = 0.0
        E_z = 0.0
        for coord, atomNum in zip(self.get_coordinates(), self.get_atomic_numbers()):
            dist = numpy.sqrt((coord[0] - pointcoord[0]) ** 2 + (coord[1] - pointcoord[1]) ** 2 + (coord[2] - pointcoord[2]) ** 2)
            E_x += atomNum * (coord[0] - pointcoord[0]) / dist ** 3
            E_y += atomNum * (coord[1] - pointcoord[1]) / dist ** 3
            E_z += atomNum * (coord[2] - pointcoord[2]) / dist ** 3

        return numpy.array([E_x, E_y, E_z]) * (Bohr_in_Angstrom * Bohr_in_Angstrom)

    def get_fragment(self, atoms, ghosts=True):
        """
        Give back a part of the molecule (a fragment)

        The fragment that is requested is specified by
        giving a list of the atoms that are included in this
        fragment. This method returns a new L{molecule},
        which contains this fragment.

        @param atoms:
           A list of the numbers of the atoms to include.
           If C{None} (default), all atoms are included.
        @type  atoms: list of int

        @param ghosts:
            Whether to include ghost atoms or not.
        @type  ghosts: bool

        @returns: the requested fragment
        @rtype:   molecule

        @exampleuse:

            >>> an = OBMolecule('an.xyz')
            >>> print an.print_coordinates()
            1)     C        2.40366        0.63303       -0.29209
            2)     C        1.77188        1.66625        0.53174
            3)     N        1.27005        2.49581        1.19175
            4)     H        2.29842       -0.34974        0.18696
            5)     H        1.92918        0.59583       -1.28199
            6)     H        3.47247        0.85113       -0.42037
            >>> mol = an.get_fragment([1,2,6])
            >>> print mol.print_coordinates()
            1)     C        2.40366        0.63303       -0.29209
            2)     C        1.77188        1.66625        0.53174
            3)     H        3.47247        0.85113       -0.42037

        """
        m = OBMolecule()
        m.add_atoms(self.get_atom_symbols(atoms, ghosts),
                    self.get_coordinates(atoms, ghosts))
        return m

    def get_residues(self, chain=None, restype=None, resnum=None, idx=None):
        """
        Obtain the individual residues (applies for pdb files or proteins).

        This method can be used for decomposing big structures that
        are read from pdb or similar files. It gives back each residue
        individually as a separate L{molecule}. Optionally, only
        certain residues (chain and/or type) can be obtained.

        @param chain: Optionally, only include residues from a certain chain.
        @type  chain: type

        @param restype: Optionally only include residues of a certain type.
        @type  restype: str

        @param resnum: Optionally only include residues with a certain number.
        @type  resnum: int

        @returns: A list of the matching residues (each as a new L{molecule})
        @rtype:   list of molecules

        @param idx: Optionally only include residues with a certain (internal) number.
        @type  idx: int

        @exampleuse:
            An example of using this method can be found in
            the test PDB_Molecules and in FDE_NMR_relax

        """

        res_list = []

        if self.mol.NumAtoms() == 0:
            return res_list

        # force Openbabel to perceive chains by inquiring the residue of atom 1
        self.mol.GetAtom(1).GetResidue()

        # collect list of the residues we are interested in
        residues = [(r, r.GetChain(), r.GetName(), r.GetNum(), r.GetIdx()) for r in openbabel.OBResidueIter(self.mol)]

        if chain:
            residues = [r for r in residues if (r[1] == chain)]
        if restype:
        # for mol2 format: only compare first three letters
            residues = [r for r in residues if (r[2][:3] == restype)]
        if resnum:
            residues = [r for r in residues if (r[3] == resnum)]
        if idx:
            residues = [r for r in residues if (r[4] == idx)]

        for res in residues:

            m = OBMolecule()

            for at in openbabel.OBResidueAtomIter(res[0]):
                m.mol.AddAtom(at)
                m.is_ghost.append(False)

            m.set_residue(res[2], res[3], res[1], res[4], atoms=None)

            res_list.append(m)

        return res_list

    def get_residue_numbers_of_atoms(self):
        """
        Return a list giving the residue number of each atom.

        Numbering of the residues starts at 0.
        """

        res_list = []

        for at in openbabel.OBMolAtomIter(self.mol):
            res_list.append(at.GetResidue().GetIdx())

        return res_list

    def delete_residue(self, restype=None, resnums=None, chain=None):
        """
        Delete certain residues from the molecule.

        @param restype: the type of the residue(s) to be deleted
        @type restype:  str

        @param resnums: the residuenumber(s) to be deleted - better use together
                       with chain, since residuenumbers are not unique
        @type resnums: list of ints

        @param chain: which chain, use together with residuenumber
        @type chain: int
        """

        self.mol.GetAtom(1).GetResidue()
        residues = [r for r in openbabel.OBResidueIter(self.mol)]

        if chain:
            residues = [r for r in residues if r.GetChain() == chain]
        if restype:
            residues = [r for r in residues if r.GetName() == restype]
        if resnums:
            residues = [r for r in residues if r.GetNum() in resnums]

        for res in residues:
            atoms = [a for a in openbabel.OBResidueAtomIter(res)]

            for a in atoms:
                res.RemoveAtom(a)
                self.mol.DeleteAtom(a)

            self.mol.DeleteResidue(res)

        self.is_ghost = [False] * self.mol.NumAtoms()

    def get_restype_resnums(self, restype, lidx=True):
        """
        Get the residue numbers for a certain residue type.

        Use that to obtain the residues for which to set charges, e.g., for all GLU.

        @param restype: the type of the residue
        @type restype:  str

        @param lidx: whether to return the internal residue number (counting starts at 0)
        @type lidx: bool

        @returns: A list of residue numbers belonging to restype, if idx == True the internal number
                  is returned

        """

        resnum_list = []

        residues = [(r, r.GetName(), r.GetNum(), r.GetIdx()) for r in openbabel.OBResidueIter(self.mol)]

        residues = [r for r in residues if (r[1][:3] == restype)]

        for res in residues:
            if lidx:
                resnum_list.append(res[3])
            else:
                resnum_list.append(res[2])

        return resnum_list

    def delete_atoms(self, atoms):
        """
        Delete a list of atoms from the molecule.

        Atoms in list are first sorted in ascending order, then
        deleting starts from highest number so that numbering does not shift

        @param atoms: the list of atom numbers
        @type  atoms: list of ints
        @Warning: not yet checked whether all information is updated
                 (i.e., number of atoms, pse, ...)

        """
        for atomnumber in sorted(atoms)[::-1]:
            a = self.mol.GetAtom(atomnumber)
            self.mol.DeleteAtom(a)

    def get_alternate_locations(self, filename):
        """
        Get alternate location atoms contained in pdb file from the molecule.

        Use delete_atoms to delete alternate locations.
        So far it is not possible to define which of the alternate locations you want,
        you will get everything that is not 'A'.

        @param filename:
            File to read (only pdb file format)
        @type filename: str

        @returns: A list of atom numbers representing alternate locations
                  (numbering of the atoms starts at 1)
        """

        # needs to open and read pdb file because openbabel does not read four letter residue names
        pdbfile = open(filename, 'r')

        # find atoms with four letter residue codes
        # and add them to atomlist
        atomlist = []
        for line in pdbfile.readlines():
            word = line.split()
            if word[0] == "ATOM":
                if len(word[2]) > 3:
                    atomtype = word[2][:3]
                    resname = word[2][3:]
                    word = word[:2] + [atomtype] + [resname] + word[3:]

        # determine whether first letter is 'A'
        # if not add them to atomlist
                if len(word[3]) == 4 and word[3][0] != 'A':
                    atomlist.append(int(word[1]))

        pdbfile.close()
        return atomlist

    def get_all_bonds(self):
        """
        Get a list of all bonds in the molecule.

        @returns: A list of pairs of atom numbers
                  (numbering of the atoms starts at 1)
        """

        bond_list = []

        for b in openbabel.OBMolBondIter(self.mol):
            bond_list.append([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])

        return bond_list

    def find_adjacent_hydrogens(self, atoms):
        """
        Return a list of all hydrogen atoms that are directly connected to one of the given atoms.

        @param atoms: the list of atoms numbers
        @type  atoms: list of ints

        @returns: the list of the numbers of the hydrogen atoms
        @rtype:   list of ints
        """

        hydrogens = []

        for i in atoms:
            a = self.mol.GetAtom(i)
            for neighbor in openbabel.OBAtomAtomIter(a):

                if (neighbor.GetAtomicNum() == 1) and not (neighbor.GetIdx() in atoms):
                    hydrogens.append(neighbor.GetIdx())

        return hydrogens

    def add_hydrogens(self, correctForPH=False, pH=7.4):
        """
        Add hydrogen atoms.

        This makes use of routines from OpenBabel.
        Please always check the result carefully, it might
        not be what you expected.

        @param correctForPH:
            Whether pH value is taken into account
        @type correctForPH: bool

        @param pH:
            pH value
        @type pH: float

        """
        self.mol.ConnectTheDots()
        self.mol.PerceiveBondOrders()

        # set all formal charges to 0 -> all residues will be neutral
        for at in openbabel.OBMolAtomIter(self.mol):
            at.SetFormalCharge(0)

        # force atom typer to work
        at = self.mol.GetAtom(1)
        at.GetHyb()
        at.IsAromatic()
        at.GetImplicitValence()

        # CJ: UGLY HACK
        # avoid double bonds in ILE residues
        sp = openbabel.OBSmartsPattern()
        sp.Init('C=C')
        sp.Match(self.mol)
        for mp in sp.GetUMapList():
            at1 = self.mol.GetAtom(mp[0])
            at2 = self.mol.GetAtom(mp[1])
            if at1.GetResidue().GetName() == 'ILE':
                b = self.mol.GetBond(mp[0], mp[1])
                b.SetBondOrder(1)
                at1.SetImplicitValence(4)
                at2.SetImplicitValence(4)
                at1.SetHyb(3)
                at2.SetHyb(3)
        # END UGLY HACK

        self.mol.AddHydrogens(False, correctForPH, pH)

        self.is_ghost += [False] * (self.mol.NumAtoms() - len(self.is_ghost))

    def set_residue(self, restype, resnum, chain=None, idx=None, atoms=None):
        """
        Set the residue information for the given atoms (or all, if not given).

        @param restype: the residue name
        @type  restype: str
        @param resnum: the residue number
        @type  resnum: int
        @param chain: the chain name/number
        @type  chain: str
        @param idx: the residue idx (internal number)
        @type  idx: int
        @param atoms: the numbers of the atoms belonging to this residue
                      (atom numbering starts at 1)
        @type  atoms: list on ints
        """

        if not atoms:
            atoms = range(1, self.mol.NumAtoms() + 1)

        res = self.mol.NewResidue()
        res.SetName(restype)
        res.SetNum(resnum)
        if chain:
            res.SetChain(chain)
        if idx:
            res.SetIdx(idx)

        for i in atoms:
            a = self.mol.GetAtom(i)
            res.AddAtom(a)
            res.SetAtomID(a, pse.get_symbol(a.GetAtomicNum()))

        self.mol.SetChainsPerceived()

    def separate(self):
        """
        Separate the molecule into disconnected fragments.

        @returns: A list of molecules, one for each disconnected fragment
        @rtype:   list of L{molecule}s
        """

        mols = []
        obmols = self.mol.Separate()
        for m in obmols:
            mols.append(OBMolecule().set_OBMol(m))

        return mols

    def translate(self, vec):
        '''
        Translate the molecule.

        @param vec: the translation vector
        @type  vec: numpy array or list of 3 floats
        '''
        obvec = openbabel.vector3(vec[0], vec[1], vec[2])
        self.mol.Translate(obvec)

    def rotate(self, rotmat):
        '''
        Rotate the molecule.

        @param rotmat: a 3x3 rotation matrix
        @type  rotmat: numpy 3x3 matrix
        '''
        obmat = openbabel.double_array(rotmat.flatten().tolist())
        self.mol.Rotate(obmat)

    def align(self, other, atoms, atoms_other=None):
        '''
        Rotate and translate the molecule such that the given atoms are maximally aligned.

        Returns the rotation matrix and translation vector that were applied
        (apply in order!: first the rotation, then the translation)

        @param other: molecule to align with
        @type  other: L{molecule}
        @param atoms: list of atoms that should be aligned
        @type  atoms: list of int
        @param atoms_other: list of atoms in other molecule to align with
        @type  atoms_other: list of int

        @returns: tuple of rotation matrix and translation vector
        @rtype:   tuple of: numpy.array((3,3)), numpy.array((3,))
        '''
        import numpy

        def quaternion_fit(coords_r, coords_f):
            # this function is based on the algorithm described in
            # Molecular Simulation 7, 113-119 (1991)

            x = numpy.zeros((3, 3))
            for r, f in zip(coords_r, coords_f):
                x = x + numpy.outer(f, r)

            c = numpy.zeros((4, 4))

            c[0, 0] = x[0, 0] + x[1, 1] + x[2, 2]
            c[1, 1] = x[0, 0] - x[1, 1] - x[2, 2]
            c[2, 2] = x[1, 1] - x[2, 2] - x[0, 0]
            c[3, 3] = x[2, 2] - x[0, 0] - x[1, 1]

            c[1, 0] = x[2, 1] - x[1, 2]
            c[2, 0] = x[0, 2] - x[2, 0]
            c[3, 0] = x[1, 0] - x[0, 1]

            c[0, 1] = x[2, 1] - x[1, 2]
            c[2, 1] = x[0, 1] + x[1, 0]
            c[3, 1] = x[2, 0] + x[0, 2]

            c[0, 2] = x[0, 2] - x[2, 0]
            c[1, 2] = x[0, 1] + x[1, 0]
            c[3, 2] = x[1, 2] + x[2, 1]

            c[0, 3] = x[1, 0] - x[0, 1]
            c[1, 3] = x[2, 0] + x[0, 2]
            c[2, 3] = x[1, 2] + x[2, 1]

            # diagonalize c
            d, v = numpy.linalg.eig(c)

            # extract the desired quaternion
            q = v[:, d.argmax()]

            # generate the rotation matrix

            u = numpy.zeros((3, 3))
            u[0, 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
            u[1, 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
            u[2, 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]

            u[1, 0] = 2.0 * (q[1] * q[2] - q[0] * q[3])
            u[2, 0] = 2.0 * (q[1] * q[3] + q[0] * q[2])

            u[0, 1] = 2.0 * (q[2] * q[1] + q[0] * q[3])
            u[2, 1] = 2.0 * (q[2] * q[3] - q[0] * q[1])

            u[0, 2] = 2.0 * (q[3] * q[1] - q[0] * q[2])
            u[1, 2] = 2.0 * (q[3] * q[2] + q[0] * q[1])

            return u

        frag_mv = self.get_fragment(atoms)
        if atoms_other is None:
            frag_ref = other.get_fragment(atoms)
        else:
            frag_ref = other.get_fragment(atoms_other)

        com_mv = frag_mv.get_center_of_mass()
        com_ref = frag_ref.get_center_of_mass()

        # move both fragments to center of mass
        frag_ref.translate(-com_ref)
        frag_mv.translate(-com_mv)

        rotmat = quaternion_fit(frag_ref.get_coordinates(), frag_mv.get_coordinates())

        transvec = com_ref - numpy.dot(rotmat, com_mv)

        self.rotate(rotmat)
        self.translate(transvec)

        return rotmat, transvec

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

        @returns:
            String representation of atomic coordinates
        @rtype: str

        @exampleuse:
            Simple printing of the coordinates:

            >>> mol = OBMolecule('h2o.xyz')
            >>> print mol.print_coordinates()
            1)     H       -0.21489        3.43542        2.17104
            2)     O       -0.89430        3.96159        2.68087
            3)     H       -0.43479        4.75018        3.07278

        @exampleuse:
            Printing of selected atoms:

            >>> mol = OBMolecule('h2o.xyz')
            >>> print mol.print_coordinates(atoms=[2])
            2)     O       -0.89430        3.96159        2.68087

        @exampleuse:
            Printing without atom numbering:

            >>> mol = OBMolecule('h2o.xyz')
            >>> print mol.print_coordinates(index=False)
            H       -0.21489        3.43542        2.17104
            O       -0.89430        3.96159        2.68087
            H       -0.43479        4.75018        3.07278

        @exampleuse:
            Printing with a suffix:

            >>> mol = OBMolecule('h2o.xyz')
            >>> print mol.print_coordinates(index=False, suffix='f=frag1')
            H       -0.21489        3.43542        2.17104    f=frag1
            O       -0.89430        3.96159        2.68087    f=frag1
            H       -0.43479        4.75018        3.07278    f=frag1

        @note:
            Coordinates are always printed in Angstrom units.

        """

        lines = ""
        if atoms == None:
            atoms = range(1, self.mol.NumAtoms() + 1)

        coords = self.get_coordinates(atoms)
        symbs = self.get_atom_symbols(atoms, prefix_ghosts=True)

        for i in range(len(atoms)):
            symb = symbs[i]
            c = coords[i]

            if index == True:
                line = "  %3i) %8s %14.5f %14.5f %14.5f" % (atoms[i], symb, c[0], c[1], c[2])
            else:
                line = "  %8s %14.5f %14.5f %14.5f" % (symb, c[0], c[1], c[2])

            line += "    " + suffix + "\n"
            lines += line

        return lines

    def get_xyz_file(self):
        """
        Return an xyz file of the molecule.

        @rtype: str
        """
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats('xyz', 'xyz')
        return conv.WriteString(self.mol)

    def get_geovar_atoms_block(self, geovar):
        """
        Print the coordinates for use in the ATOMS block of ADF, using geovars.

        @param geovar: The atoms for which geovars should be used.
        @type geovar: list

        @exampleuse:
            Printing of the atoms using geovars:

            >>> mol = OBMolecule('h2o.xyz')
            >>> print mol.get_geovar_atoms_block([1,3])
            H         atom1x         atom1y         atom1z
            O    -0.89430000     3.96159000     2.68087000
            H         atom3x         atom3y         atom3z
        """

        AtomsBlock = ""
        for i, atom in enumerate(openbabel.OBMolAtomIter(self.mol)):
            symb = pse.get_symbol(atom.GetAtomicNum())
            if self.is_ghost[i] == True:
                symb = "Gh." + symb
            if i + 1 in geovar:
                varname = "atom" + str(i + 1)
                line = "  %5s " % symb
                line += "%14s " % (varname + "x")
                line += "%14s " % (varname + "y")
                line += "%14s \n" % (varname + "z")
            else:
                line = "  %5s %14.5f %14.5f %14.5f \n" % \
                       (symb, atom.GetX(), atom.GetY(), atom.GetZ())
            AtomsBlock += line

        return AtomsBlock

    def get_geovar_block(self, geovar):
        """
        Print the GEOVAR block of ADF using the coordinates of the molecule.

        @param geovar: The atoms for which geovars should be used.
        @type geovar: list

        @exampleuse:
            Printing of the atoms using geovars:

            >>> mol = OBMolecule('h2o.xyz')
            >>> print mol.get_geovar_block([1,3])
            GEOVAR
              atom1x         -0.21489000
              atom1y          3.43542000
              atom1z          2.17104000
              atom3x         -0.43479000
              atom3y          4.75018000
              atom3z          3.07278000
            END
        """

        block = " GEOVAR\n"
        for i in geovar:
            atom = self.mol.GetAtom(i)
            block += "   atom" + str(i) + "x   %14.5f \n" % atom.GetX()
            block += "   atom" + str(i) + "y   %14.5f \n" % atom.GetY()
            block += "   atom" + str(i) + "z   %14.5f \n" % atom.GetZ()
        block += " END\n\n"
        return block

    def get_dalton_molfile(self, basis):
        """
        Returns the content of a Dalton-style molecule file.

        @param basis: the basis set to use (for all atoms)
        @type  basis: str

        @exampleuse:
            Printing of the Dalton molecule file:

            >>> mol = OBMolecule('h2o.xyz')
            >>> print mol.get_dalton_molfile('STO-3G')
            BASIS
            STO-3G
            This Dalton molecule file was generated by PyADF
             Homepage: http://www.pyadf.org
            Angstrom Nosymmetry Atomtypes=2
            Charge=1.00000000 Atoms=2
            H1         -0.21489000        3.43542000        2.17104000
            H2         -0.43479000        4.75018000        3.07278000
            Charge=8.00000000 Atoms=1
            O1         -0.89430000        3.96159000        2.68087000
        """

        molfile = 'BASIS\n'
        molfile += basis + '\n'
        molfile += 'This Dalton molecule file was generated by PyADF\n'
        molfile += ' Homepage: http://www.pyadf.org\n'

        # determine number of atomtypes
        atsyms = self.get_atom_symbols()
        atyps = set(atsyms)
        num_atomtypes = len(atyps)

        # FIXME: hardcoding NO SYMMETRY here
        molfile += 'Angstrom Nosymmetry Atomtypes=%d\n' % num_atomtypes

        for atyp in atyps:

            atoms = [i + 1 for i, at in enumerate(atsyms) if at == atyp]
            coords = self.get_coordinates(atoms)

            molfile += "Charge=%.1f Atoms=%d\n" % (pse.get_atomic_number(atyp), len(coords))
            for i, a in enumerate(coords):
                line = "%-4s %14.5f %14.5f %14.5f \n" % \
                    (atyp + str(i + 1), a[0], a[1], a[2])
                molfile += line

        return molfile

    def distance(self, other):
        """
        Measures the distance between two molecules.

        The distance is defined as the minumum distance between two
        atoms in the molecules.
        """
        coords1 = self.get_coordinates()
        coords2 = other.get_coordinates()

        dists = []

        for c1 in coords1:
            for c2 in coords2:
                d = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
                dists.append(d)

        return math.sqrt(min(dists))

    def distance_to_point(self, point, ghosts=True):
        """
        Measure the distance between the molecule and a given point.
        """
        coords = self.get_coordinates(ghosts=ghosts)

        dists = []
        for c in coords:
            d = (c[0] - point[0]) ** 2 + (c[1] - point[1]) ** 2 + (c[2] - point[2]) ** 2
            dists.append(d)
        return math.sqrt(min(dists))

    def write_dalton_molfile(self, filename, basis):
        """
        Write the molecule to a Dalton-style molecule file.

        @param filename: The name of the file to be written.
        @type  filename: str
        @param basis: The basis set to use (for all atoms).
        @type  basis: str
        """

        f = open(filename, 'w')
        f.write(self.get_dalton_molfile(basis))
        f.close()

    def get_cube_header(self):
        """
        Return a cub-file header for the molecule.
        """
        header = ""

        atoms = range(1, self.mol.NumAtoms() + 1)
        atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        for i in atoms:
            at = self.mol.GetAtom(i)
            header += "%5d%12.6f%12.6f%12.6f%12.6f\n" % (at.GetAtomicNum(), 0.0,
                                                               at.GetX() / Bohr_in_Angstrom,
                                                               at.GetY() / Bohr_in_Angstrom,
                                                               at.GetZ() / Bohr_in_Angstrom)

        return header

    def has_spin_assigned(self):
        """
        Returns a boolean stating wether spin has been assigned by user

        @author: Rosa Bulo (REB)

        @rtype: bool
        """
        return self.mol.HasSpinMultiplicityAssigned()

    def get_checksum(self, representation='xyz'):
        """
        Get a hexadecimal 128-bit md5 hash of the molecule.

        This method writes a coordinate file, digests it and returns the md5
        checksum of that file. If you think that the representation of the
        molecule matters, you can specify it explicitly via the C{representation}
        flag. Needless to say that you have to use the same representation
        to compare two molecules.

        @param representation: Molecule file format understood by I{Open Babel}.
        @returns:              Hexadicimal hash
        @rtype:                L{str}
        @author:               Moritz Klammler
        @date:                 Aug. 2011

        """

        import os
        import tempfile
        import hashlib

        # First write the  coordinates to a file. The  format obviously doesn't
        # matter as long as it it unambigous and we always use the same. We use
        # Python's  `tempfile' module to  write the  coordinates. This  has the
        # advantage that the method will also succeed if we do not have writing
        # access to the CWD and we don't risk acidently overwriting an existing
        # file. The temporary file will be  unlinked from the OS at the time of
        # disposal of the `tempfile.NamedTemporaryFile' object.

        # We  detect  one source  of  errors by  comparing  the  hash with  the
        # empty-string hash. If it matches, something must have went wrong with
        # writing and re-reading the file.

        m = hashlib.md5()
        emptyhash = m.hexdigest()

        tmp = tempfile.NamedTemporaryFile()
        tmp.file.close()

        # The file is empty now. Note  that we only call the `file' attribute's
        # `close()' method.  Saying `tmp.close()' would  immediately unlink the
        # pysical file which is not what we want.

        self.write(tmp.name, outputformat=representation)

        with open(tmp.name, 'r') as infile:
            for line in infile:
                m.update(line)

        molhash = m.hexdigest()
        if molhash == emptyhash:
            raise PyAdfError("""Error while trying to compute the md5 hash of
            the molecule. Hash equals empty-string hash.""")

        return molhash


def _setUp_doctest(test):
    # pylint: disable-msg=W0613

    import os

    # create molecule files needed for doctests

    os.mkdir('molecule_doctests')

    h2o = OBMolecule()
    h2o.add_atoms(['H', 'O', 'H'],
                  [[-0.21489, 3.43542, 2.17104],
                   [-0.89430, 3.96159, 2.68087],
                   [-0.43479, 4.75018, 3.07278]])
    h2o.write('h2o.xyz')

    an = OBMolecule()
    an.add_atoms(['C', 'C', 'N', 'H', 'H', 'H'],
                 [[2.40366, 0.63303, -0.29209],
                  [1.77188, 1.66625, 0.53174],
                  [1.27005, 2.49581, 1.19175],
                  [2.29842, -0.34974, 0.18696],
                  [1.92918, 0.59583, -1.28199],
                  [3.47247, 0.85113, -0.42037]])
    an.write('an.xyz')


def _tearDown_doctest(test):
    # pylint: disable-msg=W0613

    import os

    os.remove('h2o.xyz')
    os.remove('an.xyz')

    os.rmdir('molecule_doctests')
