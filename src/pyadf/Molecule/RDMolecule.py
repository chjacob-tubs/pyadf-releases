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
 Defines the L{RDMolecule} class.

 This module defines the class molecule, which
 is used by PyADF to represent molecules
 (i.e., atomic coordinates, charge, ...)

 The original OBMolecule heavily relies on OpenBabel

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 This RDMolecule attempts to provide the same class interface
 while using rdkit instead
 
 @author:       Lars Ridder
 @organization: Netherlands eScience Center (NLeSC)
 @contact:      lars.ridder@esciencecenter.nl

 @warning:
    RDKit does not derive bond orders for ligands in pdb files,
    therefore cannot be trusted for adding the correct number of 
    hydrogens to those
    Also, RDKit does not accout for correction according to pH
"""

from rdkit import Chem, Geometry

from ..Errors import PyAdfError
from ..Utils import pse, Bohr_in_Angstrom
from BaseMolecule import BaseMolecule
from ProteinMolecule import ProteinMoleculeMixin

import copy
import math


class RDMolecule(ProteinMoleculeMixin, BaseMolecule):
    """
    Class for representing a molecule.

    This class is used for representing molecules
    in PyADF (i.e., atomic coordinates, charge, etc.).

    B{Basic usage}

    Molecules can be constructed by reading from a file in
    any file format RDkit can handle as well as xyz
    (see L{__init__}).

    >>> mol = RDMolecule('h2o.xyz')

    They can be read and written in formats RDKit can handle + xyz
    using L{read} and L{write}, respectively.
    For printing, L{print_coordinates} can be used.

    Furthermore, L{get_residues} can be used to obtain individual
    residues from a pdb file.

    @Note:
        Under the hood, this class uses the RDKit Mol
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
        get_residues, get_residue_index_of_atoms, set_residue, delete_residue,
        get_restype_resnums
    @group Hydrogen-related methods:
        add_hydrogens, find_adjacent_hydrogens

    @undocumented: __deepcopy__, set_RDMol
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
        super(RDMolecule, self).__init__()
        self.mol = Chem.Mol()
        self.symmetry = None

        self.is_ghost = []

        # By default, the charge is initialized to zero.
        # This is necessary because openbabel does not provide a
        # HasTotalChargeAssigned() method
        self.set_charge(0)

        self.TotalSpinMultiplicity = None

        if filename is not None:
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
        Deepcopy: also copy the RDMol molecule.
        """
        new = self.__class__()
        new.mol = copy.deepcopy(self.mol)
        new.set_charge(self.get_charge())
        if self.has_spin_assigned():
            new.set_spin(self.get_spin())

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

            >>> mol = RDMolecule('h2o.xyz')
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

        >>> h2o = RDMolecule('h2o.xyz')
        >>> print h2o
        Cartesian coordinates:
        1)     H       -0.21489        3.43542        2.17104
        2)     O       -0.89430        3.96159        2.68087
        3)     H       -0.43479        4.75018        3.07278
        >>> an = RDMolecule('an.xyz')
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
        new_atoms_indices = m.add_atoms(other.get_atom_symbols(), other.get_coordinates(), return_new_indices=True)

        e = Chem.EditableMol(m.mol)
        for b in other.get_all_bonds():
            e.AddBond(new_atoms_indices[b[0] - 1] - 1, new_atoms_indices[b[1] - 1] - 1)
        m.mol = e.GetMol()

        m.set_charge(self.get_charge() + other.get_charge())
        if self.has_spin_assigned() or other.has_spin_assigned():
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

        >>> an = RDMolecule('an.xyz')
        >>> h2o = RDMolecule('h2o.xyz')
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
        new_atoms_indices = m.add_atoms(other.get_atom_symbols(), other.get_coordinates(),
                                        ghosts=True, return_new_indices=True)

        e = Chem.EditableMol(m.mol)
        for b in other.get_all_bonds():
            e.AddBond(new_atoms_indices[b[0] - 1] - 1, new_atoms_indices[b[1] - 1] - 1)
        m.mol = e.GetMol()

        m.set_charge(self.get_charge())
        if self.has_spin_assigned():
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

        >>> h2o = RDMolecule('h2o.xyz')
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

        if atom is None:
            raise PyAdfError("atom number missing in RDMolecule.displace_atom")
        elif coordinate is None:
            raise PyAdfError("coordinate missing in RDMolecule.displace_atom")
        coordinate = coordinate.lower()

        if atomicunits:
            displacement = displacement * Bohr_in_Angstrom

        m = copy.deepcopy(self)
        m.set_symmetry('NOSYM')
        c = m.mol.GetConformer()
        p = c.GetAtomPosition(atom - 1)
        if coordinate == 'x':
            p.x += displacement
        elif coordinate == 'y':
            p.y += displacement
        elif coordinate == 'z':
            p.z += displacement
        c.SetAtomPosition(atom - 1, p)

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

        @Warning:
            Reading xyz and pdb functions do not create bonds between atoms and
            therefore behaves different from the one in OBMolecule

        @exampleuse:

            >>> an = RDMolecule('an.xyz')
            >>> print an
              Cartesian coordinates: 
                1)        C        2.40366000        0.63303000       -0.29209000    
                2)        C        1.77188000        1.66625000        0.53174000    
                3)        N        1.27005000        2.49581000        1.19175000    
                4)        H        2.29842000       -0.34974000        0.18696000    
                5)        H        1.92918000        0.59583000       -1.28199000    
                6)        H        3.47247000        0.85113000       -0.42037000    
            <BLANKLINE>
        """
        import os
        import subprocess

        self.mol = None

        if inputformat == 'mol':
            self.mol = Chem.MolFromMolFile(filename, sanitize=False, removeHs=False)
        elif inputformat == 'mol2':
            self.mol = Chem.MolFromMol2File(filename, sanitize=False, removeHs=False)
        elif inputformat == 'pdb':
            self.mol = Chem.MolFromPDBFile(filename, sanitize=False, removeHs=False)
        elif (inputformat == 'xyz') or (inputformat == 'tmol'):
            # convert to PDB with Openbabel (command line interface)
            # in order to have bond and residue information auto-detected
            FNULL = open(os.devnull, 'w')
            obpdb = subprocess.check_output(['obabel', '-i' + inputformat, filename, '-opdb'],
                                            stderr=FNULL, close_fds=True)
            self.mol = Chem.MolFromPDBBlock(obpdb, sanitize=False, removeHs=False)

            c = self.mol.GetConformer()

            # PDB only uses three significant digits, so read XYZ again 
            # to recover the remaining digits
            f = open(filename, 'r')
            lines = f.readlines()
            f.close()
            natoms = self.mol.GetNumAtoms()

            if inputformat == 'xyz':
                startline = 2
            else:
                startline = 1

            for i, l in enumerate(lines[startline:natoms + startline]):
                if inputformat == 'xyz':
                    coord = [float(a) for a in l.split()[1:]]
                else:
                    coord = [float(a) * Bohr_in_Angstrom for a in l.split()[:3]]
                p = Geometry.Point3D(coord[0], coord[1], coord[2])
                c.SetAtomPosition(i, p)
        else:
            raise PyAdfError("Error reading molecule " + inputformat + " " + filename)

        numread = self.mol.GetNumAtoms() - len(self.is_ghost)
        self.is_ghost += [ghosts] * numread

    def set_RDMol(self, mol):
        """
        Initialize the molecule with an RDKit mol.

        This will destroy and replace anything that is already there.

        @param mol: the molecule
        @type  mol: RDKit Mol
        """

        self.mol = mol
        self.is_ghost = [False] * mol.GetNumAtoms()

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
        if outputformat == 'mol':
            Chem.MolToMolFile(self.mol, filename)
        elif outputformat == 'pdb':
            Chem.MolToPDBFile(self.mol, filename, flavor=12)
        elif outputformat == 'xyz':
            f = open(filename, 'w')
            f.write(self.get_xyz_file())
            f.close()
        elif outputformat == 'tmol':
            f = open(filename, 'w')
            f.write(self.get_tmol_file())
            f.close()
        else:
            raise PyAdfError("Unknown file format in RDMolecule.write")

    def get_tmol_file(self):
        ratio = 1.0 / Bohr_in_Angstrom

        tmolfile = '$coord\n'
        for at, c in zip(self.get_atom_symbols(ghosts=False), self.get_coordinates(ghosts=False)):
            tmolfile += '%20.14f %20.14f %20.14f %-8s \n' \
                        % (c[0] * ratio, c[1] * ratio, c[2] * ratio, at.lower())
        tmolfile += '$end\n'
        return tmolfile

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
        self._charge = charge

    def get_charge(self):
        """
        Returns the total charge of the molecule.

        @returns: total charge
        @rtype:   float
        """
        return self._charge

    def set_spin(self, spin):
        """
        Set the total spin of the molecule.

        @param spin: total spin
        @type spin: float

        @returns: nothing
        """
        self.TotalSpinMultiplicity = spin

    def get_spin(self):
        """
        Returns the total spin of the molecule.

        @returns: total spin
        @rtype:   float
        """

        if self.TotalSpinMultiplicity is None:
            return 0
        else:
            return self.TotalSpinMultiplicity

    def get_chain_of_residue(self, residx, chaininfo=None):
        """
        Returns the chain of a residue.

        @returns: string with the chain descriptor.
        @rtype: str or None

        @param residx: The internal index of the residue.
        @type  residx: int

        @exampleuse:
        >>> mol = RDMolecule('1PBE3.pdb', inputformat='pdb')
        >>> print mol.get_chain_of_residue(2)
        A
        >>> mol = RDMolecule('an.pdb', inputformat='pdb')
        >>> print mol.get_chain_of_residue(0)
        A
        >>> print mol.get_chain_of_residue(1)
        B
        """
        if chaininfo is None:
            chains, chain_offsets = self.get_chain_info()
        else:
            chains, chain_offsets = chaininfo

        chain_offsets_sorted = sorted(chain_offsets.items(), key=lambda x: x[1])
        for i, (chain_id, offset) in enumerate(chain_offsets_sorted[:-1]):
            if residx >= offset and (residx < chain_offsets_sorted[i+1][1]):
                return str(chain_id)

        return str(chain_offsets_sorted[-1][0])

    def add_atoms(self, atoms, coords, atomicunits=False, ghosts=False, bond_to=None, return_new_indices=False):
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
            >>> mol = RDMolecule()
            >>> atoms = ['H', 'H', 'O']
            >>> coords = [[-0.21489, 3.43542, 2.17104],
            ...           [-0.89430, 3.96159, 2.68087],
            ...           [-0.43479, 4.75018, 3.07278]]
            >>> mol.add_atoms(atoms, coords)
            >>> print mol.print_coordinates()
            1)     H       -0.21489        3.43542        2.17104
            2)     H       -0.89430        3.96159        2.68087
            3)     O       -0.43479        4.75018        3.07278

        @exampleuse:
            >>> mol = RDMolecule()
            >>> atoms = [1, 1, 8]
            >>> coords = [[-0.21489, 3.43542, 2.17104],
            ...           [-0.89430, 3.96159, 2.68087],
            ...           [-0.43479, 4.75018, 3.07278]]
            >>> mol.add_atoms(atoms, coords)
            >>> print mol.print_coordinates()
            1)     H       -0.21489        3.43542        2.17104
            2)     H       -0.89430        3.96159        2.68087
            3)     O       -0.43479        4.75018        3.07278

        """
        import numbers

        if len(atoms) != len(coords):
            raise PyAdfError('length of atoms and coords not matching')

        if atomicunits:
            for i in range(len(atoms)):
                for j in range(3):
                    coords[i][j] = coords[i][j] * Bohr_in_Angstrom
        e = Chem.EditableMol(self.mol)
        if isinstance(atoms[0], numbers.Number):
            new_atoms_indices = [e.AddAtom(Chem.Atom(int(a))) for a in atoms]
        else:
            new_atoms_indices = [e.AddAtom(Chem.Atom(a)) for a in atoms]

        if bond_to is not None:
            for ai in new_atoms_indices:
                e.AddBond(ai, bond_to - 1)

        self.mol = e.GetMol()

        if bond_to is not None:
            rinfo = self.mol.GetAtomWithIdx(bond_to - 1).GetPDBResidueInfo()
            for ai in new_atoms_indices:
                at = self.mol.GetAtomWithIdx(ai)
                if rinfo is not None:
                    at.SetMonomerInfo(Chem.AtomPDBResidueInfo('%3s ' % at.GetSymbol(),
                                                              residueName=rinfo.GetResidueName(),
                                                              residueNumber=rinfo.GetResidueNumber(),
                                                              chainId=rinfo.GetChainId()))
                else:
                    at.SetMonomerInfo(Chem.AtomPDBResidueInfo('%3s ' % at.GetSymbol()))

        if self.mol.GetNumConformers() == 0:
            c = Chem.Conformer()
        else:
            c = self.mol.GetConformer()
        for i in range(len(coords)):
            a = coords[i]
            p = Geometry.Point3D(a[0], a[1], a[2])
            c.SetAtomPosition(new_atoms_indices[i], p)
            self.is_ghost.append(ghosts)

        if self.mol.GetNumConformers() == 0:
            self.mol.AddConformer(c)

        if return_new_indices:
            return [ai + 1 for ai in new_atoms_indices]

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

            >>> mol = RDMolecule('h2o.xyz')
            >>> mol.get_coordinates()
            [[-0.21489, 3.43542, 2.17104],
             [-0.89430, 3.96159, 2.68087],
             [-0.43479, 4.75018, 3.07278]]

        """

        if atoms is None:
            atoms = range(1, self.mol.GetNumAtoms() + 1)
            if not ghosts:
                atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        coords = []
        c = self.mol.GetConformer()
        for i in atoms:
            a = c.GetAtomPosition(i - 1)
            coords.append([a.x, a.y, a.z])

        return coords

    def get_center_of_mass(self):
        """
        Return the coordinates of the center of mass.

        @exampleuse:

            >>> mol = RDMolecule('h2o.xyz')
            >>> mol.get_center_of_mass()
            array([-0.83057359,  3.97627328,  2.67427198])

        """
        import numpy
        center = numpy.zeros((3,))
        coords = self.get_coordinates()
        molwt = 0
        for atom in range(self.mol.GetNumAtoms()):
            vec = numpy.array([coords[atom][0], coords[atom][1], coords[atom][2]])
            center += self.mol.GetAtomWithIdx(atom).GetMass() * vec
            molwt += self.mol.GetAtomWithIdx(atom).GetMass()
        center = center / molwt
        return center

    def get_number_of_atoms(self):
        """
        Return the number of atoms.

        @returns: number of atoms
        @rtype: int

        @exampleuse:

            >>> mol = RDMolecule('h2o.xyz')
            >>> mol.get_number_of_atoms()
            3

        """
        return self.mol.GetNumAtoms()

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

            >>> mol = RDMolecule('an.xyz')
            >>> mol.get_atom_symbols()
            ['C', 'C', 'N', 'H', 'H', 'H']

        @exampleuse:
            Restricting the list to certain atoms.

            >>> mol = RDMolecule('an.xyz')
            >>> mol.get_atom_symbols(atoms=[1,3,5])
            ['C', 'N', 'H']

        """

        if atoms is None:
            atoms = range(1, self.mol.GetNumAtoms() + 1)
            if not ghosts:
                atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        symbols = []
        for i in atoms:
            a = self.mol.GetAtomWithIdx(i - 1)
            symb = a.GetSymbol()
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

        @exampleuse:
            Obtaining a list of all atomic symbols.

            >>> an = RDMolecule('an.xyz')
            >>> h2o = RDMolecule('h2o.xyz')
            >>> mol = an.add_as_ghosts(h2o)
            >>> mol.get_atomic_numbers(ghosts=False)
            [6, 6, 7, 1, 1, 1]

        """

        if atoms is None:
            atoms = range(1, self.mol.GetNumAtoms() + 1)
            if not ghosts:
                atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        nums = []
        for i in atoms:
            a = self.mol.GetAtomWithIdx(i - 1)
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

        @exampleuse:

            >>> mol = RDMolecule('h2o.xyz')
            >>> mol.get_nuclear_dipole_moment()
            array([-14.74757385,  75.35910311,  50.43826425])

        """
        import numpy
        voronoinucdip = []
        printsum = False

        if atoms is None:
            atoms = range(1, self.mol.GetNumAtoms() + 1)
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

        @exampleuse:

            >>> mol = RDMolecule('h2o.xyz')
            >>> mol.get_nuclear_efield_in_point([0,0,0])
            array([-0.01894430 ,  0.09856731,  0.06584592])

        """
        import numpy

        E_x = 0.0
        E_y = 0.0
        E_z = 0.0
        for coord, atomNum in zip(self.get_coordinates(), self.get_atomic_numbers()):
            dist = numpy.sqrt(
                (coord[0] - pointcoord[0])**2 + (coord[1] - pointcoord[1])**2 + (coord[2] - pointcoord[2])**2)
            E_x += atomNum * (coord[0] - pointcoord[0]) / dist**3
            E_y += atomNum * (coord[1] - pointcoord[1]) / dist**3
            E_z += atomNum * (coord[2] - pointcoord[2]) / dist**3

        return numpy.array([E_x, E_y, E_z]) * (Bohr_in_Angstrom * Bohr_in_Angstrom)

    def get_nuclear_interaction_energy(self, other):
        """
        Return the electrostatic interaction energy between the nuclei of this and another molecule.
        """
        import numpy

        inten = 0.0
        for coord1, atomNum1 in zip(self.get_coordinates(), self.get_atomic_numbers()):
            for coord2, atomNum2 in zip(other.get_coordinates(), other.get_atomic_numbers()):
                dist = numpy.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)
                dist = dist / Bohr_in_Angstrom
                inten = inten + atomNum1 * atomNum2 / dist
        return inten

    def get_fragment(self, atoms):
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

        @returns: the requested fragment
        @rtype:   molecule

        @exampleuse:

            >>> an = RDMolecule('an.xyz')
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
        bonds = []
        nobond_atoms = []

        for i in atoms:
            a = self.mol.GetAtomWithIdx(i - 1)
            num_bonds_for_a = 0
            for b in a.GetBonds():
                if (b.GetBeginAtomIdx() + 1 in atoms) and (b.GetEndAtomIdx() + 1 in atoms):
                    num_bonds_for_a += 1
                    if b.GetIdx() not in bonds:
                        bonds.append(b.GetIdx())
            if num_bonds_for_a == 0:
                nobond_atoms.append(i)

        m = RDMolecule()
        atommap = {}
        m.mol = Chem.PathToSubmol(self.mol, bonds, atomMap=atommap)

        if len(nobond_atoms) > 0:
            atsyms = [a.split('.')[0] for a in self.get_atom_symbols(atoms=nobond_atoms)]
            new_atoms_indices = m.add_atoms(atsyms, self.get_coordinates(atoms=nobond_atoms),
                                            return_new_indices=True)
            for i, j in zip(nobond_atoms, new_atoms_indices):
                atommap[i - 1] = j - 1

        m.mol = Chem.RenumberAtoms(m.mol, [atommap[i - 1] for i in atoms])

        m.is_ghost = [self.is_ghost[a - 1] for a in atoms]
        return m

    def residue_iter(self):
        reslist = []
        for at in range(1, self.get_number_of_atoms() + 1):
            chain_id, res_name, res_num = self.get_atom_resinfo(at)
            if (chain_id, res_name, res_num) not in reslist:
                reslist.append((chain_id, res_name, res_num))
                yield chain_id, res_name, res_num

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

            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> l = an.get_residues(restype='LIG')
            >>> print [r.get_nuclear_dipole_moment() for r in l]
            [array([ 42.75316384,  35.51551279, -19.00308591])]
            >>> l = an.get_residues(chain='A')
            >>> print [r.get_nuclear_dipole_moment() for r in l]
            [array([ 42.75316384,  35.51551279, -19.00308591])]
            >>> l = an.get_residues(resnum=1, chain='B')
            >>> print [r.get_nuclear_dipole_moment() for r in l]
            [array([-26.18593491,  49.01382649,  24.45683550 ])]

        @exampleuse:

            >>> diala = RDMolecule('dialanine.xyz', 'xyz')
            >>> allala = diala.get_residues(restype='ALA')
            >>> print [f.get_number_of_atoms() for f in allala]
            [11, 12]
            >>> ala1 = diala.get_residues(restype='ALA', resnum=1)
            >>> print [f.get_number_of_atoms() for f in ala1]
            [11]
            >>> ala2 = diala.get_residues(restype='ALA', resnum=2)
            >>> print [f.get_number_of_atoms() for f in ala2]
            [12]

        @exampleuse:
            An example of using this method can be found in
            the test PDB_Molecules and in FDE_NMR_relax
        """
        res_list = []
        res_index = self.get_residx_of_atoms()

        ci = self.get_chain_info()

        for chain_id, res_name, res_num in self.residue_iter():
            ridx = self.get_residx_from_resinfo(chain_id, res_name, res_num, chaininfo=ci)
            if ((chain is None) or (chain_id == chain)) and ((restype is None) or (res_name == restype)) \
                    and ((resnum is None) or (res_num == resnum)) and ((idx is None) or (idx == ridx)):
                f = self.get_fragment([i + 1 for i, j in enumerate(res_index) if j == ridx])
                res_list.append(f)

        return res_list

    def get_atom_resinfo(self, atom):
        """
        Return the residue information (chainID, name, number) for a given atom.

        @param atom: atom number (staring at 1)
        @type atom: int

        @returns: tuple of (chainid, resname, resnum)

        @exampleuse:

            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> print an.get_atom_resinfo(10)
            ('B', 'TIP', 2)
        """
        res = self.mol.GetAtomWithIdx(atom - 1).GetPDBResidueInfo()

        if res is not None:
            return res.GetChainId(), res.GetResidueName(), res.GetResidueNumber()
        else:
            return None, None, None

    def get_chain_info(self):
        """
        Figure out the number of residues in each chains.

        @returns:
            chains: a list of the residue numbers in each chain
            chain_offsets: a list of the offset for the residue indexing in each chain

        @exampleuse
            >>> dian = RDMolecule('dialanine.xyz')
            >>> print dian.get_chain_info()
            ({'A': [(1, 'ALA'), (2, 'ALA')]}, {'A': 0})
            >>> dian.write('dialanine.pdb', 'pdb')
            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> print an.get_chain_info()
            ({'A': [(1, 'LIG')], 'B': [(1, 'TIP'), (2, 'TIP'), (3, 'TIP'), (4, 'TIP'),
                                       (5, 'TIP'), (6, 'TIP'), (7, 'TIP')]}, {'A': 0, 'B': 1})
        """
        # first figure out which chains and residues exist
        chains = {}
        for chain_id, res_name, res_num in self.residue_iter():
            if not (chain_id in chains):
                chains[chain_id] = []
            if (res_num, res_name) not in chains[chain_id]:
                chains[chain_id].append((res_num, res_name))

        for v in chains.values():
            v.sort()

        chain_offsets = {}
        kk = chains.keys()
        kk.sort()
        last_offset = 0
        for k in kk:
            chain_offsets[k] = last_offset
            last_offset = last_offset + len(chains[k])

        return chains, chain_offsets

    def get_residx_of_atoms(self, atoms=None, chaininfo=None):
        """
        Return a list giving the residue index of each atom.

        Indexing of the residues starts at 0 and counts all residues
        of the molecule, does not distinguish chains..

        @exampleuse:

            >>> dian = RDMolecule('dialanine.xyz')
            >>> print dian.get_residx_of_atoms()
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            >>> dian.write('dialanine.pdb', 'pdb')
            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> print an.get_residx_of_atoms()
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]

        """
        if chaininfo is None:
            ci = self.get_chain_info()
        else:
            ci = chaininfo

        if atoms is None:
            atoms = range(1, self.get_number_of_atoms() + 1)

        res_list = []
        for at in atoms:
            chain_id, res_name, res_num = self.get_atom_resinfo(at)
            residx = self.get_residx_from_resinfo(chain_id, res_name, res_num, chaininfo=ci)
            res_list.append(residx)
        return res_list

    def get_resnums_of_atoms(self, atoms=None):
        """
        Return a list giving the residue number of given atoms.

        Numbering of the residues as in the pdb file

        @exampleuse:
            >>> dian = RDMolecule('dialanine.xyz')
            >>> print dian.get_resnums_of_atoms()
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> print an.get_resnums_of_atoms()
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]
            >>> mol = RDMolecule('1PBE3.pdb', inputformat='pdb')
            >>> print mol.get_resnums_of_atoms()
            [21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
            23, 23, 23, 23, 23, 23, 23, 23, 23]
            >>> print mol.get_resnums_of_atoms(atoms=[1,8,9,18,19,27])
            [21, 21, 22, 22, 23, 23]
        """
        if atoms is None:
            atoms = range(1, self.get_number_of_atoms() + 1)

        res_list = []
        for at in atoms:
            chain_id, res_name, res_num = self.get_atom_resinfo(at)
            res_list.append(res_num)
        return res_list

    def get_resnums(self, chain=None):
        """
        Return a list giving the residue numbers (of a chain).

        @exampleuse:
        >>> mol = RDMolecule('1PBE3.pdb', inputformat='pdb')
        >>> print mol.get_resnums()
        [21, 22, 23]
        >>> mol = RDMolecule('an.pdb', inputformat='pdb')
        >>> print mol.get_resnums()
        [1, 1, 2, 3, 4, 5, 6, 7]
        >>> print mol.get_resnums(chain='A')
        [1]
        >>> print mol.get_resnums(chain='B')
        [1, 2, 3, 4, 5, 6, 7]
        """
        return [res_num for (chain_id, res_name, res_num) in self.residue_iter()
                if (chain is None) or (chain_id == chain)]

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

        @exampleuse:

            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> an.delete_residue(restype='TIP')
            >>> print an.print_coordinates()
            1)        C        0.83000000        0.66100000       -0.44400000    
            2)        N        0.00000000        0.00000000        0.00000000    
            3)        C        1.87800000        1.55900000       -0.81900000    
            4)        H        1.78500000        2.40300000       -0.13500000    
            5)        H        1.76200000        1.94900000       -1.83000000    
            6)        H        2.82900000        1.12200000       -0.51300000    
            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> an.delete_residue(chain='B')
            >>> print an.print_coordinates()
            1)        C        0.83000000        0.66100000       -0.44400000    
            2)        N        0.00000000        0.00000000        0.00000000    
            3)        C        1.87800000        1.55900000       -0.81900000    
            4)        H        1.78500000        2.40300000       -0.13500000    
            5)        H        1.76200000        1.94900000       -1.83000000    
            6)        H        2.82900000        1.12200000       -0.51300000    
            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> an.delete_residue(resnums=range(1,8), chain='B')
            >>> print an.print_coordinates()
            1)        C        0.83000000        0.66100000       -0.44400000    
            2)        N        0.00000000        0.00000000        0.00000000    
            3)        C        1.87800000        1.55900000       -0.81900000    
            4)        H        1.78500000        2.40300000       -0.13500000    
            5)        H        1.76200000        1.94900000       -1.83000000    
            6)        H        2.82900000        1.12200000       -0.51300000    

        """
        atomlist = []
        for at in range(1, self.get_number_of_atoms() + 1):
            chain_id, res_name, res_num = self.get_atom_resinfo(at)
            if (chain is None or chain_id == chain) and \
                    (restype is None or res_name == restype) and \
                    (resnums is None or res_num in resnums):
                atomlist.append(at)
        self.delete_atoms(atomlist)

    def get_restype_resnums(self, restype):
        """
        Get the residue numbers for a certain residue type.
  
        Use that to obtain the residues for which to set charges, e.g., for all GLU.
  
        @param restype: the type of the residue
        @type restype:  str
  
        @returns: A list of tuples (chain, residue number) belonging to restype. 
  
        @exampleuse:
  
            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> print an.get_restype_resnums('LIG')
            [('A', 1)]
            >>> print an.get_restype_resnums('TIP')
            [('B', 1), ('B', 2), ('B', 3), ('B', 4), ('B', 5), ('B', 6), ('B', 7)]
  
        """
        reslist = []
        for chain_id, resname, resnum in self.residue_iter():
            if resname == restype:
                reslist.append((chain_id, resnum))
        return reslist

    def get_residx_from_resinfo(self, chainid, resname, resnum, chaininfo=None):
        """
        Get a unique residue index number for the specified residue.

        @exampleuse:

            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> print an.get_residx_from_resinfo('A', 'LIG', 1)
            0
            >>> print an.get_residx_from_resinfo('B', 'TIP', 1)
            1

        """
        if chaininfo is None:
            chains, chain_offsets = self.get_chain_info()
        else:
            chains, chain_offsets = chaininfo
        residx = chain_offsets[chainid] + chains[chainid].index((resnum, resname))
        return residx

    def get_restype_residx(self, restype, chaininfo=None):
        """
        Get the residue indicess for a certain residue type.
  
        Use that to obtain the residues for which to set charges, e.g., for all GLU.
  
        @param restype: the type of the residue
        @type restype:  str
  
        @returns: A list of residue indices belonging to restype.
            Indices start at 0 and run over all chains. 
  
        @exampleuse:
  
            >>> an = RDMolecule('an.pdb', 'pdb')
            >>> print an.get_restype_residx('LIG')
            [0]
            >>> print an.get_restype_residx('TIP')
            [1, 2, 3, 4, 5, 6, 7]
  
        """
        if chaininfo is None:
            ci = self.get_chain_info()
        else:
            ci = chaininfo

        reslist = []
        for chain_id, resname, resnum in self.residue_iter():
            if resname == restype:
                reslist.append(self.get_residx_from_resinfo(chain_id, resname, resnum, chaininfo=ci))
        return reslist

    def delete_atoms(self, atoms):
        """
        Delete a list of atoms from the molecule.

        Atoms in list are first sorted in ascending order, then
        deleting starts from highest number so that numbering does not shift

        @param atoms: the list of atom numbers
        @type  atoms: list of ints
        @Warning: not yet checked whether all information is updated
                 (i.e., number of atoms, pse, ...)

        @exampleuse:

        >>> an = RDMolecule('an.xyz')
        >>> an.delete_atoms([1,5])
        >>> print an
        Cartesian coordinates:
        1)     C        1.77188        1.66625        0.53174
        2)     N        1.27005        2.49581        1.19175
        3)     H        2.29842       -0.34974        0.18696
        4)     H        3.47247        0.85113       -0.42037

        """
        self.is_ghost = [self.is_ghost[i] for i in range(self.get_number_of_atoms()) if i - 1 not in atoms]
        e = Chem.EditableMol(self.mol)
        for atomnumber in sorted(atoms)[::-1]:
            e.RemoveAtom(atomnumber - 1)
        self.mol = e.GetMol()

    def get_all_bonds(self):
        """
        Get a list of all bonds in the molecule.

        @returns: A list of pairs of atom numbers
                  (numbering of the atoms starts at 1)

        @exampleuse:

        >>> an = RDMolecule('an.pdb', 'pdb')
        >>> print len(an.get_all_bonds())
        19
        >>> diala = RDMolecule('dialanine.xyz', 'xyz')
        >>> bonds = diala.get_all_bonds()
        >>> print len(bonds)       
        22
        >>> for b in bonds:
        ...     b.sort()
        >>> bonds.sort()
        >>> print bonds[:12] 
        [[1, 2], [1, 3], [1, 8], [4, 5], [4, 6], [4, 7], [4, 8], [8, 9], [8, 10], [9, 11], [9, 14], [12, 13]]
        >>> print bonds[12:] 
        [[12, 14], [12, 16], [12, 20], [13, 15], [13, 18], [14, 17], [18, 19], [20, 21], [20, 22], [20, 23]]
 
        """
        bond_list = []

        for b in self.mol.GetBonds():
            bond_list.append([b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1])

        return bond_list

    def find_adjacent_hydrogens(self, atoms):
        """
        Return a list of all hydrogen atoms that are directly connected to one of the given atoms.

        @param atoms: the list of atoms numbers
        @type  atoms: list of ints

        @returns: the list of the numbers of the hydrogen atoms
        @rtype:   list of ints

        @exampleuse:

            >>> m = RDMolecule('an.pdb', 'pdb')
            >>> print sorted(m.find_adjacent_hydrogens([1,3,10]))
            [4, 5, 6, 11, 12]

        """
        return self.find_adjacent_atoms(atoms, atnum=1)

    def find_adjacent_atoms(self, atoms, atnum=None):
        """
        Return a list of all atoms that are directly connected to one of the given atoms.

        @param atoms: the list of atoms numbers
        @type  atoms: list of ints

        @param atnum: only include adjacent atoms with this atomic number
        @type  atoms: int

        @returns: the list of the numbers of the adjacent atoms
        @rtype:   list of ints
        """
        adjacent = []

        for i in atoms:
            a = self.mol.GetAtomWithIdx(i - 1)
            for b in a.GetBonds():
                for n in [b.GetBeginAtom(), b.GetEndAtom()]:
                    if (n.GetIdx() + 1) not in atoms:
                        if (atnum is None) or (n.GetAtomicNum() == atnum):
                            adjacent.append(n.GetIdx() + 1)

        return adjacent

    def get_smarts_matches(self, smartspattern):
        sp = Chem.MolFromSmarts(smartspattern)
        mol_wo_implicit_h = Chem.RemoveHs(self.mol, implicitOnly=True, sanitize=False)
        for i in range(mol_wo_implicit_h.GetNumAtoms()):
            at = mol_wo_implicit_h.GetAtomWithIdx(i)
            at.SetNoImplicit(True)
        maplist = [[a + 1 for a in mp] for mp in mol_wo_implicit_h.GetSubstructMatches(sp)]
        return maplist

    def set_residue(self, restype, resnum, chain=None, atoms=None):
        """
        Set the residue information for the given atoms (or all, if not given).

        @param restype: the residue name
        @type  restype: str
        @param resnum: the residue number
        @type  resnum: int
        @param chain: the chain name/number
        @type  chain: str
        @param atoms: the numbers of the atoms belonging to this residue
                      (atom numbering starts at 1)
        @type  atoms: list on ints

        @exampleuse:

        >>> an = RDMolecule('an.pdb', 'pdb')
        >>> an.set_residue('WAT', 999, 'C', atoms=range(7,10))
        >>> print an.get_residues('C', 'WAT', 999)[0]
          Cartesian coordinates: 
            1)        O       -1.46800000        2.60500000        1.37700000    
            2)        H       -0.95200000        3.29800000        0.96500000    
            3)        H       -1.16100000        1.79900000        0.96100000    
        <BLANKLINE>
        >>> an.write('an-modres.pdb', 'pdb')
        
        """

        if not atoms:
            atoms = range(1, self.mol.GetNumAtoms() + 1)
        if chain is None:
            chain = ""

        for i in atoms:
            a = self.mol.GetAtomWithIdx(i - 1)
            if a.GetPDBResidueInfo() is not None:
                atomname = a.GetPDBResidueInfo().GetName()
            else:
                atomname = '%3s ' % a.GetSymbol()
            a.SetMonomerInfo(Chem.AtomPDBResidueInfo(atomname,
                                                     residueName=restype,
                                                     residueNumber=resnum,
                                                     chainId=chain))

    def separate(self):
        """
        Separate the molecule into disconnected fragments.

        @returns: A list of molecules, one for each disconnected fragment
        @rtype:   list of L{molecule}s

        @exampleuse:

        >>> an = RDMolecule('an.pdb', 'pdb')
        >>> print len(an.separate())
        8
        
        """
        rdfrags = Chem.GetMolFrags(self.mol, asMols=True, sanitizeFrags=False)
        return [RDMolecule().set_RDMol(m) for m in rdfrags]

    def translate(self, vec):
        """
        Translate the molecule.

        @param vec: the translation vector
        @type  vec: numpy array or list of 3 floats

        @exampleuse:

        >>> h2o = RDMolecule('h2o.xyz')
        >>> h2o.translate([-1,2,1])
        >>> print h2o
          Cartesian coordinates: 
            1)        H       -1.21489000        5.43542000        3.17104000    
            2)        O       -1.89430000        5.96159000        3.68087000    
            3)        H       -1.43479000        6.75018000        4.07278000    
        <BLANKLINE>
        
        """
        c = self.mol.GetConformer()
        for atom in range(self.mol.GetNumAtoms()):
            p = c.GetAtomPosition(atom)
            p.x += vec[0]
            p.y += vec[1]
            p.z += vec[2]
            c.SetAtomPosition(atom, p)

    #        obvec = openbabel.vector3(vec[0], vec[1], vec[2])
    #        self.mol.Translate(obvec)

    def rotate(self, rotmat):
        """
        Rotate the molecule.

        @param rotmat: a 3x3 rotation matrix
        @type  rotmat: numpy 3x3 matrix
        """

        import numpy
        coords = numpy.array(self.get_coordinates())
        newcoords = numpy.dot(coords, rotmat.T)
        c = self.mol.GetConformer()
        for atom in range(self.mol.GetNumAtoms()):
            p = c.GetAtomPosition(atom)
            p.x = newcoords[atom][0]
            p.y = newcoords[atom][1]
            p.z = newcoords[atom][2]
            c.SetAtomPosition(atom, p)

    #         obmat = openbabel.double_array(rotmat.flatten().tolist())
    #         self.mol.Rotate(obmat)

    def align(self, other, atoms, atoms_other=None):
        """
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

        @exampleuse:

        >>> from numpy import round
        >>> pdb = RDMolecule('an.pdb', 'pdb')
        >>> mols = pdb.get_residues(restype = 'TIP')
        >>> rotmat,transvec = mols[0].align(mols[1],range(1,4))
        >>> print round(rotmat,6)
        [[ 0.40120700  0.55936000   0.72536100]
         [-0.47720900 -0.54829300  0.68676500]
         [ 0.78186000  -0.62168400  0.04695200]]
        >>> print round(transvec,5)
        [ 0.53705000 -2.72789000   2.34056000]

        """
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

            >>> mol = RDMolecule('h2o.xyz')
            >>> print mol.print_coordinates()
            1)     H       -0.21489        3.43542        2.17104
            2)     O       -0.89430        3.96159        2.68087
            3)     H       -0.43479        4.75018        3.07278

        @exampleuse:
            Printing of selected atoms:

            >>> mol = RDMolecule('h2o.xyz')
            >>> print mol.print_coordinates(atoms=[2])
            2)     O       -0.89430        3.96159        2.68087

        @exampleuse:
            Printing without atom numbering:

            >>> mol = RDMolecule('h2o.xyz')
            >>> print mol.print_coordinates(index=False)
            H       -0.21489        3.43542        2.17104
            O       -0.89430        3.96159        2.68087
            H       -0.43479        4.75018        3.07278

        @exampleuse:
            Printing with a suffix:

            >>> mol = RDMolecule('h2o.xyz')
            >>> print mol.print_coordinates(index=False, suffix='f=frag1')
            H       -0.21489        3.43542        2.17104    f=frag1
            O       -0.89430        3.96159        2.68087    f=frag1
            H       -0.43479        4.75018        3.07278    f=frag1

        @note:
            Coordinates are always printed in Angstrom units.

        """

        lines = ""
        if atoms is None:
            atoms = range(1, self.mol.GetNumAtoms() + 1)

        coords = self.get_coordinates(atoms)
        symbs = self.get_atom_symbols(atoms, prefix_ghosts=True)

        for i in range(len(atoms)):
            symb = symbs[i]
            c = coords[i]

            if index:
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
        xyz = str(self.mol.GetNumAtoms()) + '\n\n'
        c = self.mol.GetConformer()
        for a in self.mol.GetAtoms():
            p = c.GetAtomPosition(a.GetIdx())
            xyz += '%-3s %14.5f %14.5f %14.5f\n' % (a.GetSymbol(), p.x, p.y, p.z)
        return xyz

    def get_geovar_atoms_block(self, geovar):
        """
        Print the coordinates for use in the ATOMS block of ADF, using geovars.

        @param geovar: The atoms for which geovars should be used.
        @type geovar: list

        @exampleuse:
            Printing of the atoms using geovars:

            >>> mol = RDMolecule('h2o.xyz')
            >>> print mol.get_geovar_atoms_block([1,3])
            H         atom1x         atom1y         atom1z
            O    -0.89430000     3.96159000     2.68087000
            H         atom3x         atom3y         atom3z
        """

        c = self.mol.GetConformer()
        AtomsBlock = ""
        for i, atom in enumerate(self.mol.GetAtoms()):
            symb = atom.GetSymbol()
            if self.is_ghost[i]:
                symb = "Gh." + symb
            if i + 1 in geovar:
                varname = "atom" + str(i + 1)
                line = "  %5s " % symb
                line += "%14s " % (varname + "x")
                line += "%14s " % (varname + "y")
                line += "%14s \n" % (varname + "z")
            else:
                p = c.GetAtomPosition(i)
                line = "  %5s %14.5f %14.5f %14.5f \n" % \
                       (symb, p.x, p.y, p.z)
            AtomsBlock += line

        return AtomsBlock

    def get_geovar_block(self, geovar):
        """
        Print the GEOVAR block of ADF using the coordinates of the molecule.

        @param geovar: The atoms for which geovars should be used.
        @type geovar: list

        @exampleuse:
            Printing of the atoms using geovars:

            >>> mol = RDMolecule('h2o.xyz')
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

        c = self.mol.GetConformer()
        block = " GEOVAR\n"
        for i in geovar:
            p = c.GetAtomPosition(i - 1)
            block += "   atom" + str(i) + "x   %14.5f \n" % p.x
            block += "   atom" + str(i) + "y   %14.5f \n" % p.y
            block += "   atom" + str(i) + "z   %14.5f \n" % p.z
        block += " END\n\n"
        return block

    def get_dalton_molfile(self, basis):
        """
        Returns the content of a Dalton-style molecule file.

        @param basis: the basis set to use (for all atoms)
        @type  basis: str

        @exampleuse:
            Printing of the Dalton molecule file:

            >>> mol = RDMolecule('h2o.xyz')
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
                d = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2
                dists.append(d)

        return math.sqrt(min(dists))

    def distance_to_point(self, point, ghosts=True):
        """
        Measure the distance between the molecule and a given point.
        """
        coords = self.get_coordinates(ghosts=ghosts)

        dists = []
        for c in coords:
            d = (c[0] - point[0])**2 + (c[1] - point[1])**2 + (c[2] - point[2])**2
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

        @exampleuse:

            >>> mol = RDMolecule('h2o.xyz')
            >>> print mol.get_cube_header()
                1    0.00000000   -0.40608300    6.49200300    4.10267100
                8    0.00000000   -1.68998200    7.48632000    5.06611000
                1    0.00000000   -0.82163400    8.97653900    5.80671300
            <BLANKLINE>

        """
        header = ""

        coords = self.get_coordinates()
        # atoms = range(1, self.mol.GetNumAtoms() + 1)
        # atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        for at in self.mol.GetAtoms():
            i = at.GetIdx()
            if not self.is_ghost[i]:
                header += "%5d%12.6f%12.6f%12.6f%12.6f\n" % (at.GetAtomicNum(), 0.0,
                                                             coords[i][0] / Bohr_in_Angstrom,
                                                             coords[i][1] / Bohr_in_Angstrom,
                                                             coords[i][2] / Bohr_in_Angstrom)

        return header

    def has_spin_assigned(self):
        """
        Returns a boolean stating wether spin has been assigned by user

        @author: Rosa Bulo (REB)

        @rtype: bool
        """
        return self.TotalSpinMultiplicity != None

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

    h2o = RDMolecule()
    h2o.add_atoms(['H', 'O', 'H'],
                  [[-0.21489, 3.43542, 2.17104],
                   [-0.89430, 3.96159, 2.68087],
                   [-0.43479, 4.75018, 3.07278]])
    h2o.write('h2o.xyz')

    an = RDMolecule()
    an.add_atoms(['C', 'C', 'N', 'H', 'H', 'H'],
                 [[2.40366, 0.63303, -0.29209],
                  [1.77188, 1.66625, 0.53174],
                  [1.27005, 2.49581, 1.19175],
                  [2.29842, -0.34974, 0.18696],
                  [1.92918, 0.59583, -1.28199],
                  [3.47247, 0.85113, -0.42037]])
    an.write('an.xyz')

    f = open('an.pdb', 'w')
    f.write("""REMARK DATE:19-Oct-06  15:59:22       created by user: bulo                     
ATOM      1  C   LIG A   1       0.830   0.661  -0.444  1.00  0.00      MOL  C
ATOM      2  N   LIG A   1       0.000   0.000   0.000  1.00  0.00      MOL  N 
ATOM      3  C   LIG A   1       1.878   1.559  -0.819  1.00  0.00      MOL  C
ATOM      4  H   LIG A   1       1.785   2.403  -0.135  1.00  0.00      MOL  H
ATOM      5  H   LIG A   1       1.762   1.949  -1.830  1.00  0.00      MOL  H
ATOM      6  H   LIG A   1       2.829   1.122  -0.513  1.00  0.00      MOL  H
ATOM      7  OH2 TIP3B   1      -1.468   2.605   1.377  1.00  0.00      WAT  O
ATOM      8  H1  TIP3B   1      -0.952   3.298   0.965  1.00  0.00      WAT  H
ATOM      9  H2  TIP3B   1      -1.161   1.799   0.961  1.00  0.00      WAT  H
ATOM     10  OH2 TIP3B   2       2.404  -2.510  -0.362  1.00  0.00      WAT  O
ATOM     11  H1  TIP3B   2       2.700  -3.419  -0.409  1.00  0.00      WAT  H
ATOM     12  H2  TIP3B   2       1.775  -2.500   0.359  1.00  0.00      WAT  H
ATOM     13  OH2 TIP3B   3      -3.228  -1.615   1.185  1.00  0.00      WAT  O
ATOM     14  H1  TIP3B   3      -3.333  -2.553   1.030  1.00  0.00      WAT  H
ATOM     15  H2  TIP3B   3      -3.142  -1.236   0.310  1.00  0.00      WAT  H
ATOM     16  OH2 TIP3B   4       0.840  -2.612   2.890  1.00  0.00      WAT  O
ATOM     17  H1  TIP3B   4       0.588  -3.437   3.305  1.00  0.00      WAT  H
ATOM     18  H2  TIP3B   4       0.025  -2.115   2.829  1.00  0.00      WAT  H
ATOM     19  OH2 TIP3B   5       2.954  -0.851   2.997  1.00  0.00      WAT  O
ATOM     20  H1  TIP3B   5       2.120  -1.224   2.712  1.00  0.00      WAT  H
ATOM     21  H2  TIP3B   5       2.718  -0.241   3.696  1.00  0.00      WAT  H
ATOM     22  OH2 TIP3B   6       3.622  -0.740  -2.193  1.00  0.00      WAT  O
ATOM     23  H1  TIP3B   6       3.051  -1.252  -1.621  1.00  0.00      WAT  H
ATOM     24  H2  TIP3B   6       4.081  -0.142  -1.602  1.00  0.00      WAT  H
ATOM     25  OH2 TIP3B   7      -3.800  -1.131  -1.711  1.00  0.00      WAT  O
ATOM     26  H1  TIP3B   7      -3.026  -0.809  -2.174  1.00  0.00      WAT  H
ATOM     27  H2  TIP3B   7      -4.316  -0.345  -1.533  1.00  0.00      WAT  H
""")
    f.close()

    f = open('an_bla.pdb', 'w')
    f.write("""REMARK DATE:19-Oct-06  15:59:22       created by user: bulo                     
ATOM      1  C   LIG B   0       0.830   0.661  -0.444  1.00  0.00      MOL  C
ATOM      2  N   LIG B   0       0.000   0.000   0.000  1.00  0.00      MOL  N 
ATOM      3  C   LIG B   0       1.878   1.559  -0.819  1.00  0.00      MOL  C
ATOM      4  H   LIG B   0       1.785   2.403  -0.135  1.00  0.00      MOL  H
ATOM      5  H   LIG B   0       1.762   1.949  -1.830  1.00  0.00      MOL  H
ATOM      6  H   LIG B   0       2.829   1.122  -0.513  1.00  0.00      MOL  H
ATOM      7  OH2 TIP3A   0      -1.468   2.605   1.377  1.00  0.00      WAT  O
ATOM      8  H1  TIP3A   0      -0.952   3.298   0.965  1.00  0.00      WAT  H
ATOM      9  H2  TIP3A   0      -1.161   1.799   0.961  1.00  0.00      WAT  H
ATOM     10  OH2 TIP3A   2       2.404  -2.510  -0.362  1.00  0.00      WAT  O
ATOM     11  H1  TIP3A   2       2.700  -3.419  -0.409  1.00  0.00      WAT  H
ATOM     12  H2  TIP3A   2       1.775  -2.500   0.359  1.00  0.00      WAT  H
ATOM     13  OH2 TIP3A   3      -3.228  -1.615   1.185  1.00  0.00      WAT  O
ATOM     14  H1  TIP3A   3      -3.333  -2.553   1.030  1.00  0.00      WAT  H
ATOM     15  H2  TIP3A   3      -3.142  -1.236   0.310  1.00  0.00      WAT  H
ATOM     16  OH2 TIP3A   4       0.840  -2.612   2.890  1.00  0.00      WAT  O
ATOM     17  H1  TIP3A   4       0.588  -3.437   3.305  1.00  0.00      WAT  H
ATOM     18  H2  TIP3A   4       0.025  -2.115   2.829  1.00  0.00      WAT  H
ATOM     19  OH2 TIP3A   8       2.954  -0.851   2.997  1.00  0.00      WAT  O
ATOM     20  H1  TIP3A   8       2.120  -1.224   2.712  1.00  0.00      WAT  H
ATOM     21  H2  TIP3A   8       2.718  -0.241   3.696  1.00  0.00      WAT  H
ATOM     22  OH2 TIP3A  12       3.622  -0.740  -2.193  1.00  0.00      WAT  O
ATOM     23  H1  TIP3A  12       3.051  -1.252  -1.621  1.00  0.00      WAT  H
ATOM     24  H2  TIP3A  12       4.081  -0.142  -1.602  1.00  0.00      WAT  H
ATOM     25  OH2 TIP3A  13      -3.800  -1.131  -1.711  1.00  0.00      WAT  O
ATOM     26  H1  TIP3A  13      -3.026  -0.809  -2.174  1.00  0.00      WAT  H
ATOM     27  H2  TIP3A  13      -4.316  -0.345  -1.533  1.00  0.00      WAT  H
""")
    f.close()

    f = open('anNOH.pdb', 'w')
    f.write("""REMARK DATE:19-Oct-06  15:59:22       created by user: bulo                     
ATOM      1  C   LIG A   1       0.830   0.661  -0.444  1.00  0.00      MOL  C
ATOM      2  N   LIG A   1       0.000   0.000   0.000  1.00  0.00      MOL  N 
ATOM      3  C   LIG A   1       1.878   1.559  -0.819  1.00  0.00      MOL  C
ATOM      7  OH2 TIP3B   1      -1.468   2.605   1.377  1.00  0.00      WAT  O
ATOM     10  OH2 TIP3B   2       2.404  -2.510  -0.362  1.00  0.00      WAT  O
ATOM     13  OH2 TIP3B   3      -3.228  -1.615   1.185  1.00  0.00      WAT  O
""")
    f.close()

    f = open('1PBE3.pdb', 'w')
    f.write("""HEADER    OXIDOREDUCTASE                          06-JUL-94   1PBE              
ATOM    137  N   LEU A  21       9.727  97.836  71.643  1.00 13.50           N  
ATOM    138  CA  LEU A  21       9.876  97.777  73.087  1.00 16.20           C  
ATOM    139  C   LEU A  21       9.339  99.042  73.792  1.00 15.11           C  
ATOM    140  O   LEU A  21       8.737  99.002  74.868  1.00 19.11           O  
ATOM    141  CB  LEU A  21      11.310  97.465  73.467  1.00 12.96           C  
ATOM    142  CG  LEU A  21      11.992  96.143  73.219  1.00 12.45           C  
ATOM    143  CD1 LEU A  21      13.462  96.207  73.626  1.00  8.78           C  
ATOM    144  CD2 LEU A  21      11.347  95.047  74.103  1.00 14.87           C  
ATOM    145  N   HIS A  22       9.591 100.173  73.211  1.00 19.90           N  
ATOM    146  CA  HIS A  22       9.197 101.467  73.810  1.00 19.34           C  
ATOM    147  C   HIS A  22       7.695 101.476  73.920  1.00 21.09           C  
ATOM    148  O   HIS A  22       7.230 101.695  75.019  1.00 21.48           O  
ATOM    149  CB  HIS A  22       9.707 102.673  72.988  1.00 25.03           C  
ATOM    150  CG  HIS A  22       9.358 103.940  73.722  1.00 24.50           C  
ATOM    151  ND1 HIS A  22      10.074 104.467  74.745  1.00 25.57           N  
ATOM    152  CD2 HIS A  22       8.301 104.772  73.489  1.00 25.55           C  
ATOM    153  CE1 HIS A  22       9.477 105.595  75.128  1.00 35.00           C  
ATOM    154  NE2 HIS A  22       8.405 105.813  74.387  1.00 23.11           N  
ATOM    155  N   LYS A  23       7.052 101.224  72.799  1.00 20.55           N  
ATOM    156  CA  LYS A  23       5.598 101.149  72.693  1.00 25.19           C  
ATOM    157  C   LYS A  23       4.980 100.123  73.659  1.00 25.76           C  
ATOM    158  O   LYS A  23       3.789 100.278  73.997  1.00 27.71           O  
ATOM    159  CB  LYS A  23       5.126 100.827  71.289  1.00 26.67           C  
ATOM    160  CG  LYS A  23       5.560 101.692  70.134  1.00 39.91           C  
ATOM    161  CD  LYS A  23       4.704 102.867  69.790  1.00 58.31           C  
ATOM    162  CE  LYS A  23       3.224 102.586  69.671  1.00 71.72           C  
ATOM    163  NZ  LYS A  23       2.915 101.249  69.098  1.00 72.06           N  
END                                                                             
""")
    f.close()

    f = open('dialanine.xyz', 'w')
    f.write("""23

N       0.97280732       0.73537547       0.03408123
H       0.33743395       0.04973960       0.45249170
H       1.73319634       0.82429562       0.71833093
C       2.39168837       1.22701642      -1.90058653
H       1.78621751       2.12994815      -2.04776317
H       3.26758017       1.49017277      -1.28867103
H       2.75668458       0.88009116      -2.87706875
C       1.54424739       0.16640181      -1.18473409
C       2.39345347      -1.09809018      -0.91593123
H       0.70597089      -0.11721773      -1.84374712
O       2.78627527      -1.39752392       0.21229639
C       3.66617759      -2.91250919      -1.98101717
C       4.72107905      -2.62836309      -3.04102219
N       2.66506334      -1.85958295      -2.02551493
O       4.56822496      -1.89170803      -3.99699206
H       4.13265868      -2.86478311      -0.98518915
H       2.50907074      -1.45042396      -2.94679813
O       5.85982586      -3.33781244      -2.81160980
H       6.46407233      -3.11928450      -3.55895889
C       3.05615196      -4.30706571      -2.19293690
H       3.82835149      -5.08472715      -2.13258703
H       2.56819458      -4.36693631      -3.17606991
H       2.30387296      -4.48773848      -1.41554000
""")
    f.close()


def _tearDown_doctest(test):
    # pylint: disable-msg=W0613

    import os

    # os.remove('h2o.xyz')
    # os.remove('an.xyz')

    os.rmdir('molecule_doctests')
