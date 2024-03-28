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
 Job and results for ADF 3-partition FDE calculations.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""

from .Errors import PyAdfError
from .Molecule import MoleculeFactory, OBMolecule, RDMolecule
from .BaseJob import metajob, results
from .ADFFragments import fragment, fragmentlist, adffragmentsjob
from .ADFSinglePoint import adfsinglepointjob, adfsinglepointresults
from pyadf.PyEmbed.Plot.Grids import cubegrid
from functools import reduce


class CapMoleculeMixin:
    """
    A molecule used in a cap or capped fragment.

    It stores information on the atoms in the cap(s) and
    on the different parts of the caps. (It used residues
    within Openbabel for this: caps are residues with the
    name 'CAP'. Each cap has two parts (1 and 2), which are
    marked with even / odd residue numbers.
    """

    def __init__(self, mol=None):
        """
        Initialize a cap molecule.

        @param mol: a standard L{molecule}
        @type mol:  L{molecule}
        """
        super().__init__()
        if mol is not None:
            self.copy(mol)

    def cap_id(self, atom):
        """
        Get the "cap id" of the given atom.

        The cap id is 1 or 2 if the atom is in part 1 or 2 of the cap,
        respectively. For disulfide bonds, cap ids are 3 or 4.
        Otherwise, it is 0.

        @param atom: the atom number (Atom numbering starts at 1)
        @type atom:  int
        """
        chain_id, resname, resnum = self.get_atom_resinfo(atom)

        if resname == 'CAP':
            # even residue number: capid = 1
            #  odd residue number: capid = 2
            capid = (resnum % 2) + 1
        elif resname == 'SCP':
            # even residue number: capid = 3
            #  odd residue number: capid = 4
            capid = (resnum % 2) + 3
        else:
            capid = 0

        return capid

    def get_atom_symbols(self, atoms=None, ghosts=True, prefix_ghosts=False):
        """
        Give back an array with the atom symbols.

        For cap atoms, the suffix cap1 or cap2 is appended to the atom symbol
        if the atom is in part 1 or 2 of the cap, respectively.
        For disulfide cap atoms, the suffix scap1 or scap2 is appended.


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

        """
        if atoms is None:
            atoms = list(range(1, self.get_number_of_atoms() + 1))
            if not ghosts:
                atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        # noinspection PyUnresolvedReferences
        symbols = super().get_atom_symbols(atoms, ghosts, prefix_ghosts)

        for i, at in enumerate(atoms):
            if self.cap_id(at) == 1:
                symbols[i] = symbols[i] + ".cap1"
            elif self.cap_id(at) == 2:
                symbols[i] = symbols[i] + ".cap2"
            elif self.cap_id(at) == 3:
                symbols[i] = symbols[i] + ".scap1"
            elif self.cap_id(at) == 4:
                symbols[i] = symbols[i] + ".scap2"

        return symbols

    def get_noncap_fragment(self):
        atoms = []
        for i in range(1, self.get_number_of_atoms() + 1):
            chain_id, resname, resnum = self.get_atom_resinfo(i)

            if not (resname == 'CAP' or resname == 'SCP'):
                atoms.append(i)

        return self.get_fragment(atoms)


if OBMolecule is not None:
    class OBCapMolecule(CapMoleculeMixin, OBMolecule.OBMolecule):
        pass

if RDMolecule is not None:
    class RDCapMolecule(CapMoleculeMixin, RDMolecule.RDMolecule):
        pass


def capmolecule(mol=None):
    if (OBMolecule is not None) and isinstance(mol, OBMolecule.OBMolecule):
        return OBCapMolecule(mol)
    elif (RDMolecule is not None) and isinstance(mol, RDMolecule.RDMolecule):
        return RDCapMolecule(mol)
    elif MoleculeFactory().molclass == "openbabel":
        return OBCapMolecule(mol)
    elif MoleculeFactory().molclass == "rdkit":
        return RDCapMolecule(mol)
    else:
        return None


class cappedfragment(fragment):
    """
    A capped fragments, as used in 3-partition FDE.

    @undocumented: __delattr__, __getattribute__, __hash__, __new__,
                   __repr__, __str__, __setattr__

    """

    def __init__(self, frag_results, mol=None, subfrag=None, isfrozen=False, fdeoptions=None,
                 occ=None, fragoptions=None, iscap=False):
        """
        Initialize a capped fragment (or more precisely, a fragment type).

        @param frag_results:
            Results of a previous ADF singlepoint job. If C{None}, this
            fragment is considered to be active, which means that it consists
            of atomic fragments for which fragment files will be prepared
            automatically.
        @type  frag_results: L{adfsinglepointresults} or None

        @param mol:
            A molecule, giving the coordinates where this fragment
            is used.
        type mol: L{molecule} or list of L{molecule}s

        @param subfrag: The name of the subfragment to be used.
        @type  subfrag: str

        @param isfrozen: Specify if this fragment is frozen (FDE)
        @type  isfrozen: bool

        @param fdeoptions:
            A dictionary of FDE options.
            Possible options are:
            For frozen fragments: USEBASIS, RELAX, XC, DENSTYPE,
            For active fragments: LBDAMP, CAPRADIUS, SCFCONVTHRESH, NOCAPSEPCONV, LBMAXSTEP
            B{Options have to be in upper case}
        @type  fdeoptions: dictionary

        @param occ: fragment occupation numbers
        @type occ: list of lists of three elements in the format
                   ['irrep', num_alpha, num_beta]

        @param fragoptions:
            For fragments, possibly a dictionary of adfsettings options
        @type fragoptions: dictionary
        """

        if mol:
            if isinstance(mol, list):
                raise PyAdfError('Only one molecule possible in capped fragment')
            else:
                mol = capmolecule(mol)

        super().__init__(frag_results, mols=mol, subfrag=subfrag, isfrozen=isfrozen,
                         fdeoptions=fdeoptions, occ=occ, fragoptions=fragoptions)

        self._iscap = iscap

        self.num_cap = -1

        self._cap_fragments = []
        self._cap_residue_nums = []
        self._overlapping_caps = []
        self._charge = 0

    @property
    def mol(self):
        return self._mols[0]

    def get_overlapping_caps(self):
        return self._overlapping_caps

    def add_cap(self, cap, cap_id):
        """
        Add a cap to this fragment.

        This will add the respective cap atoms to the molecule and
        invalidate the results of a previous fragment calculation.

        @param cap: the cap fragment
        @type  cap: L{fragment}
        @param cap_id: the id of the part of the cap to be used (1 or 2)
        @type  cap_id: int
        """

        self._cap_fragments.append(cap)

        resnum = 2 * (len(self._cap_fragments) - 1) + cap_id
        self._cap_residue_nums.append(resnum)

        m_cap = cap.get_molecules()[0].get_residues(restype="CAP", resnum=cap_id)[0]

        i1 = self.mol.get_number_of_atoms() + 1
        i2 = i1 + m_cap.get_number_of_atoms()

        self._mols[0] = self._mols[0] + m_cap
        self._mols[0].set_residue('CAP', resnum, atoms=list(range(i1, i2)))

    def add_scap(self, cap, cap_id):
        """
        Add a disulfide cap to this fragment.

        This will add the respective cap atoms to the molecule and
        invalidate the results of a previous fragment calculation.

        @param cap: the cap fragment
        @type  cap: L{fragment}
        @param cap_id: the id of the part of the cap to be used (3 or 4)
        @type  cap_id: int
        """

        self._cap_fragments.append(cap)

        resnum = 2 * (len(self._cap_fragments) - 1) + cap_id + 2
        self._cap_residue_nums.append(resnum)

        m_cap = cap.get_molecules()[0].get_residues(restype="SCP", resnum=cap_id - 2)[0]

        i1 = self.mol.get_number_of_atoms() + 1
        i2 = i1 + m_cap.get_number_of_atoms()

        self._mols[0] = self._mols[0] + m_cap
        self._mols[0].set_residue('SCP', resnum, atoms=list(range(i1, i2)))

    def get_type(self):
        if self.iscap:
            ftype = "type=FDEsubstract"
        else:
            ftype = super().get_type()
        return ftype

    @property
    def iscap(self):
        """
        Whether this fragment is a cap fragment.
        """
        return self._iscap

    @iscap.setter
    def iscap(self, value):
        self._iscap = value

    def print_fragment_options(self):
        if self.iscap:
            print(" type: frozen FDE cap fragment")
            if len(self._fdeoptions) > 0:
                print("        FDE options: ")
            for opt, value in self._fdeoptions.items():
                print("           ", opt, "  ", value)
            print()
        else:
            super().print_fragment_options()

    def get_atoms_block(self):

        atoms_block = ""

        if (not self.has_frag_results()) or (len(self._cap_fragments) == 0):
            atoms_block += super().get_atoms_block()
        else:
            if len(self._mols) > 1:
                raise PyAdfError("Capped fragments must appear only once")

            mol = self.mol.get_noncap_fragment()
            suffix = "f=" + self.fragname
            atoms_block += mol.print_coordinates(index=False, suffix=suffix)

            for cap_frag, cap_res in zip(self._cap_fragments, self._cap_residue_nums):
                if cap_res < 5:
                    mol = self.mol.get_residues(restype='CAP', resnum=cap_res)[0]
                    mol = capmolecule(mol)
                    suffix = "f=" + self.fragname + "   fs=cap" + str(cap_frag.num_cap)
                    atoms_block += mol.print_coordinates(index=False, suffix=suffix)
                elif cap_res > 4:
                    mol_s = self.mol.get_residues(restype='SCP', resnum=cap_res)[0]
                    mol_s = capmolecule(mol_s)
                    suffix = "f=" + self.fragname + "   fs=scap" + str(cap_frag.num_cap)
                    atoms_block += mol_s.print_coordinates(index=False, suffix=suffix)

        return atoms_block

    def get_cap_residue_nums(self, intersection, newcappedfragment):
        """
        returns a list of cap residue numbers

        @param intersection: cappedfragment
        @type  intersection: set(cappedfragment)
        @param newcappedfragment: cappedfragment
        @type  newcappedfragment: cappedfragment
        """
        newcoords = newcappedfragment.mol.get_coordinates()
        capresnums = []
        for c in self._cap_fragments:
            tempcapresnums = []
            if c not in intersection:
                newcappedfragment._cap_fragments.append(c)
                capcoords = c.mol.get_coordinates()
                for atomnum, coords in enumerate(newcoords):
                    if coords in capcoords:
                        chainid, resname, resnum = newcappedfragment.mol.get_atom_resinfo(atomnum + 1)
                        if resname == 'CAP' or resname == 'SCP':
                            tempcapresnums.append(resnum)
            if tempcapresnums:
                capresnums.append(tempcapresnums[0])
        return capresnums

    def merge_fragments(self, frag):
        """
        merges two cappedfragments and returns a new cappedfragment

        @param frag: cappedfragment
        @type  frag: cappedfragment
        """
        mol1 = self.mol
        mol2 = frag.mol
        newmol = mol1 + mol2
        charge = mol1.get_charge() + mol2.get_charge()

        # NONDISJOINT FRAGMENTS
        intersection = set(self._cap_fragments).intersection(set(frag._cap_fragments))
        if intersection:
            olcap = list(intersection)[0]
            capcoords = olcap.mol.get_coordinates()
            deletelist = []
            for atomnum, coords in enumerate(newmol.get_coordinates()):
                chainid, resname, resnum = newmol.get_atom_resinfo(atomnum + 1)
                if coords in capcoords:
                    if resname == 'CAP' or resname == 'SCP':
                        deletelist.append(atomnum + 1)
            newmol.delete_atoms(deletelist)
            newmol.set_spin(0)
            newcappedfragment = cappedfragment(None, newmol)
            newcappedfragment._charge += charge
            newcappedfragment.mol.set_charge(charge)
            for cap in self._overlapping_caps:
                newcappedfragment._overlapping_caps.append(cap)
            for cap in frag._overlapping_caps:
                newcappedfragment._overlapping_caps.append(cap)
            mol1_cap_res_nums = self.get_cap_residue_nums(intersection, newcappedfragment)
            mol2_cap_res_nums = frag.get_cap_residue_nums(intersection, newcappedfragment)
            capresnums = mol1_cap_res_nums + mol2_cap_res_nums
            for i in capresnums:
                newcappedfragment._cap_residue_nums.append(i)
            newcappedfragment._overlapping_caps.append(olcap)

        # DISJOINT FRAGMENTS
        else:
            newcappedfragment = cappedfragment(None, newmol)
            for cap in self._cap_fragments:
                newcappedfragment._cap_fragments.append(cap)
            for cap in frag._cap_fragments:
                newcappedfragment._cap_fragments.append(cap)
            for capresnum in self._cap_residue_nums:
                newcappedfragment._cap_residue_nums.append(capresnum)
            for capresnum in frag._cap_residue_nums:
                newcappedfragment._cap_residue_nums.append(capresnum)
        return newcappedfragment


class cappedfragmentlist(fragmentlist):
    """
    List of capped fragments and caps.

    This is needed for 3-partition FDE jobs (L{adf3fdejob}).
    """

    def __init__(self):
        """
        Create a cappedfragmentlist.
        """

        super().__init__(frags=None)

        self._caps = []

    def __iter__(self):
        """
        Iteration over all fragments (including caps).
        """
        import itertools
        return itertools.chain(self._frags, self._caps)

    def fragiter(self):
        """
        Iteration over fragments only (not over caps).
        """
        return self._frags.__iter__()

    def capiter(self):
        """
        Iteration over caps.
        """
        return self._caps.__iter__()

    @property
    def caps(self):
        return self._caps

    def set_charges(self, reschargelist=None):
        """
        sets charges for every cappedfragment

        @param reschargelist: list with charged residues [['A2', -1], ['B3', +2]]
        @type  reschargelist: list
        """
        for frag in self.fragiter():
            natoms = frag.mol.get_number_of_atoms()
            chainresnumlist = []
            for i in range(natoms):
                chain, resname, resnum = frag.mol.get_atom_resinfo(i+1)
                if resname not in ['CAP', 'SCP']:
                    chainresnumlist.append(chain + str(resnum))
            charge = 0
            for res in reschargelist:
                if res[0] in chainresnumlist:
                    charge += res[1]
            frag.mol.set_charge(charge)
        return

    def get_total_molecule(self):
        """
        Returns the total molecule.

        @rtype: L{molecule}
        """
        mols = []
        for frag in self.fragiter():
            m = frag.get_molecules()[0].get_noncap_fragment()
            mols.append(m)
        return reduce(lambda x, y: x + y, mols)

    def get_atoms_block(self):
        atoms_block = ""
        atoms_block += " ATOMS [Angstrom]\n"
        for frag in self.fragiter():
            atoms_block += frag.get_atoms_block()
        atoms_block += " END\n\n"
        atoms_block += " AllowCloseAtoms\n\n"
        return atoms_block

    def get_fragments_block(self, checksum_only):
        block = ""
        for frag in self.fragiter():
            if frag.has_frag_results():
                block += frag.get_fragments_block(checksum_only)
        for cap in self.capiter():
            block += cap.get_fragments_block(checksum_only)
        return block

    def append_cap(self, cap, frag1, frag2):
        """
        Append a cap fragment

        @param cap: the cap fragment
        @type  cap: L{fragment}

        @param frag1: the fragment capped by the first part of the cap
        @type  frag1: L{cappedfragment}

        @param frag2: the fragment capped by the second part of the cap
        @type  frag2: L{cappedfragment}
        """

        cap.isfrozen = True
        cap.iscap = True
        cap.num_cap = len(self._caps) + 1
        self._caps.append(cap)
        self._caps[-1].fragname = "cap" + str(len(self._caps))

        frag1.add_cap(cap, 1)
        frag2.add_cap(cap, 2)

    def append_scap(self, cap, frag1, frag2):
        """
        Append a disulfide cap fragment

        @param cap: the cap fragment
        @type  cap: L{fragment}

        @param frag1: the fragment capped by the first part of the cap
        @type  frag1: L{cappedfragment}

        @param frag2: the fragment capped by the second part of the cap
        @type  frag2: L{cappedfragment}
        """

        cap.isfrozen = True
        cap.iscap = True
        cap.num_cap = len(self._caps) + 1
        self._caps.append(cap)
        self._caps[-1].fragname = "scap" + str(len(self._caps))

        frag1.add_scap(cap, 3)
        frag2.add_scap(cap, 4)

    def partition_protein(self, mol, special_reslists=None, fragsize=None, caps=None):
        """
        Partition a protein into the individual amino acids and caps.

        @param mol: the protein
        @type mol:  L{molecule}

        @param special_reslists:
            lists of residue numbers that should be treated within one 3-FDE fragment
        @type special_reslists: list of lists of integers

        @param fragsize:
            number of residues in one 3-FDE fragment (not for residues in special_reslist)
        @type fragsize: int

        @param caps:
            type of cap molecule, standard is mfcc caps, option is hydrogen caps
        @type caps: str
        """

        # generate fragment list

        res_of_atoms = mol.get_residx_of_atoms()

        capped_bonds = []
        # list of bonds between residues within fragment
        non_capped_res = []
        # to which fragment belongs the residue
        frag_indices = []

        # clear the fragment list
        self._frags = []
        self._caps = []

        if fragsize is None:
            fragsize = 1
        if special_reslists is None:
            special_reslists = {}

        # ----------------------------------------------------------------------------------
        # simplest fragmentation: each residue is one fragment

        if special_reslists == {} and fragsize == 1:

            # get the individual residues
            for res in mol.get_residues():
                self.append(cappedfragment(None, res))
                frag_indices.append(len(self._frags) - 1)
        # ----------------------------------------------------------------------------------
        # more complicated fragmentation

        else:
            # careful with residue numbers (internal openbabel index idx starts at 0)

            fragmentation_list = []
            residuelist = mol.get_residues()
            frag_indices = [-1] * len(residuelist)

            # get list of residues connected by covalent bonds (tuples)
            connected_residues = []
            for b in mol.get_all_bonds():
                if not (res_of_atoms[b[0] - 1] == res_of_atoms[b[1] - 1]):
                    # -1 is not totally clear to me, but works
                    connected_residues.append({res_of_atoms[b[0] - 1], res_of_atoms[b[1] - 1]})

            # ------------------------------------------------------------------------------
            # first get fragmentation list: which residues belong to which fragment
            # something like [ [3,4,5], [11,12,14], [1, 2], [6, 7], [8, 9], [10], [13]]

            # ugly
            # dictionary that connects pdb resnums with internal idx
            # problem: resnum as key is not unique, use combination of residue name and chain as key
            res_idx = {}

            for reschain, resname, resnum in mol.residue_iter():
                reskey = 'c' + str(reschain) + str(resnum)
                res_idx[reskey] = mol.get_residx_from_resinfo(reschain, resname, resnum)

            # convert residue numbers in special_reslists to internal numbers
            for i in range(len(special_reslists)):
                for j in range(len(special_reslists[i])):
                    special_reslists[i][j] = res_idx[special_reslists[i][j]]

            # flatten list of lists so that you can check whether a residue is a special residue
            specialresidxs = []
            for slist in special_reslists:
                specialresidxs.extend(slist)

            # first add special residues to fragmentation list
            for slist in special_reslists:
                fragmentation_list.append(slist)

            fragl = []  # templist for fragments
            # add all other residues
            # use res.GetIdx() instead of enumerate
            for res, (reschain, resname, resnum) in zip(residuelist, mol.residue_iter()):
                residx = mol.get_residx_from_resinfo(reschain, resname, resnum)
                if residx not in specialresidxs:
                    # take care of fragsize
                    if fragsize == 1:
                        fragmentation_list.append([residx])
                    else:
                        # check whether there has to be a new fragment
                        if len(fragl) != 0:
                            if not {residx, fragl[-1]} in connected_residues or \
                                    len(fragl) == fragsize:
                                # no covalent connection? new fragment! fragsize reached?
                                fragmentation_list.append(fragl)
                                fragl = []
                        # new fragment
                        if len(fragl) == 0:
                            fragl.append(residx)
                        # add to fragment
                        else:
                            fragl.append(residx)
            # last fragment
            if len(fragl) != 0:
                fragmentation_list.append(fragl)

            # ------------------------------------------------------------------------------
            # now add fragments to _frags using fragmentation list
            # take care of uncapped bonds and fragindices etc.

            frag_reslist = []

            for rlist in fragmentation_list:
                for rnum in range(len(rlist)):
                    res = mol.get_residues(idx=rlist[rnum])[0]
                    frag_reslist.append(res)

                    # if residue is added to an existing fragment: add tuple of residues to
                    # list of connected (non-capped) residues (only if they are connected)

                    # take also care of disulfide bonds within one fragment
                    # check whether new residue is connected to another residue within this fragment
                    if len(frag_reslist) > 1:
                        for rd in range(len(frag_reslist)):
                            if {rlist[rnum], rlist[rd]} in connected_residues:
                                non_capped_res.append({rlist[rnum], rlist[rd]})

                    # take care of residue information here
                    frag_indices[rlist[rnum]] = len(self._frags)

                res_frag = fragment(None, mols=frag_reslist)

                self.append(cappedfragment(None, res_frag.get_total_molecule()))
                # clear frag_reslist for next fragment
                frag_reslist = []
        # ------------------------------------------------------------------------------

        # find all peptide bonds and create caps

        if caps is None:
            caps = 'mfcc'

        if caps == 'mfcc':
            sp = '[NX3,NX4][C;X4;H1,H2][CX3](=O)[NX3][C;X4;H1,H2][CX3](=O)'
            # this SMARTS pattern matches all peptide bonds
            #
            # (in this comment: atom numbering starts at 1;
            #  in the code below the numbering of the list starts at 0)
            # the peptide bond is between atoms 2 and 4
            # atoms 1-5 (+ connected hydrogens) form the cap fragment
            # part 1 (o-part): atoms 1-3
            # part 2 (n-part): atoms 4-5

        elif caps == 'hydrogen':
            sp = '[NX3,NX4][C;X4;H1,H2][CX3](=O)[NX3]([*])[C;X4;H1,H2][CX3](=O)'
            # same as above, but also matches any atom bound to the nitrogen
            # the * might not be a good idea, but [C,H] was not working...
            # (this is needed for the proper calculation of the hydrogen coordinates)

        else:
            raise PyAdfError('unsupported cap type, please specify one of the following: mfcc, hydrogen')

        maplist = mol.get_smarts_matches(sp)
        for mp in maplist:
            if {res_of_atoms[mp[4] - 1], res_of_atoms[mp[2] - 1]} not in non_capped_res:

                capped_bonds.append({mp[2], mp[4]})

                m_cap = None
                if caps == 'mfcc':
                    cap_o = mp[1:4] + mol.find_adjacent_hydrogens(mp[1:4])
                    cap_n = mp[4:6] + mol.find_adjacent_hydrogens(mp[4:6])
                    m_cap = mol.get_fragment(cap_o + cap_n)
                    m_cap.set_spin(0)
                    m_cap.set_residue('CAP', 1, atoms=list(range(1, len(cap_o) + 1)))
                    m_cap.set_residue('CAP', 2, atoms=list(range(len(cap_o) + 1, len(cap_o + cap_n) + 1)))

                    m_cap.add_hydrogens_to_sp3_c(1)
                    m_cap.add_hydrogens_to_sp2_n(len(cap_o) + 1)
                    m_cap.add_hydrogens_to_sp3_c(len(cap_o) + 2)

                elif caps == 'hydrogen':
                    ccap = mol.get_fragment([mp[2], mp[3], mp[1]])
                    ccap.add_hydrogens_to_sp2_c(1)
                    ncap = mol.get_fragment([mp[4], mp[5], mp[6], mp[7]])
                    ncap.add_hydrogens_to_sp3_n(1)

                    m_cap = capmolecule()
                    m_cap.add_atoms(['H'], ncap.get_coordinates([5]))
                    m_cap.add_atoms(['H'], ccap.get_coordinates([4]), bond_to=1)

                    m_cap.set_spin(0)
                    m_cap.set_residue('CAP', 1, atoms=[1])
                    m_cap.set_residue('CAP', 2, atoms=[2])

                    m_cap.set_symmetry('NOSYM')

                # find the capped fragments
                #   frag1: fragment that contains the N atom (mp[4])
                #   frag2: fragment that contains the C=O atom (mp[2])
                fi1 = frag_indices[res_of_atoms[mp[4] - 1]]
                fi2 = frag_indices[res_of_atoms[mp[2] - 1]]

                frag1 = self._frags[fi1]
                frag2 = self._frags[fi2]
                self.append_cap(cappedfragment(None, capmolecule(m_cap)), frag1, frag2)

        # ------------------------------------------------------------------------------
        # check here for disulfide bonds and create disulfide bond caps

        maplist_sulfide = mol.get_smarts_matches('[C;X4;H1,H2]SS[C;X4;H1,H2]')
        # SMARTS pattern for disulfde bonds

        for mps in maplist_sulfide:

            if {res_of_atoms[mps[2] - 1], res_of_atoms[mps[1] - 1]} not in non_capped_res:

                capped_bonds.append({mps[1], mps[2]})

                m_cap_s = None
                if caps == 'mfcc':
                    cap_s1 = mps[0:2] + mol.find_adjacent_hydrogens(mps[0:2])
                    cap_s2 = mps[2:] + mol.find_adjacent_hydrogens(mps[2:])
                    m_cap_s = mol.get_fragment(cap_s1 + cap_s2)
                    m_cap_s.set_spin(0)
                    m_cap_s.set_residue('SCP', 1, atoms=list(range(1, len(cap_s1) + 1)))
                    m_cap_s.set_residue('SCP', 2, atoms=list(range(len(cap_s1) + 1, len(cap_s1 + cap_s2) + 1)))

                    m_cap_s.add_hydrogens_to_sp3_c(1)
                    m_cap_s.add_hydrogens_to_sp3_c(len(cap_s1) + 2)

                elif caps == 'hydrogen':

                    import numpy as np

                    s1_coord = np.array(mol.get_coordinates([mps[1]]))
                    s2_coord = np.array(mol.get_coordinates([mps[2]]))

                    h1_coord = s1_coord + 1.34 * ((s2_coord - s1_coord) / np.linalg.norm(s2_coord - s1_coord))
                    h2_coord = s2_coord + 1.34 * ((s1_coord - s2_coord) / np.linalg.norm(s1_coord - s2_coord))

                    m_cap_s = capmolecule()
                    m_cap_s.add_atoms(['H'], h2_coord)
                    m_cap_s.add_atoms(['H'], h1_coord, bond_to=1)

                    m_cap_s.set_spin(0)
                    m_cap_s.set_residue('SCP', 1, atoms=[1])
                    m_cap_s.set_residue('SCP', 2, atoms=[2])

                    m_cap_s.set_symmetry('NOSYM')

                fi1 = frag_indices[res_of_atoms[mps[2] - 1]]
                fi2 = frag_indices[res_of_atoms[mps[1] - 1]]

                frag_s1 = self._frags[fi1]
                frag_s2 = self._frags[fi2]
                self.append_scap(cappedfragment(None, capmolecule(m_cap_s)), frag_s1, frag_s2)

        # ------------------------------------------------------------------------------

        # find bonds between different residues and check if they have been capped

        num_uncapped_bonds = 0

        for b in mol.get_all_bonds():

            if not (res_of_atoms[b[0] - 1] == res_of_atoms[b[1] - 1]):

                if not {b[0], b[1]} in capped_bonds \
                        and not {res_of_atoms[b[0] - 1], res_of_atoms[b[1] - 1]} in non_capped_res \
                        and not {res_of_atoms[b[0] - 1], res_of_atoms[b[1] - 1]}.issubset(special_reslists):
                    num_uncapped_bonds += 1
                    print("Bond between atoms ", b[0], " and ", b[1], "not capped.")
                    print("Bond between res ", res_of_atoms[b[0] - 1], " and ", res_of_atoms[b[1] - 1], "not capped.")

        if num_uncapped_bonds > 0:
            print(num_uncapped_bonds, " bonds not capped. ")
            raise PyAdfError('Not all bonds capped in partition_protein')


class mfccresults(results):

    def __init__(self, job, frags=None):
        super().__init__(job)
        self._frags = frags

    def set_fragmentlist(self, frags):
        self._frags = frags

    def get_fragmentlist(self):
        return self._frags

    def get_dipole_vector(self):
        # pylint: disable-msg=E1101
        import numpy as np
        dipole = np.zeros(3)
        for f in self._frags.fragiter():
            dipole += f.results.get_dipole_vector()
        for c in self._frags.capiter():
            dipole -= c.results.get_dipole_vector()
        return dipole

    def get_total_energy(self):
        frag_energies = [f.results.get_total_energy() for f in self._frags.fragiter()]
        cap_energies = [c.results.get_total_energy() for c in self._frags.capiter()]

        return sum(frag_energies) - sum(cap_energies)

    def get_density(self, grid=None, spacing=0.5, fit=False, order=None):
        if grid is None:
            grid = cubegrid(self.job.get_molecule(), spacing)

        posdens = [f.results.get_nonfrozen_density(grid, fit=fit, order=order) for f in self._frags.fragiter()]
        capdens = [c.results.get_nonfrozen_density(grid, fit=fit, order=order) for c in self._frags.capiter()]

        posdens = reduce(lambda x, y: x + y, posdens)

        if capdens:
            capdens = reduce(lambda x, y: x + y, capdens)
            return posdens - capdens
        else:
            return posdens

    def get_potential(self, grid=None, spacing=0.5, pot='total'):
        if grid is None:
            grid = cubegrid(self.job.get_molecule(), spacing)

        pospot = [f.results.get_nonfrozen_potential(grid, pot=pot) for f in self._frags.fragiter()]
        cappot = [c.results.get_nonfrozen_potential(grid, pot=pot) for c in self._frags.capiter()]

        posdens = reduce(lambda x, y: x + y, pospot)

        if cappot:
            capdens = reduce(lambda x, y: x + y, cappot)
            return posdens - capdens
        else:
            return posdens

    def get_mulliken_charges(self):
        """
        Returns the Mulliken charges

        @returns: the Mulliken charges
        """
        import numpy as np

        mulliken_charges = None

        for f in self._frags.fragiter():
            if mulliken_charges is None:
                mulliken_charges = np.trim_zeros(f.results.get_mulliken_charges())
            else:
                frag_mulliken_charges = f.results.get_mulliken_charges()
                mulliken_charges = np.concatenate((mulliken_charges, np.trim_zeros(frag_mulliken_charges)))

        return mulliken_charges

    def print_mulliken_charges(self, nocap=False):
        """
        Prints the Mulliken charges together with the molecule geometry

        @param nocap: whether to include cap atoms
        @type  nocap: bool

        """
        mulliken_charges = self.get_mulliken_charges()
        atomsblock = self._frags.get_atoms_block()

        for i, line in enumerate(atomsblock.splitlines()[1:len(mulliken_charges)+1]):
            if nocap:
                if line.find('cap') < 0:
                    print("{}     {:+2.3f}".format(line.split('fs')[0].rstrip(), mulliken_charges[i]))
            else:
                print("{}     {:+2.3f}".format(line.split('fs')[0].rstrip(), mulliken_charges[i]))
        # what can one do to prevent the return value None from being printed?
        return ''


class adfmfccjob(metajob):

    def __init__(self, frags, basis, settings=None, core=None, pointcharges=None, fitbas=None, options=None):
        """
        Initialize a MFCC job.

        @param frags: the list of MFCC fragment
        @type  frags: L{cappedfragmentlist}
        @param basis:
        @type  basis:
        @param settings:
        @type  settings:
        @param core:
        @type  core:
        @param pointcharges:
        @type  pointcharges:
        @param fitbas:
        @type  fitbas
        @param options:
        @type  options:
        """
        super().__init__()

        self._frags = frags

        self._basis = basis
        self._settings = settings
        self._core = core
        self._pc = pointcharges
        self._fitbas = fitbas
        self._options = options

    def create_results_instance(self):
        return mfccresults(self)

    def get_molecule(self):
        return self._frags.get_total_molecule()

    def metarun(self):
        import copy
        frags = copy.deepcopy(self._frags)

        frags.calculate_all(lambda mol:
                            adfsinglepointjob(mol, basis=self._basis, settings=self._settings,
                                              core=self._core, pointcharges=self._pc, fitbas=self._fitbas,
                                              options=self._options).run())
        r = self.create_results_instance()
        r.set_fragmentlist(frags)
        return r


class adf3fdejob(adfmfccjob):

    def __init__(self, frags, basis, settings=None, core=None, fde=None,
                 fdeoptions=None, pointcharges=None, fitbas=None, options=None):

        if settings.zlmfit:
            raise PyAdfError("3-FDE in combination with ZlmFit not implemented")

        super().__init__(frags, basis, settings, core, pointcharges, fitbas, options)

        if fde is None:
            self._fde = {}
        else:
            import copy
            self._fde = copy.copy(fde)
        if 'RELAXCYCLES' in self._fde:
            self._cycles = self._fde['RELAXCYCLES'] + 1
            del self._fde['RELAXCYCLES']
        else:
            self._cycles = 1

        if 'NORMALFT' in self._fde:
            self._normalft = True
            self._mixedft = False
            del self._fde['NORMALFT']
        elif 'MIXEDFT' in self._fde:
            self._mixedft = True
            self._normalft = False
            del self._fde['MIXEDFT']
        else:
            self._normalft = False
            self._mixedft = False

        if fdeoptions is None:
            self._fdeoptions = {}
        else:
            self._fdeoptions = fdeoptions

        # make all fragments frozen and apply fdeoptions
        for f in self._frags.__iter__():
            f.isfrozen = True
            f.set_fdeoptions(self._fdeoptions)

    def parallel_ft_run(self):

        import copy

        frags_old = copy.deepcopy(self._frags)
        frags_new = None

        for i in range(self._cycles):

            print("-" * 50)
            print("Beginning 3-FDE cycle (parallel FT)", i)

            frags_new = copy.deepcopy(frags_old)

            # frags_old: fragments of the previous cycle
            # frags_new: the updated fragments

            for f_new, f_old in zip(frags_new.fragiter(), frags_old.fragiter()):
                f_old.isfrozen = False

                fdesettings = self._fde.copy()
                for fdeoption in ['LBdamp', 'CapRadius', 'ScfConvThresh', 'NoCapSepConv', 'LBmaxStep', 'FullGrid']:
                    if f_old.has_fdeoption(fdeoption):
                        fdesettings[fdeoption] = f_old.fdeoptions[fdeoption]

                # is it possible to make it more general?
                import copy
                fragsettings = copy.copy(self._settings)

                if f_old.has_fragoption('ncycles'):
                    fragsettings.set_ncycles(f_old.fragoptions['ncycles'])
                if f_old.has_fragoption('mix'):
                    fragsettings.set_mixing(f_old.fragoptions['mix'])
                if f_old.has_fragoption('diis'):
                    fragsettings.set_diis(f_old.fragoptions['diis'])
                if f_old.has_fragoption('adiis'):
                    fragsettings.set_adiis(f_old.fragoptions['adiis'])
                if f_old.has_fragoption('vshift'):
                    fragsettings.set_lshift(f_old.fragoptions['vshift'])
                if f_old.has_fragoption('cosmo'):
                    fragsettings.set_cosmo(f_old.fragoptions['cosmo'])
                # overlap with occ?
                if f_old.has_fragoption('occupations'):
                    fragsettings.set_occupations(f_old.fragoptions['occupations'])

                job = adffragmentsjob(frags_old, self._basis, settings=fragsettings,
                                      core=self._core, pointcharges=self._pc, fitbas=self._fitbas,
                                      options=self._options, fde=fdesettings)

                f_new.results = job.run()
                f_new.results.pack_tape()

                f_old.isfrozen = True

            frags_old = copy.deepcopy(frags_new)

        r = self.create_results_instance()
        r.set_fragmentlist(frags_new)
        return r

    def normal_ft_run(self):

        # normal freeze-thaw cycles: everything is updated immediately

        import copy

        frags_new = copy.deepcopy(self._frags)
        for i in range(self._cycles):

            print("-" * 50)
            print("Beginning 3-FDE cycle (normal FT)", i)

            for f_new in frags_new.fragiter():
                f_new.isfrozen = False
                job = adffragmentsjob(frags_new, self._basis, settings=self._settings,
                                      core=self._core, pointcharges=self._pc, fitbas=self._fitbas,
                                      options=self._options, fde=self._fde)
                f_new.results = job.run()
                f_new.results.pack_tape()

                f_new.isfrozen = True

        r = self.create_results_instance()
        r.set_fragmentlist(frags_new)
        return r

    def mixed_ft_run(self):

        import copy

        frags_old = copy.deepcopy(self._frags)
        frags_new = None

        for i in range(self._cycles):

            print("-" * 50)
            print("Beginning 3-FDE cycle (mixed FT)", i)

            frags_new = copy.deepcopy(frags_old)

            # frags_old: fragments of the previous cycle
            # frags_new: the updated fragments

            for f_new, f_old in zip(frags_new.fragiter(), frags_old.fragiter()):

                if not f_old.fdeoptions['n3fde'] == 0:

                    f_old.isfrozen = False

                    # maybe put fragment information in fragment-wise fdeoptions?
                    fdesettings = self._fde.copy()
                    for fdeoption in ['LBdamp', 'CapRadius', 'ScfConvThresh', 'NoCapSepConv']:
                        if f_old.has_fdeoption(fdeoption):
                            fdesettings[fdeoption] = f_old.fdeoptions[fdeoption]

                    # is it possible to make it more general?
                    import copy
                    fragsettings = copy.copy(self._settings)

                    if f_old.has_fragoption('ncycles'):
                        fragsettings.set_ncycles(f_old.fragoptions['ncycles'])
                    if f_old.has_fragoption('mix'):
                        fragsettings.set_mixing(f_old.fragoptions['mix'])
                    if f_old.has_fragoption('diis'):
                        fragsettings.set_diis(f_old.fragoptions['diis'])
                    if f_old.has_fragoption('adiis'):
                        fragsettings.set_adiis(f_old.fragoptions['adiis'])
                    if f_old.has_fragoption('vshift'):
                        fragsettings.set_lshift(f_old.fragoptions['vshift'])
                    if f_old.has_fragoption('cosmo'):
                        fragsettings.set_cosmo(f_old.fragoptions['cosmo'])
                    # overlap with occ?
                    if f_old.has_fragoption('occupations'):
                        fragsettings.set_occupations(f_old.fragoptions['occupations'])

                    job = adffragmentsjob(frags_old, self._basis, settings=fragsettings,
                                          core=self._core, pointcharges=self._pc, fitbas=self._fitbas,
                                          options=self._options, fde=fdesettings)
                    f_new.results = job.run()
                    f_new.results.pack_tape()
                    # subtract 1 from the n3fde (number of 3fde runs for this fragment)
                    f_new.fdeoptions['n3fde'] = f_old.fdeoptions['n3fde'] - 1

                    f_old.isfrozen = True

            frags_old = copy.deepcopy(frags_new)

        r = self.create_results_instance()
        r.set_fragmentlist(frags_new)
        return r

    def metarun(self):

        if self._normalft:
            return self.normal_ft_run()
        elif self._mixedft:
            return self.mixed_ft_run()
        else:
            return self.parallel_ft_run()
