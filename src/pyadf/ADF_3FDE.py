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
 Job and results for ADF 3-partition FDE calculations.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""

from .Errors import PyAdfError
from .Molecule.OBMolecule import OBMolecule
from .BaseJob import metajob, results
from .ADFFragments import fragment, fragmentlist, adffragmentsjob
# pylint: disable-msg=W0611
from .ADFSinglePoint import adfsinglepointjob, adfsinglepointresults
from .Plot.Grids import cubegrid
from functools import reduce


class capmolecule (OBMolecule):

    """
    A molecule used in a cap or capped fragment.

    It stores information on the atoms in the cap(s) and
    on the different parts of the caps. (It used residues
    within Openbabel for this: caps are residues with the
    name 'CAP'. Each cap has two parts (1 and 2), which are
    marked with even / odd residue numbers.

    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__

    """

    def __init__(self, mol=None):
        """
        Initialize a cap molecule.

        @param mol: a standard L{molecule}
        @type mol:  L{molecule}
        """
        # pylint: disable-msg=W0231

        if mol:
            self.copy(mol)
        else:
            OBMolecule.__init__(self)

    def cap_id(self, atom):
        """
        Get the "cap id" of the given atom.

        The cap id is 1 or 2 if the atom is in part 1 or 2 of the cap,
        respectively. For disulfide bonds, cap ids are 3 or 4.
        Otherwise, it is 0.

        @param atom: the atom number (Atom numbering starts at 1)
        @type atom:  int
        """

        a = self.mol.GetAtom(atom)
        res = a.GetResidue()

        if (res and res.GetName() == 'CAP'):
            # even residue number: capid = 1
            #  odd residue number: capid = 2
            capid = (res.GetNum() % 2) + 1
        elif (res and res.GetName() == 'SCP'):
            # even residue number: capid = 3
            #  odd residue number: capid = 4
            capid = (res.GetNum() % 2) + 3
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

        if atoms == None:
            atoms = list(range(1, self.mol.NumAtoms() + 1))
            if ghosts == False:
                atoms = [i for i in atoms if not self.is_ghost[i - 1]]

        symbols = OBMolecule.get_atom_symbols(self, atoms, ghosts, prefix_ghosts)

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
            a = self.mol.GetAtom(i)
            res = a.GetResidue()

            if not res or not (res.GetName() == 'CAP' or res.GetName() == 'SCP'):
                atoms.append(i)

        return self.get_fragment(atoms)


class cappedfragment (fragment):

    """
    A capped fragments, as used in 3-partition FDE.

    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__

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
        @type  frag_results: L{adfsinglepointresults}

        @param mol:
            A molecule, giving the coordinates where this fragment
            is used.
        @type mol: list of L{molecule}s

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

        fragment.__init__(self, frag_results, mols=mol, subfrag=subfrag, isfrozen=isfrozen,
                          fdeoptions=fdeoptions, occ=occ, fragoptions=fragoptions)

        self._iscap = iscap

        self.num_cap = -1

        self._cap_fragments = []
        self._cap_residue_nums = []

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

        i1 = self._mols[0].get_number_of_atoms() + 1
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

        i1 = self._mols[0].get_number_of_atoms() + 1
        i2 = i1 + m_cap.get_number_of_atoms()

        self._mols[0] = self._mols[0] + m_cap
        self._mols[0].set_residue('SCP', resnum, atoms=list(range(i1, i2)))

    def get_type(self):

        if self.iscap:
            ftype = "type=FDEsubstract"
        else:
            ftype = fragment.get_type(self)
        return ftype

    def _set_iscap(self, iscap):
        self._iscap = iscap

    def _get_iscap(self):
        return self._iscap

    iscap = property(_get_iscap, _set_iscap, None,
                     """
                     Whether this fragment is a cap fragment.
                     """)

    def print_fragment_options(self):
        if self.iscap:
            print(" type: frozen FDE cap fragment")
            if (len(self._fdeoptions) > 0):
                print("        FDE options: ")
            for opt, value in self._fdeoptions.items():
                print("           ", opt, "  ", value)
            print()
        else:
            fragment.print_fragment_options(self)

    def get_atoms_block(self):

        AtomsBlock = ""

        if (not self.has_frag_results()) or (len(self._cap_fragments) == 0):
            AtomsBlock += fragment.get_atoms_block(self)
        else:
            if len(self._mols) > 1:
                raise PyAdfError("Capped fragments must appear only once")

            mol = self._mols[0].get_noncap_fragment()
            suffix = "f=" + self.fragname
            AtomsBlock += mol.print_coordinates(index=False, suffix=suffix)

            for cap_frag, cap_res in zip(self._cap_fragments, self._cap_residue_nums):
                if cap_res < 5:
                    mol = self._mols[0].get_residues(restype='CAP', resnum=cap_res)[0]
                    mol = capmolecule(mol)
                    suffix = "f=" + self.fragname + "   fs=cap" + str(cap_frag.num_cap)
                    AtomsBlock += mol.print_coordinates(index=False, suffix=suffix)
                elif cap_res > 4:
                    mol_s = self._mols[0].get_residues(restype='SCP', resnum=cap_res)[0]
                    mol_s = capmolecule(mol_s)
                    suffix = "f=" + self.fragname + "   fs=scap" + str(cap_frag.num_cap)
                    AtomsBlock += mol_s.print_coordinates(index=False, suffix=suffix)

        return AtomsBlock


class cappedfragmentlist (fragmentlist):

    """
    List of capped fragments and caps.

    This is needed for 3-partition FDE jobs (L{adf3fdejob}).

    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__

    """

    def __init__(self):
        """
        Create a cappedfragmentlist.
        """

        fragmentlist.__init__(self, None)

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
        AtomsBlock = ""
        for frag in self.fragiter():
            AtomsBlock += frag.get_atoms_block()
        return AtomsBlock

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

    def partition_protein(self, mol, special_reslists=None, fragsize=None):
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
        """

        import openbabel

        # generate fragment list

        res_of_atoms = mol.get_residue_numbers_of_atoms()

        capped_bonds = []
        # list of bonds between residues within fragment
        non_capped_res = []
        # to which fragment belongs the residue
        frag_indices = []

        # clear the fragment list
        self._frags = []
        self._caps = []

        if fragsize == None:
            fragsize = 1
        if special_reslists == None:
            special_reslists = {}

        #----------------------------------------------------------------------------------
        # simplest fragmentation: each residue is one fragment

        if special_reslists == {} and fragsize == 1:

            # get the individual residues
            for res in mol.get_residues():
                self.append(cappedfragment(None, res))
                frag_indices.append(len(self._frags) - 1)
        #----------------------------------------------------------------------------------
        # more complicated fragmentation

        else:
            # careful with residue numbers (internal openbabel index idx starts at 0)

            fragmentation_list = []
            residuelist = mol.get_residues()
            frag_indices = [-1 for i in range(len(residuelist))]

            # get list of residues connected by covalent bonds (tuples)
            connected_residues = []
            for b in mol.get_all_bonds():
                if not (res_of_atoms[b[0] - 1] == res_of_atoms[b[1] - 1]):
                    # -1 is not totally clear to me, but works
                    connected_residues.append(set([res_of_atoms[b[0] - 1], res_of_atoms[b[1] - 1]]))

            #------------------------------------------------------------------------------
            # first get fragmentation list: which residues belong to which fragment
            # something like [ [3,4,5], [11,12,14], [1, 2], [6, 7], [8, 9], [10], [13]]

            # ugly
            # dictionary that connects pdb resnums with internal idx
            # problem: resnum as key is not unique, use combination of residue name and chain as key
            res_idx = {}
            for res in residuelist:
                resnum = res.mol.GetResidue(0).GetNum()
                reschain = res.mol.GetResidue(0).GetChain()
                reskey = 'c' + str(reschain) + str(resnum)
                intresnum = res.mol.GetResidue(0).GetIdx()
                res_idx[reskey] = intresnum

            # convert residue numbers in special_reslists to internal numbers
            for i in range(len(special_reslists)):
                for j in range(len(special_reslists[i])):
                    special_reslists[i][j] = res_idx[special_reslists[i][j]]

            # flatten list of lists so that you can check whether a residue is a special residue
            specialresnums = []
            for slist in special_reslists:
                if isinstance(slist, (list, tuple)):
                    specialresnums.extend(slist)
                # else should not occur, but maybe it is needed later
                else:
                    specialresnums.append(slist)

            # first add special residues to fragmentation list
            for slist in special_reslists:
                fragmentation_list.append(slist)

            fragl = []  # templist for fragments
            # add all other residues
            # use res.GetIdx() instead of enumerate
            for res in residuelist:
                resnum = res.mol.GetResidue(0).GetIdx()
                if resnum not in specialresnums:
                    # take care of fragsize
                    if fragsize == 1:
                        fragmentation_list.append([resnum])
                    else:
                        # check whether there has to be a new fragment
                        if len(fragl) != 0:
                            if not set([resnum, fragl[-1]]) in connected_residues or \
                                    len(fragl) == fragsize:
                               # no covalent connection? new fragment! fragsize reached?
                                fragmentation_list.append(fragl)
                                fragl = []
                        # new fragment
                        if len(fragl) == 0:
                            fragl.append(resnum)
                        # add to fragment
                        else:
                            fragl.append(resnum)
            # last fragment
            if len(fragl) != 0:
                fragmentation_list.append(fragl)

            #------------------------------------------------------------------------------
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
                            if set([rlist[rnum], rlist[rd]]) in connected_residues:
                                non_capped_res.append(set([rlist[rnum], rlist[rd]]))

                    # take care of residue information here
                    frag_indices[rlist[rnum]] = len(self._frags)

                res_frag = fragment(None, mols=frag_reslist)

                self.append(cappedfragment(None, res_frag.get_total_molecule()))
                # clear frag_reslist for next fragment
                frag_reslist = []

        #------------------------------------------------------------------------------

        # find all peptide bonds and create caps

        sp = openbabel.OBSmartsPattern()
        sp.Init('[C;X4;H1,H2][CX3](=O)[NX3][C;X4;H1,H2][CX3](=O)')
        # this SMARTS pattern matches all peptide bonds
        #
        # (in this comment: atom numbering starts at 1;
        #  in the code below the numbering of the list starts at 0)
        # the peptide bond is between atoms 2 and 4
        # atoms 1-5 (+ connected hydrogens) form the cap fragment
        # part 1 (o-part): atoms 1-3
        # part 2 (n-part): atoms 4-5

        sp.Match(mol.mol)
        maplist = sp.GetUMapList()
        for mp in maplist:
            mp = list(mp)

            if set([res_of_atoms[mp[3] - 1], res_of_atoms[mp[1] - 1]]) not in non_capped_res:

                capped_bonds.append(set([mp[1], mp[3]]))
                cap_o = mp[0:3] + mol.find_adjacent_hydrogens(mp[0:3])
                cap_n = mp[3:5] + mol.find_adjacent_hydrogens(mp[3:5])
                m_cap = mol.get_fragment(cap_o + cap_n)
                m_cap.set_spin(0)
                m_cap.set_residue('CAP', 1, atoms=list(range(1, len(cap_o) + 1)))
                m_cap.set_residue('CAP', 2, atoms=list(range(len(cap_o) + 1, len(cap_o + cap_n) + 1)))
                m_cap.add_hydrogens()

            # find the capped fragments
            #   frag1: fragment that contains the N atom (mp[3])
            #   frag2: fragment that contains the C=O atom (mp[1])
            # why minus 1???

                fi1 = frag_indices[res_of_atoms[mp[3] - 1]]
                fi2 = frag_indices[res_of_atoms[mp[1] - 1]]

                frag1 = self._frags[fi1]
                frag2 = self._frags[fi2]
                self.append_cap(cappedfragment(None, capmolecule(m_cap)), frag1, frag2)

        #------------------------------------------------------------------------------

        # check here for disulfide bonds and create disulfide bond caps

        sp_sulfide = openbabel.OBSmartsPattern()
        sp_sulfide.Init('[C;X4;H1,H2]SS[C;X4;H1,H2]')
        # SMARTS pattern for disulfde bonds

        sp_sulfide.Match(mol.mol)
        maplist_sulfide = sp_sulfide.GetUMapList()
        for mps in maplist_sulfide:
            mps = list(mps)

            if set([res_of_atoms[mps[2] - 1], res_of_atoms[mps[1] - 1]]) not in non_capped_res:

                capped_bonds.append(set([mps[1], mps[2]]))
                cap_s1 = mps[0:2] + mol.find_adjacent_hydrogens(mps[0:2])
                cap_s2 = mps[2:] + mol.find_adjacent_hydrogens(mps[2:])
                m_cap_s = mol.get_fragment(cap_s1 + cap_s2)
                m_cap_s.set_spin(0)
                m_cap_s.set_residue('SCP', 1, atoms=list(range(1, len(cap_s1) + 1)))
                m_cap_s.set_residue('SCP', 2, atoms=list(range(len(cap_s1) + 1, len(cap_s1 + cap_s2) + 1)))
                m_cap_s.add_hydrogens()

                fi1 = frag_indices[res_of_atoms[mps[2] - 1]]
                fi2 = frag_indices[res_of_atoms[mps[1] - 1]]

                frag_s1 = self._frags[fi1]
                frag_s2 = self._frags[fi2]
                self.append_scap(cappedfragment(None, capmolecule(m_cap_s)), frag_s1, frag_s2)

        #------------------------------------------------------------------------------

        # find bonds between different residues and check if they have been capped

        num_uncapped_bonds = 0
        for b in mol.get_all_bonds():

            if not (res_of_atoms[b[0] - 1] == res_of_atoms[b[1] - 1]):

                if not set([b[0], b[1]]) in capped_bonds \
                        and not set([res_of_atoms[b[0] - 1], res_of_atoms[b[1] - 1]]) in non_capped_res \
                        and not set([res_of_atoms[b[0] - 1], res_of_atoms[b[1] - 1]]).issubset(special_reslists):

                    num_uncapped_bonds += 1
                    print("Bond between atoms ", b[0], " and ", b[1], "not capped.")
                    print("Bond between res ", res_of_atoms[b[0] - 1], " and ", res_of_atoms[b[1] - 1], "not capped.")

        if num_uncapped_bonds > 0:
            print(num_uncapped_bonds, " bonds not capped. ")
            raise PyAdfError('Not all bonds capped in partition_protein')


class mfccresults (results):

    def __init__(self, job, frags=None):
        results.__init__(self, job)
        self._frags = frags

    def set_fragmentlist(self, frags):
        self._frags = frags

    def get_fragmentlist(self):
        return self._frags

    def get_dipole_vector(self):
        # pylint: disable-msg=E1101
        import numpy
        dipole = numpy.zeros(3)
        for f in self._frags.fragiter():
            dipole += f.results.get_dipole_vector()
        for c in self._frags.capiter():
            dipole -= c.results.get_dipole_vector()
        return dipole

    def get_density(self, grid=None, spacing=0.5, fit=False):
        if grid == None:
            grid = cubegrid(self.job.get_molecule(), spacing)

        posdens = [f.results.get_nonfrozen_density(grid, fit=fit) for f in self._frags.fragiter()]
        capdens = [c.results.get_nonfrozen_density(grid, fit=fit) for c in self._frags.capiter()]

        posdens = reduce(lambda x, y: x + y, posdens)
        capdens = reduce(lambda x, y: x + y, capdens)

        return posdens - capdens

    def get_potential(self, grid=None, pot='total'):
        if grid == None:
            grid = cubegrid(self.job.get_molecule(), spacing)

        pospot = [f.results.get_nonfrozen_potential(grid, pot=pot) for f in self._frags.fragiter()]
        cappot = [c.results.get_nonfrozen_potential(grid, pot=pot) for c in self._frags.capiter()]

        posdens = reduce(lambda x, y: x + y, pospot)
        capdens = reduce(lambda x, y: x + y, cappot)

        return posdens - capdens

    def get_mulliken_charges(self):
        """
        Returns the Mulliken charges

        @returns: the Mulliken charges
        """
        import numpy

        mulliken_charges = None

        for f in self._frags.fragiter():
            if mulliken_charges == None:
                mulliken_charges = numpy.trim_zeros((f.results.get_mulliken_charges()))
            else:
                frag_mulliken_charges = f.results.get_mulliken_charges()
                mulliken_charges = numpy.concatenate((mulliken_charges, numpy.trim_zeros(frag_mulliken_charges)))

        return mulliken_charges

    def print_mulliken_charges(self, nocap=False):
        """
        Prints the Mulliken charges together with the molecule geometry

        @param nocap: whether to include cap atoms
        @type  nocap: bool

        """
        import numpy

        mulliken_charges = None

        for f in self._frags.fragiter():
            if mulliken_charges == None:
                mulliken_charges = numpy.trim_zeros((f.results.get_mulliken_charges()))
            else:
                frag_mulliken_charges = f.results.get_mulliken_charges()
                mulliken_charges = numpy.concatenate((mulliken_charges, numpy.trim_zeros(frag_mulliken_charges)))

        atomsblock = self._frags.get_atoms_block()

        for i, line in enumerate(atomsblock.splitlines()):
            if (nocap):
                if line.find('cap') < 0:
                    print("%s     %+2.3f" % (line.split('fs')[0].rstrip(), mulliken_charges[i]))
            else:
                print("%s     %+2.3f" % (line.split('fs')[0].rstrip(), mulliken_charges[i]))
        # what can one do to prevent the return value None from being printed?
        return ''


class adfmfccjob (metajob):

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
        metajob.__init__(self)

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


class adf3fdejob (adfmfccjob):

    def __init__(self, frags, basis, settings=None, core=None,
                 fde=None, fdeoptions=None, pointcharges=None, fitbas=None, options=None):

        adfmfccjob.__init__(self, frags, basis, settings, core, pointcharges, fitbas, options)

        if fde == None:
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

        if fdeoptions == None:
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
                        fdesettings[fdeoption] = f_old._fdeoptions[fdeoption]

                # is it possible to make it more general?
                import copy
                fragsettings = copy.copy(self._settings)

                if f_old.has_fragoption('ncycles'):
                    fragsettings.set_ncycles(f_old._fragoptions['ncycles'])
                if f_old.has_fragoption('mix'):
                    fragsettings.set_mixing(f_old._fragoptions['mix'])
                if f_old.has_fragoption('diis'):
                    fragsettings.set_diis(f_old._fragoptions['diis'])
                if f_old.has_fragoption('adiis'):
                    fragsettings.set_adiis(f_old._fragoptions['adiis'])
                if f_old.has_fragoption('vshift'):
                    fragsettings.set_lshift(f_old._fragoptions['vshift'])
                if f_old.has_fragoption('cosmo'):
                    fragsettings.set_cosmo(f_old._fragoptions['cosmo'])
                # overlap with occ?
                if f_old.has_fragoption('occupations'):
                    fragsettings.set_occupations(f_old._fragoptions['occupations'])

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
        for i in range(self._cycles):

            print("-" * 50)
            print("Beginning 3-FDE cycle (mixed FT)", i)

            frags_new = copy.deepcopy(frags_old)

            # frags_old: fragments of the previous cycle
            # frags_new: the updated fragments

            for f_new, f_old in zip(frags_new.fragiter(), frags_old.fragiter()):

                if not f_old._fdeoptions['n3fde'] == 0:

                    f_old.isfrozen = False

                    # maybe put fragment information in fragment-wise fdeoptions?
                    fdesettings = self._fde.copy()
                    for fdeoption in ['LBdamp', 'CapRadius', 'ScfConvThresh', 'NoCapSepConv']:
                        if f_old.has_fdeoption(fdeoption):
                            fdesettings[fdeoption] = f_old._fdeoptions[fdeoption]

                    # is it possible to make it more general?
                    import copy
                    fragsettings = copy.copy(self._settings)

                    if f_old.has_fragoption('ncycles'):
                        fragsettings.set_ncycles(f_old._fragoptions['ncycles'])
                    if f_old.has_fragoption('mix'):
                        fragsettings.set_mixing(f_old._fragoptions['mix'])
                    if f_old.has_fragoption('diis'):
                        fragsettings.set_diis(f_old._fragoptions['diis'])
                    if f_old.has_fragoption('adiis'):
                        fragsettings.set_adiis(f_old._fragoptions['adiis'])
                    if f_old.has_fragoption('vshift'):
                        fragsettings.set_lshift(f_old._fragoptions['vshift'])
                    if f_old.has_fragoption('cosmo'):
                        fragsettings.set_cosmo(f_old._fragoptions['cosmo'])
                    # overlap with occ?
                    if f_old.has_fragoption('occupations'):
                        fragsettings.set_occupations(f_old._fragoptions['occupations'])

                    job = adffragmentsjob(frags_old, self._basis, settings=fragsettings,
                                          core=self._core, pointcharges=self._pc, fitbas=self._fitbas,
                                          options=self._options, fde=fdesettings)
                    f_new.results = job.run()
                    f_new.results.pack_tape()
                    # subtract 1 from the n3fde (number of 3fde runs for this fragment)
                    f_new._fdeoptions['n3fde'] = f_old._fdeoptions['n3fde'] - 1

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
