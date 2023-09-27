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
 Job and results for ADF fragment analysis calculations, including NewFDE.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""


from .ADFSinglePoint import adfsinglepointjob, adfsinglepointresults
from .ADFSinglePoint import use_default_grid
from .ADF_Densf import densfjob
from .Errors import PyAdfError

from .Plot.Properties import PlotPropertyFactory
from .Plot.GridFunctions import GridFunctionDensityWithDerivatives

import copy
import os.path
from functools import reduce


class fragment:
    """
    A class representing a fragment (a fragment type, to be more precise).

    This is used to setup L{adffragmentsjob}s.

    @undocumented: __deepcopy__
    @undocumented: __delattr__, __getattribute__, __hash__, __new__,
                   __repr__, __str__, __setattr__
    """

    def __init__(self, frag_results, mols=None, subfrag=None, isfrozen=False,
                 fdeoptions=None, occ=None, fragoptions=None):
        """
        Initialize a fragment (or more precisely, a fragment type).

        @param frag_results:
            Results of a previous ADF singlepoint job. If C{None}, this
            fragment is considered to be active, which means that it consists
            of atomic fragments for which fragment files will be prepared
            automatically.
        @type  frag_results: L{adfsinglepointresults} or None

        @param mols:
            A list of molecules, giving the coordinates where this fragment
            is used. Fragments can be used multiple times, but for each
            molecule it must be possible to rotate the fragment to this
            position.

        @param subfrag: The name of the subfragment to be used.
        @type  subfrag: str or None

        @param isfrozen: Specify if this fragment is frozen (FDE)
        @type  isfrozen: bool

        @param fdeoptions:
            A dictionary of FDE options.
            Possible options are:
            For frozen fragments: USEBASIS, RELAX, XC, DENSTYPE,
            For active fragments: LBDAMP, CAPRADIUS, SCFCONVTHRESH, NOCAPSEPCONV, LBMAXSTEP
            B{Options can be upper, lower or mixed case}
        @type  fdeoptions: dictionary or None

        @param occ: fragment occupation numbers
        @type occ: list of lists of three elements in the format
                   ['irrep', num_alpha, num_beta]

        @param fragoptions:
            For fragments, possibly a dictionary of adfsettings options
        @type fragoptions: dictionary or None
        """

        self._frag_results = copy.deepcopy(frag_results)
        if mols:
            self._mols = mols
        elif self._frag_results:
            self._mols = [self._frag_results.get_molecule()]

        if not isinstance(self._mols, list):
            self._mols = [self._mols]

        self._subfrag = subfrag

        self._isfrozen = isfrozen

        if fdeoptions is None:
            self._fdeoptions = {}
        else:
            self._fdeoptions = {}
            for k, v in fdeoptions.items():
                if k.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE']:
                    # these options should always be uppercase
                    self._fdeoptions[k.upper()] = v
                else:
                    # all other options are case-sensitive!
                    self._fdeoptions[k] = v

        if occ is None:
            self._occ = []
        else:
            self._occ = occ

        if fragoptions is None:
            self._fragoptions = {}
        else:
            self._fragoptions = {}
            for k, v in fragoptions.items():
                self._fragoptions[k.lower()] = v

        # the name use for this fragment. This is assigned by fragmentlist
        self.fragname = None

    def __deepcopy__(self, memo):
        # pylint: disable=W0613

        # make sure that deepcopies are not deeper than needed
        # (this can otherwise use up an incredible amount of time)
        other = copy.copy(self)
        other._fdeoptions = copy.copy(self._fdeoptions)
        other._fragoptions = copy.copy(self._fragoptions)

        return other

    @property
    def isfrozen(self):
        """
        Whether this is a frozen fragment.
        """
        return self._isfrozen

    @isfrozen.setter
    def isfrozen(self, isfrozen):
        self._isfrozen = isfrozen

    @property
    def results(self):
        """
         The results associated with this fragment.
         """
        return self._frag_results

    @results.setter
    def results(self, frag_results):
        self._frag_results = frag_results

    @property
    def fragoptions(self):
        return self._fragoptions

    @property
    def fdeoptions(self):
        return self._fdeoptions

    def has_frag_results(self):
        return self._frag_results is not None

    def get_molecules(self):
        """
        Return the list of molecules.

        @returns: the list of molecules
        @rtype:   list of L{molecule}s
        """
        return self._mols

    def get_total_molecule(self):
        """
        Return the total molecule.

        A molecule consisting of all the copies of this fragment is returned.

        @returns: the total molecule
        @rtype:   L{molecule}s
        """
        return reduce(lambda x, y: x + y, self._mols)

    def get_fragment_filename(self):
        """
        Return the name of the fragment file.

        @returns: name of fragment TAPE21 file
        @rtype:   str
        """
        if self._frag_results:
            filename = os.path.basename(self._frag_results.get_tape_filename(21))
        else:
            filename = None
        return filename

    def copy_fragment_file(self):
        """
        Copies the fragment file to the working directory.
        """

        if self._frag_results:
            filename = os.path.basename(self._frag_results.get_tape_filename(21))
            if not os.path.exists(filename):
                self._frag_results.link_tape(name=filename)

    def delete_fragment_file_copy(self):
        """
        Delete the fragment file that was previously copied to the working directory.
        """

        if self._frag_results:
            filename = os.path.basename(self._frag_results.get_tape_filename(21))
            if os.path.exists(filename):
                os.remove(filename)

    def has_occupations(self):
        """
        Returns True if fragoccupation data is present.
        """
        return len(self._occ) > 0

    def get_num_frags(self):
        """
        Returns whether this fragment type is only used once.
        """
        return len(self._mols)

    def is_fde_fragment(self):

        if self._frag_results:
            return self._frag_results.is_fde_job()
        else:
            return False

    def set_fdeoptions(self, opts):
        for k, v in opts.items():
            if k.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE']:
                self._fdeoptions[k.upper()] = v
            else:
                self._fdeoptions[k] = v

    def has_fdeoption(self, opt):
        if opt.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE']:
            return opt.upper() in self._fdeoptions
        else:
            return opt in self._fdeoptions

    def delete_fdeoption(self, opt):
        if opt.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE']:
            if opt.upper() in self._fdeoptions:
                del self._fdeoptions[opt.upper()]
        else:
            if opt in self._fdeoptions:
                del self._fdeoptions[opt]

    def add_fdeoption(self, opt, val):
        if opt.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE']:
            self._fdeoptions[opt.upper()] = val
        else:
            self._fdeoptions[opt] = val

    def set_fragoptions(self, opts):
        for k, v in opts.items():
            self._fragoptions[k.lower()] = v

    def has_fragoption(self, opt):
        return opt.lower() in self._fragoptions

    def delete_fragoption(self, opt):
        if opt.lower() in self._fragoptions:
            del self._fragoptions[opt.lower()]

    def add_fragoption(self, opt, val):
        if opt.lower() in self._fragoptions:
            self._fragoptions[opt.lower()] = val
        else:
            self._fragoptions[opt.lower()] = val

    def print_fragment_options(self):
        """
        Print information about the options used for this fragment.
        """
        if self.isfrozen:
            print(" type: frozen FDE fragment")

            FrozenFDEOptions = ""

            for opt, value in self._fdeoptions.items():
                if opt not in ['LBdamp', 'CapRadius', 'ScfConvThresh', 'NoCapSepConv', 'LBmaxStep', 'FullGrid']:
                    if isinstance(value, float):
                        FrozenFDEOptions += f"           {opt}    {value:.2f} \n"
                    if isinstance(value, str):
                        FrozenFDEOptions += f"           {opt}    {value} \n"
            if FrozenFDEOptions != "":
                print("        FDE options: ")
                print(FrozenFDEOptions)
        else:
            print(" type: nonfrozen fragment")

            NonFrozenFDEOptions = ""
            for fdeoption in ['LBdamp', 'CapRadius', 'ScfConvThresh', 'NoCapSepConv', 'LBmaxStep', 'FullGrid']:
                if fdeoption in self._fdeoptions:
                    if isinstance(self._fdeoptions[fdeoption], float):
                        NonFrozenFDEOptions += f"           {fdeoption}    {self._fdeoptions[fdeoption]:.2f} \n"
                    if isinstance(self._fdeoptions[fdeoption], str):
                        NonFrozenFDEOptions += f"           {fdeoption}    {self._fdeoptions[fdeoption]}   \n"
            if NonFrozenFDEOptions != "":
                print("        FDE options: ")
                print(NonFrozenFDEOptions)

            if len(self._fragoptions) > 0:
                print("        Fragment settings : ")
            for opt, value in self._fragoptions.items():
                print("           ", opt, "  ", value)
        print()

    def get_atoms_block(self):
        """
        Return the list of atoms in this fragment,
        to be used in the ATOMS block of the ADF input.

        @returns: a string that can be used in the ATOMS block
        @rtype:   str
        """

        AtomsBlock = ""

        if self.has_frag_results():
            for num_frag, m in enumerate(self._mols):
                if len(self._mols) > 1:
                    suffix = "f=" + self.fragname + "/" + str(num_frag + 1)
                else:
                    suffix = "f=" + self.fragname

                AtomsBlock += m.print_coordinates(index=False, suffix=suffix)
        else:
            for m in self._mols:
                AtomsBlock += m.print_coordinates(index=False)

        return AtomsBlock

    def get_type(self):
        ftype = ""
        if self.isfrozen:
            ftype = "type=FDE"
        return ftype

    def get_fragments_block(self, checksumonly):
        """
        Return a line that can be used in the FRAGMENTS block
        of the ADF input for this fragment.

        @returns: a string that can be used in the FRAGMENTS block
        @rtype:   str
        """
        block = ""
        if self.has_frag_results():
            block += "  " + self.fragname + "  "

            if checksumonly and self._frag_results is not None:
                block += self._frag_results.checksum
            else:
                block += self.get_fragment_filename()

            if self.is_fde_fragment():
                if self._subfrag:
                    block += " subfrag=" + self._subfrag
                else:
                    block += " subfrag=active"
            if self.isfrozen:
                block += " " + self.get_type()

                fdeoptions_block = self.get_fdeoptions_block()
                if len(fdeoptions_block) > 0:
                    block += "  & \n"
                    block += fdeoptions_block
                    block += "  SubEnd"

            block += "\n"

        return block

    def get_fdeoptions_block(self):
        block = ""
        if ('USEBASIS' in self._fdeoptions) or ('RELAX' in self._fdeoptions) or \
                ('DENSTYPE' in self._fdeoptions) or ('XC' in self._fdeoptions):
            options = ''
            if 'USEBASIS' in self._fdeoptions:
                options += 'USEBASIS '
            if 'RELAX' in self._fdeoptions:
                options += 'RELAX '
            if len(options) > 0:
                block += "     fdeoptions " + options + "\n"
            if 'DENSTYPE' in self._fdeoptions:
                block += "     fdedenstype "
                block += self._fdeoptions['DENSTYPE'] + "\n"
            if 'XC' in self._fdeoptions:
                block += "     XC " + self._fdeoptions['XC'] + "\n"
        return block

    def get_fragoccupations_block(self):
        block = ''
        if self.has_occupations():
            block += f"  {self.fragname} \n"
            for irrep in self._occ:
                block += f"   {irrep[0]} {irrep[1]:d} // {irrep[2]:d}\n"
            block += "  SubEnd \n"

        return block

    def get_special_options_block(self):
        return ""

    def calculate(self, func, context=None):
        """
        Calulate the results for this fragment.

        @param func: The function that performs the actual calculation.
        @type  func: A function with the signature func(mol, **context), where
                     mol is a L{molecule} and context are additional keyword
                     arguments, returning an instance of L{adfsinglepointresults}
                     or of a derived class.
        @param context: An optional dictionary of additional keyword arguments to func.
        @type  context: dict
        """

        if context is None:
            context = {}
        # pylint: disable-msg=W0142
        self.results = func(self._mols[0], **context)


class FrozenDensFragment(fragment):

    def __init__(self, mol, density, coulpot):

        if density.grid is not coulpot.grid:
            raise PyAdfError('Density and potential must use same grid in FrozenDensFragment')

        self._density = density
        self._coulpot = coulpot

        super().__init__(None, mols=mol, subfrag='active', isfrozen=True,
                         fdeoptions={'DENSTYPE': 'SCFexact'}, occ=None)

    def has_frag_results(self):
        return True

    def is_fde_fragment(self):
        return False

    def get_fragment_filename(self):
        return "DummyFragment.t21"

    @staticmethod
    def get_fileimport_filename():
        return "FrozenDensFragment.t10"

    def get_fdeoptions_block(self):
        block = super().get_fdeoptions_block()
        block += "     fileimport " + self.get_fileimport_filename() + "\n"
        return block

    def get_special_options_block(self):
        block = super().get_special_options_block()
        block += " IMPORTGRID " + self.get_fileimport_filename() + "\n\n"
        return block

    def write_dummy_fragment_file(self):
        """
        Writes a TAPE21 file with the minimum information ADF needs, even though
        none of the fragment information is used (the density and potential are
        imported from a separate T10-like file.
        """
        from .Utils import pse, Bohr_in_Angstrom

        import numpy as np
        import kf
        filename = self.get_fragment_filename()

        with kf.kffile(filename, buffered=True) as f:

            f.writechars('General', 'title', 'Dummy fragment file generated by PyADF')
            f.writeints('General', 'jobid', 42)
            f.writereals('General', 'electrons', 0)
            f.writeints('General', 'nspin', 1)
            f.writeints('General', 'ioprel', 0)

            atnums = self._mols[0].get_atomic_numbers()

            ntyps = len(set(atnums))
            natoms = len(atnums)

            f.writeints('Geometry', 'nr of atomtypes', ntyps)
            f.writeints('Geometry', 'nr of atoms', natoms)

            atcharges = []
            attyps = []
            masses = []
            for i in atnums:
                if i not in atcharges:
                    atcharges.append(i)
                    attyps.append(pse.get_symbol(i))
                    masses.append(pse.get_mass(i))
            f.writechars('Geometry', 'atomtype', attyps)
            f.writereals('Geometry', 'mass', masses)

            xyz = np.zeros((natoms, 3))
            coords = self._mols[0].get_coordinates()

            natoms_per_typ = np.zeros((ntyps,), dtype=int)

            pos = 0
            for i in range(ntyps):
                for j in range(natoms):
                    if atnums[j] == atcharges[i]:
                        xyz[pos, :] = coords[j]
                        pos = pos + 1
                        natoms_per_typ[i] = natoms_per_typ[i] + 1

            f.writereals('Geometry', 'xyz', xyz / Bohr_in_Angstrom)

            f.writereals('Geometry', 'atomtype total charge', atcharges)
            f.writereals('Geometry', 'atomtype effective charge', atcharges)

            natomt = np.zeros((ntyps+1,), dtype=int)

            for i in range(ntyps):
                natomt[i+1] = natomt[i] + natoms_per_typ[i]

            f.writeints('Geometry', 'cum nr of atoms', natomt)
            f.writeints('Geometry', 'nnuc', natoms)
            f.writeints('Geometry', 'ntyp', ntyps)

            f.writeints('Symmetry', 'nsym', 1)
            f.writeints('Symmetry', 'nsetat', natoms)
            f.writechars('Symmetry', 'grouplabel', 'NOSYM')

            f.writeints('Symmetry', 'ngr', 1)
            f.writeints('Symmetry', 'igr', 999)
            f.writeints('Symmetry', 'nogr', 1)
            npeq = natoms*(natoms+1) // 2
            f.writeints('Symmetry', 'npeq', npeq)
            f.writeints('Symmetry', 'nratst', [1] * natoms)

            notyps = []
            for i in range(ntyps):
                notyps = notyps + [i+1] * natoms_per_typ[i]
            f.writeints('Symmetry', 'notyps', notyps)
            f.writeints('Symmetry', 'noat', list(range(1, natoms+1)))

            f.writechars('Symmetry', 'symlab', 'A')

            f.writeints('Symmetry', 'jsyml', 1)
            f.writeints('Symmetry', 'jasym', [1] * npeq)
            f.writeints('Symmetry', 'ja1ok', [1] * npeq)

            f.writereals('Symmetry', 'faith', np.eye(3))

            f.writeints('Basis', 'nbset', 1)
            f.writeints('Basis', 'nbaspt', [1] + [2]*ntyps)
            f.writeints('Basis', 'nqbas', [1])
            f.writeints('Basis', 'lqbas', [0])
            f.writereals('Basis', 'alfbas', [1.0])

            f.writeints('Core', 'ncset', 0)
            f.writeints('Core', 'ncorpt', [1]*(ntyps + 1))
            f.writeints('Core', 'nqcor', [])
            f.writeints('Core', 'lqcor', [])
            f.writereals('Core', 'alfcor', [])

            f.writeints('Core', 'nrcset', [0]*(4*ntyps))
            f.writeints('Core', 'nrcorb', [0]*(4*ntyps))

            f.writeints('Fit', 'nfset', 1)
            f.writeints('Fit', 'nfitpt', [1] + [2]*ntyps)
            f.writeints('Fit', 'nqfit', [1])
            f.writeints('Fit', 'lqfit', [0])
            f.writereals('Fit', 'alffit', [1.0])

            f.writeints('Fit', 'nsfos', 1)
            f.writeints('Fit', 'niskf', 1)
            f.writeints('Fit', 'na1cof', 1)
            f.writelogicals('Fit', 'lnosymfit', True)
            f.writeints('Fit', 'na1ptr', [1] + [2]*natoms)
            f.writeints('Fit', 'iskf', [1]*4)
            f.writeints('Fit', 'numcom', [1])
            f.writereals('Fit', 'cofcom', [1.0])

            f.writereals('Fit', 'coef_SCF', [1.0]*natoms)

            f.writeints('Symmetry', 'norb', 2)
            f.writeints('Symmetry', 'nfcn', 2)

            f.writeints('A', 'npart', [1, 2])
            f.writeints('A', 'nmo_A', [2])
            f.writereals('A', 'eps_A', [0.0, 0.0])
            f.writereals('A', 'froc_A', [2.0, 0.0])
            f.writereals('A', 'Eigen-Bas_A', [1.0, 1.0, 1.0, 1.0])

            for i, at in enumerate(attyps):
                section = f'Atyp{i + 1:3d} {at}'
                f.writereals(section, 'rx val', 1.0)
                f.writeints(section, 'nrint val', 5)

                f.writereals(section, 'rup val', np.ones(5))
                f.writeints(section, 'ncheb val', np.ones(5, dtype=int))
                f.writereals(section, 'ccheb val', np.ones(88))

                f.writeints(section, 'nrad', 10)
                f.writereals(section, 'rmin', 1.0e-5)
                f.writereals(section, 'rfac', 1.0)

                f.writereals(section, 'valence den', np.zeros(10))
                f.writereals(section, 'valence pot', np.zeros(10))
                f.writereals(section, 'qval', atcharges[i])

    def copy_fragment_file(self):
        """
        Copies the fragment file to the working directory.
        """
        from .Plot.FileWriters import GridWriter, GridFunctionWriter

        self.write_dummy_fragment_file()

        filename = self.get_fileimport_filename()
        GridWriter.write_tape10(self._density.grid, filename)
        GridFunctionWriter.write_t10(self._density[0], filename, "FrozenDensity", "rhoffd")
        GridFunctionWriter.write_t10(self._density[1], filename, "FrozenDensityFirstDer", "drhoffd")
        GridFunctionWriter.write_t10(self._density[2], filename, "FrozenDensitySecondDer", "d2rhoffd")

        nucpot = self._mols[0].get_nuclear_potential(self._density.grid)
        GridFunctionWriter.write_t10(nucpot + self._coulpot, filename, "FrozenDensityElpot", "ElpotFD")

    def delete_fragment_file_copy(self):
        """
        Delete the fragment file that was previously copied to the working directory.
        """
        filename = self.get_fragment_filename()
        if os.path.exists(filename):
            os.remove(filename)

        filename = self.get_fileimport_filename()
        if os.path.exists(filename):
            os.remove(filename)

    def print_fragment_options(self):
        """
        Print information about the options used for this fragment.
        """
        print(" type: Imported Frozen Density Fragment")
        print()

    def get_fragments_block(self, checksumonly):
        """
        Return a line that is can be used in the FRAGMENTS block
        of the ADF input for this fragment.

        @returns: a string that can be used in the FRAGMENTS block
        @rtype:   str
        """

        block = super().get_fragments_block(self)
        if checksumonly:
            block += self._density.checksum
            block += self._coulpot.checksum
        return block


class fragmentlist:
    """
    A class repesenting a list of fragments, as used in L{adffragmentsjob}s.

    @undocumented: __delattr__, __getattribute__, __hash__, __new__,
                   __repr__, __str__, __setattr__
    """

    def __init__(self, frags=None):
        """
        Create a fragmentlist.

        @param frags: the list of L{fragment}s or C{None}
        @type frags: list or None
        """
        if frags is None:
            self._frags = []
        else:
            self._frags = frags
            for num_ftyp, frag in enumerate(self._frags):
                frag.fragname = "frag" + str(num_ftyp + 1)

    def append(self, f):
        """
        Append a fragment to the list.

        @param f: the fragment
        @type f:  L{fragment}
        """
        self._frags.append(copy.copy(f))
        self._frags[-1].fragname = "frag" + str(len(self._frags))

    def __iter__(self):
        """
        Iterator for fragmentlist.

        Iteration loops over the list of fragments.
        """
        return self._frags.__iter__()

    @property
    def frags(self):
        return self._frags

    def get_frozen_frags(self):
        """
        Returns a list of the frozen fragments.
        """
        return [f for f in self._frags if f.isfrozen]

    def get_nonfrozen_frags(self):
        """
        Returns a list of the nonfrozen fragments.
        """
        return [f for f in self._frags if not f.isfrozen]

    def get_total_molecule(self):
        """
        Returns the total molecule.

        @rtype: L{molecule}
        """
        mols = [frag.get_total_molecule() for frag in self._frags]
        return reduce(lambda x, y: x + y, mols)

    def get_frozen_molecule(self):
        """
        Returns the frozen part as a molecule.

        @rtype: L{molecule}
        """
        mols = [frag.get_total_molecule() for frag in self.get_frozen_frags()]
        return reduce(lambda x, y: x + y, mols)

    def get_nonfrozen_molecule(self):
        """
        Returns the nonfrozen part as a molecule.

        @rtype: L{molecule}
        """
        mols = [frag.get_total_molecule() for frag in self.get_nonfrozen_frags()]
        return reduce(lambda x, y: x + y, mols)

    def get_atomtypes_without_fragfile(self):
        """
        Return a list of the atomtypes for which no fragment file is available.

        For these atom types, atomic fragment files (or a BASIS block)
        have top be provided
        """

        mols = [frag.get_total_molecule() for frag in self._frags if not frag.has_frag_results()]
        mols = [m for m in mols if m]

        if len(mols) == 0:
            atomtypes = []
        else:
            mol = reduce(lambda x, y: x + y, mols)
            atomtypes = mol.get_atom_symbols(prefix_ghosts=True)

        return atomtypes

    def has_fde_fragments(self):
        """
        Whether there is at least one fragment obtained from an FDE calculation in the list.

        This FDE fragment does not have to be frozen. The distiction refers to the
        way the fragment was calculated previously.
        """
        return reduce(lambda x, y: x or y, [f.is_fde_fragment() for f in self._frags])

    def has_fragoccupations(self):
        """
        Whether fragment occupations have been specified for at least one fragment in the list.
        """
        return reduce(lambda x, y: x or y, [f.has_occupations() for f in self._frags])

    def get_num_frozen_frags(self):
        """
        Returns the number of frozen fragments.
        """
        numfrozenfrags = 0
        for f in self._frags:
            if f.isfrozen:
                numfrozenfrags += f.get_num_frags()
        return numfrozenfrags

    def has_frozen_fragment(self):
        """
        Whether there is at least one frozen fragment in the list.
        """
        return reduce(lambda x, y: x or y, [f.isfrozen for f in self._frags])

    def has_relax_fragment(self):
        """
        Whether there is at least one frozen fragment with RELAX option in the list.
        """
        fderelax = False
        for f in self._frags:
            fderelax = fderelax or (f.isfrozen and f.has_fdeoption('RELAX'))
        return fderelax

    def get_atoms_block(self):
        AtomsBlock = ""
        AtomsBlock += " ATOMS [Angstrom]\n"
        for frag in self.__iter__():
            AtomsBlock += frag.get_atoms_block()
        AtomsBlock += " END\n\n"
        return AtomsBlock

    def get_fragments_block(self, checksum_only):

        block = ""
        for frag in self.__iter__():
            if frag.has_frag_results():
                block += frag.get_fragments_block(checksum_only)
        return block

    def get_fragoccupations_block(self):
        block = ""
        for frag in self.__iter__():
            block += frag.get_fragoccupations_block()
        return block

    def get_special_options_block(self):
        block = ""
        for frag in self.__iter__():
            block += frag.get_special_options_block()
        return block

    def copy_fragment_files(self):
        """
        Copy all fragment files to the current working directory.
        """
        for f in self.__iter__():
            f.copy_fragment_file()

    def delete_fragment_files_copy(self):
        """
        Delete copies of fragment files in the current working directory.
        """
        for f in self.__iter__():
            f.delete_fragment_file_copy()

    def calculate_all(self, func, context=None):
        """
        Loop over all fragments and calculated the results for the fragments.

        @param func: The function that performs the actual calculation.
        @type  func: A function with the signature func(mol, **context), where
                     mol is a L{molecule} and context are additional keyword
                     arguments, returning an instance of L{adfsinglepointresults}
                     or of a derived class.
        @param context: An optional dictionary of additional keyword arguments to func.
        @type  context: dict
        """
        for f in self.__iter__():
            f.calculate(func, context)


class adffragmentsresults(adfsinglepointresults):
    """
    Results of a L{adffragmentsjob}.

    @undocumented: __delattr__, __getattribute__, __hash__, __new__,
                   __repr__, __str__, __setattr__
    """

    def __init__(self, j=None):
        # pylint: disable-msg=W0231
        super().__init__(j)

        # cache the results checksum
        _ = self.checksum

        for frag in self.job.get_fragmentlist():
            # remove density / potential gridfunctions from copied jobs,
            # because these will fill up the memory and are not needed
            if isinstance(frag, FrozenDensFragment):
                frag._density = None
                frag._potential = None

    def get_subfragments(self):
        """
        Returns a list of the subfragments of this calculation.

        @returns: A list with the subfragments used in this calculation
        @rtype:   list of L{fragment}s
        """

        frags = [fragment(None, self.job.get_nonfrozen_molecule(), subfrag="active")]

        return frags

    def export_embedding_data(self, filename_potential='embpot_adfgrid', filename_density='frzdens_adfgrid',
                              hessian=False):
        """
        Export the embedding potential and optionally frozen densities and gradients to
        a text file (for use with Dalton/Dirac)

        The embedding potentail will be, together with the integration grid
        and weights, written to a text file with the following format:
        First line: number of points
        Following lines: x,y,z coordinates of the grid point, w in this point, v_emb in this point
        Last line: -42

        The density and gradient will be given in a separate file, that also contains
        the integration grid and weights, in the following format
        First line: number of points, number of properties exported (densities, gradients)
        Following lines: x,y,z coordinates of the grid point, w in this point, density(a+b),
        grd.grd (averaged) in this point
        TODO: in the case of unrestricted calculations, see whether to export
        density_a, density_b, grd_a.grd_a, grd_b.grd_b, grd_a.grd_b in
        this point separately (that is, without generating the total ones)
        Last line: -42

        @param filename_potential: name of the file where the potential will be written
        @type filename_potential:  str
        @param filename_density: name of the file where the potential will be written
        @type filename_density:  str
        @param hessian: Whether to include the density Hessian in the density file
        @type hessian: str
        """

        if not self.job.is_fde_job():
            raise PyAdfError('export_embedding_data called, but calculation is no FDE job')

        nspin = self.get_result_from_tape('General', 'nspin')

        vemb_outfile = open(filename_potential, 'w')
        dens_outfile = open(filename_density, 'w')

        nblock = self.get_result_from_tape('Points', 'nblock', tape=10)
        npoints = self.get_result_from_tape('Points', 'lblock', tape=10)

        vemb_outfile.write(f"{nblock * npoints:d} \n")

        # exported_properties reflects the number of quantities, apart from the grid and weights, we are exporting
        #
        # this number is related to the following at the moment (columns mean the columns after the grid point
        # coordinates and weights):
        #
        # columns 5-6: electrostatic potentials (5:coulomb+nuclear,
        #              6:point charges) of frozen subsystems/surroundings
        # columns 6-9: 6:frozen density and  7:x, 8:y, 9:z frozen density gradient components
        # columns 10-15: frozen density hessian components, if we are to export that

        if hessian:
            exported_properties = 10
        else:
            exported_properties = 6

        dens_outfile.write(f"{nblock * npoints:d}     {exported_properties:d}\n")

        PointsData = self.get_result_from_tape('Points', 'Data', tape=10, always_array=True)
        # andre: as far as i see, v_nuc is always together with v_H
        elpotFD = self.get_result_from_tape('FrozenDensityElpot', 'ElpotFD', tape=10, always_array=True)
        efieldpot = self.get_result_from_tape('Efield', 'Efield', tape=10, always_array=True)
        xckinpotFD = self.get_result_from_tape('FrozenDensityXcKinpot', 'XcKinPotFD', tape=10, always_array=True)

        frzdens = self.get_result_from_tape('FrozenDensity', 'rhoffd', tape=10, always_array=True)
        frzdensgrd = self.get_result_from_tape('FrozenDensityFirstDer', 'drhoffd', tape=10, always_array=True)
        frzdenshes = self.get_result_from_tape('FrozenDensitySecondDer', 'd2rhoffd', tape=10, always_array=True)

        vemb_points_written = 0
        dens_points_written = 0

        for iblock in range(1, nblock + 1):
            ipoint = npoints * (iblock - 1)

            coords = PointsData[ipoint * 4:ipoint * 4 + 3 * npoints]
            coords = coords.reshape((npoints, 3), order='F')
            weights = PointsData[ipoint * 4 + 3 * npoints:ipoint * 4 + 4 * npoints]

            potel = elpotFD[ipoint:ipoint + npoints]
            potefld = efieldpot[ipoint:ipoint + npoints]
            potxc = xckinpotFD[ipoint * nspin:ipoint * nspin + npoints * nspin]

            dens = frzdens[ipoint * nspin:(ipoint + npoints) * nspin]
            densgrd = frzdensgrd[ipoint * nspin * 3:(ipoint + npoints) * nspin * 3]
            densgrd = densgrd.reshape((npoints, 3, nspin), order='F')

            if nspin == 2:
                potxc = potxc.reshape((npoints, nspin), order='F')
                potxc = 0.5 * (potxc[:, 0] + potxc[:, 1])
                # we export the total density now, so we sum alpha and beta components
                dens = dens.reshape((npoints, nspin), order='F')
                dens = dens[:, 0] + dens[:, 1]
                # and we also average the gradients..? (todo)
                densgrd = densgrd[:, :, 0] + densgrd[:, :, 1]
            else:
                densgrd = densgrd[:, :, 0]

            for c, w, e, ef, x in zip(coords, weights, potel, potefld, potxc):
                vemb_outfile.write(
                    f"{c[0]:25.18e}  {c[1]:25.18e}  {c[2]:25.18e}  {w:25.18e} {(e + ef + x):25.18e} \n")
                vemb_points_written += 1

            if hessian:
                denshes = frzdenshes[ipoint * nspin * 6:(ipoint + npoints) * nspin * 6]
                denshes = denshes.reshape((npoints, 6, nspin), order='F')
                if nspin == 2:
                    denshes = denshes[:, :, 0] + denshes[:, :, 1]
                else:
                    denshes = denshes[:, :, 0]

                for c, w, e, ef, n, dg, dh in zip(coords, weights, potel, potefld, dens, densgrd, denshes):
                    dens_outfile.write(" {:25.18e}   {:25.18e}   {:25.18e}   {:25.18e}      "
                                       " {:25.18e}   {:25.18e}     {:25.18e}  "
                                       " {:25.18e}  {:25.18e}  {:25.18e}   {:25.18e}  {:25.18e}  {:25.18e}"
                                       " {:25.18e}  {:25.18e}  {:25.18e}\n"
                                       .format(*c, w, e, ef, n, *dg, *dh))
                    dens_points_written += 1
            else:
                for c, w, e, ef, n, dg in zip(coords, weights, potel, potefld, dens, densgrd):
                    dens_outfile.write(" {:25.18e}   {:25.18e}   {:25.18e}   {:25.18e}       "
                                       "{:25.18e}   {:25.18e}      {:25.18e}   {:25.18e}  {:25.18e}  {:25.18e}\n"
                                       .format(*c, w, e, ef, n, *dg))
                    dens_points_written += 1

        vemb_outfile.write("-42\n")
        dens_outfile.write("-42\n")

        if not (vemb_points_written == nblock * npoints):
            raise PyAdfError('wrong number of points written for embedding potential export')
        if not (dens_points_written == nblock * npoints):
            raise PyAdfError('wrong number of points written for frozen density export')

        vemb_outfile.close()
        dens_outfile.close()

    def get_nonfrozen_molecule(self):
        return self.job.get_nonfrozen_molecule()

    def get_frozen_molecule(self):
        return self.job.get_frozen_molecule()

    #   @use_default_grid
    def get_fragment_density(self, grid=None, fit=False, orbs=None, order=None, frag=None):

        if (orbs is not None) and ('Loc' not in list(orbs.keys())):
            if (order is not None) and (order >= 1):
                raise PyAdfError("Derivatives not implemented for orbital densities.")
            if (frag is not None) and not (frag == 'active'):
                raise PyAdfError("Orbital densities only available for acyive fragment")
            return self.get_orbital_density(grid, orbs)
        elif (order is not None) and (order >= 1):
            # the density itself
            prop = PlotPropertyFactory.newDensity('dens', fit=fit, orbs=orbs)
            dres = densfjob(self, prop, grid=grid, frag=frag).run()
            gfs = [dres.get_gridfunction()]

            # density gradient
            if order >= 1:
                prop = PlotPropertyFactory.newDensity('grad', fit=fit, orbs=orbs)
                dres = densfjob(self, prop, grid=grid, frag=frag).run()
                gfs.append(dres.get_gridfunction())

            # density hessian
            if order >= 2:
                prop = PlotPropertyFactory.newDensity('hess', fit=fit, orbs=orbs)
                dres = densfjob(self, prop, grid=grid, frag=frag).run()
                gfs.append(dres.get_gridfunction())

            return GridFunctionDensityWithDerivatives(gfs)

        else:
            prop = PlotPropertyFactory.newDensity('dens', fit, orbs)
            dres = densfjob(self, prop, grid=grid, frag=frag).run()
            return dres.get_gridfunction()

    @use_default_grid
    def get_nonfrozen_density(self, grid=None, fit=False, orbs=None, order=None):
        """
        Returns the electron density of the nonfrozen (active) subsystem.

        For details on the processing of the electron density,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the fit density is returned, otherwise
                    the exact density.
        @type  fit: bool
        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict

        @param order: order of derivatives of the density to calculate (1 and 2 possible)
        @type order: int
        """
        return self.get_fragment_density(grid=grid, fit=fit, orbs=orbs, order=order, frag='Active')

    @use_default_grid
    def get_frozen_density(self, grid=None, fit=False, orbs=None, order=None):
        """
        Returns the frozen electron density.

        For details on the processing of the electron density,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the fit density is returned, otherwise
                    the exact density.
        @type  fit: bool
        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict

        @param order: order of derivatives of the density to calculate (1 and 2 possible)
        @type order: int
        """
        frozendens = []
        for f in self.job.get_fragmentlist().get_frozen_frags():
            fragdens = self.get_fragment_density(grid=grid, fit=fit, orbs=orbs,
                                                 order=order, frag=f.fragname)
            frozendens.append(fragdens)

        frozendens = reduce(lambda x, y: x + y, frozendens)

        return frozendens

    @use_default_grid
    def get_density(self, grid=None, fit=False, orbs=None, order=None):
        """
        Returns the electron density.

        For details on the processing of the electron density,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the fit density is returned, otherwise
                    the exact density.
        @type  fit: bool
        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict
        @param order: order of derivatives of the density to calculate (1 and 2 possible)
        @type order: int
        """
        if not self.job.is_fde_job() or orbs is not None:
            return self.get_nonfrozen_density(grid=grid, fit=fit, orbs=orbs, order=order)
        else:
            return self.get_nonfrozen_density(grid=grid, fit=fit, order=order) + \
                   self.get_frozen_density(grid=grid, fit=fit, order=order)

    @use_default_grid
    def get_embedding_potential(self, grid=None, pot="total"):
        """
        Returns the FDE embedding potential.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @param pot: Which potential to calculate. One of: total, nuc, coul, elstat, xc
        @type  pot: str
        """

        if not self.job.is_fde_job():
            raise PyAdfError("Can get embedding potenial only for FDE jobs")

        if pot.lower() == "total":
            prop = PlotPropertyFactory.newPotential("embpot")
        elif pot.lower() == "nuc":
            prop = PlotPropertyFactory.newPotential("embnuc")
        elif pot.lower() == "coul":
            prop = PlotPropertyFactory.newPotential("embcoul")
        elif pot.lower() == "xc":
            prop = PlotPropertyFactory.newPotential("nadxc")
        elif pot.lower() == "kin":
            prop = PlotPropertyFactory.newPotential("nadkin")
        elif pot.lower().startswith("kinpot"):
            s1, s2 = pot.split(None, 1)
            prop = PlotPropertyFactory.newPotential("nadkin", func=s2)
        else:
            raise PyAdfError("Unknown potential requested")

        res = densfjob(self, prop, grid=grid, frag="ALL").run()
        return res.get_gridfunction()

    @use_default_grid
    def get_frozen_potential(self, grid=None, pot="total"):
        """
        Returns the potential belonging to the frozen electron density.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @param pot: Which potential to calculate. One of: total, nuc, coul, elstat, xc
        @type  pot: str
        """
        if not self.job.is_fde_job():
            raise PyAdfError("Can get frozen potential only for FDE jobs")

        prop = PlotPropertyFactory.newPotential(pot.lower())

        frozenpot = []
        for f in self.job.get_fragmentlist().get_frozen_frags():
            afrozenpot = densfjob(self, prop, grid=grid, frag=f.fragname).run()
            frozenpot.append(afrozenpot.get_gridfunction())

        frozenpot = reduce(lambda x, y: x + y, frozenpot)

        return frozenpot

    @use_default_grid
    def get_nonfrozen_potential(self, grid=None, pot="total"):
        """
        Returns the potential of the electron density of the nonfrozen (active) subsystem.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}

        @param pot: Which potential to calculate. One of: total, nuc, coul, elstat, xc
        @type  pot: str
        """
        prop = PlotPropertyFactory.newPotential(pot.lower())

        res = densfjob(self, prop, grid=grid, frag='Active').run()
        return res.get_gridfunction()


class adffragmentsjob(adfsinglepointjob):
    """
    ADF fragments analysis job.

    @undocumented: __delattr__, __getattribute__, __hash__, __new__,
                   __repr__, __str__, __setattr__
    """

    def __init__(self, fragments, basis=None, settings=None, core=None,
                 fde=None, pointcharges=None, fitbas=None, options=None):
        """
        Initialize ADF fragments analysis / FDE job.

        @param fragments:
            the list of fragments to be used in this job.
        @type fragments: list of L{fragment}s or L{fragmentlist}

        @param fde:
            a dictionary with options for FDE. They will be added in the FDE
            block of the input.
            The functional for the non-additive kinetic energy can be specified
            under the key TNAD. If is is not specified, PW91k will be
            chosen as default.
        @type fde: dict

        """

        if isinstance(fragments, list):
            self._fragments = fragmentlist(fragments)
        else:
            self._fragments = fragments

        if fde is None:
            self._fde = {}
        else:
            self._fde = {}
            for k, v in fde.items():
                self._fde[k.upper()] = v

        super().__init__(None, basis, core=core, settings=settings,
                         pointcharges=pointcharges, fitbas=fitbas, options=options)

        if self._fragments.has_fde_fragments():
            if 'ALLOW PARTIALSUPERFRAGS' not in self._options:
                self._options.append('ALLOW PARTIALSUPERFRAGS')

    def create_results_instance(self):
        return adffragmentsresults(self)

    def is_fde_job(self):
        return self._fragments.has_frozen_fragment()

    def is_fde_relax_job(self):
        return self._fragments.has_relax_fragment()

    def get_fragmentlist(self):
        return self._fragments

    def get_molecule(self):
        return self._fragments.get_total_molecule()

    def get_frozen_molecule(self):
        return self._fragments.get_frozen_molecule()

    def get_nonfrozen_molecule(self):
        return self._fragments.get_nonfrozen_molecule()

    def get_atomtypes_without_fragfile(self):
        return self._fragments.get_atomtypes_without_fragfile()

    def print_jobtype(self):

        if self.is_fde_job():
            jobtype = "ADF NewFDE job"
        else:
            jobtype = "ADF fragments analysis job"

        return jobtype

    def print_molecule(self):

        print("   Total Molecule")
        print("   ==============")
        print()
        print(self.get_molecule())
        print()
        print("   Fragments ")
        print("   ==========")
        print()
        for num_ftyp, frag in enumerate(self._fragments):
            print("     Fragment Typ ", num_ftyp + 1, "  ", end=' ')
            frag.print_fragment_options()
            for num_frag, m in enumerate(frag.get_molecules()):
                print("     Fragment Typ ", num_ftyp + 1, ", Fragment ", num_frag + 1)
                print(m)
        print()

        print("   Fragment Files ")
        print("   ============== ")
        print()
        for num_ftyp, frag in enumerate(self._fragments):
            filename = frag.get_fragment_filename()
            if filename:
                print("     Fragment Typ ", num_ftyp + 1, ": ", filename)

        print()

    def print_settings(self):

        print("   Settings")
        print("   ========")
        print()
        print(self.settings)
        print()

        if self.is_fde_job():
            print("   FDE settings")
            print("   ============")
            print()
            if 'TNAD' not in self._fde:
                print('   TNAD  PW91k')
            for k, v in self._fde.items():
                print(f'   {k}  {v}')
            print()

    def get_atoms_block(self):
        """
        give back the atoms input key for ADF input files
        """
        return self._fragments.get_atoms_block()

    def get_charge_block(self):
        block = ""
        if isinstance(self.get_molecule().get_charge(), int):
            block += f" CHARGE {self._fragments.get_nonfrozen_molecule().get_charge():d}"
        else:
            block += f" CHARGE {self._fragments.get_nonfrozen_molecule().get_charge():8.4f}"
        block += " \n\n"
        return block

    def get_spin_block(self):
        block = ""
        if self.get_molecule().get_spin() > 0:
            block += f"SPINPOLARIZATION {self._fragments.get_nonfrozen_molecule().get_spin():2d}"
            block += " \n\n"
        return block

    def get_fragments_block(self):
        block = " FRAGMENTS\n"
        block += self._fragments.get_fragments_block(self._checksum_only)
        block += " END\n\n"
        return block

    def get_fragoccupations_block(self):
        block = ""
        block += " FRAGOCCUPATIONS\n"
        block += self._fragments.get_fragoccupations_block()
        block += " END\n\n"
        return block

    def get_other_blocks(self):
        block = super().get_other_blocks()
        block += self._fragments.get_special_options_block()
        if self.is_fde_job():
            block += self.get_fde_block()
        if self._fragments.has_fragoccupations():
            block += self.get_fragoccupations_block()
        return block

    def get_fde_block(self):

        for k, v in self._fde.items():
            if k in ['COULOMB', 'THOMASFERMI', 'PW91K', 'PW91Kscaled', 'TW02', 'TF9W', 'PBE2', 'PBE3', 'PBE4'] \
                    and v == '':
                raise PyAdfError('   Incorrect definition of KEF. Use the TNAD keyword to specify the functional')

        block = " FDE\n"
        if 'TNAD' in self._fde:
            block += '   ' + self._fde['TNAD'] + '\n'
        else:
            block += "   PW91k\n"
        for opt, val in self._fde.items():
            if opt == 'TNAD':
                continue
            block += "   " + opt + " " + str(val) + "\n"
        block += " END\n\n"
        return block

    def before_run(self):
        super().before_run()
        self._fragments.copy_fragment_files()

    def after_run(self):
        super().after_run()
        self._fragments.delete_fragment_files_copy()
