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
 WFT-in-DFT calculations

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""

from ADFFragments import fragment, fragmentlist, adffragmentsjob
from BaseJob import metajob
from Dirac import diracsinglepointjob

import os


class diracfragment (fragment):

    def __init__(self, dirac_results, mols=None, subfrag=None, isfrozen=False, fdeoptions=None, occ=None):

        self._dirac_results = dirac_results

        adf_results = self._dirac_results.fdein
        fragment.__init__(self, adf_results, mols, subfrag, isfrozen, fdeoptions, occ)

    def get_dirac_tape10_filename(self):
        return os.path.basename(self._dirac_results.get_gridout_filename()) + ".dirac"

    def get_fdeoptions_block(self):
        block = fragment.get_fdeoptions_block(self)
        block += "     fileimport " + self.get_dirac_tape10_filename() + "\n"
        return block

    def get_special_options_block(self):
        block = fragment.get_special_options_block(self)
        block += " IMPORTGRID " + self.get_dirac_tape10_filename() + "\n\n"
        return block

    def copy_fragment_file(self):
        """
        Copies the fragment file to the working directory.
        """
        fragment.copy_fragment_file(self)

        filename = self.get_dirac_tape10_filename()
        if not os.path.exists(filename):
            self._dirac_results.export_dirac_tape10(filename)

    def delete_fragment_file_copy(self):
        """
        Delete the fragment file that was previously copied to the working directory.
        """
        fragment.delete_fragment_file_copy(self)

        filename = self.get_dirac_tape10_filename()
        if os.path.exists(filename):
            os.remove(filename)

    def print_fragment_options(self):
        """
        Print information about the options used for this fragment.
        """
        if self.isfrozen:
            print " type: frozen Dirac fragment"
        else:
            print " type: nonfrozen Dirac fragment"
        print

    def get_fragments_block(self, checksumonly):
        """
        Return a line that is can be used in the FRAGMENTS block
        of the ADF input for this fragment.

        @returns: a string that can be used in the FRAGMENTS block
        @rtype:   str
        """

        block = fragment.get_fragments_block(self, checksumonly)
        if checksumonly:
            block += self._dirac_results.get_checksum()
        return block


class wftindftjob (metajob):

    def __init__(self, frags, adfoptions, diracoptions):
        metajob.__init__(self)

        if isinstance(frags, list):
            self._frags = fragmentlist(frags)
        else:
            self._frags = frags

        self._adfoptions = adfoptions
        self._diracoptions = diracoptions

        self._cycles = 0
        if "fde" in self._adfoptions:
            if 'RELAXCYCLES' in self._adfoptions["fde"]:
                self._cycles = self._adfoptions["fde"]['RELAXCYCLES']
                del self._adfoptions["fde"]['RELAXCYCLES']

    def metarun(self):

        import copy

        print "-" * 50
        print "Beginning WFT-in-DFT embedding calculation "
        print
        print "Performing %i RELAX-cycles" % self._cycles
        print

        # first we do a normal DFT-in-DFT run
        initial_frags = copy.deepcopy(self._frags)
        for f in initial_frags:
            # rename RELAX option to RELAXsave so that we can handle it, instead of ADF
            if f.has_fdeoption("RELAX"):
                f.delete_fdeoption("RELAX")
                f.add_fdeoption("RELAXsave", "")

        # pylint: disable-msg=W0142
        dftindft_res = adffragmentsjob(initial_frags, **self._adfoptions).run()

        # now construct a new list of fragments
        frags = copy.deepcopy(fragmentlist(initial_frags.get_frozen_frags()))
        for f in frags:
            if f.has_fdeoption("RELAXsave"):
                f.isfrozen = False
        dirac_frag = fragment(dftindft_res, dftindft_res.get_nonfrozen_molecule(), subfrag="active", isfrozen=True)
        dirac_mol = dirac_frag.get_total_molecule()
        frags.append(dirac_frag)

        # now run a DFT-in-DFT calculation with the Dirac-fragment frozen,
        # and all DFT-relax fragments nonfrozen in order to get a grid that
        # can be used for all of them
        # (the Dirac step is expensive, so we want to do it only for one common grid)
        #
        # FIXME: Alternatively, Dirac could use several grids for exporting
        #
        # FIXME: we do the full DFT-in-DFT calculation, but only extracting
        #        the grid would be enough; maybe use some STOPAFTER option?

        # pylint: disable-msg=W0142
        outgrid_res = adffragmentsjob(frags, **self._adfoptions).run()

        for i in range(self._cycles):

            print "-" * 50
            print "Performing cycle ", i + 1
            print

            # do the Dirac calculation
            # pylint: disable-msg=W0142
            dirac_res = diracsinglepointjob(dirac_mol, fdein=dftindft_res, fdeout=outgrid_res, **self._diracoptions).run()

            # construct new list of fragments, update the Dirac fragment,
            frags = fragmentlist(initial_frags.get_frozen_frags())
            dirac_frag = diracfragment(dirac_res, dirac_mol, isfrozen=True)
            frags.append(dirac_frag)

            # loop over all DFT-relax fragments and update them
            for f in frags:
                if f.has_fdeoption("RELAXsave"):
                    f.isfrozen = False
                    # pylint: disable-msg=W0142
                    f.results = adffragmentsjob(frags, **self._adfoptions).run()
                    f.isfrozen = True

            frozenfrags = frags.get_frozen_frags()
            frozenfrags = [f for f in frozenfrags if not isinstance(f, diracfragment)]
            initial_frags = copy.deepcopy(fragmentlist(frozenfrags))
            dirac_frag = fragment(dftindft_res, dirac_mol, subfrag="active", isfrozen=False)
            initial_frags.append(dirac_frag)

            # run ADF DFT-in-DFT again to get a new embedding potential
            #
            # FIXME: rho1 is calculated by ADF in this step again.
            #        Instead, the rho1 calculated by Dirac should be used
            dftindft_res = adffragmentsjob(initial_frags, **self._adfoptions).run()

        print "-" * 50
        print "Performing final DIRAC calculation "
        print

        dirac_res = diracsinglepointjob(dirac_mol, fdein=dftindft_res, fdeout=outgrid_res, **self._diracoptions).run()

        return dirac_res
