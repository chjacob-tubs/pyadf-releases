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
 Dalton CC2 excitation energy calculations

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    daltonCC2job
 @group Settings:
    daltonCC2settings
 @group Results:
    daltonCC2results
"""

from .DaltonSinglePoint import daltonjob, daltonsinglepointjob, daltonresults, \
                              daltonsinglepointresults, daltonsettings

import re


class daltonCC2results (daltonsinglepointresults):

    """
    Class for results of an Dalton CC2 calculation.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_excitation_energies, get_oscillator_strengths
    """

    def __init__(self, j=None):
        """
        Constructor for daltonCC2results.
        """
        daltonsinglepointresults.__init__(self, j)

    def get_excitation_energies(self):
        """
        Returns the CC2 excitation energies (in eV).

        @returns: list of the calculated excitation energies, in eV
        @rtype: list of float
        """
        exens = []

        output = self.get_output()

        start = re.compile(r".*CC2\s*Excitation energies")
        for i, l in enumerate(output):
            m = start.match(l)
            if m:
                startline = i

        exen = re.compile(r"""\s* \| \s* \^1A \s* \|
                                   \s* (\d+)  \s* \|
                               \s*(?P<exenau>[-+]?(\d+(\.\d*)?|\d*\.\d+)) \s* \|
                               \s*(?P<exeneV>[-+]?(\d+(\.\d*)?|\d*\.\d+)) \s* \|
                           """, re.VERBOSE)

        for i in range(self.job.nexci):
            m = exen.match(output[startline + 4 + i])
            exens.append(float(m.group("exeneV")))

        return exens

    def get_oscillator_strengths(self):
        """
        Returns the CC2 excitation energies (in eV).

        @returns: a list of the calculated oszillator strengths (FIZME: units?)
        @rtype:   list of float
        """
        os = []

        output = self.get_output()

        start = re.compile(r".*(CC2\s*Transition properties|CC2\s*Length\s+Gauge Oscillator Strength)")
        for i, l in enumerate(output):
            m = start.match(l)
            if m:
                startline = i

        oscstr = re.compile(r"""\s* \| \s* \^1A \s* \|
                                   \s* (\d+)  \s* \|
                               \s*(?P<dipstr>[-+]?(\d+(\.\d*)?|\d*\.\d+)) \s* \|
                               \s*(?P<oscstr>[-+]?(\d+(\.\d*)?|\d*\.\d+)) \s* \|
                           """, re.VERBOSE)

        for i in range(self.job.nexci):
            m = oscstr.match(output[startline + 4 + i])
            os.append(float(m.group("oscstr")))

        return os


class daltonCC2settings (daltonsettings):

    """
    Class that holds the settings for a Dalton CC2 calculation..

    @group Initialization:
        set_nexci, set_freeze
    @group Input Generation:
        get_wavefunction_block
    @group Other Internals:
        __str__
    """

    def __init__(self, nexci=10, freeze_occ=0, freeze_virt=0,):
        """
        Constructor for daltonCC2settings.

        All arguments are optional, leaving out an argument will choose default settings.

        @param nexci: Number of excitations to calculate, see L{set_nexci}.
        @type  nexci: int

        @param freeze_occ: number of occupied orbitals to freeze, see L{set_freeze}.
        @type  freeze_occ: int

        @param freeze_virt: number of virtual orbitals to freeze, see L{set_freeze}.
        @type  freeze_virt: int
        """
        daltonsettings.__init__(self, method='CC')

        self.nexci = None
        self.freeze_occ = None
        self.freeze_virt = None

        self.set_nexci(nexci)
        self.set_freeze(freeze_occ, freeze_virt)

    def set_nexci(self, nexci):
        """
        Set the number of excitations to calculated.

        @param nexci: the number of excitations to calculate
        @type  nexci: int
        """
        self.nexci = nexci

    def set_freeze(self, freeze_occ, freeze_virt):
        """
        Set the number of orbitals to freeze in the CC2 calculation.

        @param freeze_occ: number of frozen occupied orbitals
        @type  freeze_occ: int
        @param freeze_virt: number of frozen virtual orbitals
        @type  freeze_virt: int
        """
        self.freeze_occ = freeze_occ
        self.freeze_virt = freeze_virt

    def __str__(self):
        """
        Returns a human-readable description of the settings.
        """
        s = "  Method: CC2 \n\n"
        s += "  Number of excitations: %i \n" % self.nexci
        s += "  Number of frozen occupied orbitals: %i \n" % self.freeze_occ
        s += "  Number of frozen virtual orbitals:  %i \n" % self.freeze_virt
        return s


class daltonCC2job (daltonsinglepointjob):

    """
    A class for Dalton CC2 excitation energy calculations.

    See the documentation of L{__init__} and L{daltonCC2settings }
    for details on the available options.

    Corresponding results class: L{daltonCC2results}

    @group Initialization:
        __init__
    @group Input Generation:
        get_cc_block
    @undocumented: _get_nexci
    """

    def __init__(self, mol, basis, settings=None, fdein=None, options=None):
        """
        Constructor for Dalton CC2 jobs.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}

        @param basis:
            A string specifying the basis set to use (e.g. C{basis='cc-pVDZ'}).
        @type basis: str

        @param settings: The settings for the Dalton CC2 job.
        @type  settings: L{daltonCC2settings}

        @param fdein:
            Results of an ADF FDE calculation. The embedding potential from this
            calculation will be imported into Dalton (requires modified Dalton version).
        @type  fdein: L{adffragmentsresults}

        @param options:
            Additional options.
            These will each be included directly in the Dalton input file.
        @type options: list of str
        """

        if settings == None:
            self.settings = daltonCC2settings()
        else:
            self.settings = settings

        daltonsinglepointjob.__init__(self, mol, basis, fdein=fdein,
                                      settings=self.settings, options=options)

    def get_runscript(self):
        # CC is not available in MPI parallel runs
        return daltonjob.get_runscript(self, nproc=None, memory=self.settings.memory)

    def _get_nexci(self):
        return self.settings.nexci
    nexci = property(_get_nexci, None, None, """
    The number of excitations that were calculated.

    @type: int
    """)

    def create_results_instance(self):
        return daltonCC2results(self)

    def get_integral_block(self):
        block = "**INTEGRAL\n"
        block += ".DIPLEN\n"
        return block

    def get_cc_block(self):
        block = "*CC INPUT \n"
        block += ".CC2 \n"
        block += ".PRINT \n"
        block += "2 \n"
        block += ".NSYM \n"
        block += "1 \n"
        # freeze orbitals
        block += ".FREEZE\n"
        block += "%i %i\n" % (self.settings.freeze_occ, self.settings.freeze_virt)
        block += "*CCEXCI \n"
        block += ".NCCEXCI \n"
        block += str(self.settings.nexci) + "\n"
        block += "*CCLRSD \n"
        block += ".DIPOLE \n"
        # properties
        block += "*CCFOP\n"
        block += ".NONREL\n"
        block += ".DIPMOM\n"
        block += "*CCEXGR\n"
        block += ".DIPOLE\n"
        return block

    def get_other_blocks(self):
        blocks = daltonsinglepointjob.get_other_blocks(self)
        blocks += self.get_cc_block()
        return blocks

    def print_jobtype(self):
        return "Dalton Excitations (CC2) job"
