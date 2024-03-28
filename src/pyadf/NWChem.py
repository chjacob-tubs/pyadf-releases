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
 The basics needed for NWChem calculations

 @author:       Lucas Visscher and others
 @organization: VU University Amsterdam

 @group Jobs:
    nwchemjob, nwchemsinglepointjob
 @group Settings:
    nwchemsettings
 @group Results:
    nwchemresults, nwchemsinglepointresults
"""

from .Errors import PyAdfError
from .BaseJob import results, job
from pyadf.PyEmbed.DensityEvaluator import GTODensityEvaluatorMixin

import os
import re


class nwchemresults(results):
    """
    Class for results of an NWChem calculation.
    """

    def __init__(self, j=None):
        """
        Constructor for nwchemresults.
        """
        super().__init__(j=j)
        self.resultstype = 'NWChem results'


class nwchemsinglepointresults(nwchemresults, GTODensityEvaluatorMixin):
    """
    Class for results of an NWChem single point calculation.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_molecule, get_energy
    @undocumented: _get_fdein
    """

    def __init__(self, j=None):
        """
        Constructor for nwchemsinglepointresults.
        """
        super().__init__(j=j)

    def get_molecule(self):
        """
        Return the molecular geometry after the NWChem job.

        @returns: The molecular geometry.
        @rtype:   L{molecule}

        """
        return self.job.mol

    def read_molden_file(self):
        """
        Returns Molden results file as a string.
        """

        try:
            molden_filename = self.files.get_results_filename(self.fileid, tape=41)
        except PyAdfError:
            raise PyAdfError("NWChem Molden file not found")

        with open(molden_filename, encoding='utf-8') as f:
            content = f.read()

        return content

    def get_dipole_vector(self):
        """
        Return the dipole moment vector.

        @returns: the dipole moment vector, in atomic units
        @rtype: float[3]
        """

        dipole = [0.0, 0.0, 0.0]

        output = self.get_output()

        startline = None
        start = re.compile(r"\s*Dipole Moment")
        for i, l in enumerate(output):
            m = start.match(l)
            if m:
                startline = i

        if startline is None:
            raise PyAdfError('NWChem dipole moment not found in output')

        for i, c in enumerate(['X', 'Y', 'Z']):
            dip = re.compile(r"\s*DM" + c + r"\s*(?P<dip>[-+]?(\d+(\.\d*)?|\d*\.\d+))\s+DM" + c + "EFC")
            m = dip.match(output[startline + 7 + i])
            dipole[i] = float(m.group('dip'))

        return dipole

    def get_energy(self):
        """
        Return the total energy

        @returns: the total energy in atomic units
        @rtype: float
        """

        energy = 0.0

        output = self.get_output()
        en_re = re.compile(r"^ +Total (SCF|DFT) energy = *(?P<energy>-?\d+\.\d+)")
        if self.job.settings.method.upper() == 'CCSD':
            en_re = re.compile(r"^ Total CCSD energy: *(?P<energy>-?\d+\.\d+)")
        elif self.job.settings.method.upper() == 'CCSD(T)':
            en_re = re.compile(r"^ Total CCSD\(T\) energy: *(?P<energy>-?\d+\.\d+)")
        elif self.job.settings.method.upper() == 'CCSD+T(CCSD)':
            en_re = re.compile(r"^ Total CCSD\+T\(CCSD\) energy: *(?P<energy>-?\d+\.\d+)")

        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group("energy"))
                break
        return energy

    def get_nuclear_repulsion_energy(self):
        """
        Return the nuclear repulsion energy

        @returns: the nuclear repulsion energy in atomic units
        @rtype: float
        """

        energy = 0.0

        output = self.get_output()
        en_re = re.compile(r"^ +Nuclear repulsion energy = *(?P<energy>-?\d+\.\d+)")

        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group("energy"))
                break
        return energy

    def get_total_energy(self):
        return self.get_energy()

    @property
    def fdein(self):
        """
        The results of the ADF FDE calculation from that the embedding potential was imported.

        @type: L{adffragmentsresults}
        """
        return self.job.fdein


class nwchemjob(job):
    """
    An abstract base class for NWChem jobs.

    Corresponding results class: L{nwchemresults}

    @group Initialization:
        __init__
    @group Running Internals:
        get_nwchemfile
    """

    def __init__(self):
        """
        Constructor for NWChem jobs.
        """
        super().__init__()

    def create_results_instance(self):
        return nwchemresults(self)

    def print_jobtype(self):
        pass

    def get_nwchemfile(self):
        """
        Abstract method. Should be overwritten to return the NWChem input file.
        """
        return ""

    @property
    def checksum(self):
        import hashlib
        m = hashlib.md5()

        m.update(self.get_nwchemfile().encode('utf-8'))
        return m.hexdigest()

    def get_runscript(self, nproc=1):
        runscript = ""

        runscript += "cat <<eor >NWCHEM.INP\n"
        runscript += self.get_nwchemfile()
        runscript += "eor\n"
        runscript += "cat NWCHEM.INP \n"

        runscript += f"mpirun -np {nproc:d} $NWCHEMBIN/nwchem NWCHEM.INP >NWCHEM.OUT \n"
        runscript += "retcode=$?\n"

        runscript += "if [[ -f NWCHEM.OUT ]]; then \n"
        runscript += "  cat NWCHEM.OUT \n"
        runscript += "else \n"
        runscript += "  cat NWCHEM.out \n"
        runscript += "fi \n"

        runscript += "rm NWCHEM.INP \n"
        runscript += "exit $retcode \n"

        return runscript

    def result_filenames(self):
        return ['NWCHEM.db', 'NWCHEM.gridpts.0', 'NWCHEM.molden']

    def check_success(self, outfile, errfile):
        # check that NWChem terminated normally
        if not (os.path.exists('NWCHEM.OUT') or os.path.exists('NWCHEM.out')):
            raise PyAdfError('NWChem output file does not exist')
        return True


class nwchemsettings:
    """
    Settings for a NWChem calculation.

    @Note:
        Currently, this class is just a placeholder. It should be
        extended once more NWChem functionality is added.

    @group Initialization:
        __init__
    @group Other Internals:
        __str__
    """

    def __init__(self, method='DFT', functional='LDA', dftgrid=None, properties=None, memory=None):
        """
        Constructor for nwchemsettings.

        All arguments are optional, leaving out an argument will choose default settings.

        @param method: the computational method, see L{set_method}
        @type method: str
        @param functional:
            exchange-correlation functional for DFT calculations, see L{set_functional}
        @type  functional: str
        @param dftgrid: the numerical integration grid for the xc part in DFT, see L{set_dftgrid}
        @type  dftgrid: None or str
        @param properties: a list of properties to calculate (e.g. 'dipole')
        @type  properties: list of str
        @param memory: the maximum total memory to use (in MB)
        @type  memory: integer
        """
        self.method = None
        self._functional = None
        self.dftgrid = None
        self.properties = None
        self.memory = None

        self.set_method(method)
        if self.method == 'DFT':
            self.set_functional(functional)
        self.set_dftgrid(dftgrid)
        self.set_properties(properties)
        self.set_memory(memory)

    def set_method(self, method):
        """
        Select the computational method.

        Available options are: C{'HF'}, C{'DFT'}

        @param method: string identifying the selected method
        @type  method: str
        """
        self.method = method

    @property
    def functional(self):
        if self.method == 'DFT':
            return self._functional
        else:
            return None

    def set_functional(self, functional):
        """
        Select the exchange-correlation functional for DFT.

        @param functional:
            A string identifying the functional.
            See Dalton manual for available options.
        @type functional: str
        """
        if self.method == 'DFT':
            self._functional = functional
        else:
            raise PyAdfError('Functional can only be set for DFT calculations')

        if self._functional.upper() == 'LDA':
            self._functional = 'slater vwn_5'

    def set_dftgrid(self, dftgrid):
        """
        Select the numerical integration grid.
        """
        self.dftgrid = dftgrid

    def set_properties(self, properties):
        """
        Select which properties to calculate.

        @param properties: A list of properties to calculate
        @type  properties: list of str
        """
        self.properties = properties

    def set_memory(self, memory):
        """
        Set total memory to use.

        @param memory: the maximum total memory to use (in MB)
        @type  memory: integer
        """
        self.memory = memory

    def get_scftask_block(self):
        if self.method.upper() == 'HF':
            block = "task scf"
        elif self.method.upper() == 'DFT':
            block = "task dft"
        elif self.method.upper() == 'CCSD':
            block = "task ccsd"
        elif self.method.upper() == 'CCSD(T)':
            block = "task ccsd(t)"
        elif self.method.upper() == 'CCSD+T(CCSD)':
            block = "task ccsd+t(ccsd)"
        else:
            raise PyAdfError('Unknown method in NWChem job')
        block += ' energy'
        block += ' property'
        block += '\n'
        return block

    def get_properties_block(self):
        block = 'property\n'
        block += '   MOLDENFILE \n'
        block += '   molden_norm janpa\n'
        if self.properties is not None:
            for p in self.properties:
                block += '   ' + p + '\n'
        block += 'end\n'
        return block

    def get_memory_block(self):
        if self.memory is not None:
            block = f'MEMORY total {self.memory:d} mb \n'
        else:
            block = ''
        return block

    def get_dft_block(self):
        block = ""
        block += "   xc " + self.functional + "\n"
        if self.dftgrid is not None:
            block += "   grid " + self.dftgrid + "\n"
        return block

    def __str__(self):
        """
        Returns a human-readable description of the settings.
        """
        return "Default NWChem settings"


class nwchemsinglepointjob(nwchemjob):
    """
    A class for NWChem single point runs.

    See the documentation of L{__init__} and L{nwchemsettings} for details
    on the available options.

    Corresponding results class: L{nwchemsinglepointresults}

    @Note: Right now, HF, DFT, and CC jobs are supported.

    @Note: Importing of embedding potential requires a modified NWChem version.

    @group Initialization:
        set_restart
    @group Input Generation:
        get_molecule, get_nwchem_block, get_molecule_block,
        get_basis_block, get_scftask_block, get_dft_block,
        get_options_block, get_other_blocks
    @group Other Internals:
        print_extras, print_molecule, print_settings

    """

    def __init__(self, mol, basis, settings=None, fdein=None, options=None):
        """
        Constructor for NWChem single point jobs.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}

        @param basis:
            A string specifying the basis set to use (e.g. C{basis='cc-pVDZ'}).
        @type basis: str

        @param settings: The settings for the Dalton job. Currently not used.
        @type  settings: L{nwchemsettings}

        @param fdein:
            Results of an ADF FDE calculation. The embedding potential from this
            calculation will be imported into NWChem (requires modified NWChem version).
        @type  fdein: L{adffragmentsresults}

        @param options:
            Additional options.
            These will each be included directly in the NWChem input file.
        @type options: list of str
        """
        super().__init__()

        self.mol = mol
        self.basis = basis

        if settings is None:
            self.settings = nwchemsettings()
        else:
            self.settings = settings

        self.restart = None
        self.set_restart(None)

        self.fdein = fdein

        # FIXME: Symmetry hardcoded
        if self.mol:
            self.mol.set_symmetry('NOSYM')

        if options is None:
            self._options = []
        else:
            self._options = options

    def create_results_instance(self):
        return nwchemsinglepointresults(self)

    @property
    def checksum(self):
        import hashlib
        m = hashlib.md5()

        m.update(self.get_nwchemfile().encode('utf-8'))

        if self.restart is not None:
            m.update(b'Restarted from NWChem job \n')
            m.update(self.restart.checksum.encode('utf-8'))

        if self.fdein is not None:
            m.update(b'Embedding potential imported from ADF job \n')
            m.update(self.fdein.checksum.encode('utf-8'))

        return m.hexdigest()

    # FIXME: restart with NWChem not implemented
    def set_restart(self, restart):
        """
        Set restart file. (NOT IMPLEMENTED)

        @param restart: results object of previous Dalton calculation
        @type  restart: L{nwchemsinglepointresults} or None

        @Note: restarts with NWChem are not implemented!
        """
        self.restart = restart

    def get_molecule(self):
        return self.mol

    def get_nwchem_block(self):
        block = ""
        block += self.settings.get_memory_block()
        block += 'title "Title Input generated by PyADF" \n'
        block += "start NWCHEM\n"
        return block

    def get_molecule_block(self):
        block = "geometry units angstrom noautoz nocenter noautosym\n"
        block += self.get_molecule().print_coordinates(index=False)
        block += 'end\n'
        block += 'charge ' + str(self.mol.get_charge()) + '\n'
        return block

    def get_basis_block(self):
        block = "basis spherical\n"
        block += "   * library " + self.basis + "\n"
        block += 'end\n'
        return block

    def get_dft_block(self):
        block = "dft\n"
        block += "   direct\n"
        block += self.settings.get_dft_block()
        if self.fdein is not None:
            block += '   frozemb\n'
        block += 'end\n'
        return block

    def get_scf_block(self):
        block = "scf\n"
        block += "   direct\n"
        if self.fdein is not None:
            block += '   frozemb\n'
        block += 'end\n'
        return block

    @staticmethod
    def get_cc_block():
        block = "ccsd\n"
        block += "   freeze atomic\n"
        block += '   maxiter 100\n'
        block += 'end\n'
        return block

    def get_options_block(self):
        block = ""
        for opt in self._options:
            block += opt + "\n"
        return block

    def get_scftask_block(self):
        return self.settings.get_scftask_block()

    def get_other_blocks(self):
        return ""

    def get_nwchemfile(self):
        # The following blocks should always be present: define the geometry and basis
        nwchemfile = "echo\n"
        nwchemfile += self.get_nwchem_block()
        nwchemfile += self.get_molecule_block()
        nwchemfile += self.get_basis_block()
        if self.settings.method.upper() == 'DFT':
            nwchemfile += self.get_dft_block()
        elif self.settings.method.upper().startswith('CCSD'):
            nwchemfile += self.get_cc_block()
        else:
            nwchemfile += self.get_scf_block()

        nwchemfile += self.settings.get_properties_block()

        nwchemfile += self.get_options_block()

        nwchemfile += self.get_scftask_block()

        # Here we make room for optional blocks
        nwchemfile += self.get_other_blocks()

        return nwchemfile

    def print_jobtype(self):
        return "NWChem single point job"

    def before_run(self):
        super().before_run()
        if self.fdein is not None:
            self.fdein.export_embedding_data('EMBPOT', 'FRZDNS')

    def after_run(self):
        super().after_run()
        if self.fdein is not None:
            os.remove('EMBPOT')
            os.remove('FRZDNS')

    def print_molecule(self):

        print("   Molecule")
        print("   ========")
        print()
        print(self.get_molecule())
        print()

    def print_settings(self):

        print("   Settings")
        print("   ========")
        print()
        print(self.settings)
        print()

    def print_extras(self):
        pass

    def print_jobinfo(self):
        print(" " + 50 * "-")
        print(" Running " + self.print_jobtype())
        print()

        self.print_molecule()

        self.print_settings()

        self.print_extras()
