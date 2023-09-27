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
 The basics needed for Dalton calculations: single point jobs

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    daltonjob, daltonsinglepointjob
 @group Settings:
    daltonsettings
 @group Results:
    daltonresults, daltonsinglepointresults
"""

from .Errors import PyAdfError
from .BaseJob import results, job
from .DensityEvaluator import GTODensityEvaluatorMixin
from .Utils import newjobmarker

import os
import re


class daltonresults(results):
    """
    Class for results of a Dalton calculation.
    """

    def __init__(self, j=None):
        """
        Constructor for daltonresults.
        """
        super().__init__(j=j)
        self.resultstype = 'Dalton results'
        self.compression = 'gz'


class daltonsinglepointresults(daltonresults, GTODensityEvaluatorMixin):
    """
    Class for results of a Dalton single point calculation.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_molecule, get_energy
    """

    def __init__(self, j=None):
        """
        Constructor for daltonsinglepointresults.
        """
        super().__init__(j=j)

    def get_molecule(self):
        """
        Return the molecular geometry after the Dalton job.

        @returns: The molecular geometry.
        @rtype:   L{molecule}

        @note: implemented even though the coordinates should not change
        """

        return self.job.mol

    def read_molden_file(self):
        """
        Returns Molden results file as a string.
        """
        moldenfile = self.files.read_file_from_archive(self.fileid, 'molden.inp')
        if moldenfile is None:
            raise PyAdfError('Dalton Molden file not found.')
        return moldenfile

    def get_dipole_vector(self):
        """
        Return the dipole moment vector.

        @returns: the dipole moment vector, in atomic units
        @rtype: float[3]
        """

        dipole = [0.0, 0.0, 0.0]

        output = self.get_output()

        startline = None
        start = re.compile(r"\s*Dipole moment components")
        for i, l in enumerate(output):
            m = start.match(l)
            if m:
                startline = i

        if startline is None:
            raise PyAdfError('Dalton dipole moment not found in output')

        for i, c in enumerate(['x', 'y', 'z']):
            dip = re.compile(r"\s*" + c + r"\s*(?P<dip>[-+]?(\d+(\.\d*)?|\d*\.\d+))")
            m = dip.match(output[startline + 5 + i])
            dipole[i] = float(m.group('dip'))

        return dipole

    def get_energy(self, level='SCF'):
        """
        Return the total energy.

        @returns: the total energy in atomic units
        @rtype: float
        """

        if self.job.settings.method in ['HF', 'DFT']:
            regexp = r"^ {5}Total energy *(?P<energy>-?\d+\.\d+)"
        elif level == 'CCSD(T)':
            regexp = r"^ {12}Total energy CCSD\(T\): *(?P<energy>-?\d+\.\d+)"
        else:
            regexp = r"^ {12}Total " + level + r" *energy: *(?P<energy>-?\d+\.\d+)"

        energy = float(0)
        output = self.get_output()
        en_re = re.compile(regexp)
        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group("energy"))
                break
        return energy

    def get_scf_energy(self):
        return self.get_energy(level='SCF')

    def get_total_energy(self):
        return self.get_energy(level=self.job.settings.method)


class daltonjob(job):
    """
    An abstract base class for Dalton jobs.

    Corresponding results class: L{daltonresults}

    @group Initialization:
        __init__
    @group Running Internals:
        get_daltonfile, get_moleculefile
    """

    def __init__(self):
        """
        Constructor for Dalton jobs.
        """
        super().__init__()

    def create_results_instance(self):
        return daltonresults(self)

    def print_jobtype(self):
        pass

    def get_daltonfile(self):
        """
        Abstract method. Should be overwritten to return the Dalton input file.
        """
        return ""

    def get_moleculefile(self):
        """
        Abstract method. Should be overwritten to return the Dalton molecule file.
        """
        return ""

    @property
    def checksum(self):
        import hashlib
        m = hashlib.md5()

        m.update(self.get_daltonfile().encode('utf-8'))
        m.update(self.get_moleculefile().encode('utf-8'))

        return m.hexdigest()

    def get_runscript(self, nproc=1, memory=None):
        put_files = [f for f in ['EMBPOT', 'FRZDNS'] if os.path.exists(f)]

        runscript = ""

        runscript += "cat <<eor >DALTON.dal\n"
        runscript += self.get_daltonfile()
        runscript += "eor\n"
        runscript += "cat DALTON.dal \n"

        runscript += "cat <<eor >MOLECULE.mol\n"
        runscript += self.get_moleculefile()
        runscript += "eor\n"
        runscript += "cat MOLECULE.mol \n"

        if len(put_files) > 0:
            runscript += 'tar cf dalfiles.tar ' + ' '.join(put_files) + '\n'
            runscript += 'gzip dalfiles.tar\n'

        runscript += "$DALTONBIN/dalton "
        if nproc > 1:
            runscript += f'-N {nproc:d} '
        if memory is not None:
            runscript += f'-M {memory:d} '
        if len(put_files) > 0:
            runscript += '-f dalfiles'
        runscript += " DALTON MOLECULE\n"
        runscript += "retcode=$? \n"

        runscript += "if [[ -f DALTON_MOLECULE.OUT ]]; then \n"
        runscript += "  cat DALTON_MOLECULE.OUT \n"
        runscript += "else \n"
        runscript += "  cat DALTON_MOLECULE.out \n"
        runscript += "fi \n"

        runscript += "rm DALTON.dal \n"
        runscript += "rm MOLECULE.mol \n"
        runscript += "exit $retcode \n"

        return runscript

    def result_filenames(self):
        return ['DALTON_MOLECULE.tar.gz']

    def check_success(self, outfile, errfile):
        # check that Dalton terminated normally
        if not (os.path.exists('DALTON_MOLECULE.OUT') or os.path.exists('DALTON_MOLECULE.out')):
            raise PyAdfError('Dalton output file does not exist')

        f = open(errfile, encoding='utf-8')
        err = f.readlines()
        for line in reversed(err):
            if "SEVERE ERROR" in line:
                raise PyAdfError("Error running Dalton job")
            if line == newjobmarker:
                break
        f.close()
        return True


class daltonsettings:
    """
    Class that holds the settings for a Dalton calculation..

    @group Initialization:
        __init__,
        set_method, set_functional
    @group Input Generation:
        get_wavefunction_block
    @group Other Internals:
        __str__
    """

    def __init__(self, method='DFT', functional='LDA', dftgrid=None, freeze_occ=0, freeze_virt=0, memory=None):
        """
        Constructor for daltonsettings.

        All arguments are optional, leaving out an argument will choose default settings.

        @param method: the computational method, see L{set_method}
        @type method: str

        @param functional:
            exchange-correlation functional for DFT calculations, see L{set_functional}
        @type  functional: str

        @param dftgrid: the numerical integration grid for the xc part in DFT, see L{set_dftgrid}
        @type  dftgrid: None or str

        @param freeze_occ: number of occupied orbitals to freeze, see L{set_freeze}.
        @type  freeze_occ: int

        @param freeze_virt: number of virtual orbitals to freeze, see L{set_freeze}.
        @type  freeze_virt: int

        @param memory: the maximum total memory to use (in MB)
        @type  memory: integer
        """
        self.method = None
        self._functional = None
        self.dftgrid = None
        self.freeze_occ = None
        self.freeze_virt = None
        self.memory = None

        self.set_method(method)
        if self.method == 'DFT':
            self.set_functional(functional)
        self.set_dftgrid(dftgrid)
        self.set_freeze(freeze_occ, freeze_virt)
        self.set_memory(memory)

    def set_method(self, method):
        """
        Select the computational method.

        Available options are: C{'HF'}, C{'DFT'}, C{'CC'}, C{'CCSD'}, C{'CCSD(T)'}

        @param method: string identifying the selected method
        @type  method: str
        """
        self.method = method.upper()

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

    def set_dftgrid(self, dftgrid):
        """
        Select the numerical integration grid.
        """
        self.dftgrid = dftgrid

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

    def set_memory(self, memory):
        """
        Set total memory to use.

        @param memory: the maximum total memory to use (in MB)
        @type  memory: integer
        """
        self.memory = memory

    def get_wavefunction_block(self):
        block = "**WAVE FUNCTIONS\n"
        if self.method == 'HF':
            block += ".HF\n"
        elif self.method == 'DFT':
            block += ".DFT\n"
            block += " " + self.functional + "\n"
            if self.dftgrid is not None:
                block += "*DFT INPUT\n"
                block += "." + self.dftgrid.upper() + "\n"
        elif self.method in ['CC', 'CCSD', 'CCSD(T)']:
            block += '.CC\n'
        else:
            raise PyAdfError('Unknown method in Dalton job')
        return block

    def __str__(self):
        """
        Returns a human-readable description of the settings.
        """
        s = f'  Method: {self.method} \n'
        if self.method == 'DFT':
            s += f'  Exchange-correlation functional: {self.functional} \n'
        elif self.method.startswith('CC'):
            s += f"  Number of frozen occupied orbitals: {self.freeze_occ:d} \n"
            s += f"  Number of frozen virtual orbitals:  {self.freeze_virt:d} \n"
        return s


class daltonsinglepointjob(daltonjob):
    """
    A class for Dalton single point runs.

    See the documentation of L{__init__} and L{daltonsettings} for details
    on the available options.

    Corresponding results class: L{daltonsinglepointresults}

    @Note: Right now, HF, DFT, and CC jobs are supported.

    @Note: Importing of embedding potential requires a modified Dalton version.

    @group Initialization:
        set_restart
    @group Input Generation:
        get_dalton_block, get_integral_block, get_molecule, get_options_block,
        get_other_blocks, get_properties_block
    @group Other Internals:
        print_extras, print_molecule, print_settings

    """

    def __init__(self, mol, basis, settings=None, fdein=None, options=None, response=None):
        """
        Constructor for Dalton single point jobs.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}

        @param basis:
            A string specifying the basis set to use (e.g. C{basis='cc-pVDZ'}).
        @type basis: str

        @param settings: The settings for the Dalton job. Currently not used.
        @type  settings: L{daltonsettings}

        @param fdein:
            Results of an ADF FDE calculation. The embedding potential from this
            calculation will be imported into Dalton (requires modified Dalton version).
        @type  fdein: L{adffragmentsresults}

#        @param response: The response settings for a Dalton job.
#        @type response: list of str

        @param options:
            Additional options.
            These will each be included directly in the Dalton input file.
        @type options: list of str
        """
        super().__init__()

        self.mol = mol
        self.basis = basis
        if self.mol and (self.basis is None):
            raise PyAdfError("Missing basis set in Dalton single point job")

        if settings is None:
            self.settings = daltonsettings()
        else:
            self.settings = settings

        self.restart = None
        self.set_restart(None)

        self.fdein = fdein

        # FIXME: Symmetry in Dalton hardcoded
        if self.mol:
            self.mol.set_symmetry('NOSYM')

        if options is None:
            self._options = []
        else:
            self._options = options

        if response is None:
            self._response = None
        else:
            self._response = response

    # CC is not available in MPI parallel runs
    @property
    def only_serial(self):
        return self.settings.method.startswith('CC')

    def create_results_instance(self):
        return daltonsinglepointresults(self)

    @property
    def checksum(self):
        import hashlib
        m = hashlib.md5()

        m.update(self.get_daltonfile().encode('utf-8'))
        m.update(self.get_moleculefile().encode('utf-8'))

        if self.restart is not None:
            m.update(b'Restarted from Dalton job \n')
            m.update(self.restart.checksum.encode('utf-8'))

        if self.fdein is not None:
            m.update(b'Embedding potential imported from ADF job \n')
            m.update(self.fdein.checksum.encode('utf-8'))

        return m.hexdigest()

    # noinspection PyMethodOverriding
    def get_runscript(self, nproc=1):
        return super().get_runscript(nproc=nproc, memory=self.settings.memory)

    # FIXME: restart with Dalton not implemented
    def set_restart(self, restart):
        """
        Set restart file. (NOT IMPLEMENTED)

        @param restart: results object of previous Dalton calculation
        @type  restart: L{daltonsinglepointresults} or None

        @Note: restarts with Dalton are not implemented!
        """
        self.restart = restart

    def get_molecule(self):
        return self.mol

    def get_dalton_block(self):
        block = ".RUN WAVE FUNCTIONS\n"
        block += ".DIRECT\n"
        if self.settings.method in ('HF', 'DFT'):
            block += ".RUN PROPERTIES\n"
        if self._response is not None:
            block += '.RUN RESPONSE\n'
        if self.fdein is not None:
            block += '.FDE\n'
            block += '*FDE\n'
            block += '.PRINT\n 1\n'
            block += '.EMBPOT\nEMBPOT\n'
        return block

    def get_integral_block(self):
        return ""

    # noinspection PyMethodMayBeStatic
    def get_properties_block(self):
        return ""

    def get_response_block(self):
        block = ""
        if self._response:
            block += "**RESPONSE\n"
            for rsp in self._response:
                block += rsp + "\n"
        return block

    def get_options_block(self):
        block = ""
        for opt in self._options:
            block += opt + "\n"
        return block

    def get_cc_block(self):
        block = "*CC INPUT \n"
        if self.settings.method == 'CCSD':
            block += ".CCSD\n"
        elif self.settings.method == 'CCSD(T)':
            block += ".CC(T)\n"
        else:
            block += ".CC2\n"
        block += ".PRINT \n"
        block += "2 \n"
        block += ".NSYM \n"
        block += "1 \n"
        # freeze orbitals
        block += ".FREEZE\n"
        block += f"{self.settings.freeze_occ:d} {self.settings.freeze_virt:d}\n"
        return block

    def get_other_blocks(self):
        return ""

    def get_daltonfile(self):
        daltonfile = "**DALTON INPUT\n"
        daltonfile += self.get_dalton_block()
        daltonfile += self.get_integral_block()

        daltonfile += self.settings.get_wavefunction_block()
        daltonfile += self.get_properties_block()
        daltonfile += self.get_response_block()
        daltonfile += self.get_options_block()

        if self.settings.method.startswith('CC'):
            daltonfile += self.get_cc_block()

        daltonfile += self.get_other_blocks()
        daltonfile += "**END OF DALTON INPUT\n"

        return daltonfile

    def get_moleculefile(self):
        return self.mol.get_dalton_molfile(self.basis)

    def print_jobtype(self):
        return "Dalton single point job"

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
