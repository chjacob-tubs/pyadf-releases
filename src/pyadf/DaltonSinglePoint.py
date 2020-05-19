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

from Errors import PyAdfError
from BaseJob import results, job
from Utils import newjobmarker
import os
import re


class daltonresults (results):

    """
    Class for results of a Dalton calculation.
    """

    def __init__(self, j=None):
        """
        Constructor for daltonresults.
        """
        results.__init__(self, j)


class daltonsinglepointresults (daltonresults):

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
        daltonresults.__init__(self, j)

    def get_molecule(self):
        """
        Return the molecular geometry after the Dalton job.

        @returns: The molecular geometry.
        @rtype:   L{molecule}

        @note: currently not implemented
        """
        pass

    def get_dipole_vector(self):
        """
        Return the dipole moment vector.

        @returns: the dipole moment vector, in atomic units
        @rtype: float[3]
        """

        dipole = [0.0, 0.0, 0.0]

        output = self.get_output()

        start = re.compile("\s*Dipole moment components")
        for i, l in enumerate(output):
            m = start.match(l)
            if m:
                startline = i

        for i, c in enumerate(['x', 'y', 'z']):
            dip = re.compile("\s*" + c + "\s*(?P<dip>[-+]?(\d+(\.\d*)?|\d*\.\d+))")
            m = dip.match(output[startline + 5 + i])
            dipole[i] = float(m.group('dip'))

        return dipole

    def get_energy(self):
        """
        Return the total energy.

        @returns: the total energy in atomic units
        @rtype: float
        """

        energy = float(0)
        output = self.get_output()
        en_re = re.compile("^ {5}Total energy *(?P<energy>-?\d+\.\d+)")
        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group("energy"))
                break
        return energy


class daltonjob (job):

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
        job.__init__(self)
        self._checksum_only = False

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

    def get_checksum(self):
        import hashlib
        m = hashlib.md5()

        self._checksum_only = True
        m.update(self.get_daltonfile())
        m.update(self.get_moleculefile())
        self._checksum_only = False

        return m.digest()

    def get_runscript(self, nproc=None, memory=None):
        put_files = [f for f in ['EMBPOT', 'FRZDNS'] if os.path.exists(f)]

        runscript = "#!/bin/bash \n\n"

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
        if nproc is not None:
            runscript += '-N %i ' % nproc
        if memory is not None:
            runscript += '-M %i ' % memory
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

    def check_success(self, outfile, errfile):
        # check that Dalton terminated normally
        if not (os.path.exists('DALTON_MOLECULE.OUT') or os.path.exists('DALTON_MOLECULE.out')):
            raise PyAdfError('Dalton output file does not exist')

        f = open(errfile)
        err = f.readlines()
        for l in reversed(err):
            if "SEVERE ERROR" in l:
                raise PyAdfError("Error running Dalton job")
            if l == newjobmarker:
                break
        f.close()
        return True


class daltonsettings (object):

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

    def __init__(self, method='DFT', functional='LDA', dftgrid=None, memory=None):
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
        @param memory: the maximum total memory to use (in MB)
        @type  memory: integer
        """
        self.functional = None
        self.method = None
        self.dftgrid = None
        self.memory = None

        self.set_method(method)
        self.set_functional(functional)
        self.set_dftgrid(dftgrid)
        self.set_memory(memory)

    def set_method(self, method):
        """
        Select the computational method.

        Available options are: C{'HF'}, C{'DFT'}, C{'CC'}

        @param method: string identifying the selected method
        @type  method: str
        """
        self.method = method

    def set_functional(self, functional):
        """
        Select the exchange-correlation functional for DFT.

        @param functional:
            A string identifying the functional.
            See Dalton manual for available options.
        @type functional: str
        """
        self.functional = functional

    def set_dftgrid(self, dftgrid):
        """
        Select the numerical integration grid.
        """
        self.dftgrid = dftgrid

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
        elif self.method == 'CC':
            block += '.CC\n'
        else:
            raise PyAdfError('Unknown method in Dalton job')
        return block

    def __str__(self):
        """
        Returns a human-readable description of the settings.
        """
        s = '  Method: %s \n' % self.method
        if self.method == 'DFT':
            s += '  Exchange-correlation functional: %s \n' % self.functional
        return s


class daltonsinglepointjob (daltonjob):

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

    def __init__(self, mol, basis, settings=None, fdein=None, options=None):
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

        @param options:
            Additional options.
            These will each be included directly in the Dalton input file.
        @type options: list of str
        """
        daltonjob.__init__(self)

        self.mol = mol
        self.basis = basis
        if self.mol and (self.basis == None):
            raise PyAdfError("Missing basis set in Dalton single point job")

        if settings == None:
            self.settings = daltonsettings()
        else:
            self.settings = settings

        self.set_restart(None)

        self.fdein = fdein

        self.post2017code = True
        if 'DALTON_POST2017_VERSION' in os.environ:
            self.post2017code = False 

        # FIXME: Symmetry in Dalton hardcoded
        if self.mol:
            self.mol.set_symmetry('NOSYM')

        if options is None:
            self._options = []
        else:
            self._options = options

    def create_results_instance(self):
        return daltonsinglepointresults(self)

    def get_runscript(self):
        if 'NSCM' in os.environ :
            nproc = int(os.environ['NSCM'])
        else:
            nproc = None
        return daltonjob.get_runscript(self, nproc=nproc, memory=self.settings.memory)

    # FIXME: restart with Dalton not implemented
    def set_restart(self, restart):
        """
        Set restart file. (NOT IMPLEMENTED)

        @param restart: results object of previous Dalton calculation
        @type  restart: L{daltonsinglepointresults}

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
        if not self.fdein == None:
            block += '.FDE\n'
        # for versions prior to the 2018 dalton release,
        # the input below is not necessary to run the calculations.
        # from the 2018 release onwards, the definition of *FDE and
        # some of its keywords will be mandatory
            if not self.post2017code:
                block += '*FDE\n'
                block += '.PRINT\n 1\n'
                block += '.EMBPOT\nEMBPOT\n'
        return block

    def get_integral_block(self):
        return ""

    def get_properties_block(self):
        return ""

    def get_options_block(self):
        block = ""
        for opt in self._options:
            block += opt + "\n"
        return block

    def get_other_blocks(self):
        return ""

    def get_daltonfile(self):
        daltonfile = "**DALTON INPUT\n"
        daltonfile += self.get_dalton_block()
        daltonfile += self.get_integral_block()

        daltonfile += self.settings.get_wavefunction_block()
        daltonfile += self.get_properties_block()

        daltonfile += self.get_options_block()

        daltonfile += self.get_other_blocks()
        daltonfile += "**END OF DALTON INPUT\n"

        return daltonfile

    def get_moleculefile(self):
        return self.mol.get_dalton_molfile(self.basis)

    def print_jobtype(self):
        return "Dalton single point job"

    def before_run(self):
        daltonjob.before_run(self)
        if not self.fdein == None:
            self.fdein.export_embedding_data('EMBPOT', 'FRZDNS')

    def after_run(self):
        daltonjob.after_run(self)
        if not self.fdein == None:
            os.remove('EMBPOT')
            os.remove('FRZDNS')

    def print_molecule(self):

        print "   Molecule"
        print "   ========"
        print
        print self.get_molecule()
        print

    def print_settings(self):

        print "   Settings"
        print "   ========"
        print
        print self.settings
        print

    def print_extras(self):
        pass

    def print_jobinfo(self):
        print " " + 50 * "-"
        print " Running " + self.print_jobtype()
        print

        self.print_molecule()

        self.print_settings()

        self.print_extras()
