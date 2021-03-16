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


from Errors import PyAdfError
from BaseJob import results, job


class OrcaResults(results):
    """

    Class for results of a Orca calculation.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_singlepointenergy, get_optimized_molecule,
        get_dipole_vector, get_dipole_magnitude

    """

    def __init__(self, j=None):
        """
        Constructor
        """
        self.resultstype = "Orca results"
        results.__init__(self, j)

    def get_gwb_filename(self):
        return self.files.get_results_filename(self.fileid, tape=21)

    def get_prop_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=67)
        except PyAdfError:
            raise PyAdfError("ORCA Properties file not found")
        return fn

    def get_engrad_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=68)
        except PyAdfError:
            raise PyAdfError("ORCA engrad gradients file not found")
        return fn

    def get_traj_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=13)
        except PyAdfError:
            raise PyAdfError("ORCA trajectory file not found")
        return fn

    def get_xyz_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=69)
        except PyAdfError:
            raise PyAdfError("ORCA geometry optimization xyz file not found")
        return fn

    def get_energy(self):
        import re

        output = self.get_output()
        energy = None
        en_re = re.compile(r"(FINAL SINGLE POINT ENERGY) *(?P<energy>-?\d+\.\d+)")
        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group('energy'))
        return energy

    def get_molecule(self):
        from Molecule import molecule

        try:
            xyz_fn = self.get_xyz_filename()
        except PyAdfError:
            return self.job.mol
        return molecule(xyz_fn)

    def get_dipole_vector(self):
        prop_fn = self.get_prop_filename()

        with open(prop_fn) as f:
            lines = f.readlines()
        found = None
        for i, ll in enumerate(lines):
            if 'Total Dipole moment:' in ll:
                found = i
        if found is None:
            raise PyAdfError('Dipole moment not found in ORCA properties file')
        vector = lines[found+2].strip().split()
        vector = [float(vector[ii]) for ii in range(1,4)]
        return vector


class OrcaSettings(object):
    """
    Class that holds the settings for a orca calculation
    """

    def __init__(self, method='DFT', basis='def2-SVP', functional='LDA', ri=True, maxiter=None):
        """
        Constructor for OrcaSettings.

        All arguments are optional, leaving out an argument will choose default settings.

        @param method: the computational method
        @type method: str
        @param functional:
            exchange-correlation functional for DFT calculations, see L{set_functional}
        @type  functional: str
        @param basis: the basis set for the calculation
        @type  basis: str
        @param ri:  C{True} to switch on RI approximation, C{False} to switch off.
        @type  ri:  L{bool}
        @param maxiter: the maximum number of SCF iterations
        @type  maxiter: int
        """
        # declare all
        self.method = None
        self.basis = None
        self.functional = None
        self.ri = None
        self.maxiter = None
        self.extra_keywords = None
        self.extra_blocks = None

        # initialize the setter
        self.set_method(method)
        self.set_basis(basis)
        self.set_functional(functional)
        self.set_ri(ri)
        self.set_maxiter(maxiter)

    def set_basis(self, basisset):
        """
        Select a basis set to be used for all atoms.

        @param basisset: Name of a basis set (e.g. C{def2-TZVP}).
        @type  basisset: str
        """
        self.basis = basisset

    def set_functional(self, functional):
        """
        Select the exchange-correlation functional for DFT.

        @param functional:
            A string identifying the functional.
            See Orca manual for available options.
        @type functional: str
        """
        self.functional = functional

    def set_ri(self, value):
        """
        Switch RI approximation on or off.

        @param value:  C{True} to switch on RI approximation, C{False} to switch off.
        @type  value:  L{bool}
        """
        self.ri = value

    def set_maxiter(self, maxiter):
        """
        Set the maximum number of iteration cycles to perform before giving up
        the optimization.

        @param maxiter: Max. number of cycles
        @type  maxiter: int
        """
        self.maxiter = maxiter

    def set_extra_keywords(self, keywords, append=False):
        """
        Set additional keywords for the calculation

        @param keywords: Avaible keywords. See Orca manual for available keywords.
        @type  keywords: str or list or None
        @param append: If True, append the extra keywords to those already set.
        @type  append: bool
        """
        if keywords is None:
            if not append:
                self.extra_keywords = None
        else:
            kws = keywords
            if not isinstance(keywords, list):
                kws = [keywords]
            if self.extra_keywords is None:
                self.extra_keywords = kws
            elif append:
                self.extra_keywords = self.extra_keywords + kws
            else:
                self.extra_keywords = kws

    def set_method(self, method):
        """
        Select the computational method.

        @param method: string identifying the selected method.
        @type  method: str
        """
        self.method = method.upper()

    def set_extra_block(self, block, append=False):
        """
        Set additional input blocks for the calculation

        @param block: Avaible keywords. See Orca manual for available keywords.
        @type  block: str or None
        @param append: If True, append the extra keywords to those already set.
        @type  append: bool
        """
        if block is None:
            if not append:
                self.extra_blocks = None
        else:
            if self.extra_blocks is None:
                self.extra_blocks = [block]
            elif append:
                self.extra_blocks = self.extra_blocks.append(block)
            else:
                self.extra_blocks = [block]

    def __str__(self):
        """
        Get a nicely formatted text block summarizing the settings.

        @returns: Text block
        @rtype:   L{str}
        """
        s = '   Method: %s ' % self.method
        if self.ri:
            s += '(RI: ON) \n'
        else:
            s += '(RI: OFF) \n'
        s += '   Basis Set: %s \n' % self.basis
        if self.method == 'DFT':
            s += '   Exchange-correlation functional: %s \n' % self.functional
        if self.maxiter is not None:
            s += '   Maximum number of SCF iterations %i \n' % self.maxiter
        if self.extra_keywords is not None:
            s += '   Extra keywords: ' + str(self.extra_keywords)
        if self.extra_blocks is not None:
            s += '   Extra input blocks: \n'
            s += '\n'.join(self.extra_blocks)
        return s

    def get_keywords(self):
        kws = [self.basis]
        if self.ri:
            kws.append('RI')
        else:
            kws.append('NoRI')
        if self.extra_keywords is not None:
            kws = kws + self.extra_keywords
        return kws

    def get_method_block(self):
        block = ''
        block += "%method\n"
        block += "method " + self.method + "\n"
        if self.method == 'DFT':
            block += "functional " + self.functional + "\n"
        block += "end\n"
        return block

    def get_scf_block(self):
        block = ''
        if self.maxiter is not None:
            block += "%scf\n"
            block += "MaxIter %i \n" % self.maxiter
            block += "end\n"
        return block

    def get_input_blocks(self):
        blocks = ''
        blocks += self.get_method_block()
        blocks += self.get_scf_block()

        if self.extra_blocks is not None:
            blocks += '\n'.join(self.extra_blocks) + '\n'

        return blocks


class OrcaJob(job):
    """
    A class for Orca jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None):
        """
        Constructor
        """
        self.settings = None
        self.mol = mol

        if settings is None:
            self.settings = OrcaSettings()
        else:
            self.settings = settings

        job.__init__(self)

    def print_jobtype(self):
        return "Orca job"

    def get_molecule(self):
        return self.mol

    def get_runscript(self, nproc=1):

        runscript = ""
        runscript += "cat <<eor >INPUT.inp\n"
        runscript += self.get_orcafile(nproc)
        runscript += "eor\n"
        runscript += "cat INPUT.inp\n"

        runscript += 'if [ -z "$OPAL_PREFIX" ]; then\n'
        runscript += "  $ORCA_PATH/orca INPUT.inp\n"
        runscript += "else\n"
        runscript += '  $ORCA_PATH/orca INPUT.inp "--prefix $OPAL_PREFIX"\n'
        runscript += "fi\n"
        runscript += "retcode=$?\n"

        runscript += "rm INPUT.inp\n"
        runscript += "exit $retcode\n"

        return runscript

    def result_filenames(self):
        return ['INPUT.gbw', 'INPUT_property.txt', 'INPUT.engrad', 'INPUT.xyz', 'INPUT_trj.xyz']

    def check_success(self, outfile, errfile):
        f = open(outfile)
        success = False
        for ll in f.readlines()[-10:]:
            if '**ORCA TERMINATED NORMALLY**' in ll:
                success = True
            if 'ORCA finished by error termination' in ll:
                raise PyAdfError("Error termination in ORCA")
        return success

    def get_keywords(self):
        return self.settings.get_keywords()

    def get_input_blocks(self, nproc=1):
        blocks = ''
        blocks += self.get_parallel_block(nproc)
        blocks += self.settings.get_input_blocks()
        return blocks

    @staticmethod
    def get_parallel_block(nproc):
        block = ''
        if nproc > 1:
            block += "%pal\n"
            block += "nprocs %i \n" % nproc
            block += "end\n"
        return block

    def get_orcafile(self, nproc=1):

        orcafile = "! " + ' '.join(self.get_keywords()) + "\n"
        orcafile += self.get_input_blocks(nproc) + "\n"

        orcafile += "*xyz %i %i \n" % (self.mol.get_charge(), self.mol.get_spin()+1)
        xyz_file = self.mol.get_xyz_file()
        orcafile += ''.join(xyz_file.splitlines(True)[2:])
        orcafile += "*\n"

        return orcafile

    def create_results_instance(self):
        return OrcaResults(self)

    def get_checksum(self):
        import hashlib

        m = hashlib.md5()
        m.update(self.get_orcafile())
        return m.digest()

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


class OrcaSinglePointJob(OrcaJob):
    """
    A class for Orca single point jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None):
        OrcaJob.__init__(self, mol, settings)

    def print_jobtype(self):
        return "Orca single point job"

    def get_keywords(self):
        return ['SP'] + self.settings.get_keywords()


class OrcaGeometryOptimizationJob(OrcaJob):
    """
    A class for Orca geometry optimization jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None):
        OrcaJob.__init__(self, mol, settings)

    def print_jobtype(self):
        return "Orca geometry optimization job"

    def get_keywords(self):
        return ['Opt'] + self.settings.get_keywords()
