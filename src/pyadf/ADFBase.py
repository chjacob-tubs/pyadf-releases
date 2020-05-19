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
 The basics needed for ADF calculations: simple jobs and results.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    adfjob
 @group Results:
    adfresults
"""

import kf
import os

from Errors import PyAdfError
from Molecule import molecule
from BaseJob import results, job


class amssettings(object):

    def __init__(self):
        pass

    def __str__(self):
        ss = ''
        return ss


class adfresults (results):

    """
    Class for results of an ADF calculation.

    Information stored:
        - deepcopy of the job object itself
        - output: filename and line numbers (start and end)
        - reference to filemanager
        - results id for filemanager

    Specific results are always obtained from
    the available files when they are requested

    @group Initialization:
      import_tape_files
    @group Access to result files:
      get_tape_filename, get_tapes_copy, copy_tape, link_tape, pack_tape
    @group Retrieval of specific results:
      get_result_from_tape
    @group Access to internal properties:
      get_checksum
    """

    def __init__(self, j=None):
        """
        Constructor for adfresults.
        """
        results.__init__(self, j)

    def import_tape_files(self, fn_list, tape_list):
        """
        Initialize adfresults by importing tape files.

        @param fn_list: a list of the files to import
        @type  fn_list: list of str
        @param tape_list: the tape numbers (i.e., 21 for TAPE21 etc.) corresponding to these files
        @type  tape_list: list of int
        """

        from Files import adf_filemanager
        self.files = adf_filemanager()

        name = ''
        for fn, tape in zip(fn_list, tape_list):
            os.symlink(fn, 'TAPE' + str(tape))
            name += fn + 'TAPE' + str(tape) + '\n'

        # use the name of the imported file as checksum
        self._checksum = os.path.abspath(name)

        self.files.add_results(self)

    def get_tape_filename(self, tape=21):
        """
        Return the file name of a TAPE file belonging to the results.

        @param tape: The tape number (i.e., 21 for TAPE21)
        @type  tape: int
        """
        return self.files.get_results_filename(self.fileid, tape)

    def get_tapes_copy(self):
        """
        Copy all TAPE files belonging to this job to the working directory.
        """
        self.files.copy_job_result_files(self.fileid)

    def copy_tape(self, tape=21, name="TAPE21"):
        """
        Copy result TAPE file to the working directory.

        @param tape: The tape number (i.e., 21 for TAPE21)
        @type  tape: int
        @param name: The name of the copied file
        @type  name: str
        """
        self.files.copy_result_file(self.fileid, tape, name)

    def link_tape(self, tape=21, name="TAPE21"):
        """
        Make a symbolic link to a results TAPE file in the working directory.
        """
        self.files.link_result_file(self.fileid, tape, name)

    def pack_tape(self):
        """
        Pack the result tape files belonging to this job.

        For details, see L{adf_filemanager.pack_results}.
        """
        self.files.pack_results(self.fileid)

    def get_result_from_tape(self, section, variable, tape=21, always_array=False):
        """
        Get a specific variable from a tape.

        @param section: the section on tape to be read
        @type  section: str

        @param variable: the variable on tape to be read
        @type  variable: str

        @param tape: the number of the tape to use, default is 21
        @type  tape: int

        @param always_array: always return a numpy array, even if these have only one element
        @type  always_array: bool

        @returns: the contents of the variable as read.
        @rtype:   depends on the variable to be read
        """

        f = kf.kffile(self.get_tape_filename(tape))
        result = f.read(section, variable)
        f.close()

        if result is None:
            raise PyAdfError("Variable " + section + "%" + variable + " not found in tape file")

        if (not always_array) and (len(result) == 1):
            result = result[0]

        return result


class amsresults(adfresults):

    def __init__(self, j=None):
        """
        Constructor for amsresults.
        """
        adfresults.__init__(self, j)

    def get_molecule(self):
        """
        Return the molecular geometry after the ADF job.

        This can be changes with respect to the input geometry
        because it was optimized in the calculation.

        @returns: The molecular geometry.
        @rtype:   L{molecule}
        """
        nnuc = self.get_result_from_tape('Molecule', 'nAtoms')

        atnums = self.get_result_from_tape('Molecule', 'AtomicNumbers')
        xyz = self.get_result_from_tape('Molecule', 'Coords')
        xyznuc = xyz.reshape(nnuc, 3)

        m = molecule()
        m.add_atoms(atnums, xyznuc, atomicunits=True)

        return m

    def get_dipole_vector(self):
        """
        Return the dipole moment vector.

        @returns: the dipole moment vector, in atomic units
        @rtype: float[3]
        """
        return self.get_result_from_tape('AMSResults', 'DipoleMoment')

    def get_energy(self):
        """
        Return the bond energy.

        @returns: the bond energy in atomic units
        @rtype: float
        """
        return self.get_result_from_tape('AMSResults', 'Energy')

    def get_frequencies(self):
        """
        Return the vibrational frequencies
        """
        return self.get_result_from_tape('Vibrations', 'Frequencies[cm-1]')

    def get_normalmodes_c(self):
        """
        Return the normal modes (not normalized, not mass-weighted)
        """
        import numpy

        natoms = self.get_result_from_tape('Molecule', 'nAtoms')
        nmodes = self.get_result_from_tape('Vibrations', 'nNormalModes')

        modes_c = numpy.zeros((nmodes, 3*natoms))
        for i in range(nmodes):
            modes_c[i, :] = self.get_result_from_tape('Vibrations',
                                                      'NoWeightNormalMode(%i)' % (i+1))

        return modes_c


class adfjob (job):

    """
    An abstract base class for ADF jobs (ADF and related programs).

    Corresponding results class: L{adfresults}

    @group Initialization:
        __init__
    @group Input Generation:
        get_input
    @group Running:
        run
    """

    def __init__(self):
        """
        Constructor for adfjob.
        """
        job.__init__(self)
        self._checksum_only = False

    def create_results_instance(self):
        """
        Create an instance of the matching results object for this job.
        """
        return adfresults(self)

    def get_input(self):
        """
        Abstract method to obtain an input file for ADF or a related program.
        """
        return None

    def get_checksum(self):
        """
        Obtain a checksum for the job.

        This uses a MD5 checksum of the input file.
        """

        self._checksum_only = True
        inp = self.get_input()
        self._checksum_only = False

        if inp is not None:
            import hashlib
            m = hashlib.md5()
            m.update(inp)
            return m.digest()
        else:
            return None

    def get_runscript(self, program="adf", serial=False, inputfile=None):
        """
        Return a runscript for ADF or a related program.

        The input for this program is either taken from the
        given C{inputfile} or generated by L{get_input}.

        @param program: The program to run (by default: adf).
        @type  program: str
        @param serial: Whether to run the program in serial (default: parallel).
        @type  serial: bool
        @param inputfile: The input file to use. If None (default), L{get_input} is called.
        @type  inputfile: str or C{None}
        """
        if inputfile is None:
            inp = "<<eor"
        else:
            inp = "<" + inputfile
        runscript = "#!/bin/bash \n\n"
        runscript += 'if [ -e ../FOCKMATRIX ]; then \n'
        runscript += '  cp ../FOCKMATRIX . \n'
        runscript += 'fi \n'
        if serial:
            runscript += "$ADFBIN/" + program + " -n1 " + inp + " || exit $? \n"
        else:
            runscript += "$ADFBIN/" + program + " " + inp + " || exit $? \n"
        if inputfile is None:
            runscript += self.get_input()
            runscript += "eor\n"
        runscript += "\n"
        return runscript

    def check_success(self, outfile, errfile, logfile=None):
        if logfile is None:
            logfilename = 'logfile'
        else:
            logfilename = logfile

        # check if the ADF run was successful
        f = open(logfilename, 'r')
        lastline = ''.join(f.readlines()[-3:])
        f.close()
        if lastline.find('ERROR') >= 0:
            raise PyAdfError('ERROR DETECTED in ADF run')
        elif lastline.find('NORMAL TERMINATION') == -1:
            raise PyAdfError('Unknown Error in ADF run')

        # check for warnings in PyAdf
        f = open(logfilename, 'r')
        for line in f.readlines():
            if line.find('WARNING') >= 0:
                print " Found WARNING in ADF logfile:"
                warning = line.split()
                print ' '.join(warning[3:])
        f.close()
        print

        os.remove(logfilename)

        return True

    def result_filenames(self):
        return ['TAPE21', 'TAPE10', 'TAPE41', 'FOCKMATRIX']


class amsjob(adfjob):
    """
    Base class for ADF jobs using the AMS engine.
    """
    def __init__(self, mol, task='SinglePoint', settings=None):
        adfjob.__init__(self)
        self.mol = mol
        self.task = task

        if settings is None:
            self.settings = amssettings()
        else:
            self.settings = settings

    def create_results_instance(self):
        return amsresults(self)

    def get_input(self):
        amsinput = ""
        amsinput += self.get_task_block()
        amsinput += self.get_properties_block()
        amsinput += self.get_system_block()
        amsinput += self.get_engine_block()
        return amsinput

    def get_task_block(self):
        block = " TASK %s \n\n" % self.task
        return block

    def get_system_block(self):
        block = " SYSTEM\n"
        block += "  ATOMS [Angstrom]\n"
        block += self.mol.print_coordinates(index=False)
        block += "  END\n"
        block += " END\n\n"
        return block

    def get_properties_block(self):
        return ""

    def get_engine_block(self):
        pass

    def get_runscript(self):
        """
        Return a runscript for AMS.
        """
        return adfjob.get_runscript(self, program='ams', serial=False)

    def check_success(self, outfile, errfile, logfile=None):
        if logfile is None:
            logfilename = os.path.join('ams.results', 'ams.log')
        else:
            logfilename = logfile
        return adfjob.check_success(self, outfile, errfile, logfilename)

    def result_filenames(self):
        return [os.path.join('ams.results', f) for f in ['ams.rkf', 'dftb.rkf']]
