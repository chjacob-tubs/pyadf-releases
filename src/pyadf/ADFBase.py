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

from .Errors import PyAdfError
from .Molecule import molecule
from .BaseJob import results, job


class amssettings:

    def __init__(self):
        pass

    def __str__(self):
        ss = ''
        return ss


class scmresults(results):
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
      checksum
    """

    def __init__(self, j=None):
        """
        Constructor for amsresults.
        """
        super().__init__(j)

    def import_tape_files(self, fn_list, tape_list):
        """
        Initialize amsresults by importing tape files.

        @param fn_list: a list of the files to import
        @type  fn_list: list of str
        @param tape_list: the tape numbers (i.e., 21 for TAPE21 etc.) corresponding to these files
        @type  tape_list: list of int
        """

        from .Files import adf_filemanager
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


class amsresults(scmresults):

    def __init__(self, j=None):
        """
        Constructor for amsresults.
        """
        super().__init__(j)

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

        modes_c = numpy.zeros((nmodes, 3 * natoms))
        for i in range(nmodes):
            modes_c[i, :] = self.get_result_from_tape('Vibrations',
                                                      f'NoWeightNormalMode({i + 1:d})')
        return modes_c


class adfresults(amsresults):

    def __init__(self, j=None):
        """
        Constructor for adfresults.
        """
        super().__init__(j)


class scmjob(job):
    """
    An abstract base class for SCM jobs (AMS and related programs).

    Corresponding results class: L{scmresults}

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
        super().__init__()
        self._checksum_only = False

    def create_results_instance(self):
        """
        Create an instance of the matching results object for this job.
        """
        return scmresults(self)

    def get_input(self):
        """
        Abstract method to obtain an input file for ADF or a related program.
        """
        raise NotImplementedError

    @property
    def checksum(self):
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
            m.update(inp.encode('utf-8'))
            return m.hexdigest()
        else:
            return None

    def get_runscript(self, nproc=1, program="ams", inputfile=None):
        """
        Return a runscript for ADF or a related program.

        The input for this program is either taken from the
        given C{inputfile} or generated by L{get_input}.

        @param nproc: Number of processes to use.
        @type  nproc: int
        @param program: The program to run (by default: adf).
        @type  program: str
        @param inputfile: The input file to use. If None (default), L{get_input} is called.
        @type  inputfile: str or C{None}
        """
        runscript = ''
        if inputfile is None:
            inp = "<<eor"
            runscript += "cat <<eor\n"
            runscript += self.get_input()
            runscript += "eor\n"
        else:
            inp = "<" + inputfile
            runscript += "cat " + inputfile + "\n"
        if nproc == 1:
            runscript += "$AMSBIN/" + program + " -n1 " + inp + " || exit $? \n"
        else:
            runscript += f"export NSCM={nproc:d} \n\n"
            runscript += 'if [ -n "$SLURM_JOB_ID" ]; then \n'
            runscript += f'   export SCM_MPIOPTIONS="-np {nproc:d}" \n'
            runscript += 'fi \n'
            runscript += "$AMSBIN/" + program + " " + inp + " || exit $? \n"
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
        f = open(logfilename)
        lastline = ''.join(f.readlines()[-3:])
        f.close()
        if lastline.find('ERROR') >= 0:
            raise PyAdfError('ERROR DETECTED in AMS run')
        elif lastline.find('NORMAL TERMINATION') == -1:
            raise PyAdfError('Unknown Error in AMS run')

        # check for warnings in PyAdf
        all_warnings = []
        f = open(logfilename)
        for line in f.readlines():
            if line.find('WARNING') >= 0:
                print(" Found WARNING in AMS logfile:")
                warning = ' '.join(line.split()[3:])
                print(warning)
                all_warnings.append(warning)
        f.close()
        print()

        for warning in all_warnings:
            if "NOT CONVERGED" in warning:
                raise PyAdfError('NOT CONVERGED in AMS run')

        os.remove(logfilename)

        return True

    def result_filenames(self):
        return ['TAPE21', 'TAPE10', 'TAPE41']


class amsjob(scmjob):
    """
    Base class for ADF jobs using the AMS engine.
    """

    def __init__(self, mol, task='SinglePoint', settings=None):
        super().__init__()
        self.mol = mol
        self.task = task

        if settings is None:
            self.settings = amssettings()
        else:
            self.settings = settings

    def create_results_instance(self):
        return amsresults(self)

    def get_molecule(self):
        return self.mol

    def get_input(self):
        amsinput = ""
        amsinput += self.get_system_block()
        if self.symmetrize:
            amsinput += self.get_symtol_block()
        amsinput += self.get_task_block()
        amsinput += self.get_properties_block()
        amsinput += self.get_other_amsblocks()
        amsinput += self.get_engine_block()
        return amsinput

    def get_task_block(self):
        block = f"TASK {self.task} \n\n"
        return block

    def get_charge_block(self):
        return ""

    def get_efield_block(self):
        return ""

    @property
    def symmetrize(self):
        return False

    @property
    def symtol(self):
        return None

    def get_atoms_block(self):
        block = " ATOMS [Angstrom]\n"
        block += self.get_molecule().print_coordinates(index=False)
        block += " END\n"
        return block

    def get_system_block(self):
        block = "SYSTEM\n"
        if self.symmetrize:
            block += " SYMMETRIZE\n"
            if self.get_molecule().symmetry is not None:
                block += f" SYMMETRY {self.get_molecule().symmetry} \n"
        block += self.get_atoms_block()
        block += self.get_charge_block()
        block += self.get_efield_block()
        block += "END\n\n"
        return block

    def get_symtol_block(self):
        block = ""
        if self.symtol is not None:
            block = "SYMMETRY \n "
            block += " SymmetrizeTolerance " + str(self.symtol) + "\n"
            block += "END \n\n"
        return block

    def get_properties_block(self):
        return ""

    def get_other_amsblocks(self):
        return ""

    def get_engine_block(self):
        pass

    # noinspection PyMethodOverriding
    def get_runscript(self, nproc=1):
        """
        Return a runscript for AMS.
        """
        runscript = super().get_runscript(nproc=nproc, program='ams')
        runscript += "retcode=$?\n"
        runscript += "cat ams.results/ams.log\n"

        runscript += "exit $retcode\n"
        return runscript

    def check_success(self, outfile, errfile, logfile=None):
        if logfile is None:
            logfilename = os.path.join('ams.results', 'ams.log')
        else:
            logfilename = logfile
        return super().check_success(outfile, errfile, logfilename)

    def result_filenames(self):
        return [os.path.join('ams.results', f) for f in ['ams.rkf']]


class adfjob(amsjob):
    """
    An abstract base class for ADF jobs.

    Corresponding results class: L{adfresults}

    @group Initialization:
        __init__
    @group Input Generation:
        get_input
    @group Running:
        run
    """

    def __init__(self, mol, settings=None):
        if settings is None:
            mysettings = amssettings()
        else:
            mysettings = settings

        super().__init__(mol, task='SinglePoint', settings=mysettings)

    def create_results_instance(self):
        """
        Create an instance of the matching results object for this job.
        """
        return adfresults(self)

    def result_filenames(self):
        fns = super().result_filenames()
        return fns + [os.path.join('ams.results', f) for f in ['adf.rkf', 'TAPE10']]

    def get_adf_input(self):
        pass

    def get_engine_block(self):
        block = "Engine ADF\n"
        block += self.get_adf_input()
        block += "EndEngine\n\n"
        return block
