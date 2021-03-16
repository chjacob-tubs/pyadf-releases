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
"""
Support for I{SNF} computations using I{Turbomole} (other programs may be added).

It might be desireable to make this module more general and support any kind of
I{SNF} computations. However, my time to write it was rather limited and I
needed it only for a specific purpose. And last but not least, I don't know how
to use I{SNF} with other backends but I{Turbomole}. Adding an additional layer
of abstraction to derive more C{SNF...Job} classes should (at most) be a
moderately difficult task.

@author:  Moritz Klammler
@contact: U{moritz.klammler@gmail.com<mailto:moritz.klammler@gmail.com>}
@date:    Aug. 2011

"""

import os

from Errors import PyAdfError
from Turbomole import TurbomoleJob, TurbomoleResults, \
    TurbomoleSinglePointResults, TurbomoleGeometryOptimizationResults


class SNFJob(TurbomoleJob):
    """
    A I{SNF} job using I{Turbomole} as backend.

    """

    def __init__(self, converged_results, deuterium=None):

        """
        Initializes a L{SNFJob}.

        @param converged_results: Results from a converged geometry
                                  optimization to use as the equilibrium
                                  configuration.
        @type converged_results:  L{TurbomoleGeometryOptimizationResults}

        @param deuterium: list of hydrogen atom numbers that should be replaced by deuterium
        @param deuterium: list
        """

        import hashlib

        # A SNF computation  makes no sense if we don't  start from a converged
        # geometry computed with the same tool (i.e. Turbomole).

        if not isinstance(converged_results, TurbomoleGeometryOptimizationResults):
            if isinstance(converged_results, TurbomoleSinglePointResults):
                print "WARNING: SNF is not started with a converged Turbomole geometry optimization."
                print "         This might be correct, but be sure that you know what you are doing."
            else:
                raise PyAdfError("SNF job requires a converged Turbomole calculation")

        # We simply  use the setup from  the converged results.  To access them
        # later,  we  get us  a  reference  to that  object  and  use the  file
        # extraction method later.

        self.converged_predecessor = converged_results

        super(SNFJob, self).__init__(self.converged_predecessor.get_molecule())

        self.settings = self.converged_predecessor.job.settings
        self.deuterium = deuterium

        self.jobtype = "SNF / Turbomole first-order vibration job"

        # We compose our checksum from the one of our converged predecessor and
        # our job type.

        md5 = hashlib.md5()
        md5.update(self.converged_predecessor.get_checksum())
        md5.update(self.jobtype)
        self.checksum = md5.hexdigest()

    def set_restart(self, restart):
        """
        Not implemented!

        @raises NotImplementedError: This feature is not implemented (yet).
        @bug: Not implemented.

        """
        raise NotImplementedError()  # FIX THIS! (or maybe not - who needs that?)

    def get_checksum(self):
        """
        Get a quasi unique checksum for this job.

        @returns: Quasi unique checksum.
        @rtype:   L{str}
        
        """
        return self.checksum

    def before_run(self):

        """
        Does stuff that has to be done I{before} the run script is executed.

        Copies the needed files from the coverged geometry optimization into
        the current working directory and run I{snfdefine}.
        
        @raises PyAdfError: If I{snfdefine} quit in error.

        """

        import shutil
        from subprocess import Popen, PIPE
        from JobRunner import DefaultJobRunner

        # First we copy  the files we need from  our converged predecessor into
        # our working directory. Since the results object's method always makes
        # temporary files, we simply move them via high level OS operations.

        filenames = ['coord', 'control', 'basis', 'auxbasis', 'mos']
        for filename in filenames:
            tempfilename = self.converged_predecessor.get_temp_result_filename(filename)
            shutil.move(tempfilename, filename)

        # The next  step is to run  `snfdefine'. We don't care  much about this
        # but simply assemble  a standard input sequence (Answer  `tm' once and
        # press `Enter'  in all other situations.)   and pipe it  to it.  (Note
        # that a `subprocess.communicate(...)'  will  never hang in infinite IO
        # loops but  rather terminate  in error.)  If the return  code is  0 we
        # believe that everything is fine.

        snfdefine_input = ['', 'tm']

        if self.deuterium is not None:
            snfdefine_input.append('iso')
            for atom in self.deuterium:
                snfdefine_input.append('%s' %(atom))
                snfdefine_input.append('2')
            snfdefine_input.append('')

        for i in xrange(5):
            snfdefine_input.append('')
        snfdefine_input.append('8')  # sets scfconv to 8 as requested by SNFdefine
        snfdefine_input.append('m4')  # sets gridzize to m4 as requested by SNFdefine
        snfdefine_input.append('')
        snfdefine_stdin = ''
        for item in snfdefine_input:
            snfdefine_stdin += item + '\n'

        env = DefaultJobRunner().get_environ_for_local_command(SNFJob)

        try:
            # Create the subprocess
            sub = Popen(['snfdefine'], stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env)

            # Pass it the input and wait for it to finish.
            snfdefine_stdout, snfdefine_stderr = sub.communicate(input=snfdefine_stdin)
        except OSError:
            raise PyAdfError("""Couldn't start a `snfdefine' subprocess. Have
                             you even installed it?""")

        with open('snfdefine_stdin' + os.extsep + 'log', 'w') as report:
            report.write(snfdefine_stdin)
        with open('snfdefine_stdout' + os.extsep + 'log', 'w') as report:
            report.write(snfdefine_stdout)
        with open('snfdefine_stderr' + os.extsep + 'log', 'w') as report:
            report.write(snfdefine_stderr)

        if sub.returncode != 0:
            raise PyAdfError("""`snfdefine' quit on error. There is no
                             point in going on.""")

    def get_runscript(self, nproc=1):

        # this will probably fail if only one core is available
        runscript = 'mpirun -np %i snfdc \n' % nproc

        return runscript

    def after_run(self):
        """
        Does stuff that has to be done I{after} the runscript has been
        (successfully) executed.

        Runs I{snf}.  If the postprocessing  step fails, (i.e. I{snf}  can't be
        executed or quits  in error), a warning is printed  and the incident is
        forgotten.
        """
        # This is preferably done here  rather than in the runscript because it
        # needs some utilities that may not be available on a cluster.

        from JobRunner import DefaultJobRunner
        from subprocess import Popen, PIPE

        env = DefaultJobRunner().get_environ_for_local_command(SNFJob)

        try:
            sub = Popen(['snf'], stdout=PIPE, stderr=PIPE, env=env)
            sub.communicate()
            if sub.returncode != 0:
                print "Warning: `snf' (the post processing script) quit in error but I don't care."
        except OSError:
            print "Failed to run `snf'. Have you even installed it?"

        super(SNFJob, self).after_run()

    def create_results_instance(self):
        """
        Get a L{SNFResults} object from this job.

        @returns: Results of this job.
        @rtype:   L{SNFResults}

        """
        return SNFResults(self)

    def print_settings(self):
        """
        Prints the settings for this job to I{stdout}.

        Since this class doesn't know much about itself but rather goes on
        working with someone else's results and settings, the output is very
        puristic.

        """
        print ("  Using settings from converged structure from job "
               + str(self.converged_predecessor.fileid) + ".")
        print "  I'm an SNF job. I have nothing like an own will. Do you?"


class SNFResults(TurbomoleResults):
    """
    Results of a I{SNF} computation (using I{Turbomole}).

    @group Retrieval of specific results: get_eigenvectors,
                                          get_carthesian_modes, 
                                          get_wave_numbers
    @group Access to result files:        _read_snf_output

    """

    def __init__(self, j=None):
        """
        Initialize a new L{SNFResults} object.

        @param j: L{job} object of the corresponding job.
        @type  j: L{job}

        """

        super(SNFResults, self).__init__(j=j)
        self.vibs = None

    def get_wave_numbers(self):
        """
        Get a list of the wave numbers for each mode.

        List convention::

            vibs = res.get_wave_numbers()
            v = vibs[n]

        is the wave number (in M{cm^(-1)}) of the M{n}th mode.

        @returns: List of wave numbers.
        @rtype:   L{float}C{[3 M{N} - 6]}
        @bug:     Doesn't work for linear molecules. You'll get weired errors
                  in that case.

        """
        self._read_snf_output()
        return self.vibs.modes.freqs

    def get_vibs(self):
        self._read_snf_output()
        return self.vibs

    def get_ir_ints(self):
        self._read_snf_output()
        return self.vibs.get_ir_intensity()

    def _read_snf_output(self, readagain=False):
        """
        Read in the I{SNF} results from the output file. 

        The read values are stored in attributes. This speeds up further
        retrievals of results.

        @param readagain:   If this is set C{True}, the data will be re-read from
                            the I{SNF} output file even if this has allready been
                            done before.
        @bug:               The used C{VibTools} routine doesn't work for linear
                            molecules. You'll get weired errors in that case.
        @requires:          C{VibTools}
        @raises PyAdfError: For diatomic molecules.
        """

        # First we see if the `VibTools' are available. If that import failes,
        # we re-raise a more descriptive exception.

        try:
            import VibTools
        except ImportError:
            raise PyAdfError("""I couldn't import the `VibTools' module. Please
            make sure it can be found along your `$PYTONPATH'. I need that
            module to read in the I{SNF} results.""")

        # Next see, if we need to (re-)read  the data or if we have it allready
        # handy. Re-reading will be a costy thing since we'll have to extract a
        # total of three  files from our tarball to  temporary files, read them
        # and delete them afterwards. Lots of OS level IO.

        if (self.vibs is None) or readagain:

            # We rely  on the `VibTools' module  by Christoph Jacob  to read the
            # `SNF' output. That module has a bug (see below) but we hope that
            # it will  be fixed  soon.  Since the  `VibTools.SNFResults' object
            # needs reading access to the `snf.out', `restart' and `coord' file
            # on instanziation,  we get us  temporary copies of those  and pass
            # the initializator  the filenames.  We'll delete  these files once
            # everything is done. For the sake of easier handling, we make us a
            # dictionary that mappes the  original file names (e.g. `coord') to
            # the names of the temporary copies (e.g. `/tmp/tmpBGr7Hh').
            # (Using  the `__del__'  method for  this cleanup  and  keeping the
            # files as a class attribute  is not a safe solution since Python's
            # way to invoke garbage collection is REALLY unreliable.)

            needed_files = ['snf.out', 'restart', 'coord']
            temp_copies = {}
            try:
                for needed_file in needed_files:
                    temp_copies[needed_file] = self.get_temp_result_filename(needed_file)
                vibs = VibTools.SNFResults(outname=temp_copies['snf.out'],
                                           restartname=temp_copies['restart'],
                                           coordfile=temp_copies['coord'])
                vibs.read()

                self.vibs = vibs
            finally:
                for key, value in temp_copies.iteritems():
                    try:
                        os.remove(value)
                    except OSError:
                        pass  # okay, that tempfile doesn't bother us that much
