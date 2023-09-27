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
This is an non-user level module to handle I{Turbomole}'s I{define}.

@author: Moritz Klammler
@contact: U{moritz.klammler@gmail.com<mailto:moritz.klammler@gmail.com>}

@bug: The module is horrible spaghetti code. It probably punches sweet little
      ponnies and eats small children. It should not be released without
      careful review!
@bug: The classes herein were originally written for a different setup.
@bug: The way exceptions are handled is mostly a mess.
@bug: Much redundant code.
@bug: Long strings are splitted among several lines in an ugly fashion using
      concatenation instead of implicite line joining.
"""

import sys
import os
import re
from subprocess import Popen, PIPE

from .Errors import PyAdfError


class TurboObject:
    """
    Mother class for historical reason.

    """

    def __init__(self, verbose_level=1):
        """
        Initializes a new instance.

        You'll always find the complete debugging information in a file named
        C{turbo.log}.

        @param verbose_level: The higher the verbose level is the more output
                              will be generated.
        @type  verbose_level: L{int}

        """

        self.verbose_level = verbose_level

    def _report(self, message, level):
        """
        Report (debugging) information.

        Presents an abstract way to report good or bad news. C{level} is
        non-negative for good news and negative for errors. The greater the
        magnitude of C{level} the less important the C{message} was. Messages
        whos magintude exceeds C{verbose_level} are written to C{turbo.log} but
        not displayed.

        Level convention::

             0:  essential
             1:  important
             2:  interesting
             3:  hardly interesting
            >3:  debugging only

            -1:  fatal error
            -2:  non-fatal error or additional information on a fatal one
            -3:  debugging information

        @param message: Text to report
        @type  message: L{str}
        @param level:   Importance (see text)
        @type  level:   L{int}

        """

        if abs(level) < self.verbose_level:
            if level < 0:
                self._reportBadNews(message)
            else:
                self._reportGoodNews(message)

        self.logfilename = 'turbo' + os.extsep + 'log'
        with open(self.logfilename, 'a', encoding='utf-8') as logfile:
            logfile.write(message + '\n')

    @staticmethod
    def _reportGoodNews(message):
        """
        Prints to I{stdout}.

        @param message: Text to show
        @type  message: L{str}

        """
        print(message)

    @staticmethod
    def _reportBadNews(message):
        """
        Prints to I{stderr}.

        @param message: Text to show
        @type  message: L{str}

        """
        sys.stderr.write(message + '\n')


class TurboCosmoprep(TurboObject):
    """
    Actually handles I{cosmoprep} on a sub-user level.

    """

    def __init__(self, settings):
        """
        Creates a new instance and sets default values for the C{cosmoprep}
        session.

        @param settings: Settings for this job.
        @type  settings: L{TurbomoleSettings}
        @bug: This method is actually much older than the L{TurbomoleSettings}
              are. It works together with them but they probably don't love
              each other.

        """

        super().__init__(verbose_level=settings.verbose_level)
        self.settings = settings

        self.cosmoprep_stdin = ''
        self.cosmoprep_stdout = ''
        self.cosmoprep_stderr = ''

    def runcosmoprep(self):
        """
        Runs I{cosmoprep} and sanitizes the output.

        @returns:           0 if successful
        @rtype:             L{int}
        @raises PyAdfError: If not successfull

        """

        returncode = self._tmcosmoprep()
        tmcosmoprep_status = (returncode == 0)
        if not tmcosmoprep_status:
            info = "ERROR: `cosmoprep' quit on error. Skipping output check."
            self._report(info, -2)
            raise PyAdfError(info)

        # Check the results.
        sanitize_status = self._cosmosanitize()  # not yet implemented

        # Write our I/O  to / from `cosmoprep' to log  files.  We convert them
        # to  strings  explicitly  to  account  for  the  possibility  that
        # `cosmoprep' crashed and they are `None' rather than strings.

        with open('cosmoprep_stdin' + os.extsep + 'log', 'wb') as report:
            report.write(self.cosmoprep_stdin.encode('utf-8'))
        with open('cosmoprep_stdout' + os.extsep + 'log', 'wb') as report:
            report.write(self.cosmoprep_stdout.encode('utf-8'))
        with open('cosmoprep_stderr' + os.extsep + 'log', 'wb') as report:
            report.write(self.cosmoprep_stderr.encode('utf-8'))

        if tmcosmoprep_status:
            self._report("`cosmoprep' successfully quit on exit status "
                         + str(returncode) + ".", 2)
            if sanitize_status:
                self._report("I've checked `cosmoprep's output and it looked fine. "
                             + "This is certainly no gurantee that it really "
                             + "did the right thing. If I were able to check "
                             + "this, I needn't even call it.", 2)
            else:
                self._report("ERROR: I've checked the results and they don't "
                             + "seem quite okay. I'll refuse "
                             + "accepting this as a success.", -2)
                self._report('Input for cosmoprep:', 4)
                self._report(self.cosmoprep_stdin, 4)
                self._report('Output from cosmoprep:', 4)
                self._report(self.cosmoprep_stdout, 4)
                self._report('Error-output from cosmoprep:', 4)
                self._report(self.cosmoprep_stderr, 4)
        else:
            self._report("ERROR: `cosmoprep' quit on exit status `"
                         + str(returncode) + "'. I didn't even look at "
                         + "the results.", -2)

        if tmcosmoprep_status:  # and sanitize_status:
            return returncode
        else:
            message = ("ERROR: Some checks of the output of `cosmoprep' were not successful. There is no point "
                       + "in starting a computation.")
            if self.verbose_level <= 1:
                message += ('\n' + "Consider looking at the file `{log}' "
                            + "or run me with `verbose_level > 1' to "
                            + "get more output.").format(log=self.logfilename)
            raise PyAdfError(message)

    def _tmcosmoprep(self):
        """
        Runs I{cosmoprep} and returns its exit status but doesn't check the
        output.

        Communication with I{cosmoprep} will be stored in C{cosmoprep_stdin},
        C{cosmoprep_stdout} and C{cosmoprep_stderr} respectively.
        The results should be sanitized afterwards.

        """
        from .JobRunner import DefaultJobRunner
        from .Turbomole import TurbomoleJob

        # Generate the input sequence to be passed.
        self.cosmoprep_stdin = self._assembleCosmoInput()

        env = DefaultJobRunner().get_environ_for_local_command(TurbomoleJob)

        try:
            # Create the subprocess
            CosmoProcess = Popen(['cosmoprep'], stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env)

            # Pass it the input and wait for it to finish.
            self.cosmoprep_stdout, self.cosmoprep_stderr \
                = CosmoProcess.communicate(input=self.cosmoprep_stdin.encode('utf-8'))
            self.cosmoprep_stdout = self.cosmoprep_stdout.decode('Latin-1')
            self.cosmoprep_stderr = self.cosmoprep_stderr.decode('Latin-1')
            self._report("Successfully started a `cosmoprep' subprocess. ", 2)
        except OSError:
            self._report("Couldn't start a `cosmoprep' subprocess. ", -2)
            return None
        return CosmoProcess.returncode

    def _assembleCosmoInput(self):
        """
        Generates the input string to be passed to I{cosmoprep}.

        @returns: Input sequence
        @rtype:   L{str}

        """

        sequence = []
        if self.settings.cosmo_epsilon:
            sequence.append(str(self.settings.cosmo_epsilon))
        else:
            sequence.append('')
        for i in range(6):
            sequence.append('')
        if self.settings.cosmo_rsolv:
            sequence.append(str(self.settings.cosmo_rsolv))
        else:
            sequence.append('')
        for i in range(3):
            sequence.append('')
        if self.settings.cosmo_radii:
            sequence.append(str(self.settings.cosmo_radii))
        else:
            sequence.append('r all o')  # radius definition menu

        sequence.append('*')
        for i in range(2):
            sequence.append('')

        inputstring = ''
        for item in sequence:
            inputstring += item + '\n'

        return inputstring

    def _cosmosanitize(self):
        """
        Runs some tests on the output of I{cosmoprep} and compares it with what
        would be to be expected from the given setup.

        If expectation and observation match, it returns C{True}. Needless to
        say that calling this method makes only sense after I{cosmoprep} has been
        executed and without any properties having been redefined since then.

        @returns: Status of check
        @rtype:   L{bool}

        """

        # We will always  try to run through the entire test  to supply as much
        # information as  possible. However, we a single  error issufficient to
        # let us return `success = False'.
        success = True

        try:
            # We print a promt in front of every line for the verbose output.
            sanitizeprompt = "checking output: "

            # The very  first thing  to do is  to look  at the finaly  words of
            # `define'.  I have no clue why, but `define' ALWAYS writes them to
            # `stderr'.

            if re.search("cosmoprep ended normally", self.cosmoprep_stderr):
                self._report(sanitizeprompt + "`cosmoprep' claims to have ended "
                             + "normally.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `cosmoprep' says he ended "
                             + "abnormally.", -3)

            # Next we check if the cosmo part is found in the `control` file.

            cosmo_keywords = ['cosmo', 'cosmo_atoms', 'cosmo_out']
            if self.settings.cosmo_epsilon:
                cosmo_keywords.append(str(self.settings.cosmo_epsilon))
            if self.settings.cosmo_rsolv:
                cosmo_keywords.append(str(self.settings.cosmo_rsolv))
            controlfile = 'control'

            if os.path.isfile(controlfile):
                self._report(sanitizeprompt + "File `" + controlfile
                             + "' still exists.", 3)
                with open(controlfile) as checkfile:
                    contents = checkfile.read()
                    for keyword in cosmo_keywords:
                        if keyword in contents:
                            pass
                        else:
                            success = False
                            self._report(sanitizeprompt + "ERROR: File `"
                                         + controlfile
                                         + "' doesn't look as I "
                                         + "expected it to.", -3)
                    self._report(sanitizeprompt + "File `" + controlfile
                                 + "' looks fine at a first glance. (I've "
                                 + "been checking for these variables: "
                                 + str(cosmo_keywords) + ".)", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: File `" + controlfile +
                             "' doesn't " + "exist anymore.", -3)

            self.temp = TurboDefinition._compact(self.cosmoprep_stdout)

            # Check if the `Set COSMO parameters` menu still exists:
            if self._checkfor("Set COSMO parameters"):
                self._report(sanitizeprompt + "`cosmoprep' `set COSMO parameters' menu exists.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `cosmoprep' did not find `set COSMO parameters menu'.", -3)

            # Check if the `Set COSMO parameters` menu asks for epsilon:
            if self._checkfor("epsilon"):
                self._report(sanitizeprompt + "`cosmoprep' still asks for the epsilon value. "
                                              "(Uses default value if not specified by user.)", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `cosmoprep' did not ask for the epsilon value.", -3)

            # Check if the `Set COSMO parameters` menu asks for rsolv:
            if self._checkfor("rsolv"):
                self._report(sanitizeprompt + "`cosmoprep' still asks for the rsolv value. "
                                              "(Uses default value if not specified by user.)", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `cosmoprep' did not ask for the rsolv value.", -3)

            # Check if the `radius definition menu` menu still exists:
            if self._checkfor("radius definition menu"):
                self._report(sanitizeprompt + "`cosmoprep' `radius definition menu' exists.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `cosmoprep' did not find `radius definition menu'.", -3)

            # Check if the `radius definition menu` menu worked:
            if self._checkfor("Group   atom   mtype   radius"):
                self._report(sanitizeprompt + "`cosmoprep' found radii for the atoms.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `cosmoprep' did not find radii for the atoms", -3)

            # Check if the `Cosmo output definition` menu still exists:
            if self._checkfor("Cosmo output definition"):
                self._report(sanitizeprompt + "`cosmoprep' `Cosmo output definition' exists.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `cosmoprep' did not find `Cosmo output definition'.", -3)

            # Look for  the final statement. The number  of asterisks shouldn't
            # bother us.
            if self._checkfor(r"\*+  cosmoprep : all done  \*+"):
                self._report(sanitizeprompt + "I found the familiar "
                             + "`**** cosmoprep : all done ****' statement.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: I'm missing the familiar "
                             + "`**** cosmoprep : all done ****' statement.", -3)

        except Exception as e:
            # Make any exception during the process of sanitization a reason to
            #  return `False'.
            self._report("ERROR: There was an unexpected exception caught "
                         + "while checking `cosmoprep's output. It says: "
                         + str(e), -2)
            raise

        finally:
            # Get rid of the temporary variable before returning.
            del self.temp

        return success

    def _checkfor(self, pattern):
        """
        Copied function!!
        TO DO: make one static function?

        Sees if C{pattern} occurs in C{cosmoprep_stdout}.

        Both strings are "compacted" (using L{_compact}) first. This method will
        always return C{False} if C{self.temp} has not been defined to be the
        compacted version of C{self.cosmoprep_stdout}.

        @returns: C{True} if the pattern occurs, otherwise C{False}.
        @rtype:   L{bool}

        """

        pattern = TurboDefinition._compact(pattern)
        pattern = re.compile(pattern)
        try:
            if re.search(pattern, self.temp):
                return True
            else:
                return False
        except NameError:
            # <self.temp> not defined!
            return False


class TurboDefinition(TurboObject):
    """
    Actually handles I{define} on a sub-user level.

    """

    def __init__(self, settings):
        """
        Creates a new instance and sets default values for the C{define}
        session.

        @param settings: Settings for this job.
        @type  settings: L{TurbomoleSettings}
        @bug: This method is actually much older than the L{TurbomoleSettings}
              are. It works together with them but they probably don't love
              each other.

        """

        super().__init__(verbose_level=settings.verbose_level)
        self.settings = settings

        self.define_stdin = ''
        self.define_stdout = ''
        self.define_stderr = ''

        try:
            self.setCoordFile(self.settings.coordfilename)
        except OSError:
            self._report("Error setting `" + str(self.settings.coordfilename) + "' as input file.", -3)
            self.coordfilename = None
            self.atom_checksum = float('NaN')

    def setCoordFile(self, filename):
        """
        Select the name for the C{coord} file.

        Set the file from which the coordinates (in proper I{Turbomole} format
        should be read from. This method automatically computes and stores the
        checksum (number of atoms) in the file. Any exceptions will raise
        through.

        @param filename: Name of the coordinate file
        @type  filename: L{str}

        """

        self.coordfilename = filename
        self.atom_checksum = self._countAtoms(filename)

    def run(self):
        """
        Runs I{define} and sanitizes the output.

        @returns:           0 if successful
        @rtype:             L{int}
        @raises PyAdfError: If not successfull

        """

        returncode = self._tmdefine()
        tmdefine_status = (returncode == 0)
        if not tmdefine_status:
            info = "ERROR: `define' quit on error. Skipping output check."
            self._report(info, -2)
            raise PyAdfError(info)

        # Postprocess the `control file.
        self._postprocess()

        # Check the results.
        sanitize_status = self._sanitize()

        # Start a cosmoprep job, if specified to use cosmo
        if self.settings.cosmo:
            self._report("***** Starting `cosmoprep` menu", 2)
            TurboCosmoprep(self.settings).runcosmoprep()
            self._report("***** Finishing `cosmoprep` menu", 2)

        # Write our I/O  to / from `define' to log  files.  We convert them
        # to  strings  explicitly  to  account  for  the  possibility  that
        # `define' crashed and thy are `None' rather than strings.

        with open('define_stdin' + os.extsep + 'log', 'wb') as report:
            report.write(self.define_stdin.encode('utf-8'))
        with open('define_stdout' + os.extsep + 'log', 'wb') as report:
            report.write(self.define_stdout.encode('utf-8'))
        with open('define_stderr' + os.extsep + 'log', 'wb') as report:
            report.write(self.define_stderr.encode('utf-8'))

        if tmdefine_status:
            self._report("`define' successfully quit on exit status "
                         + str(returncode) + ".", 2)
            if sanitize_status:
                self._report("I've checked `define's output and it looked fine. "
                             + "This is certainly no gurantee that it really "
                             + "did the right thing. If I were able to check "
                             + "this, I needn't even call it.", 2)
            else:
                self._report("ERROR: I've checked the results and they don't "
                             + "seem quite okay. I'll refuse "
                             + "accepting this as a success.", -2)
                self._report('Input for define:', 2)
                self._report(self.define_stdin, 2)
                self._report('Output from define:', 2)
                self._report(self.define_stdout, 2)
                self._report('Error-output from define:', 2)
                self._report(self.define_stderr, 2)
        else:
            self._report("ERROR: `define' quit on exit status `"
                         + str(returncode) + "'. I didn't even look at "
                         + "the results.", -2)

        if tmdefine_status and sanitize_status:
            return returncode
        else:
            message = ("ERROR: Some checks of the output of `define' were not successful. There is no point "
                       + "in starting a computation.")
            if self.verbose_level <= 1:
                message += ('\n' + "Consider looking at the file `{log}' "
                            + "or run me with `verbose_level > 1' to "
                            + "get more output.").format(log=self.logfilename)
            raise PyAdfError(message)

    def _tmdefine(self):
        """
        Runs I{define} and returns its exit status but doesn't check the
        output.

        Communication with I{define} will be stored in C{define_stdin},
        C{define_stdout} and C{define_stderr} respectively. This method will
        always succeed in the sense that it finishes in finite time. (Except
        the I{Turbomole} developers introduce and endless loop into I{define}.)
        It will never hang on I/O promts.  The results should be sanitized
        afterwards.

        """
        from .JobRunner import DefaultJobRunner
        from .Turbomole import TurbomoleJob

        # Generate the input sequence to be passed.
        self.define_stdin = self._assembleInput()

        env = DefaultJobRunner().get_environ_for_local_command(TurbomoleJob)

        try:
            # Create the subprocess
            D = Popen(['define'], stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env)

            # Pass it the input and wait for it to finish.
            self.define_stdout, self.define_stderr = D.communicate(input=self.define_stdin.encode('utf-8'))
            self.define_stdout = self.define_stdout.decode('Latin-1')
            self.define_stderr = self.define_stderr.decode('Latin-1')
            self._report("Successfully started  a `define' subprocess. ", 2)
        except OSError:
            self._report("Couldn't start a `define' subprocess. "
                         + "Have you even installed it?", -2)
            return None
        return D.returncode

    def _postprocess(self):
        """
        Make some changes in the C{control} file that can't be done via
        I{define}.

        """

        import tempfile

        toadd = ''

        if self.settings.disp is None:
            pass
        elif self.settings.disp == 'dft-d1':
            toadd += '$olddisp' + '\n'
        elif self.settings.disp == 'dft-d2':
            toadd += '$disp' + '\n'
        elif self.settings.disp == 'dft-d3':
            toadd += '$disp3' + '\n'
        else:
            raise PyAdfError("Unknown value `" + str(self.settings.disp) + "' for dispersion correction.")

        if self.settings.scfconv is not None:
            toadd += '$scfconv ' + str(self.settings.scfconv) + '\n'

        if self.settings.scfiterlimit is not None:
            toadd += '$scfiterlimit ' + str(self.settings.scfiterlimit) + '\n'

        if self.settings.pointcharges is not None:
            self._report("Adding " + str(self.settings.num_pointcharges) + " point charges to control file.", 2)
            toadd += '$point_charges list \n'
            for i in range(self.settings.num_pointcharges):
                toadd += ' '.join(map(str, self.settings.pointcharges[i, :])) + '\n'

        with tempfile.NamedTemporaryFile(mode='a', delete=False) as tf:
            with open('control') as infile:
                for line in infile:
                    line = line.replace('$end', toadd + '$end')
                    tf.write(line + '\n')
            tf.file.close()
            open('control', 'w').close()
            with open(tf.name) as infile:
                with open('control', 'a') as outfile:
                    for line in infile:
                        outfile.write(line + '\n')
            os.remove(tf.name)

    def _sanitize(self):
        """
        Runs some tests on the output of I{define} and compares it with what
        would be to be expected from the given setup.

        If expectation and observation match, it returns C{True}. Needless to
        say that calling this method makes only sense after I{define} has been
        executed and without any properties having been redefined since then.

        @returns: Status of check
        @rtype:   L{bool}

        """

        # We will always  try to run through the entire test  to supply as much
        # information as  possible. However, we a single  error issufficient to
        # let us return `success = False'.
        success = True

        try:
            # We print a promt in front of every line for the verbose output.
            sanitizeprompt = "checking output: "

            # The very  first thing  to do is  to look  at the finaly  words of
            # `define'.  I have no clue why, but `define' ALWAYS writes them to
            # `stderr'.

            if re.search("define ended normally", self.define_stderr):
                self._report(sanitizeprompt + "`define' claims to have ended "
                             + "normally.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `define' says he ended "
                             + "abnormally.", -3)

            # Next we see if the  standard files `control', `coord' and `basis'
            # are  present  and look  "normal".   If  we  (erm, `define')  have
            # guessed  some initial  occupation, there  should also  be  a file
            # `mos'  and with  `ri' we  expect a  file `auxbasis'.   If  we use
            # redundant coordinates, the `coord' file should show that too.  We
            # use  a dictionary  to  run a  VERY  rudimentary check,  if a  few
            # characteristic words appear in  the files.  This are the keywords
            # we look for.  (A `$' will be prepended to each.)

            control_keywords = ['title', 'coord', 'atoms', 'basis', 'end']
            coord_keywords = ['coord', 'end']
            if self.settings.ired:
                coord_keywords.append('redundant')
            basis_keywords = ['basis', 'end']
            auxbasis_keywords = ['jbas', 'end']
            mos_keywords = ['scfmo', 'end']
            alpha_keywords = ['uhfmo_alpha', 'end']
            beta_keywords = ['uhfmo_beta', 'end']

            keywords = {'control': control_keywords,
                        'coord': coord_keywords,
                        'basis': basis_keywords,
                        'auxbasis': auxbasis_keywords,
                        'mos': mos_keywords,
                        'alpha': alpha_keywords,
                        'beta': beta_keywords}

            checkfilenames = ['control', 'coord', 'basis']
            if self.settings.guess_initial_occupation_by is not None:
                if not self.settings.unrestricted:
                    checkfilenames.append('mos')
                else:
                    checkfilenames.append('alpha')
                    checkfilenames.append('beta')
            if self.settings.ri:
                checkfilenames.append('auxbasis')

            for checkfilename in checkfilenames:
                if os.path.isfile(checkfilename):
                    self._report(sanitizeprompt + "File `" + checkfilename
                                 + "' exists.", 3)
                    with open(checkfilename) as checkfile:
                        contents = checkfile.read()
                        for keyword in keywords[checkfilename]:
                            if '$' + keyword in contents:
                                pass
                            else:
                                success = False
                                self._report(sanitizeprompt + "ERROR: File `"
                                             + checkfilename
                                             + "' doesn't look as I "
                                             + "expected it to.", -3)
                        self._report(sanitizeprompt + "File `" + checkfilename
                                     + "' looks fine at a first glance. (I've "
                                     + "been checking for these variables: "
                                     + str(keywords[checkfilename]) + ".)", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: File `" + checkfilename +
                                 "' doesn't " + "exists.", -3)

            # We  will spend  more time  investigating the  standard  output of
            # `define' and run more conditionally tests.

            # We will not panic if a line is wrapped at a different position or
            # three blanks  are seen instead  of two nor if  capitalization has
            # changed.  Therefore,  we "compact" the  output.  Namely, removing
            # any whitespace including line  breaks and converting all to lower
            # case.

            # Since  `define' creates  A LOT  of output,  we'd better  not make
            # unnecessarily  many  copies  of  it.   Since we  want  to  use  a
            # subroutine to  check if a  charactaristic pattern is  present, we
            # want to be able to pass  it by reference.  Therefore we make it a
            # temporary data member of our  class and delete it after the check
            # is done.

            self.temp = self._compact(self.define_stdout)

            # We  now use the  method `_checkfor(pattern)'  to see  if <pattern>
            # occurs in <define_stdout>.  (In  the "compacted" form of course.)
            # <pattern> certainly has  to be compacted too but  the method will
            # do that for us. This method  is a little tricky in the sense that
            # it takes  a sting,  runs a regexp  on it  to compact it  and then
            # compiles the resulting  pattern to a regexp itself.  Hence, if we
            # need more than a literal match, we have to be somewhat careful.

            # See if  we were using `turbomole  6.3.x'.  We have  to escape the
            # `.' in  the regexp! We hope that  a "wrong" version is  not a big
            # problem.
            if self._checkfor(r"TURBOMOLE rev. V7\.5"):
                self._report(sanitizeprompt + "Found `turbomole 7.5' as "
                             + "expected.", 3)
            else:
                self._report(sanitizeprompt + "Found different " + "`turbomole' "
                             + "version (expecting 7.5.x). Check output if "
                             + "everything still works.", -3)

            # See if `define' states to have added the correct number of atoms:
            if self._checkfor("CARTESIAN COORDINATES FOR "
                              + str(self.atom_checksum) + " ATOMS HAVE "
                              + "SUCCESSFULLY BEEN ADDED"):
                self._report(sanitizeprompt + "`define' claims to have added "
                             + str(self.atom_checksum) + " atoms as expected.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: I expected `define' to add "
                             + str(self.atom_checksum)
                             + " atoms but he didn't.", -3)

            # The following  two checks only  matter if we were  actually using
            # `ired' coordinates.
            if self.settings.ired:
                # See if `define' still accepts `ired' as an input.
                if self._checkfor("ired : REDUNDANT INTERNAL COORDINATES"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `ired' as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I can't find an explanation for "
                                 + "`ired' but I have used it.", -3)

                # See if internal coordinates have been written. (This check is
                # somewhat  redundant since  we've allready  been  checking the
                # `coord'  file  but  since  we're  talking  about  "redundant"
                # coordinates,  this  is   probably  okay.   Version  6.3.1  of
                # `turbomole' said  "write onto file".  Since this  is a little
                # strange  and hence likely  to change,  we also  accept "write
                # into file".
                if self._checkfor(r" writing data block \$user-defined bonds "
                                  + "(o|i)nto file <coord>"):
                    self._report(sanitizeprompt + "`define' says, he wrote "
                                 + "user-defined bonds to the `coord' file.", 3)
                else:
                    # success = False
                    self._report(sanitizeprompt + "`define' says "
                                 + "nothing about user-defined bonds, this "
                                 + "should be okay in versions >6.3.x.", 3)

            # To my  tired ears it would  sound more logical  to say "supplying
            # basis sets _for_ N atoms" so we'll also accept that. Even though,
            # version 6.3.1 says "supplying to".
            if self._checkfor("SUPPLYING BASIS SETS (TO|FOR) "
                              + str(self.atom_checksum) + " ATOMS"):
                self._report(sanitizeprompt + "`define' is concerned about basis "
                             + "sets for " + str(self.atom_checksum) + " atoms as "
                             + "expected.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `define' doesn't say anything "
                             + "about supplying basis sets for our number of "
                             + "atoms.", -3)

            # TODO: CHECK IF `define' LOADED  BASIS SET LIBRARIES FOR ALL ATOMS
            #       INVOLVED.

            # See if `define' still accepts `b' as an input to assign basis sets.
            if self._checkfor("b : ASSIGN ATOMIC BASIS SETS"):
                self._report(sanitizeprompt + "`define' still seems to accept `b' "
                             + "as input to assign atomic basis sets.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `define's menu might have "
                             + "changed. I can't find an explanation for `b' "
                             + "but I have used it.", -3)

            # To see if  `define' knew the basis set we've  chosen, we look for
            # the  string he  uses to  complain about  the opposite  to  be NOT
            # present.  We accept  anything for the path of  the basis setsa nd
            # do not look for our explicit set since a changed menue might have
            # messed this up as well.
            if self._checkfor("THERE ARE NO DATA SETS CATALOGUED IN "
                              + "FILE.*CORRESPONDING TO NICKNAME"):
                success = False
                self._report(sanitizeprompt + "ERROR: `define' didn't know the basis "
                             + "set `" + str(self.settings.basis_set_all) + "' I wanted "
                             + "to assign to all atoms.", -3)
            else:
                self._report(sanitizeprompt + "At least, `define' didn't "
                             + " complain about using `"
                             + str(self.settings.basis_set_all)
                             + "' as basis set for all atoms.", 3)

            # See if  `define' writes the basis  sets to the  file. (We've also
            # checked that before.)
            if self._checkfor(" BASIS SETS WILL BE WRITTEN TO FILE basis BY "
                              + "DEFAULT"):
                self._report(sanitizeprompt + "`define'  says, he write basis "
                             + "sets to `basis' as I expected him to.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: `define' doesn't say anything "
                             + "about writing basis sets to `basis'.", -3)

            # The  next check makes  ony sense  if we  are actually  using EHT.
            # There is  a strange `&&'  in the text  of version 6.3.1.  Ma they
            # replace it by whatever they want -- we'll still accept it.
            if str(self.settings.guess_initial_occupation_by) == 'eht':
                if self._checkfor("eht : PROVIDE MOS.*OCCUPATION NUMBERS FROM "
                                  + "EXTENDED HUECKEL GUESS"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `eht' as input to select EHT for "
                                 + "initial guess.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I can't find an explanation for "
                                 + "`eht' but I have used it.", -3)

                # Note that we have to escape the parenthesis.
                if self._checkfor(r"ENTER THE MOLECULAR CHARGE \(DEFAULT=0\)") or \
                        self._checkfor(r"ENTER THE ATOMIC CHARGE \(DEFAULT=0\)"):
                    self._report(sanitizeprompt + "`define' asked for the charge as I expected.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I wasn't asked for the molecules charge.", -3)
                # Trivial occupation check for closed-shell atoms
                if not self.settings.unrestricted and self._countAtoms(self.coordfilename) == 1:
                    if self._checkfor("OCCUPATION NUMBER SELECTION IS TRIVIAL") and \
                       self._checkfor("SINCE ALL MOLECULAR ORBITALS ARE DOUBLY") and \
                       self._checkfor("OCCUPIED IN THE MINIMAL EHT BASIS SET"):
                        self._report(sanitizeprompt + "`define' says, he found a "
                                     + "trivial atomic occupation.", 3)
                    else:
                        success = False
                        self._report(sanitizeprompt + "ERROR: `define' doesn't say anything "
                                     + "about trivial occupation for closed-shell atoms.", -3)
                # Standard occupation check for molecules and open-shell atoms
                else:
                    if self._checkfor("AUTOMATIC OCCUPATION NUMBER ASSIGNMENT ESTABLISHED"):
                        self._report(sanitizeprompt + "`define' says, he found an "
                                     + "initial guess for the occupation.", 3)
                    else:
                        success = False
                        self._report(sanitizeprompt + "ERROR: `define' doesn't say anything "
                                     + "about successfully determined occupation.", -3)

                    # The default value is not arethesized here (as commonly done).
                    # We will accept both.  (Note the `?'  is escaped once!)
                    if self._checkfor(r"DO YOU ACCEPT THIS OCCUPATION\? \(?DEFAULT=y\)?"):
                        self._report(sanitizeprompt + "`define' was asking me to "
                                     + "accept the occupation just as expected.", 3)
                    else:
                        success = False
                        self._report(sanitizeprompt + "ERROR: `define' didn't ask me to "
                                     + "accept the occupation but I said blindly `yes'!", -3)
                # Natural occupation
                if self.settings.unrestricted:
                    if self._checkfor(r"DO YOU REALLY WANT TO WRITE OUT NATURAL ORBITALS\? "
                                      + r"\(?DEFAULT=n\)?"):
                        self._report(sanitizeprompt + "`define' was asking me to "
                                     + "decline natural occupation just as expected.", 3)

                    else:
                        success = False
                        self._report(sanitizeprompt + "ERROR: `define' didn't ask me to "
                                     + "accept the natural occupation but I said blindly `no'!", -3)

            # Look for the truth.
            if self._checkfor("God is great, beer is good and people are crazy"):
                self._report(sanitizeprompt + "`define' is making true statements about beer and people.", 3)

            # The following checks only apply if we are using DFT.
            if self.settings.dft:
                # See in `define' still accepts the `dft' keyword.
                if self._checkfor("dft : DFT Parameters"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + " accept `dft' as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I can't find an explanation for "
                                 + "`dft' but I have used it.", -3)

                if self._checkfor("on: TO SWITCH ON DFT"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `on' (for DFT) as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I can't find an explanation for "
                                 + "`on' (with DFT) but I have used it.", -3)

                if self._checkfor("func : TO CHANGE TYPE OF FUNCTIONAL"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `func' (to select the DFT"
                                 + " functional as input.)", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I can't find an explanation for "
                                 + "`func' (to select the DFT functional) but "
                                 + "I have used it.", -3)

                if self._checkfor("grid : TO CHANGE GRID SIZE"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `grid' (to select the DFT"
                                 + " Grid as input.)", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I can't find an explanation for "
                                 + "`grid' (to select the DFT grid) but "
                                 + "I have used it.", -3)

                    # The following check is  dangerous to behave unexpected if
                    # the  user  selects  a  functinoal whos  name  contains  a
                    # regular expression.  However, all legal functionals as of
                    # version 6.3.1 do not contain such charcters.
                    if self._checkfor("DFT is used functional " +
                                      self.settings.dft_functional):
                        self._report(sanitizeprompt + "`define' recognized my "
                                     + "request for functional `"
                                     + self.settings.dft_functional + "'.", 3)
                    else:
                        success = False
                        self._report(sanitizeprompt + "ERROR: `define's menu might "
                                     + "have changed. I said `func "
                                     + self.settings.dft_functional + "'. But he "
                                     + "just didn't care.", -3)

            # The following checks only appply if we are using RI.
            if self.settings.ri:
                # See in `define' still accepts the `ri' keyword.
                if self._checkfor("ri : RI Parameters"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `ri' as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I can't find an explanation for "
                                 + "`ri' but I have used it.", -3)

                if self._checkfor("on: TO SWITCH ON  RI"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `on' (for RI) as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I can't find an explanation for "
                                 + "`on' (with RI) but I have used it.", -3)

                if self._checkfor("m: CHANGE MEMORY FOR RI"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `m' (for RI) as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I can't find an explanation for "
                                 + "`m' (with RI) but I have used it.", -3)

                # See if `define' conformed our RI memory setting.
                if self._checkfor("Memory for RI: " + str(self.settings.ri_memory)
                                  + "Mb"):
                    self._report(sanitizeprompt + "`define' confirmed using "
                                 + str(self.settings.ri_memory) + " MB for RI as "
                                 + "expected.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define' never confirmed my "
                                 + "settings for the RI memory.", -3)

            # The following checks only apply if we are running on MP2 level.
            if self.settings.mp2:
                if (self._checkfor("cc : OPTIONS AND DATA GROUPS FOR ricc2") and
                        self._checkfor("INPUT MENU FOR CALCULATIONS WITH ricc2")):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `cc' as keyword to enter the "
                                 + "`ricc2' menu.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I couldn't enter the `ricc2' "
                                 + "menu.", -3)
                # Note the escape of `(' and `)'.
                if (self._checkfor(r"cbas : ASSIGN AUXILIARY \(CBAS\) BASIS SETS") and
                        self._checkfor("AUXILIARY BASIS SET DEFINITION MENU")):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `cbas' as keyword to assign auxiliary "
                                 + "basis sets", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I couldn't enter the `cbas' "
                                 + "menu.", -3)
                # Note the escape of `(' and `)'.
                if (self._checkfor(r"memory : SET MAXIMUM CORE MEMORY") and
                        self._checkfor(r"memory which should be used")):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `memory' as keyword to enter the "
                                 + "menu to specify the memory for MP2.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I couldn't enter the menu to "
                                 + "set the memory for MP2.", -3)
                # Note the escape of `$', `(' and `)'.
                if (self._checkfor(r"ricc2 : DATA GROUP \$ricc2 \(MODELS AND GLOBAL OPTIONS\)") and
                        self._checkfor(r"\*,end : write \$ricc2 to file and leave the menu")):
                    self._report(sanitizeprompt + "`define' still seems to "
                                 + "accept `ricc2' as keyword to enter the "
                                 + "menu for `ricc2' models and global options.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "ERROR: `define's menu might have "
                                 + "changed. I couldn't enter the menu to "
                                 + "specify `riccs's models and global options.", -3)

            # Look for  the final statement. The number  of asterisks shouldn't
            # bother us.
            if self._checkfor(r"\*+  define : all done  \*+"):
                self._report(sanitizeprompt + "I found the familiar "
                             + "`**** define : all done ****' statement.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "ERROR: I'm missing the familiar "
                             + "`**** define : all done ****' statement.", -3)

        except Exception as e:
            # Make any exception during the process of sanitization a reason to
            #  return `False'.
            self._report("ERROR: There was an unexpected exception caught "
                         + "while checking `define's output. It says: "
                         + str(e), -2)
            raise

        finally:
            # Get rid of the temporary variable before returning.
            del self.temp

        return success

    def _checkfor(self, pattern):
        """
        Sees if C{pattern} occurs in C{define_stdout}.

        Both strings are "compacted" (using L{_compact}) first. This method will
        always return C{False} if C{self.temp} has not been defined to be the
        compacted version of C{self.define_stdout}.

        @returns: C{True} if the pattern occurs, otherwise C{False}.
        @rtype:   L{bool}

        """

        pattern = self._compact(pattern)
        pattern = re.compile(pattern)
        try:
            if re.search(pattern, self.temp):
                return True
            else:
                return False
        except NameError:
            # <self.temp> not defined!
            return False

    @staticmethod
    def _compact(text):
        """
        Converts C{text} to all lowercase and removes any sequence of
        whitespce (including line breaks).

        Use this method to compare strings for "similarity".

        @param text: Text to compact
        @type  text: L{str}
        @returns:    Compacted text
        @rtype:      L{str}
        """
        whitespace_pattern = re.compile(r'\s+')  # includes `\n'
        text = re.sub(whitespace_pattern, '', text)
        text = text.lower()
        return text

    def _assembleInput(self):
        """
        Generates the input string to be passed to I{define}.

        @returns: Input sequence
        @rtype:   L{str}

        """

        sequence = self._assembleDontReadExistingControlFileInputSequence()
        sequence.extend(self._assembleSetTitleInputSequence())
        sequence.extend(self._assembleGeometryMenuInputSequence())
        sequence.extend(self._assembleAtomicDefinitionMenuInputSequence())
        sequence.extend(self._assembleMolecularOrbitalMenuInputSequence())
        sequence.extend(self._assembleGeneralMenuInputSequence())

        inputstring = ''
        for word in sequence:
            inputstring += word + '\n'
        return inputstring

    @staticmethod
    def _assembleDontReadExistingControlFileInputSequence():
        """
        Generates an input sequence to be passed to I{define} making it not use
        an existing C{control} file.

        @returns: Input sequence
        @rtype:   L{list}
        """
        sequence = ['']
        return sequence

    @staticmethod
    def _assembleSetTitleInputSequence():
        """
        Generates an input sequence to be passed to I{define} making it
        forget the idea about a title whatsoever.

        @returns: Input sequence
        @rtype:   L{list}
        """
        sequence = ['']
        return sequence

    def _assembleGeometryMenuInputSequence(self):
        """
        Generates an input sequence to be passed to I{define}'s "SPECIFICATION
        OF MOLECULAR GEOMETRY" menu.

        @returns: Input sequence
        @rtype:   L{list}
        """

        sequence = ['a ' + self.coordfilename]

        if self.settings.idef:
            sequence.append('idef')
            for icoord in self.settings.idef_list:
                sequence.append(icoord)
            sequence.append('')
            sequence.append('')
            sequence.append('')
            self.settings.ired = True

        if self.settings.ired:
            sequence.append('ired')
            sequence.append('*')
        else:
            sequence.append('*')
            sequence.append('no')
        return sequence

    def _assembleAtomicDefinitionMenuInputSequence(self):
        """
        Generates an input sequence to be passed to I{define}'s "ATOMIC
        ATTRIBUTE DEFINITION MENU".

        @returns: Input sequence
        @rtype:   L{list}

        """

        # TODO: IMPLEMENT  FULL   FUNCTIONALITY  OF  THIS  `define'  MENU  LIKE
        #       ASSIGNING DIFFERENT BASIS  SETS TO DIFFERENT ATOMS, SPECIFIEING
        #       NON-STANDARD  ATOMIC MASSES,  USING  EFFECTIVE CORE  POTENTIALS
        #       ETC.

        sequence = ['b', 'all ' + self.settings.basis_set_all, '*']
        return sequence

    def _assembleMolecularOrbitalMenuInputSequence(self):
        """
        Generates an input sequence to be passed to I{define}'s "OCCUPATION
        NUMBER & MOLECULAR ORBITAL DEFINITION MENU" menu.

        @returns: Input sequence
        @rtype:   L{list}

        """

        # TODO: IMPLEMENT ALL METHODS  TO GUESS THE INITIAL OCCUPATION SUPPLIED
        #       BY `define'.

        sequence = []
        if str(self.settings.guess_initial_occupation_by) == 'eht':
            sequence.append('eht')
            sequence.append('y') # sometimes there is more than one SUITED DEFINITION
            sequence.append('y')  # use default (typing y once instead of charge does no harm)
            sequence.append(str(self.settings.charge))
            # Closed-shell
            if not self.settings.unrestricted:
                # Accept default occupation
                if self._countAtoms(self.coordfilename) > 1:
                    sequence.append('y')
                # Decline automatic atomic occupation
                else:
                    sequence.append('n')
            # Unrestricted
            else:
                # Decline automatic atomic occupation
                if self._countAtoms(self.coordfilename) == 1:
                    sequence.append('n')
                # Setting spin manually
                sequence.append('n')
                sequence.append('u %i' % self.settings.spin)
                sequence.append('*')
                sequence.append('n')  # additional question about natural orbitals for unrestricted calculations
        else:
            raise PyAdfError(("I'm not programmed to handle the case "
                              + "`initial occupation guessed by: {0}'. "
                              + "Sorry.").format(self.settings.guess_initial_occupation_by))
        return sequence

    def _assembleGeneralMenuInputSequence(self):
        """
        Generates an input sequence to be passed to I{define}s "GENERAL
        MENU".

        @returns: Input sequence
        @rtype:   L{list}

        """

        sequence = []
        if self.settings.dft:
            sequence.append('dft')
            sequence.append('on')
            sequence.append('func ' + self.settings.dft_functional)
            sequence.append('grid ' + self.settings.dft_grid)
            sequence.append('')
        if self.settings.ri:
            sequence.append('ri')
            sequence.append('on')
            sequence.append('m ' + str(self.settings.ri_memory))
            sequence.append('')
        if self.settings.mp2:
            sequence.append('cc')
            sequence.append('cbas')
            sequence.append('*')
            sequence.append('memory')
            sequence.append(str(self.settings.cc_memory))
            sequence.append('ricc2')
            sequence.append('mp2')
            if self.settings.mp2:
                sequence.append('geoopt mp2')
            sequence.append('*')
            sequence.append('*')
        sequence.append('*')
        return sequence

    @staticmethod
    def _countAtoms(coordfilename):
        """
        Counts and returns the number of atoms in the C{coord} file.

        File I/O
        exceptions are reraised.

        @param coordfilename: File name to count through
        @type  coordfilename: L{str}
        @returns:             Number of atoms
        @rtype:               L{int}
        @raises PyAdfError:   If the file can be read but doesn't match its
                              definition.
        @deprecated:          Should use the L{molecule} class' functionality
                              instead.
        """

        i = float('NaN')  # number of atoms
        with open(coordfilename) as coordfile:
            for line in coordfile:
                if re.match(r'\$coord.*', line):
                    i = -1  # this line doesn't count
                if i > 0 and re.match(r'\$.*', line):
                    return i
                i += 1
            raise PyAdfError("The file `" + coordfilename + "' is corrupted.")
