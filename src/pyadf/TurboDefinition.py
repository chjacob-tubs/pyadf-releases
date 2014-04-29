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

from Errors import PyAdfError

class TurboObject(object):
    
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
        with open(self.logfilename, 'a') as logfile:
            logfile.write(message + '\n')
    
    
    def _reportGoodNews(self, message):
        
        """
        Prints to I{stdout}.
        
        @param message: Text to show
        @type  message: L{str}
        
        """
        
        print message
    
    
    def _reportBadNews(self, message):
        
        """
        Prints to I{stderr}.
        
        @param message: Text to show
        @type  message: L{str}
        
        """
        
        sys.stderr.write(message + '\n')



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
        
        super(TurboDefinition, self).__init__(verbose_level=settings.verbose_level)
        self.settings = settings
        
        self.define_stdin    = ''
        self.define_stdout   = ''
        self.define_stderr   = ''
        
        try:
            self.setCoordFile(self.settings.coordfilename)
        except:
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
    
        returncode      = None
        tmdefine_status = None
        sanitize_status = None
        
        try:
            # Fire off the process.
            returncode = self._tmdefine()
            tmdefine_status = (returncode == 0)
            if not tmdefine_status:
                info = "`define' quit on error. Skipping sanitization."
                self._report(info, -2)
                raise PyAdfError(info)
            
            # Postprocess the `control file.
            self._postprocess()
            
            # Check the results.
            sanitize_status = self._sanitize()
            
        except Exception as e:
            self._report(str(e), -2)
        
        finally:
            # Write our I/O  to / from `define' to log  files.  We convert them
            # to  strings  explicitly  to  account  for  the  possibility  that
            # `define' crashed and thy are `None' rather than strings.
            
            with open('define_stdin' + os.extsep + 'log', 'w') as report:
                report.write(str(self.define_stdin))
            with open('define_stdout' + os.extsep + 'log', 'w') as report:
                report.write(str(self.define_stdout))
            with open('define_stderr' + os.extsep + 'log', 'w') as report:
                report.write(str(self.define_stderr))            
            
            if tmdefine_status:
                self._report("`define' successfully quit on exit status " 
                            + str(returncode) + ".", 2)
                if sanitize_status:
                    self._report("I've checked `define's output and it looked fine. "
                                + "This is certainly no gurantee that it really "
                                + "did the right thing. If I were able to check "
                                + "this, I needn't even call it.", 2)
                else:
                    self._report("I've checked the results and they don't "
                                + "seem quite okay. I'll refuse "
                                + "accepting this as a success.", -2)
            else:
                self._report("`define' quit on exit status `" 
                            + str(returncode) + "'. I didn't even look at "
                            + "the results.", -2)
        
            if (tmdefine_status and sanitize_status):
                return returncode
            else:
                message = ("Unable to run `define'. There is no point "
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
        
        # Generate the input sequence to be passed.
        self.define_stdin = self._assembleInput()
        
        try:
            # Create the subprocess
            D = Popen([ 'define' ], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            
            # Pass it the input and wait for it to finish.
            self.define_stdout, self.define_stderr = D.communicate(input=self.define_stdin)
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
            return
        elif self.settings.disp == 'dft-d1':
            toadd += '$olddisp' + '\n'
        elif self.settings.disp == 'dft-d2':
            toadd += '$disp' + '\n'
        elif self.settings.disp == 'dft-d3':
            toadd += '$disp3' + '\n'
        else:
            raise PyAdfError("Unknown value `" + str(self.settings.disp) + "' for dispersion correction.")
       
        if self.settings.scfiterlimit is None:
            return
        else:
            toadd += '$scfiterlimit ' + str(self.settings.scfiterlimit) + '\n'

        tf = tempfile.NamedTemporaryFile(mode='a', delete=False)
        with open('control', 'r') as infile:
            for line in infile:
                line = line.replace('$end', toadd + '$end')
                tf.write(line + '\n')
        tf.file.close()
        open('control', 'w').close()
        with open(tf.name, 'r') as infile:
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
            sanitizeprompt = "sanitize: "
            
            # The very  first thing  to do is  to look  at the finaly  words of
            # `define'.  I have no clue why, but `define' ALWAYS writes them to
            # `stderr'.
        
            if re.search("define ended normally", self.define_stderr):
                self._report(sanitizeprompt + "`define' claimes to have ended "
                            + "normally.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "`define' says he ended "
                            + "abnormally.", -3)
            
            # Next we see if the  standard files `control', `coord' and `basis'
            # are  present  and look  "normal".   If  we  (erm, `define')  have
            # guessed  some initial  occupation, there  should also  be  a file
            # `mos'  and with  `ri' we  expect a  file `auxbasis'.   If  we use
            # redundant coordinates, the `coord' file should show that too.  We
            # use  a dictionary  to  run a  VERY  rudimentary check,  if a  few
            # characteristic words appear in  the files.  This are the keywords
            # we look for.  (A `$' will be prepended to each.)
            
            control_keywords  = [ 'title', 'coord', 'atoms', 'basis', 'end' ]
            coord_keywords    = [ 'coord', 'end' ]
            if self.settings.ired:
                coord_keywords.append('redundant')
            basis_keywords    = [ 'basis', 'end' ]
            auxbasis_keywords = [ 'jbas', 'end' ]
            mos_keywords      = [ 'scfmo', 'end' ]
            
            keywords = { 'control'  : control_keywords, 
                         'coord'    : coord_keywords, 
                         'basis'    : basis_keywords, 
                         'auxbasis' : auxbasis_keywords, 
                         'mos'      : mos_keywords }
            
            checkfilenames = [ 'control', 'coord', 'basis' ]
            if self.settings.guess_initial_occupation_by is not None:
                checkfilenames.append('mos')
            if self.settings.ri:
                checkfilenames.append('auxbasis')
            
            for checkfilename in checkfilenames:
                if os.path.isfile(checkfilename):                    
                    self._report(sanitizeprompt + "File `" + checkfilename 
                                + "' exists.", 3)
                    with open(checkfilename, 'r') as checkfile:
                        contents = checkfile.read()
                        for keyword in keywords[checkfilename]:
                            if '$' + keyword in contents:
                                pass
                            else:
                                success = False
                                self._report(sanitizeprompt + "File `"
                                            + checkfilename 
                                            + "' doesn't look as I "
                                            + "expected it to.", -3)
                        self._report(sanitizeprompt + "File `" + checkfilename 
                                    + "' looks fine at a first glance. (I've "
                                    + "been checking for these variables: "
                                    + str(keywords[checkfilename]) + ".)", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "File `" + checkfilename + 
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
            if self._checkfor("TURBOMOLE V6\.3"):
                self._report(sanitizeprompt + "Found `turbomole 6.3' as "
                            + "expected.", 3)
            else:
                self._report(sanitizeprompt + "Found wrong " + "`turbomole' "
                            + "version (expecting 6.3.x). But I don't "
                            + "mind.", -3)
            
            # See if `define' states to have added the correct number of atoms:
            if self._checkfor("CARTESIAN COORDINATES FOR "
                             + str(self.atom_checksum) + " ATOMS HAVE "
                             + "SUCCESSFULLY BEEN ADDED"):
                self._report(sanitizeprompt + "`define' claims to have added "
                            + str(self.atom_checksum) + " atoms as expected.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "I expected `define' to add " 
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
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I can't find an explanation for "
                                + "`ired' but I have used it.", -3)
                
                # See if internal coordinates have been written. (This check is
                # somewhat  redundant since  we've allready  been  checking the
                # `coord'  file  but  since  we're  talking  about  "redundant"
                # coordinates,  this  is   probably  okay.   Version  6.3.1  of
                # `turbomole' said  "write onto file".  Since this  is a little
                # strange  and hence likely  to change,  we also  accept "write
                # into file".
                if self._checkfor(" writing data block \$user-defined bonds "
                                 + "(o|i)nto file <coord>"):
                    self._report(sanitizeprompt + "`define' says, he wrote " 
                                + "user-defined bonds to the `coord' file.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "Strange, `define' says "
                                + "nothing about user-defined bonds.", -3)
            
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
                self._report(sanitizeprompt + "`define' doesn't say anything "
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
                self._report(sanitizeprompt + "`define's menu might have "
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
                self._report(sanitizeprompt + "`define' didn't know the basis "
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
                self._report(sanitizeprompt + "`define' doesn't say anything "
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
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I can't find an explanation for "
                                + "`eht' but I have used it.", -3)
                
                # Note that we have to escape the parenthesis.
                if self._checkfor("ENTER THE MOLECULAR CHARGE \(DEFAULT=0\)"):
                    self._report(sanitizeprompt + "`define' asked for the chagre "
                                + "as I expected.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I wasn't asked for the molecules "
                                + "charge.", -3)
                
                if self._checkfor("AUTOMATIC OCCUPATION NUMBER ASSIGNMENT "
                                 + "ESTABLISHED"):
                    self._report(sanitizeprompt + "`define' says, he found an "
                                + "initial guess for the occupation.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define' doesn't say anything "
                                + "about successfully determined occupation.", -3)
                
                # The default value is not arethesized here (as commonly done).
                # We will accept both.  (Note the `?'  is escaped once!)
                if self._checkfor("DO YOU ACCEPT THIS OCCUPATION\? "
                                 + "\(?DEFAULT=y\)?"):
                    self._report(sanitizeprompt + "`define' was asking me to "
                                + "accept the occupation just as expected.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define' didn't ask me to "
                                + "accept the occupation but I said blindly "
                                + "`yes'!", -3)
    
            # Look for the truth.
            if self._checkfor("God is great, beer is good and people "
                             + "are crazy"):
                self._report(sanitizeprompt + "`define' is making true "
                            + "statements about beer and people.", 3)
            
            # The following checks only appply if we are using DFT.
            if self.settings.dft:
                # See in `define' still accepts the `dft' keyword.
                if self._checkfor("dft : DFT Parameters"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                + " accept `dft' as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I can't find an explanation for "
                                + "`dft' but I have used it.", -3)
                
                if self._checkfor("on: TO SWITCH ON DFT"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                + "accept `on' (for DFT) as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I can't find an explanation for "
                                + "`on' (with DFT) but I have used it.", -3)
                
                if self._checkfor("func : TO CHANGE TYPE OF FUNCTIONAL"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                + "accept `func' (to select the DFT "
                                + " functional as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I can't find an explanation for "
                                + "`func' (to select the DFT functional) but "
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
                        self._report(sanitizeprompt + "`define's menu might "
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
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I can't find an explanation for "
                                + "`ri' but I have used it.", -3)
    
                if self._checkfor("on: TO SWITCH ON  RI"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                + "accept `on' (for RI) as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I can't find an explanation for "
                                + "`on' (with RI) but I have used it.", -3)
                
                if self._checkfor("m: CHANGE MEMORY FOR RI"):
                    self._report(sanitizeprompt + "`define' still seems to "
                                + "accept `m' (for RI) as input.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define's menu might have "
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
                    self._report(sanitizeprompt + "`define' never confirmed my "
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
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I couldn't enter the `ricc2' "
                                + "menu.", -3)
                # Note the escape of `(' and `)'.
                if (self._checkfor("cbas : ASSIGN AUXILIARY \(CBAS\) BASIS SETS") and
                    self._checkfor("AUXILIARY BASIS SET DEFINITION MENU")):
                    self._report(sanitizeprompt + "`define' still seems to "
                                + "accept `cbas' as keyword to assign auxiliary "
                                + "basis sets", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I couldn't enter the `cbas' "
                                + "menu.", -3)
                # Note the escape of `(' and `)'.
                if (self._checkfor("memory : SET MAXIMUM CORE MEMORY") and
                    self._checkfor("memory which should be used \(in MB\)")):
                    self._report(sanitizeprompt + "`define' still seems to "
                                + "accept `memory' as keyword to enter the "
                                + "menu to specify the memory for MP2.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I couldn't enter the menu to "
                                + "set the memory for MP2.", -3)
                # Note the escape of `$', `(' and `)'.
                if (self._checkfor("ricc2 : DATA GROUP \$ricc2 \(MODELS AND GLOBAL OPTIONS\)") and
                    self._checkfor("\*,end : write \$ricc2 to file and leave the menu")):
                    self._report(sanitizeprompt + "`define' still seems to "
                                + "accept `ricc2' as keyword to enter the "
                                + "menu for `ricc2' models and global options.", 3)
                else:
                    success = False
                    self._report(sanitizeprompt + "`define's menu might have "
                                + "changed. I couldn't enter the menu to "
                                + "specify `riccs's models and global options.", -3)
            
            # Look for  the final statement. The number  of asterisks shouldn't
            # bother us.
            if self._checkfor("\*+  define : all done  \*+"):
                self._report(sanitizeprompt + "I found the familiar "
                            + "`**** define : all done ****' statement.", 3)
            else:
                success = False
                self._report(sanitizeprompt + "I'm missing the familiar "
                            + "`**** define : all done ****' statement.", -3)
        
        except Exception as e:
            # Make any exception during the process of sanitization a reason to
            #  return `False'.
            self._report("There was an unexpected exception caught "
                        + "while sanitizing `define's output. It says: " 
                        + str(e), -2)
            success = False
        
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
    
    
    def _compact(self, text):
        
        """
        Converts C{text} to all lowercase and removes any sequence of
        whitespce (including line breaks).
        
        Use this method to compare strings for "similarity".
        
        @param text: Text to compact
        @type  text: L{str}
        @returns:    Compacted text
        @rtype:      L{str}
        @deprecated: Redundant, c.f. L{_compressPattern}
        
        """
        
        whitespace_pattern = re.compile('\s+') # includes `\n'
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
    
    
    def _assembleDontReadExistingControlFileInputSequence(self):
        
        """
        Generates an input sequence to be passed to I{define} making it not use
        an existing C{control} file.
        
        @returns: Input sequence
        @rtype:   L{str}
        
        """
        
        sequence = []
        sequence.append('') # skip it...
        return sequence
    
    
    def _assembleSetTitleInputSequence(self):
        
        """
        Generates an input sequence to be passed to I{define} making it
        forget the idea about a title whatsoever.
        
        @returns: Input sequence
        @rtype:   L{str}
        
        """
        
        sequence = []
        sequence.append('') # skip it...
        return sequence
    
    
    def _assembleGeometryMenuInputSequence(self):
        
        """
        Generates an input sequence to be passed to I{define}'s "SPECIFICATION
        OF MOLECULAR GEOMETRY" menu.
        
        @returns: Input sequence
        @rtype:   L{str}
        
        """
        
        sequence = []
        sequence.append('a ' + self.coordfilename)
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
        @rtype:   L{str}
        
        """
        
        # TODO: IMPLEMENT  FULL   FUNCTIONALITY  OF  THIS  `define'  MENU  LIKE
        #       ASSIGNING DIFFERENT BASIS  SETS TO DIFFERENT ATOMS, SPECIFIEING
        #       NON-STANDARD  ATOMIC MASSES,  USING  EFFECTIVE CORE  POTENTIALS
        #       ETC.
        
        sequence = []
        sequence.append('b')
        sequence.append('all ' + self.settings.basis_set_all)
        sequence.append('*')
        return sequence
    
    
    def _assembleMolecularOrbitalMenuInputSequence(self):
        
        """
        Generates an input sequence to be passed to I{define}'s "OCCUPATION
        NUMBER & MOLECULAR ORBITAL DEFINITION MENU" menu.
        
        @returns: Input sequence
        @rtype:   L{str}
        
        """
        
        # TODO: IMPLEMENT ALL METHODS  TO GUESS THE INITIAL OCCUPATION SUPPLIED
        #       BY `define'.
        
        sequence = []
        if str(self.settings.guess_initial_occupation_by) == 'eht':
            sequence.append('eht')
            sequence.append('y') # use default
            sequence.append(str(self.settings.charge))
            sequence.append('y') # blindly accept
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
        @rtype:   L{str}
        
        """
        
        sequence = []
        if self.settings.dft:
            sequence.append('dft')
            sequence.append('on')
            sequence.append('func ' + self.settings.dft_functional)
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
    
    
    def _compressPattern(self, pattern):
        
        """
        Strip whitespace and the like from a string.
        
        Converts a string into a form such that two initially not necessairly
        identical string which -- for humans -- would look equivalent in
        meaning are more likely to match afterwards. The returned object is a
        String not a compiled regexp!
        
        @param pattern: Pattern with whitespace, mixed case, etc.
        @type  pattern: L{str}
        @returns:       Compressed Pattern
        @rtype:         L{str}
        @deprecated:    Redundant, c.f. L{_compact}
        
        """
        
        # It is likely that future versions of `define' have minor changes such
        # as more / less indent or  spacing. Giving up just because of an extra
        # blank would be cowardy.
        pattern = re.sub('\s', '', pattern)
        
        # The same is true for capitalization.
        pattern = pattern.lower()
        
        return pattern
    
    
    def _countAtoms(self, coordfilename):
        
        """
        Counts and returns the number of atoms in the C{coord} file.
        
        File I/O
        exceptions re reraised.
        
        @param coordfilename: File name to count through
        @type  coordfilename: L{str}
        @returns:             Number of atoms
        @rtype:               L{int}
        @raises PyAdfError:   If the file can be read but doesn't match its
                              definition.
        @deprecated:          Should use the L{molecule} class' functionality
                              instead.
        
        """
        
        i = float('NaN') # number of atoms
        with open(coordfilename, 'r') as coordfile:
            for line in coordfile:
                if re.match('\$coord.*', line):
                    i = -1 # this line doesn't count
                if i > 0 and re.match('\$.*', line):
                    return i
                i += 1
            raise PyAdfError("The file `" + coordfilename
                            + "' is corrupted.")
