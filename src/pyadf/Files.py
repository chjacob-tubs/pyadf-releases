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
 The file manager L{adf_filemanager} which is used internally.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @undocumented: newjobmarker

"""
__all__ = ['filemanager', 'adf_filemanager']

from .Errors import PyAdfError
from .Utils import newjobmarker
from .PatternsLib import Singleton
import os
import shutil
import pickle


class filemanager(metaclass=Singleton):
    """
    Base class for file managers.

    This class implements a basic file manager that can be used for different
    types of calculations.
    The file manager has the following tasks:

        1. it manages the output and error files used for the calculations.

        2. it manages files produced by the calculations by keeping track
           of their path and file name. The file manager further performs
           simple file-related tasks like copying, moving and deleting.

    @warning:
    the file manager always assumes that path and file names (except for output
    and error files) are given relative to the current working directory.
    The current working directory should not change while the file manager is used.

    @group Constructor:
        __init__
    @group Manage output and error files:
        set_outputfilename, get_outputfilename, get_errfilename
    @group Manage job files:
        add_file, have_file, rename_file, copy_file, delete_file, cleanup
    """

    def __init__(self, outdir=None, jobid=os.getpid()):
        """
        Construct a file manager instance.

        @param outdir: directory where the output and error files should be written.
        @type  outdir: str
        @param jobid: a (hopefully unique) job id that is used in the name of the output file.
                      Default is the PID of the current process.
        @type  jobid: str
        """
        self._files = set()  # set with all managed files
        if outdir is None:
            self._outdir = os.getcwd()
        else:
            self._outdir = outdir
        self._jobid = str(jobid)
        self._outfilename = ""
        self._errfilename = ""
        self.outputfilename = "pyadf_joboutput"
        self._cwd = os.getcwd()

    def __copy__(self):
        # never copy or deepcopy a filemanager
        return self

    def __deepcopy__(self, memo):
        # pylint: disable-msg=W0613
        # never copy or deepcopy a filemanager
        return self

    @property
    def outputfilename(self):
        """
        Name of the file where the output of the calculation is written.

        Assigning to C{outputfilename} will also set L{errfilename} accordingly.
        The filename is always prepended with the job id and the extension '.out',
        see L{set_outputfilename}
        """
        return self._outfilename

    @outputfilename.setter
    def outputfilename(self, outname):
        self._outfilename = os.path.join(self._outdir, outname + "." + self._jobid + ".out")
        self._errfilename = os.path.join(self._outdir, outname + "." + self._jobid + ".err")

    @property
    def errfilename(self):
        """
        Name of the file where the stderr of the calculation is written.

        This is read-only, because C{errfilename} is always set according to
        L{outputfilename}.
        """
        return self._errfilename

    def have_file(self, filename):
        """
        Return whether the given file exists in the file manager.

        @param filename: the file name
        @type  filename: str
        """
        return filename in self._files

    def change_to_basedir(self):
        os.chdir(self._cwd)

    def add_file(self, filename):
        """
        Add an existing file to the file manager.

        @param filename: the file name
        @type  filename: str
        """
        if not (os.getcwd() == self._cwd):
            print(os.getcwd(), self._cwd)
            raise PyAdfError("add_file not called in base working directory")
        if not os.path.exists(filename):
            raise PyAdfError("file " + filename + " not found")
        self._files.add(os.path.abspath(filename))

    def delete_file(self, filename):
        """
        Delete a managed file.

        @param filename: the file name
        @type  filename: str
        """
        if not os.path.abspath(filename) in self._files:
            raise PyAdfError("file " + filename + " not known")
        if not (os.getcwd() == self._cwd):
            raise PyAdfError("delete_file not called in base working directory")
        if os.path.exists(os.path.abspath(filename)):
            os.remove(os.path.abspath(filename))

    def rename_file(self, oldfilename, newfilename):
        """
        Rename a managed file.

        @param oldfilename: the old file name
        @type  oldfilename: str
        @param newfilename: the new file name
        @type  newfilename: str
        """
        if not (os.getcwd() == self._cwd):
            raise PyAdfError("rename_file not called in base working directory")
        if not os.path.abspath(oldfilename) in self._files:
            raise PyAdfError("file " + oldfilename + " not known")
        if not os.path.exists(oldfilename):
            raise PyAdfError("file " + oldfilename + " not found")
        if os.path.exists(newfilename):
            raise PyAdfError("file " + newfilename + " already exists")
        os.rename(oldfilename, newfilename)
        self._files.add(os.path.abspath(newfilename))
        self._files.remove(os.path.abspath(oldfilename))

    def copy_file(self, oldfilename, newfilename):
        """
        Copy a managed file.

        The copied file will B{not} be managed by the file manager.

        @param oldfilename: the old filename of the managed file
        @type  oldfilename: str
        @param newfilename: the new filename of the copied file
        @type  newfilename: str
        """
        if not os.path.join(self._cwd, oldfilename) in self._files:
            raise PyAdfError("file " + oldfilename + " not known")
        if not os.path.exists(os.path.join(self._cwd, oldfilename)):
            raise PyAdfError("file " + oldfilename + " not found")
        if os.path.exists(newfilename):
            raise PyAdfError("file " + newfilename + " already exists")
        shutil.copyfile(os.path.join(self._cwd, oldfilename), newfilename)

    def link_file(self, oldfilename, newfilename):
        """
        Make a symlink to a managed file.

        The symlink will B{not} be managed by the file manager.

        @param oldfilename: the old filename of the managed file
        @type  oldfilename: str
        @param newfilename: the new filename of the copied file
        @type  newfilename: str
        """
        if not os.path.join(self._cwd, oldfilename) in self._files:
            raise PyAdfError("file " + oldfilename + " not known")
        if not os.path.exists(os.path.join(self._cwd, oldfilename)):
            raise PyAdfError("file " + oldfilename + " not found")
        if os.path.exists(newfilename):
            raise PyAdfError("file " + newfilename + " already exists")
        os.symlink(os.path.join(self._cwd, oldfilename), newfilename)

    def cleanup(self):
        """
        Cleanup the file manager.

        This deletes all the managed files.
        """
        for f in self._files:
            self.delete_file(f)
        self.__class__.instance = None


class adf_filemanager(filemanager):
    """
    A file manager for ADF-related files.

    The following ADF-related files are managed:

        - The TAPE files and output files produced by ADF.
          These are organized by job id, a unique id that is assigned to them
          when they are added.

    The most important tasks for external users are:

        - L{copy_all_results_to_dir} for saving all results
        - L{import_resultsdir} for importing results

    Related to ADF results the most important methods are:

        - L{get_results_filename} to get the filename of a specific results file
        - L{copy_result_file} to copy a results file to the working directory
        - L{get_output} to get the output of an ADF calculation

    @group Constructor:
        __init__

    @group Manage result TAPE files:
        add_results, have_results,
        get_results_filename, copy_job_result_files, copy_result_file,
        get_output, get_id,
        copy_all_results_to_dir, import_resultsdir
    """

    def __init__(self, outdir=None, jobid=os.getpid()):
        """
        Construct an ADF file manager instance.

        @param outdir: directory where the output and error files should be written.
        @type  outdir: str
        @param jobid: a (hopefully unique) job id that is used in the name of the output file.
                      Default is the PID of the current process.
        @type  jobid: str
        """
        super().__init__(outdir, jobid)
        self._resultfiles = []  # list of lists of result files, id gives index
        self._output = []  # a list giving for each result the name of the
        #                               output file and first and last line of the output
        #                               result id gives index
        self._id = {}  # dictionary giving results id for job checksums
        self._ispacked = []
        os.mkdir('resultfiles')

    def cleanup(self):
        """
        Cleanup the file manager.

        This deletes all the managed files.
        """
        super().cleanup()
        shutil.rmtree('resultfiles', True)

    def add_outputfiles(self, results):
        """
        Add the output files produces by an ADF or Dalton job.
        """

        fileid = len(self._resultfiles)
        results.fileid = fileid

        # save checksum
        checksum = results.checksum

        if checksum in self._id:
            raise PyAdfError("Checksum error when adding results")
        self._id[checksum] = fileid

        # the output files

        if results.job is not None:
            outfilename = self.outputfilename

            f = open(outfilename, encoding='utf-8')
            outputstart = -1
            outputend = -1
            for i, ll in enumerate(f.readlines()):
                if ll == newjobmarker:
                    outputstart = i
                outputend = i
            if outputstart == -1:
                raise PyAdfError("PyADF marker not found in output")

            self._output.append((outfilename, outputstart, outputend))
        else:
            self._output.append(None)

    def add_resultfiles_as_tapes(self, fnlist, fileid):
        for filename, tapenr in fnlist:
            if os.path.exists(filename):
                self.add_file(filename)
                fn = f'resultfiles/t{tapenr:d}.results.{fileid:04d}'
                self.rename_file(filename, fn)
                self._resultfiles[fileid].append(fn)

    def add_orca_results(self, results):
        """
        Add the result files of a Orca job.
        """
        self.add_outputfiles(results)
        self._resultfiles.append([])
        self._ispacked.append(True)

        fnlist = [('INPUT.gbw', 21), ('INPUT_property.txt', 67), ('INPUT.engrad', 68),
                  ('INPUT.xyz', 69), ('INPUT_trj.xyz', 13), ('INPUT.molden.input', 41),
                  ('INPUT.mdci.optorb', 47), ('INPUT.scfp', 47), ('INPUT.hess', 42)]
        self.add_resultfiles_as_tapes(fnlist, results.fileid)

    def add_molcas_results(self, results):
        """
        Add the result files of a Molcas job.
        """
        self.add_outputfiles(results)
        self._resultfiles.append([])
        self._ispacked.append(True)

        fnlist = [('molcasjob.ScfOrb', 21), ('molcasjob.RasOrb', 22),
                  ('molcasjob.scf.molden', 41), ('xmldump', 66)]
        self.add_resultfiles_as_tapes(fnlist, results.fileid)

    def add_quantumespresso_results(self, results):
        """
        Add the result files of a QE job.
        """
        self.add_outputfiles(results)
        self._resultfiles.append([])
        self._ispacked.append(True)

        fnlist = [('pwscf.save.tar', 21), ('data-file.xml', 66), ('ppjob.cube', 10)]
        self.add_resultfiles_as_tapes(fnlist, results.fileid)

    def add_dalton_results(self, results):
        """
        Add the result files of a Dalton job.
        """
        self.add_outputfiles(results)
        self._resultfiles.append([])
        self._ispacked.append(True)

        self.add_resultfiles_as_tapes([('DALTON_MOLECULE.tar.gz', 21)], results.fileid)

    def add_dirac_results(self, results):
        """
        Add the result files of a Dirac job.
        """
        self.add_outputfiles(results)
        self._resultfiles.append([])
        self._ispacked.append(True)

        # we store DFCOEF as TAPE21, and GRIDOUT as TAPE10, dirac.xml as TAPE66
        fnlist = [('DFCOEF', 21), ('CHECKPOINT.h5', 22), ('CHECKPOINT.noh5.tar.gz', 23),
                  ('GRIDOUT', 10), ('dirac.xml', 66)]
        self.add_resultfiles_as_tapes(fnlist, results.fileid)

    def add_nwchem_results(self, results):
        """
        Add the result files of a NWChem job.
        """
        self.add_outputfiles(results)
        self._resultfiles.append([])
        self._ispacked.append(True)

        # we store NWCHEM.db as TAPE21, gridpts.0 as TAPE10 and molden as TAPE41
        fnlist = [('NWCHEM.db', 21), ('NWCHEM.gridpts.0', 10), ('NWCHEM.molden', 41)]
        self.add_resultfiles_as_tapes(fnlist, results.fileid)

    def add_turbomole_results(self, results):
        """
        Add the result files (yes, plural!) of a I{Turbomole} job.

        Adds a C{tar} archive to the result files containing all files from the
        I{Turbomole} working directory. This archive is generated in TurbomoleJob's
        after_run method.

        @param results: Results object from the I{Turbole} job.
        @type  results: L{TurbomoleResults}
        @author:        Moritz Klammer
        @date:          Aug. 2011

        """
        self.add_outputfiles(results)
        self._resultfiles.append([])
        self._ispacked.append(True)

        self.add_resultfiles_as_tapes([('archive.tar', 21)], results.fileid)

    def add_scm_results(self, results):
        """
        Add the result files of an ADF job.

        The TAPE files in the working directory will be added to the
        file manager. The id the results are stored under will be stored
        in the passed results object. This results object will usually
        also be used for accessing the results, see methods of L{adfresults}.

        In addition, the file manager will also remember the name of the
        output file of the job and the corresponding line numbers.

        @param results: the results object of the job that was run
        @type  results: derived from L{adfresults}
        """

        self.add_outputfiles(results)

        # now the tape files
        self._resultfiles.append([])
        self._ispacked.append(False)

        fnlist = [('TAPE21', 21), ('TAPE10', 10), ('TAPE41', 41)]
        self.add_resultfiles_as_tapes(fnlist, results.fileid)

    def add_dftb_results(self, results):
        """
        Add the result files of an AMS DFTB job.

        The rkf files in the working directory will be added to the
        file manager. The id the results are stored under will be stored
        in the passed results object. This results object will usually
        also be used for accessing the results, see methods of L{adfresults}.

        In addition, the file manager will also remember the name of the
        output file of the job and the corresponding line numbers.

        @param results: the results object of the job that was run
        @type  results: derived from L{amsresults}
        """

        self.add_outputfiles(results)

        # now the tape files
        self._resultfiles.append([])
        self._ispacked.append(False)

        fnlist = [('ams.rkf', 13), ('dftb.rkf', 21)]
        self.add_resultfiles_as_tapes(fnlist, results.fileid)

    def add_adf_results(self, results):
        """
        Add the result files of an AMS ADF job.

        The rkf files in the working directory will be added to the
        file manager. The id the results are stored under will be stored
        in the passed results object. This results object will usually
        also be used for accessing the results, see methods of L{adfresults}.

        In addition, the file manager will also remember the name of the
        output file of the job and the corresponding line numbers.

        @param results: the results object of the job that was run
        @type  results: derived from L{amsresults}
        """

        self.add_outputfiles(results)

        # now the tape files
        self._resultfiles.append([])
        self._ispacked.append(False)

        fnlist = [('ams.rkf', 13), ('adf.rkf', 21), ('TAPE10', 10)]
        self.add_resultfiles_as_tapes(fnlist, results.fileid)

    def add_results(self, results):
        """
        Add results.

        Based on the type of the results argument, ADF, Dalton, or Dirac results are added
        """
        from .ADFBase import scmresults, adfresults
        from .ADFSinglePoint import adfsinglepointresults
        from .ADF_DFTB import dftbresults
        from .DaltonSinglePoint import daltonresults
        from .Orca import OrcaResults
        from .Molcas import MolcasResults
        from .Dirac import diracresults
        from .NWChem import nwchemresults
        from .Turbomole import TurbomoleResults
        from .QuantumEspresso import QEResults

        if isinstance(results, adfsinglepointresults):
            self.add_adf_results(results)
        elif isinstance(results, dftbresults):
            self.add_dftb_results(results)
        elif isinstance(results, scmresults):
            self.add_scm_results(results)
        elif isinstance(results, daltonresults):
            self.add_dalton_results(results)
        elif isinstance(results, diracresults):
            self.add_dirac_results(results)
        elif isinstance(results, OrcaResults):
            self.add_orca_results(results)
        elif isinstance(results, MolcasResults):
            self.add_molcas_results(results)
        elif isinstance(results, nwchemresults):
            self.add_nwchem_results(results)
        elif isinstance(results, TurbomoleResults):
            self.add_turbomole_results(results)
        elif isinstance(results, QEResults):
            self.add_quantumespresso_results(results)
        elif isinstance(results, adfresults):
            pass
        else:
            raise PyAdfError("Unknown results class in add_results")

    def have_results(self):
        """
        Return whether there are any results in the file manager.
        """
        return len(self._files) > 0

    def get_id(self, checksum):
        """
        Get the results file id from the job checksum.

        If no results exist for the given checksum, C{None} is reurned.

        @param checksum: the job checksum
        @type  checksum: str
        """

        if checksum is None:
            return None
        elif checksum in self._id:
            return self._id[checksum]
        else:
            return None

    def get_results_filename(self, fileid, tape=21, no_check=False):
        """
        Returns the file name of a results TAPE file.

        @param fileid: the file id of the job results
        @type  fileid: int
        @param tape: the number of the requested TAPE file (default: TAPE21)
        @type  tape: int
        @param no_check: if True, check that the file actually exists
        @type no_check: bool
        """
        # tape: number of the tape to get

        fn = os.path.join(self._cwd, 'resultfiles', f't{tape:2d}.results.{fileid:04d}')

        if self.have_file(fn) or no_check:
            return fn
        else:
            raise PyAdfError("results file not found")

    def get_tempfile_from_archive(self, fileid, filename, tape=21):
        """
        Access a result file from the archived (.tar.gz) results.

        Extracts a result file from the archived data and writes its contents
        to a temporary file. The file name of this temporary file is returned
        and may be used to open / read as if it were the original file. You're
        self responsible to delete the temporary file afterwards, once you
        don't need it any longer.

        If the C{filename} can't be found in the archive, L{None} will be
        returned and no exception will raise.

        @param fileid: the file id of the job results
        @type  fileid: int
        @param filename: Name of the file to extract. (E.g. C{energy}.)
        @type  filename: L{str}
        @param tape: the number of the requested TAPE file containing the archive.
        @type  tape: int
        @returns:        Absolute path to the temporary file.
        @rtype:          L{str}
        """

        # comment from original Turbomole version of this method by Moritz Klammler
        #
        # We want a  somewhat wicked thing from this method:  It should give us
        # access to a file  in a `tar' archive but we don't  want to care about
        # that fact. Python  can read files from `tar' archives  as long as the
        # corresponding `TarFile' is open. If  we simply not close it, we might
        # risk the  archive with our  entire results becoming  inaccessible. We
        # could read  the content of the  archived file to memory  and return a
        # string.  But  that would be very  inconvenient if  the  user wants to
        # iterate  over the  lines  of  the file.   (She  would then,  instead,
        # iterate over the characters of the string maybe without even noting.)
        # A solid solution is to write the file of interest to an external file
        # such  that the user  can re-read  its content  from there.  But where
        # should we write  such a file? Can  we trust that there will  not be a
        # file named,  say, `energy' in  the CWD? Maybe  it isn't but  who will
        # think of this when someday  PyADF's file manager changes? What if two
        # jobs run  in parallel and  want to access  the same result  file from
        # different  computations  simultanously?  Python's  `tempfile'  module
        # provides a nice tool for that. The solution implemented is to write a
        # temporary file  and return its absolute path. Hence insetead of
        #
        # filename = 'foo.txt'
        # for line in open(filename, 'r'):
        #     print line
        #
        # a user would write
        #
        # filename = get_temp_result_filename('foo.txt')
        # for line in open(filename, 'r'):
        #     print line
        # os.remove(filename)

        import tempfile

        content = self.read_file_from_archive(fileid, filename, tape=tape)
        if content is not None:
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            temp_file.file.write(content)
            temp_file.file.close()

            return temp_file.name
        else:
            return None

    def read_file_from_archive(self, fileid, filename, tape=21):
        """
        Access a result file from the archived (.tar.gz) results.

        Extracts a result file from the archived data and reads its contents.

        If the C{filename} can't be found in the archive, L{None} will be
        returned and no exception will raise.

        @param fileid: the file id of the job results
        @type  fileid: int
        @param filename: Name of the file to extract. (E.g. C{energy}.)
        @type  filename: L{str}
        @param tape: the number of the requested TAPE file containing the archive.
        @type  tape: int
        @returns:        Absolute path to the temporary file.
        @rtype:          L{str}
        """
        import tarfile
        import sys

        content = None
        tar = None
        tarfilename = self.get_results_filename(fileid, tape=tape)
        try:
            tar = tarfile.open(name=tarfilename)
            resultfile = tar.extractfile(filename)
            content = resultfile.read().decode('utf-8')
            resultfile.close()
        except (KeyError, OSError) as e:
            sys.stderr.write(str(e) + '\n')
        finally:
            tar.close()

        return content

    def get_output(self, fileid):
        """
        Get the output of the specified job.

        @param fileid: the file id of the job results
        @type  fileid: int
        """

        f = open(self._output[fileid][0], encoding='utf-8')
        outp = f.readlines()[self._output[fileid][1]:self._output[fileid][2]]
        f.close()

        return outp

    def copy_result_file(self, fileid, tape=21, name="TAPE21"):
        """
        Copy a specific results file to working directory.

        @param fileid: the file id of the job results
        @type  fileid: int
        @param tape: the number of the TAPE file to copy (default: TAPE21)
        @type  tape: int
        @param name: the name of the copied file (default: TAPE21)
        @type  name: str
        """
        fn = self.get_results_filename(fileid, tape)
        self.copy_file(fn, name)

    def link_result_file(self, fileid, tape=21, name="TAPE21"):
        """
        Make a symlink to results file in working directory.

        @param fileid: the file id of the job results
        @type  fileid: int
        @param tape: the number of the TAPE file to copy (default: TAPE21)
        @type  tape: int
        @param name: the name of the copied file (default: TAPE21)
        @type  name: str
        """
        fn = self.get_results_filename(fileid, tape)
        self.link_file(fn, name)

    def copy_job_result_files(self, fileid):
        """
        Copy all the results of the specified job to the working directory.

        @param fileid: the file id of the job results
        @type  fileid: int
        """
        for f in self._resultfiles[fileid]:
            tapenr = os.path.basename(f)[1:3]
            self.copy_file(f, 'TAPE' + tapenr)

    def copy_all_results_to_dir(self, dirname):
        """
        Copy all results files to the specified directory.

        In addition to the result files, an index (C{adffiles.pickle}) will be
        created, so that the directory can later be imported using L{import_resultsdir}

        @param dirname: the directory where the results will be copied.
        @type  dirname: str
        """

        for filelist in self._resultfiles:
            for f in filelist:
                self.copy_file(f, os.path.join(dirname, os.path.basename(f)))

        f = open(os.path.join(dirname, 'adffiles.pickle'), 'wb')
        pickle.dump(self._id, f)
        pickle.dump(self._resultfiles, f)
        pickle.dump(self._output, f)
        f.close()

    def import_resultsdir(self, dirname):
        """
        Import results for the specified directory.

        The directory must have been saved previously with L{copy_all_results_to_dir}
        and the file manager must be empty.

        @param dirname: the directory to import from
        @type  dirname: str
        """

        f = open(os.path.join(dirname, 'adffiles.pickle'), 'rb')
        if len(self._resultfiles) > 0:
            # this means, that we already have results
            new_id = pickle.load(f)
            new_resultfiles = pickle.load(f)
            new_output = pickle.load(f)
            duplicates = 0
            checksums = new_id.copy().keys()

            # modify the ids for the checksums
            for checksum in checksums:
                if checksum not in self._id:
                    # this is a new result, we can just modify the id for this
                    # checksum
                    new_id[checksum] += (len(self._resultfiles) - duplicates)
                else:
                    # this result was already present in some form, the new
                    # version will be dropped and the duplicate will be noted
                    dupl_index = new_id[checksum]
                    drop_index = (dupl_index - duplicates)
                    new_resultfiles.pop(drop_index)
                    new_output.pop(drop_index)
                    new_id.pop(checksum)
                    duplicates += 1
            self._id = {**self._id, **new_id}
            self._output += new_output

            # the paths have to be modified one by one
            old_abs_path_dict = {}
            for result_number in range(len(new_resultfiles)):
                result_list = new_resultfiles[result_number]
                new_number = result_number + len(self._resultfiles)
                for file_number in range(len(result_list)):
                    old_path = result_list[file_number]
                    new_path = old_path[:old_path.rindex('.')+1] + f'{new_number:04d}'
                    old_abs_path = os.path.abspath(os.path.join(dirname, os.path.basename(old_path)))
                    old_abs_path_dict[new_path] = old_abs_path
                    new_resultfiles[result_number][file_number] = new_path

            self._resultfiles += new_resultfiles
            f.close()
            for filelist in new_resultfiles:
                for f in filelist:
                    f1 = old_abs_path_dict[f]
                    f2 = os.path.join('resultfiles', os.path.basename(f))
                    os.symlink(f1, f2)
                    self.add_file(f2)
        else:
            self._id = pickle.load(f)
            self._resultfiles = pickle.load(f)
            self._output = pickle.load(f)
            f.close()
            for filelist in self._resultfiles:
                for f in filelist:
                    f1 = os.path.abspath(os.path.join(dirname, os.path.basename(f)))
                    f2 = os.path.join('resultfiles', os.path.basename(f))
                    os.symlink(f1, f2)
                    self.add_file(f2)

        self._ispacked = [True] * len(self._resultfiles)

    def pack_results(self, fileid):
        """
        Pack a TAPE21 results file of an FDE job.

        This will reduce the size of a FDE TAPE21 file by
        only keeping the ActiceFragment section and related
        essential information. All information about frozen
        fragments will be deleted. This is useful for manual
        freeze-and-thaw calculations with many subsystems,
        where this packing can reduce the size of the stored
        files significantly.

        @param fileid: The results ID for the coresponding job.
        @type  fileid: int
        """

        if self._ispacked[fileid]:
            return

        import subprocess
        from xml.dom.minidom import parseString
        import kf

        fn = self.get_results_filename(fileid, 21)

        os.rename(fn, fn + ".orig")

        if kf.kffile.env is None:
            env = os.environ
        else:
            env = kf.kffile.env

        toc = subprocess.Popen([os.path.join(env['AMSBIN'], 'dmpkf'), fn + ".orig", '--xmltoc'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env).communicate()[0]
        dom = parseString(toc)

        f = kf.kffile(fn + ".orig")
        atomtypeIndices = f.read('ActiveFrag', 'atomtypeIndices')
        f.close()

        keepsections = ['General', 'Properties', 'Num Int Params', 'LinearScaling',
                        'Geometry%nr of fragmenttypes']
        keepsections_if_exist = ['Total Energy', 'Energy']

        for section in dom.getElementsByTagName('section'):
            secname = section.getAttribute('id')
            if secname in keepsections_if_exist:
                keepsections.append(secname)
            for i in atomtypeIndices:
                if secname.startswith(f'Atyp{i:3d}'):
                    keepsections.append(secname)
                # fixme: use a more general format expression
                elif secname.startswith(f'Atyp{i:4d}'):
                    keepsections.append(secname)
                elif secname.startswith(f'Atyp{i:5d}'):
                    keepsections.append(secname)
            if secname.startswith('ActiveFrag'):
                keepsections.append(secname)

        subprocess.Popen([os.path.join(env['AMSBIN'], 'cpkf'), fn + ".orig", fn] + keepsections,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env).wait()

        os.remove(fn + ".orig")
