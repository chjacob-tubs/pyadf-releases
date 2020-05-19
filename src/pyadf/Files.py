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
 The file manager L{adf_filemanager} which is used internally.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @undocumented: newjobmarker

"""

__all__ = ['filemanager', 'adf_filemanager']

from Errors import PyAdfError
from Utils import newjobmarker
from PatternsLib import Singleton
import os
import shutil
import pickle


class filemanager (object):

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

    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__
    """
    __metaclass__ = Singleton

    def __init__(self, outdir=None, jobid=os.getpid()):
        """
        Construct a file manager instance.

        @param outdir: directory where the output and error files should be written.
        @type  outdir: str
        @param jobid: a (hopefully unique) job id that is used in the name of the output file.
                      Default is the PID of the current process.
        @type  jobid: str
        """
        self._files = set([])       # set with all managed files
        if outdir is None:
            self._outdir = os.getcwd()
        else:
            self._outdir = outdir
        self._jobid = str(jobid)
        self._outfilename = ""
        self._errfilename = ""
        self.set_outputfilename("pyadf_joboutput")
        self._cwd = os.getcwd()

    def __copy__(self):
        # never copy or deepcopy a filemanager
        return self

    def __deepcopy__(self, memo):
        # pylint: disable-msg=W0613
        # never copy or deepcopy a filemanager
        return self

    def set_outputfilename(self, outname):
        """
        Set the name of the output and error file.

        The output and error files will be written to the output directory
        (see L{__init__}). The given name will be prepended with the job id
        and the extension C{'.out'} and C{'.err'}, respectively.

        @param outname: the name to be used for the output and error file.
        @type  outname: str
        """
        self._outfilename = os.path.join(self._outdir, outname + "." + self._jobid + ".out")
        self._errfilename = os.path.join(self._outdir, outname + "." + self._jobid + ".err")

    def get_outputfilename(self):
        """
        Get the name of the output file, including the full path.
        """
        return self._outfilename

    def get_errfilename(self):
        """
        Get the name of the output file, including the full path.
        """
        return self._errfilename

    def have_file(self, filename):
        """
        Return whether the given file exists in the file manager.

        @param filename: the file name
        @type  filename: str
        """
        return (filename in self._files)

    def change_to_basedir(self):
        os.chdir(self._cwd)

    def add_file(self, filename):
        """
        Add an existing file to the file manager.

        @param filename: the file name
        @type  filename: str
        """
        if not os.path.exists(filename):
            raise PyAdfError("file " + filename + " not found")
        if not (os.getcwd() == self._cwd):
            print os.getcwd(), self._cwd
            raise PyAdfError("add_file not called in base working directory")
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

    outputfilename = property(get_outputfilename, set_outputfilename, None,
                              """
                           Name of the file where the output of the calculation is written.

                           Assigning to C{outputfilename} will also set L{errfilename} accordingly.
                           The filename is always prepended with the job id and the extension '.out',
                           see L{set_outputfilename}
                           """)

    errfilename = property(get_errfilename, None, None,
                           """
                           Name of the file where the stderr of the calculation is written.

                           This is read-only, because C{errfilename} is always set according to
                           L{outputfilename}.
                           """)


class adf_filemanager (filemanager):

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

    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__
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
        filemanager.__init__(self, outdir, jobid)
        self._resultfiles = []        # list of lists of result files, id gives index
        self._output = []             # a list giving for each result the name of the
        #                               output file and first and last line of the output
        #                               result id gives index
        self._id = {}                 # dictionary giving results id for job checksums
        self._ispacked = []
        os.mkdir('resultfiles')

    def cleanup(self):
        """
        Cleanup the file manager.

        This deletes all the managed files.
        """
        filemanager.cleanup(self)
        shutil.rmtree('resultfiles', True)

    def add_outputfiles(self, results):
        """
        Add the output files produces by an ADF or Dalton job.
        """

        fileid = len(self._resultfiles)
        results.fileid = fileid

        # save checksum
        checksum = results.get_checksum()

        if checksum in self._id:
            raise PyAdfError("Checksum error when adding results")
        self._id[checksum] = fileid

        # the output files

        if not results.job == None:
            outfilename = self.outputfilename

            f = open(outfilename, 'r')
            outputstart = -1
            for i, l in enumerate(f.readlines()):
                if l == newjobmarker:
                    outputstart = i
                outputend = i
            if outputstart == -1:
                raise PyAdfError("PyADF marker not found in output")

            self._output.append((outfilename, outputstart, outputend))
        else:
            self._output.append(None)

    def add_dalton_results(self, results):
        """
        Add the result files of a Dalton job.
        """

        self.add_outputfiles(results)
        self._resultfiles.append([])
        self._ispacked.append(True)

    def add_dirac_results(self, results):
        """
        Add the result files of a Dirac job.
        """

        self.add_outputfiles(results)

        self._resultfiles.append([])
        self._ispacked.append(True)

        # we store DFCOEF as TAPE21, and GRIDOUT as TAPE10, dirac.xml as TAPE66
        for diracfile, tapenr in [('DFCOEF', '21'), ('GRIDOUT', '10'), ('dirac.xml', '66')]:
            if os.path.exists(diracfile):
                os.rename(diracfile, 'TAPE' + tapenr)
                self.add_file('TAPE' + tapenr)
                fn = 'resultfiles/t' + tapenr + '.results.%04i' % results.fileid
                self.rename_file('TAPE' + tapenr, fn)
                self._resultfiles[results.fileid].append(fn)

    def add_nwchem_results(self, results):
        """
        Add the result files of a NWChem job.
        """

        self.add_outputfiles(results)

        self._resultfiles.append([])
        self._ispacked.append(True)

        # we store NWCHEM.db as TAPE21, and GRIDOUT as TAPE10
        for nwchemfile, tapenr in [('NWCHEM.db', '21'), ('GRIDOUT', '10')]:
            if os.path.exists(nwchemfile):
                os.rename(nwchemfile, 'TAPE' + tapenr)
                self.add_file('TAPE' + tapenr)
                fn = 'resultfiles/t' + tapenr + '.results.%04i' % results.fileid
                self.rename_file('TAPE' + tapenr, fn)
                self._resultfiles[results.fileid].append(fn)

    def add_turbomole_results(self, results):
        """
        Add the result files (yes, plural!) of a I{Turbomole} job.

        Adds a C{tar} archive to the result files containing all files from the
        I{Turbomole} working directory. The archive is compressed with the
        format specified by the results object's C{compression} attribute which
        must be a proper string. See the U{C{tarfile} module's
        API<http://docs.python.org/library/tarfile.html#module-tarfile>} for a
        list of these.

        @param results: Results object from the I{Turbole} job.
        @type  results: L{TurbomoleResults}
        @author:        Moritz Klammer
        @date:          Aug. 2011

        """

        import tarfile

        self.add_outputfiles(results)
        self._resultfiles.append([])
        self._ispacked.append(True)

        # Make  a `archive.tar'  from `jobtempdir'.  The files  will be  in the
        # arcive directly  with no containing directory. They  can therefore be
        # extracted  via,  say,   `tar.extractfile'energy')'  (if  `tar'  is  a
        # `TarFile'  object  opened in  reading  mode  with properly  specified
        # compressin.

        jobdirname = 'jobtempdir'
        archivename = 'archive.tar'

        tar = tarfile.open(name=archivename, mode='w:' + results.compression)
        for filename in os.listdir(jobdirname):
            tar.add(jobdirname + os.sep + filename, arcname=filename)
        tar.close()

        self.add_file(archivename)
        fn = 'resultfiles/t21.results.%04i' % results.fileid
        self.rename_file(archivename, fn)
        self._resultfiles[results.fileid].append(fn)

    def add_adf_results(self, results):
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

        for tapenr in ['21', '10', '41']:
            if os.path.exists('TAPE' + tapenr):
                self.add_file('TAPE' + tapenr)
                fn = 'resultfiles/t' + tapenr + '.results.%04i' % results.fileid
                self.rename_file('TAPE' + tapenr, fn)
                self._resultfiles[results.fileid].append(fn)

        if os.path.exists('TAPE15'):
            os.remove('TAPE15')

    def add_ams_results(self, results):
        """
        Add the result files of an AMS job.

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

        for tapenr, rkfname in [(13, 'ams.rkf'), (21, 'dftb.rkf')]:
            if os.path.exists(rkfname):
                self.add_file(rkfname)
                fn = 'resultfiles/t%i.results.%04i' % (tapenr, results.fileid)
                self.rename_file(rkfname, fn)
                self._resultfiles[results.fileid].append(fn)

    def add_results(self, results):
        """
        Add results.

        Based on the type of the results argument, ADF, Dalton, or Dirac results are added
        """
        from ADFBase import adfresults, amsresults
        from ADFSinglePoint import adfsinglepointresults
        from ADF_Densf import densfresults
        from ADF_NMR import adfnmrresults
        from ADF_CPL import adfcplresults
        from DaltonSinglePoint import daltonresults
        from Dirac import diracresults
        from NWChem import nwchemresults
        from Turbomole import TurbomoleResults

        if isinstance(results, adfsinglepointresults):
            self.add_adf_results(results)
        elif isinstance(results, amsresults):
            self.add_ams_results(results)
        elif isinstance(results, densfresults):
            self.add_adf_results(results)
        elif isinstance(results, adfnmrresults):
            self.add_adf_results(results)
        elif isinstance(results, adfcplresults):
            self.add_adf_results(results)
        elif isinstance(results, daltonresults):
            self.add_dalton_results(results)
        elif isinstance(results, diracresults):
            self.add_dirac_results(results)
        elif isinstance(results, nwchemresults):
            self.add_nwchem_results(results)
        elif isinstance(results, TurbomoleResults):
            self.add_turbomole_results(results)
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

    def get_results_filename(self, fileid, tape=21):
        """
        Returns the file name of a results TAPE file.

        @param fileid: the file id of the job results
        @type  fileid: int
        @param tape: the number of the requested TAPE file (default: TAPE21)
        @type  tape: int
        """
        # tape: number of the tape to get

        fn = os.path.join(self._cwd, 'resultfiles', 't%2i.results.%04i' % (tape, fileid))

        if self.have_file(fn):
            return fn
        else:
            raise PyAdfError("results file not found")

    def get_output(self, fileid):
        """
        Get the output of the specified job.

        @param fileid: the file id of the job results
        @type  fileid: int
        """

        f = open(self._output[fileid][0], 'r')
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

        f = open(os.path.join(dirname, 'adffiles.pickle'), 'w')
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

        if (len(self._resultfiles) > 0):
            raise PyAdfError('Error importing resultsdir')

        f = open(os.path.join(dirname, 'adffiles.pickle'), 'r')
        self._id = pickle.load(f)
        self._resultfiles = pickle.load(f)
        self._output = pickle.load(f)
        f.close()

        self._ispacked = [True] * len(self._resultfiles)

        for filelist in self._resultfiles:
            for f in filelist:
                f1 = os.path.join(dirname, os.path.basename(f))
                f2 = os.path.join('resultfiles', os.path.basename(f))
                os.symlink(f1, f2)
                self.add_file(f2)

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

        if self._ispacked[fileid] == True:
            return

        import subprocess
        from xml.dom.minidom import parseString
        import kf

        fn = self.get_results_filename(fileid, 21)

        os.rename(fn, fn + ".orig")

        toc = subprocess.Popen([os.path.join(os.environ['ADFBIN'], 'dmpkf'), fn + ".orig", '--xmltoc'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
        dom = parseString(toc)

        f = kf.kffile(fn + ".orig")
        atomtypeIndices = f.read('ActiveFrag', 'atomtypeIndices')
        f.close()

        keepsections = ['General', 'Properties', 'Num Int Params', 'LinearScaling', 'Geometry%nr of fragmenttypes']
        for section in dom.getElementsByTagName('section'):
            secname = section.getAttribute('id')
            for i in atomtypeIndices:
                if secname.startswith('Atyp%3i' % i):
                    keepsections.append(secname)
                # fixme: use a more general format expression
                elif secname.startswith('Atyp%4i' % i):
                    keepsections.append(secname)
                elif secname.startswith('Atyp%5i' % i):
                    keepsections.append(secname)
            if secname.startswith('ActiveFrag'):
                keepsections.append(secname)

        subprocess.Popen([os.path.join(os.environ['ADFBIN'], 'cpkf'), fn + ".orig", fn] + keepsections,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()

        os.remove(fn + ".orig")
