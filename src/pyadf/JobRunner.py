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
 JobRunner classes for PyADF.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""

from PatternsLib import Singleton
from Errors import PyAdfError

class JobRunner (object):
    """
    Abstract base class for job runners.
    """
    __metaclass__ = Singleton
    
    def __init__ (self):
        pass
    
    def run_job (self, job):
        pass
    
    
class SerialJobRunner (JobRunner):
    """
    Serial job Runner.
    
    Jobs are executed instantly on the local machine in a serial fashion.
    """
    def __init__ (self):
        JobRunner.__init__(self)

        from Files import adf_filemanager
        self._files = adf_filemanager()
    
    def write_runscript_and_execute (self, job):

        import os, stat, subprocess
        from Utils import newjobmarker

        job.before_run ()
        
        runscript = job.get_runscript()

        rsname = './pyadf_runscript'
        f = open(rsname, 'w')
        f.write (runscript)
        f.close()
        os.chmod(rsname, stat.S_IRWXU)
        
        outfile = open(self._files.outputfilename, 'a')
        outfile.write (newjobmarker)
        outfile.flush()

        errfile = open(self._files.errfilename, 'a')
        errfile.write (newjobmarker)
        errfile.flush()

        retcode = subprocess.call(rsname, shell=False, stdout=outfile, stderr=errfile)

        outfile.close()
        errfile.close()

        os.remove(rsname)

        if (retcode != 0) :
            raise PyAdfError ("Error running job (non-zero return code)")
        if not job.check_success(self._files.outputfilename, self._files.errfilename) :
            raise PyAdfError ("Error running job (check_sucess failed)")

        job.after_run ()
    
    def run_job (self, job):

        import os, shutil
        
        job.print_jobinfo()

        print "   Output will be written to : ", \
            os.path.basename(self._files.outputfilename)
        print

        checksum = job.get_checksum()
        fileid = self._files.get_id (checksum)

        if (fileid == None):
            print " Running main job ..."

            cwd = os.getcwd()
            os.mkdir('jobtempdir')
            os.chdir('jobtempdir')

            if 'TC_HPMPI_MACHINE_FILE' in os.environ :
                os.system('$PYADFHOME/src/scripts/create_pwd_on_nodes.py')

            try:
                self.write_runscript_and_execute(job)

                for f in job.result_filenames() :
                    if os.path.exists(f): 
                        shutil.copy(f, cwd)
            finally:
                os.chdir(cwd)

            r = job.create_results_instance()
            self._files.add_results(r)

            os.system('rm -rf jobtempdir')

        else :
            print "Job was found in results archive - not running it again"

            r = job.create_results_instance()
            r.fileid = fileid

        r._checksum = checksum

        print " Done with "+ job.print_jobtype()
        print
        print " Results file id is ", r.fileid
        print

        return r

DefaultJobRunner = SerialJobRunner
