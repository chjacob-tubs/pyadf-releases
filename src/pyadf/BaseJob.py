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
 The base classes for jobs and results.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    job, metajob
 @group Results:
    results    
"""

class results (object) :
    """
    An abstract class for results of a calculation.

    Results objects are returned by the run() method
    of a L{job} object. This abstract class only stores
    the most basis information, a copy of the job that
    generated these results and a reference to the filemanager.

    The class hierarchy of the results classes corresponds to 
    the one of the job classes.
    The constructor of the child classes are always passed
    an object of the parent class, plus additional information
    if necessary.

    @group Initialization: 
      __init__
    @group Retrieval of specific results:
      get_dipole_magnitude, get_dipole_vector
    @group Access to result files:
      get_output
    @group Access to internal properties:
      get_checksum  

    @undocumented:
        __deepcopy__
    """

    def __init__ (self, j=None) :
        """
        Constructor for results class.

        @param j: L{job} object of the corresponding job
        """
        import copy
        
        from Files import adf_filemanager
        self.files = adf_filemanager()
        
        if j == None:
            self.job   = None
        else :
            self.job       = copy.deepcopy(j)
        self._checksum = None
        self.fileid    = None

    def __deepcopy__ (self, memo):
        #pylint: disable-msg=W0613
        #
        # make sure deepcopy are not deeper than needed -
        #  they might become very expensive
        # there is no need to deepcopy results objects, since they should not change 
        import copy
        return copy.copy(self)

    def get_checksum (self) :
        """
        Return the checksum associated with the results.
        
        @returns: the checksum
        @rtype:   str
        """
        if self._checksum == None :
            self._checksum = self.job.get_checksum()
        return self._checksum

    def get_output (self) :
        """
        Return the output file associated with the results.
        
        @returns: the lines of the job output
        @rtype:   list of str
        """
        if self.fileid == None:
            return None
        else :
            return self.files.get_output (self.fileid)

    def get_dipole_vector (self):
        """
        Return the dipole moment vector.
        
        This is an abstract method that has to be overridden 
        be child classes.
        
        @returns: the dipole moment vector, in atomic units.
        @rtype:   list[3] of float
        """
        return None
    
    def get_dipole_magnitude (self) :
        """
        Return the magnitude of the dipole moment.

        @returns: the magnitude of the dipole moment, in atomic units
        @rtype:   float
        """
        import math
        #pylint: disable-msg=W1111
        dipole = self.get_dipole_vector ()
        if dipole:
            return math.sqrt(dipole[0]**2 + dipole[1]**2 + dipole[2]**2)
        else:
            return float('NaN')


class job (object):
    """
    An abstract job base class.
    
    @group Initialization:
        __init__
    @group Running:
        run
    @group Running Internals:
        get_runscript, before_run, after_run, check_success
    @group Other Internals:
        get_checksum, create_results_instance, result_filenames,
        print_jobinfo, print_jobtype

    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__
    """

    def __init__ (self) :
        """
        Constructor for job.
        """
        pass

    def create_results_instance (self):
        """
        Create an instance of the matching results object for this job.
        
        This method should be overwritten in derived classes.
        """
        return results(self)

    def get_runscript (self) :
        """
        Abstract method that returns a run script for the job.
        """
        return ""

    def before_run (self) :
        """
        Template method that is executed before executing the runscript.
        """
        pass

    def after_run (self) :
        """
        Template method that is executed after executing the runscript.
        """
        pass
    
    def result_filenames (self):
        """
        Template method which should return the names of the result files.
        """
        return []

    def check_success (self, outfile, errfile):
        """
        Template method to check whether job was run successfully.
        """
        return True

    def get_checksum (self) :
        """
        Abstract method for obtaining a checksum of the job.
        """
        return None

    def print_jobtype (self) :
        """
        Abstract method to print the job type.
        """
        pass

    def print_jobinfo (self) :
        """
        Abstract method to print information about the job.
        """
        pass
    
    def run (self, job_runner=None) :
        """
        Run the job.

        @returns: An object representing the results of the calculation.
        @rtype: subclass of L{results}
        """
        
        if job_runner is None :
            from JobRunner import DefaultJobRunner
            res = DefaultJobRunner().run_job(self)
        else :
            res = job_runner.run_job(self)
            
        return res


class metajob (job) :
    """
    An abstract base class for meta jobs.
    
    A meta job is a job that does not actually run anything
    itself, but that consists of several jobs that are executed
    together.
    
    @group Running:
        metarun
    """
    
    def __init__ (self):
        """
        Constructor for metajob.
        
        """
        job.__init__ (self)

    def metarun (self):
        """
        Abstract method to perform the meta run.
        
        This should call the C{run} methods of the required child jobs.
        """
        return self.create_results_instance()

    def run (self, job_runner=None):
        """
        Run the job.
        
        This overrides the parent L{job.run} method to call
        L{metarun} instead of executing a run script.

        @returns: An object representing the results of the calculation.
        @rtype: L{results}
        """
        self.before_run ()
        res = self.metarun ()
        self.after_run()
        return res
