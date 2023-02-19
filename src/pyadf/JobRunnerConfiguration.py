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
 JobRunner configuration classes for PyADF.

 @author:       Christoph Jacob and others
 @organization: TU Braunschweig
 @contact:      c.jacob@tu-braunschweig.de

"""

import os
from .Errors import PyAdfError


class JobRunnerConfiguration:
    """
    Configuration class for job runners.
    """

    def __init__(self, conffile=None, jobbasedir=None):

        # set default values, might be overwritten by initialize_from_conffile
        self._read_from_file = None

        self.default_shell = '/bin/bash'
        self.env_modules_load = {}
        self.parallel_config = {'job': {'parallel': True}}

        # number of cores to use for the parallel execution of jobs (using MPI etc.)
        # The following environment variables are evaluated (those appearing earlier
        # in this list take precedence over those appearing later)
        # - $PYADF_NPROC
        # - $TC_NUM_PROCESSES
        # - $NSCM (included to preserve previous behavior; could be removed in the future)
        #
        # All of these are overridden by nproc options included in parallel_config
        self.default_nproc = 1
        if 'PYADF_NPROC' in os.environ:
            self.default_nproc = int(os.environ['PYADF_NPROC'])
        elif 'TC_NUM_PROCESSES' in os.environ:
            self.default_nproc = int(os.environ['TC_NUM_PROCESSES'])
        elif 'NSCM' in os.environ:
            self.default_nproc = int(os.environ['NSCM'])

        if conffile is not None:
            self.initialize_from_conffile(conffile)
        else:
            if jobbasedir is None:
                default_conffile = os.path.join(os.environ['HOME'], '.pyadfconfig')
            else:
                default_conffile = self._find_default_conffile_in_parentdirs(jobbasedir)
            if (default_conffile is not None) and os.path.exists(default_conffile):
                self.initialize_from_conffile(default_conffile)

        if not self._validate_jobclass_dict(self.env_modules_load):
            raise PyAdfError('Error in JobRunnerConfiguration (env_module_load)')
        if not self._validate_jobclass_dict(self.parallel_config):
            raise PyAdfError('Error in JobRunnerConfiguration (parallel_config)')

    # noinspection PyUnresolvedReferences
    def initialize_from_conffile(self, fn):
        if not os.path.exists(fn):
            raise PyAdfError(f'JobRunner configuration file {fn} does not exist')
        self._read_from_file = os.path.abspath(fn)

        import importlib.util
        from importlib.machinery import SourceFileLoader
        spec = importlib.util.spec_from_loader("pyadf.jobrunnerconf.conffile",
                                               SourceFileLoader("pyadf.jobrunnerconf.conffile", fn))
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        if 'default_shell' in dir(conf):
            self.default_shell = conf.default_shell
        if 'env_modules_load' in dir(conf):
            self.env_modules_load = conf.env_modules_load
        if 'parallel_config' in dir(conf):
            self.parallel_config.update(conf.parallel_config)

    def print_configuration(self):
        print("JobRunner configuration read from file: ", self._read_from_file)
        print("     default shell: ", self.default_shell)
        print("     env_modules_load: ", self.env_modules_load)

    @staticmethod
    def _find_default_conffile_in_parentdirs(jobbasedir):
        cur_dir = jobbasedir
        while not cur_dir == '/':
            test_file = os.path.join(cur_dir, '.pyadfconfig')
            if os.path.exists(test_file):
                return test_file
            else:
                cur_dir = os.path.dirname(cur_dir)
        return None

    @staticmethod
    def _pyadf_class_from_string(clsname):
        import inspect
        import importlib
        m = importlib.import_module('pyadf')

        try:
            cls = getattr(m, clsname)
        except AttributeError:
            cls = None

        if not inspect.isclass(cls):
            cls = None

        return cls

    def _get_relevant_entries_for_cls_from_dict(self, config_dict, cls):
        relevant_entries = []
        for jobclassname in config_dict:
            jobcls = self._pyadf_class_from_string(jobclassname)
            if issubclass(cls, jobcls):
                relevant_entries.append((jobclassname, jobcls))

        # sort list of relevant entries such that parent classes appear before their childs
        relevant_entries.sort(key=lambda e: len(e[1].mro()))

        return relevant_entries

    def _validate_jobclass_dict(self, jobclass_dict):

        pyadf_base_jobclass = self._pyadf_class_from_string('job')

        for jobclass in list(jobclass_dict.keys()):
            cls = self._pyadf_class_from_string(jobclass)

            if (cls is None) or not (issubclass(cls, pyadf_base_jobclass)):
                print(f'Error: {jobclass} is not a valid job (base) class')
                return False

        return True

    def get_env_modules_for_class(self, cls):
        """
        Return a list with the environment module names that should be loaded
        for a given job class.
        """
        relevant_entries = self._get_relevant_entries_for_cls_from_dict(self.env_modules_load, cls)

        if len(relevant_entries) == 0:
            mods = []
        else:
            mods = self.env_modules_load[relevant_entries[-1][0]]
            if not isinstance(mods, list):
                mods = [mods]

        return mods

    def get_env_modules_for_job(self, job):
        """
        Return a list with the environment module names that should be loaded
        for a given job object.
        """
        return self.get_env_modules_for_class(job.__class__)

    @staticmethod
    def _env_output_to_environ_dict(env_output):
        environ_dict = {}
        for line in env_output.splitlines():
            if line:
                e = line.strip().split('=', 1)
                if len(e) == 2:
                    environ_dict[e[0]] = e[1]
        return environ_dict

    def get_environ_for_class(self, cls):
        """
        Return a dictionary of the environment variables for a given job class.
        """
        from subprocess import Popen, PIPE

        env_modules = self.get_env_modules_for_class(cls)

        if len(env_modules) > 0:
            module_cmd = ''
            for mod in env_modules:
                module_cmd += f'module load {mod}; '

            DEVNULL = open(os.devnull, 'wb')
            env_output_new = Popen(module_cmd+'env', shell=True, executable=self.default_shell,
                                   stdout=PIPE, stderr=DEVNULL).communicate()[0].decode('utf-8')
            environ = self._env_output_to_environ_dict(env_output_new)
        else:
            environ = os.environ
        return environ

    def get_environ_for_job(self, job):
        """
        Return a dictionary of the environment variables for a given job.
        """
        return self.get_environ_for_class(job.__class__)

    def get_parallel_conf_for_class(self, cls):
        """
        Return dictionary with parallel configuration for a given job class.
        """

        relevant_entries = self._get_relevant_entries_for_cls_from_dict(self.parallel_config, cls)
        return self.parallel_config[relevant_entries[-1][0]]

    def get_parallel_conf_for_job(self, job):
        """
        Return dictionary with parallel configuration for a given job.
        """
        return self.get_parallel_conf_for_class(job.__class__)

    def get_nproc_for_class(self, cls):
        """
        Return the number of parallel processes for a given job class.
        """
        nproc = self.default_nproc

        pconf = self.get_parallel_conf_for_class(cls)
        if 'nproc' in pconf:
            nproc = pconf['nproc']
        if not pconf.get('parallel', True):
            nproc = 1

        return nproc

    def get_nproc_for_job(self, job):
        """
        Return the number of parallel processes for a given job.
        """
        return self.get_nproc_for_class(job.__class__)
