# PyADF JobRunner example configuration file

# the shell that is used for executing the job scripts
# if env_modules_load is used, this shell must be configured
# such that the module command is available
# PyADF's default is to use bash

default_shell = '/bin/zsh'

# dictionary specifying environment modules to load for specific types of
# jobs (with 'module load'). The keys of the dictionary give the name of
# a job class, and the modules will be loaded for all jobs that are derived
# from this class (i.e., the 'adfjob' modules will be loaded for all ADF-like
# jobs, such as adfsinglepointjob, densfjob etc.). Options given for a subclass
# override those of parent classes.

# this is a local variable to simplify the definition of env_modules_load below
_intel_mods = ['ifort/19.1.053', 'icc/19.1.053', 'openmpi-intel19/3.1.6']

env_modules_load = {'adfjob': _intel_mods + ['adf-openmpi/trunk-20200115'],
                    'diracjob': _intel_mods + ['dirac/dirac-master'],
                    'daltonjob': _intel_mods + ['dalton/master-public'],
                    'TurbomoleJob': 'turbomole/7.1mpi',
                    'nwchemjob': _intel_mods + ['nwchem/6.8.1'],
                    'SNFJob': ['snf/20190701', 'turbomole/7.1']}

# dictionary specifying the parallel configuration for specific types of jobs.
# The keys give the names of job classes (see env_modules_load above), the
# values are dictionaries that can contain the following entries:
# - 'parallel': True or False (False means always execute in serial)
# - 'nproc': the number of processors; will always override the default
#            obtained from the environment
# more possible entries (e.g. which parallel environment a job uses) could
# be added in the future

parallel_config = {'adfjob': {'parallel': True},
                   'nwchemjob': {'parallel': False, 'nproc': 1}}  # nproc = 1 is redundant here
