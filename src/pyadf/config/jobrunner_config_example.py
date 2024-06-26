# PyADF JobRunner example configuration file

# the shell that is used for executing the job scripts
# if env_modules_load is used, this shell must be configured
# such that the module command is available

# You can put this as .pyadfconfig into your $HOME after
# adjusting it to your set-up

# PyADF's default is to use bash

default_shell = '/bin/zsh'

# dictionary specifying environment modules to load for specific types of
# jobs (with 'module load'). The keys of the dictionary give the name of
# a job class, and the modules will be loaded for all jobs that are derived
# from this class (i.e., the 'adfjob' modules will be loaded for all ADF-like
# jobs, such as adfsinglepointjob, densfjob etc.). Options given for a subclass
# override those of parent classes.

# this is a local variable to simplify the definition of env_modules_load below
_intel19_mods = ['ifort/19.1.053', 'icc/19.1.053', 'openmpi-intel19/3.1.6']
_intel20_mods = ['ifort/20.0.088', 'openmpi-intel20/3.1.6']
_intel22_mods = ['intel-oneapi/compiler/2022.2.0', 'intel-oneapi/mkl/2022.2.0', 'openmpi-intel22/4.1.4']

env_modules_load = {'scmjob': _intel20_mods + ['adf-openmpi/trunk-20221230'],
                    'diracjob': _intel22_mods + ['dirac/master-20221231'],
                    'daltonjob': _intel22_mods + ['dalton/master-20221231'],
                    'TurbomoleJob': ['turbomole/7.6mpi'],
                    'OrcaJob': ['openmpi/4.1.1', 'orca/5.0.3'],
                    'nwchemjob': _intel22_mods + ['nwchem/7.0.2'],
                    'MolcasJob': _intel22_mods + ['OpenMolcas/v20.10'],
                    'QEJob': ['openmpi/4.1.4', 'espresso/eqe-master-20210204'],
                    'SNFJob': ['openmpi/4.1.4', 'snf/20190701', 'turbomole/7.5.1']}

# dictionary specifying the parallel configuration for specific types of jobs.
# The keys give the names of job classes (see env_modules_load above), the
# values are dictionaries that can contain the following entries:
# - 'parallel': True or False (False means always execute in serial)
# - 'nproc': the number of processors; will always override the default
#            obtained from the environment
# more possible entries (e.g. which parallel environment a job uses) could
# be added in the future

# parallel_config = {'adfjob': {'parallel': True},
#                    'nwchemjob': {'parallel': False, 'nproc': 1}}  # nproc = 1 is redundant here
