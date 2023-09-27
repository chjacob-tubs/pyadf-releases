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

import os.path
import shutil


class env:

    def __init__(self, startdir, outdir, jobid, inputname, options):

        self.startdir = startdir
        self.outdir = outdir
        self.jobid = jobid
        self.inputname = os.path.join(outdir, os.path.basename(inputname))
        self.options = options
        self.pyadfout = os.path.join(outdir,
                                     'pyadf.' + os.path.basename(inputname) + '.' + jobid + '.out')

    def print_summary(self):
        # directory pyadf was started from (where the input is)
        print(" Directory of input file:")
        print("     ", self.startdir)
        print()

        # directory where the output goes
        print(" Directory of output files:")
        print("     ", self.outdir)
        print()

        # name of the input file
        print(" " + 40 * "-")
        print(" PyADF input file: ", os.path.abspath(self.inputname))
        print()

    def cleanup(self):

        if (os.getcwd() != self.startdir) and os.path.exists(self.startdir):
            os.chdir(self.startdir)
        if os.path.exists('pyadftempdir'):
            shutil.rmtree('pyadftempdir')


def setup_pyadfenv():
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] inputfile")

    parser.add_option("--interactive", "-i", action="store_true", default=False,
                      dest="interactive", help="run interactively")
    parser.add_option("--force", "-f", action="store_true", default=False,
                      dest="force", help="force deletion of temporary directories")
    parser.add_option("--save", "-s", action="store_true", default=False,
                      dest="save_results", help="save the result files after termination")
    parser.add_option("--restartdir", "-r", type="string", metavar="DIR",
                      dest="restartdir", help="restart using results in directory DIR")
    parser.add_option("--profile", "-p", action="store_true", default=False,
                      dest="profile", help="run using the python profiler")
    parser.add_option("--molclass", choices=["openbabel", "rdkit", "obfree"], default=None,
                      help="select molecule class to use [available: openbabel, obfree, rdkit]")
    parser.add_option("--jobrunnerconf", "-c", action="store", type='string', metavar='FILE', default=None,
                      dest="jobrunnerconf", help="set job runner configuration file [default: $HOME/.pyadfconfig]")

    (options, args) = parser.parse_args()

    opts = {}
    if options.interactive:
        opts['interactive'] = True

    if options.save_results:
        opts['save_results'] = True

    if options.profile:
        opts['profile'] = True

    opts['molclass'] = options.molclass

    if options.jobrunnerconf is None:
        opts['jobrunner_conffile'] = options.jobrunnerconf
    else:
        opts['jobrunner_conffile'] = os.path.abspath(options.jobrunnerconf)

    if not (options.restartdir is None):
        opts['restartdir'] = os.path.abspath(options.restartdir)

    if 'TC_SUBMISSION_DIR' in os.environ:
        # Using TC tools
        pyadfenv = env(os.getcwd(),
                       os.environ['TC_SUBMISSION_DIR'],
                       os.environ['TC_JOB_IDENTIFIER'],
                       args[0], opts)

    elif 'PBS_O_HOME' in os.environ:
        # LISA / PBS
        pyadfenv = env(os.getcwd(),
                       os.environ['PBS_O_WORKDIR'],
                       os.environ['PBS_JOBID'],
                       args[0], opts)
    else:
        # run locally
        from datetime import datetime
        cwd = os.getcwd()

        if options.force and os.path.exists('pyadftempdir'):
            os.system('rm -r pyadftempdir')

        os.mkdir('pyadftempdir')
        os.chdir('pyadftempdir')

        now = datetime.now()
        jobid = now.strftime('%Y%m%d-%H%M%S')

        pyadfenv = env(cwd, cwd, jobid, args[0], opts)

    return pyadfenv


def setup_test_pyadfenv(pyadfinput, molclass=None, jobrunnerconf=None, save_results=False):

    # run locally
    from datetime import datetime
    cwd = os.getcwd()

    options = {'unittesting': True, 'molclass': molclass}
    if save_results:
        options['save_results'] = True

    if jobrunnerconf is None:
        options['jobrunner_conffile'] = jobrunnerconf
    else:
        options['jobrunner_conffile'] = os.path.abspath(jobrunnerconf)

    os.mkdir('pyadftempdir')
    os.chdir('pyadftempdir')

    now = datetime.now()
    jobid = now.strftime('%Y%m%d-%H%M%S') + "-" + str(os.getpid())

    pyadfenv = env(cwd, cwd, jobid, pyadfinput, options)

    return pyadfenv
