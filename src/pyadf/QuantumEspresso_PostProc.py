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
 Post-processing of Quantum Espresso calculations

 @author:       Andre Gomes and others
 @organization: CNRS

 @group Jobs:
    qepostprocjob
 @group Settings:
    qepostprocsettings
 @group Results:
    qepostprocresults
"""
from abc import abstractmethod

from .QuantumEspresso import QEJob, QEResults
from .Errors import PyAdfError


class QEPostProcResults(QEResults):
    """
    Class for results of an Quantum Espresso single point calculation.
    """

    def __init__(self, j=None):
        """
        Constructor for qesinglepointresults.
        """
        super().__init__(j)

    def get_cube_filename(self):
        """
        Return the file name of the cube file belonging to the results.
        """
        return self.files.get_results_filename(self.fileid, 10)


class QEPostProcSettings:
    """
    Settings for a Quantum Espresso post-processing calculation.
    """

    def __init__(self, inputpp=None, plot=None):
        self.runtype = 'pp'

        self.inputpp = {}
        self.plot = {}
        self.is_potential = None
        self.is_density = None

        self.set_pp_defaults()
        self.set_pp(inputpp)
        self.set_plot(plot)

    def set_pp_defaults(self):
        """
        Default input options for QE. Here None is used to flag optional keywords
        """

        # for the dictionaries with options below, the value None will be used to flag
        # which variables will remain with the quantum espresso defaults.

        self.inputpp = {
            'prefix': "'pwscf'",
            'plot_num': 0,
            'spin_component': None,
            'sample_bias': None,
            'kpoint': None,
            'kband': None,
            'lsign': None
        }

        self.plot = {
            'nfile': None,
            'filepp': None,
            'weight': None,
            'iflag': 3,
            'output_format': 6,
            'x0': [0.0, 0.0, 0.0],
            'e1': [0.1, 0.0, 0.0],
            'e2': [0.0, 0.1, 0.0],
            'e3': [0.0, 0.0, 0.1],
            'interpolation': "'fourier'",
            'fileout': "'ppjob.cube'",
            'nx': 10,
            'ny': 10,
            'nz': 10
        }

    def set_pp(self, options):
        if options is not None:
            self.inputpp.update(options)

        val = self.inputpp['plot_num']
        if val in [0, 3, 4, 6, 7, 9, 13, 17, 19, 20]:
            self.is_density = True
            self.is_potential = False
        elif val in [1, 2, 11, 12, 18]:
            self.is_density = False
            self.is_potential = True
        else:
            self.is_density = False
            self.is_potential = False

    def set_plot(self, options):
        if options is not None:
            self.plot.update(options)

    @staticmethod
    def iter_not_none(d):
        return ((k, v) for k, v in list(d.items()) if v is not None)

    def get_inputpp_block(self):
        block = " &inputpp\n"
        for opt, val in self.iter_not_none(self.inputpp):
            block += f"    {opt}={val}\n"
        block += " /\n"
        return block

    def get_plot_block(self):
        block = " &plot\n"
        for opt, val in self.iter_not_none(self.plot):
            if opt in ['nfile', 'filepp']:
                # filepp should be a list with filenames, but if there's more than one element
                # we may also want to support weight files. but for the time being we only support
                # a single file
                raise PyAdfError('Option:' + opt + 'not yet supported by PyADF')
            elif opt in ['x0', 'e1', 'e2', 'e3']:
                block += "    " + str(opt) + "="
                if len(val) > 1:
                    separator = ", "
                else:
                    separator = "  "
                for e in self.plot[opt]:
                    block += str(e) + separator
                block += "\n"
            else:
                block += f"    {opt}={val}\n"
        block += " /\n"
        return block

    def __str__(self):
        """
        Returns a human-readable description of the settings.
        """
        ss = '   QE Post-processing settings: summary of options\n'
        ss += '    inputpp   : ' + str(self.inputpp) + '\n'
        ss += '    plot      : ' + str(self.plot) + '\n'
        return ss


class QEPostProcJob(QEJob):
    """
    A class for Quantum Espresso post-processing jobs.
    """

    def __init__(self, qeres, settings=None, options=None):
        super().__init__()

        self._qeresults = qeres

        if settings is None:
            self.settings = QEPostProcSettings()
        else:
            self.settings = settings
        self.runtype = 'pp'

        if options is None:
            self._options = []
        else:
            self._options = options

    def create_results_instance(self):
        return QEPostProcResults(self)

    def get_options_block(self):
        return '\n'.join(self._options)

    @abstractmethod
    def get_other_blocks(self):
        """
        Abstract method. Allows extending the QE input file in subclasses.
        """
        return ""

    def get_qefile(self):
        qefile = self.settings.get_inputpp_block()
        qefile += self.settings.get_plot_block()
        qefile += self.get_options_block()
        qefile += self.get_other_blocks()
        return qefile

    def result_filenames(self):
        filelist = ['ppjob.cube']
        return filelist

    def print_jobtype(self):
        return "Quantum Espresso Post-processing job"

    def before_run(self):
        archivename = self._qeresults.get_data_filename()

        import tarfile
        tar = tarfile.open(name=archivename, mode='r')
        tar.extractall()
        tar.close()

        super().before_run()

    def print_settings(self):

        print("   Settings")
        print("   ========")
        print()
        print(self.settings)
        print()

    def print_extras(self):
        pass

    def print_jobinfo(self):
        print(" " + 50 * "-")
        print(" Running " + self.print_jobtype())
        print()
        print("   SCF taken from Quantum Espresso job ", self._qeresults.fileid, " (results id)")
        print()
