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
 The basics needed for ADF calculations: simple jobs and results.

 @author:       Christoph Jacob and others
 @organization: TU Braunschweig
 @contact:      c.jacob@tu-braunschweig.de

 @group Jobs:
    dftbjob
 @group Results:
    dftbresults
"""

import os

from .Molecule import molecule
from .ADFBase import amssettings, amsresults, amsjob


class dftbsettings(amssettings):

    def __init__(self):
        super().__init__()

    def __str__(self):
        ss = super().__str__()
        return ss


class dftbresults(amsresults):

    def __init__(self, j=None):
        """
        Constructor for dftbresults.
        """
        super().__init__(j)

    def get_molecule(self):
        """
        Return the molecular geometry after the ADF job.

        This can be changes with respect to the input geometry
        because it was optimized in the calculation.

        @returns: The molecular geometry.
        @rtype:   L{molecule}
        """
        nnuc = self.get_result_from_tape('Molecule', 'nAtoms')

        atnums = self.get_result_from_tape('Molecule', 'AtomicNumbers')
        xyz = self.get_result_from_tape('Molecule', 'Coords')
        xyznuc = xyz.reshape(nnuc, 3)

        m = molecule()
        m.add_atoms(atnums, xyznuc, atomicunits=True)

        return m

    def get_dipole_vector(self):
        """
        Return the dipole moment vector.

        @returns: the dipole moment vector, in atomic units
        @rtype: float[3]
        """
        return self.get_result_from_tape('AMSResults', 'DipoleMoment')

    def get_energy(self):
        """
        Return the bond energy.

        @returns: the bond energy in atomic units
        @rtype: float
        """
        return self.get_result_from_tape('AMSResults', 'Energy')


class dftbjob(amsjob):
    """
    Generic DFTB job
    """

    def __init__(self, mol, model='SCC-DFTB', parameters='Dresden', settings=None, amstask='SinglePoint'):
        if settings is None:
            mysettings = dftbsettings()
        else:
            mysettings = settings

        super().__init__(mol, task=amstask, settings=mysettings)
        self.model = None
        self.parameters = None
        self.init_dftb_model(model, parameters)
        self.init_dftb_settings(settings)

    def create_results_instance(self):
        """
        Create an instance of the matching results object for this job.
        """
        return dftbresults(self)

    def result_filenames(self):
        fns = super().result_filenames()
        return fns + [os.path.join('ams.results', f) for f in ['dftb.rkf']]

    def init_dftb_settings(self, settings=None):
        if settings is None:
            self.settings = dftbsettings()
        else:
            self.settings = settings

    def init_dftb_model(self, model='SCC-DFTB', parameters='Dresden'):
        self.model = model
        self.parameters = parameters

    def get_engine_block(self):
        block = " Engine DFTB\n"
        block += f"  ResourcesDir {self.parameters} \n"
        block += f"  Model {self.model} \n"
        block += " EndEngine\n\n"
        return block

    def print_jobtype(self):
        raise NotImplementedError

    def print_molecule(self):

        print("   Molecule")
        print("   ========")
        print()
        print(self.mol)
        print()

    def print_settings(self):

        print("   Settings")
        print("   ========")
        print()
        print(f"   Model: {self.model}")
        print(f"   DFTB Parameters: {self.parameters}")
        print()
        print(self.settings)
        print()

    def print_extras(self):
        pass

    def print_jobinfo(self):
        print(" " + 50 * "-")
        print(" Running " + self.print_jobtype())
        print()

        self.print_molecule()

        self.print_settings()

        self.print_extras()


class dftbsinglepointjob(dftbjob):
    """
    DFTB geometry optimization
    """

    def __init__(self, mol, model='SCC-DFTB', parameters='Dresden', settings=None):
        super().__init__(mol, model=model, parameters=parameters, settings=settings, amstask='SinglePoint')

    def print_jobtype(self):
        return "AMS DFTB single point job"


class dftbgeometryjob(dftbjob):
    """
    DFTB geometry optimization
    """

    def __init__(self, mol, model='SCC-DFTB', parameters='Dresden', settings=None):
        super().__init__(mol, model=model, parameters=parameters, settings=settings, amstask='GeometryOptimization')

    def print_jobtype(self):
        return "AMS DFTB geomtery optimization job"


class dftbfreqjob(dftbsinglepointjob):

    def __init__(self, mol, model='SCC-DFTB', parameters='Dresden', settings=None):
        super().__init__(mol, model, parameters, settings)

    def get_properties_block(self):
        block = " Properties\n"
        block += "  NormalModes true\n"
        block += " End\n\n"
        return block
