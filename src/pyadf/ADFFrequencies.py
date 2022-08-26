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
 Job and results for ADF frequency calculations.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    adffreqjob
"""

from .ADFSinglePoint import adfsinglepointjob
from .ADFGeometry import adfgeometryjob, adfgeometrysettings


class adffreq_mixin(adfsinglepointjob):
    """
    ADF frequencies job mixin, can be combined with either L{adfsinglepointjob} and L{adfgeometryjob}.
    """

    def __init__(self, *arg, **kwargs):
        self.deuterium_list = kwargs['deuterium']
        del kwargs['deuterium']

        super().__init__(*arg, **kwargs)

    def get_atoms_block(self):
        block = " ATOMS [Angstrom]\n"
        if self.deuterium_list is not None:
            for atom in range(1, self.get_molecule().get_number_of_atoms() + 1):
                if atom in self.deuterium_list:
                    block += self.get_molecule().print_coordinates(atoms=[atom], index=False, suffix='mass=2.014101778')
                else:
                    block += self.get_molecule().print_coordinates(atoms=[atom], index=False)
        else:
            block += self.get_molecule().print_coordinates(index=False)
        block += " END\n"
        return block

    def get_properties_block(self):
        block = super().get_properties_block()
        block += "Properties \n"
        block += " NormalModes Yes \n"
        block += "END\n\n"
        return block

    def get_other_blocks(self):
        block = super().get_other_blocks()
        block += self.get_frequencies_block()
        return block

    # noinspection PyMethodMayBeStatic
    def get_frequencies_block(self):
        block = " AnalyticalFreq \n"
        block += "  PrintNormalModeAnalysis Yes \n"
        block += " END\n\n"
        return block

    def print_molecule(self):

        print("   Molecule")
        print("   ========")
        print()
        print(self.get_molecule())
        print()

        if self.deuterium_list:

            print("   List of Deuterium atoms: ", self.deuterium_list)
            print()


class adffreqjob(adffreq_mixin, adfgeometryjob):
    """
    A job class for ADF frequency calculations WITH geometry optimization
    """
    def __init__(self, mol, basis, settings=None, geometrysettings=None, core=None, frozen_atoms=None,
                 pointcharges=None, electricfield=None, deuterium=None, options=None):

        if geometrysettings is None:
            gs = adfgeometrysettings(converge={'Gradients': '1e-4'})
        else:
            gs = geometrysettings

        super().__init__(mol, basis, settings, geometrysettings=gs, core=core, frozen_atoms=frozen_atoms,
                         pointcharges=pointcharges, electricfield=electricfield,
                         deuterium=deuterium, options=options)

    # noinspection PyMethodMayBeStatic
    def print_jobtype(self):
        return "ADF geometry optimization frequency job (Analytical Frequencies)"


class adfsinglepointfreqjob(adffreq_mixin):
    """
    A job class for single-point ADF frequency calculations.
    """
    def __init__(self, mol, basis, settings=None, core=None, pointcharges=None, electricfield=None,
                 deuterium=None, options=None):

        super().__init__(mol, basis, settings, core=core,
                         pointcharges=pointcharges, electricfield=electricfield,
                         deuterium=deuterium, options=options)

    # noinspection PyMethodMayBeStatic
    def print_jobtype(self):
        return "ADF single-point frequency job (Analytical Frequencies)"
