# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2024 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Maria Chekmeneva,
# Thomas Dresselhaus, Kevin Focke, Andre S. P. Gomes, Andreas Goetz,
# Michal Handzlik, Karin Kiewisch, Moritz Klammler, Lars Ridder,
# Jetze Sikkema, Lucas Visscher, Johannes Vornweg, Michael Welzel,
# and Mario Wolter.
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
 Defines the L{molecule} class.

 This module defines the class molecule, which
 is used by PyADF to represent molecules
 (i.e., atomic coordinates, charge, ...)

 This file provides a kind of factory that creates
 either an OBFreeMolecule (default) or a OBMolecule

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""

from .. import PatternsLib
from ..Errors import PyAdfError

from . import OBFreeMolecule

try:
    from . import OBMolecule
except ImportError:
    OBMolecule = None

try:
    from . import RDMolecule
except ImportError:
    RDMolecule = None


class MoleculeFactory(metaclass=PatternsLib.Singleton):
    def __init__(self):
        if OBMolecule is None:
            self.molclass = "obfree"
        else:
            self.molclass = "openbabel"

    def use_molclass(self, molclass):
        """
        Set the molecule class that shall be used. Choices: "obfree", "openbabel", "rdkit".
        """
        if molclass is not None:
            self.molclass = molclass

    def makeMolecule(self, *args, **kwargs):
        if self.molclass == "openbabel":
            if OBMolecule is None:
                raise PyAdfError("Error: Could not import OBMolecule")
            else:
                return OBMolecule.OBMolecule(*args, **kwargs)
        elif self.molclass == "rdkit":
            if RDMolecule is None:
                raise PyAdfError("Error: Could not import RDMolecule")
            else:
                return RDMolecule.RDMolecule(*args, **kwargs)
        else:
            return OBFreeMolecule.OBFreeMolecule(*args, **kwargs)


def molecule(*args, **kwargs):
    return MoleculeFactory().makeMolecule(*args, **kwargs)
