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

import OBFreeMolecule
try:
    import OBMolecule
except ImportError:
    OBMolecule = None


class MoleculeFactory(object):
    __metaclass__ = PatternsLib.Singleton

    def __init__(self):
        self.openbabel = (OBMolecule is not None)

    def use_openbabel(self, ob):
        """
        Select whether Openbabel or Openbabel-free molecule class is used.

        Possible choices:
        ob = True : use Openbael; ob = False : do not use Openbabel;
        ob = None: detect whether Openbabel is available
        """
        if ob is None:
            self.openbabel = not (OBMolecule is None)
        else:
            self.openbabel = ob

    def makeMolecule(self, *args, **kwargs):
        if not self.openbabel:
            return OBFreeMolecule.Molecule(*args, **kwargs)
        else:
            return OBMolecule.OBMolecule(*args, **kwargs)


def molecule(*args, **kwargs):
    return MoleculeFactory().makeMolecule(*args, **kwargs)
