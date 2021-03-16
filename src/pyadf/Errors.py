# -*- coding: utf-8 -*-

# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2021 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Thomas Dresselhaus,
# Andre S. P. Gomes, Andreas Goetz, Michal Handzlik, Karin Kiewisch,
# Moritz Klammler, Lars Ridder, Jetze Sikkema, Lucas Visscher, and
# Mario Wolter.
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
 Defines exceptions used in PyADF.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""


class PyAdfError(Exception):
    """ Base class for exceptions in PyADF. """
    pass


class FileError(PyAdfError):
    """ Base class for exceptions in PyADF. """
    pass


class PTError(PyAdfError):
    """ Periodic table exception. """
    pass


class UnitsError(PyAdfError):
    """ Unit converter exception. """
    pass


class MoleculeError(PyAdfError):
    """ Molecule exception. """
    pass


class AtomError(PyAdfError):
    """ Atom exception. """
    pass


class BondError(PyAdfError):
    """ Bond exception. """
    pass
