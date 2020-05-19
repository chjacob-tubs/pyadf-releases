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
 The basics needed for ADF calculations: simple jobs and results.

 @author:       Christoph Jacob and others
 @organization: TU Braunschweig
 @contact:      c.jacob@tu-braunschweig.de

 @group Jobs:
    dftbjob
 @group Results:
    dftbresults
"""

from Errors import PyAdfError
from ADFBase import amssettings, amsresults, amsjob


class dftbsettings(amssettings):

    def __init__(self):
        pass

    def __str__(self):
        ss = amssettings.__str__(self)
        return ss


class dftbsinglepointjob(amsjob):
    """
    DFTB single point job
    """
    def __init__(self, mol, model='SCC-DFTB', parameters='Dresden', settings=None):
        if settings is None:
            mysettings = dftbsettings()
        else:
            mysettings = settings

        amsjob.__init__(self, mol, task='SinglePoint', settings=mysettings)
        self.init_dftb_model(model, parameters)
        self.init_dftb_settings(settings)

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
        block += "  ResourcesDir %s \n" % self.parameters
        block += "  Model %s \n" % self.model
        block += " EndEngine\n\n"
        return block

    def print_jobtype(self):
        return "AMS DFTB single point job"

    def print_molecule(self):

        print "   Molecule"
        print "   ========"
        print
        print self.mol
        print

    def print_settings(self):

        print "   Settings"
        print "   ========"
        print
        print "   Model: %s" % self.model
        print "   DFTB Parameters: %s" % self.parameters
        print
        print self.settings
        print

    def print_extras(self):
        pass

    def print_jobinfo(self):
        print " " + 50 * "-"
        print " Running " + self.print_jobtype()
        print

        self.print_molecule()

        self.print_settings()

        self.print_extras()


class dftbgeometryjob(dftbsinglepointjob):
    """
    DFTB geometry optimization
    """
    def __init__(self, mol, model='SCC-DFTB', parameters='Dresden', settings=None):
        if settings is None:
            mysettings = dftbsettings()
        else:
            mysettings = settings

        amsjob.__init__(self, mol, task='GeometryOptimization', settings=mysettings)
        self.init_dftb_model(model, parameters)

    def print_jobtype(self):
        return "AMS DFTB geomery optimization job"


class dftbfreqjob(dftbsinglepointjob):

    def __init__(self, mol, model='SCC-DFTB', parameters='Dresden', settings=None):
        dftbsinglepointjob.__init__(self, mol, model, parameters, settings)

    def get_properties_block(self):
        block = " Properties\n"
        block += "  NormalModes true\n"
        block += " End\n\n"
        return block

