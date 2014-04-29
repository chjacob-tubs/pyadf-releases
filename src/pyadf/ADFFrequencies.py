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
 Job and results for ADF frequency calculations.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    adffreqjob
"""

from ADFGeometry import adfgeometryjob

class adffreqjob (adfgeometryjob) :
    """
    A job class for ADF frequency calculations.

    Corresponding results class: L{adfsinglepointresults}
    
    @group Initialization:
        __init__
    """

    def __init__ (self, mol, basis, settings=None) :
        """
        Constructor for ADF frequency jobs.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}
        
        @param basis: 
            A string specifying the basis set to use (e.g. C{basis='TZ2P'}).
            Alternatively, a dictionary can be given listing different basis sets
            for different atom types. Such a dictionary must contain an entry "default"
            giving the basis set to use for all other atom types
            (e.g. C{basis={default:'DZP', 'C':'TZ2P'}}).
        @type basis: str or dict
        
        @param settings: The settings for this calculation, see L{adfsettings}
        @type settings: L{adfsettings}        
        """        
        adfgeometryjob.__init__ (self, mol, basis, settings)

    def get_geometry_block (self) :
        block  = " AnalyticalFreq \n"
        block += " END\n\n"
        return block

    def print_jobtype (self) :
        return "ADF frequency job (Analytical Frequencies)"
