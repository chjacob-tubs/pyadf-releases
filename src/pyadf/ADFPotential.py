# This file is part of 
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2012 by Christoph R. Jacob, S. Maya Beyhan,
# Rosa E. Bulo, Andre S. P. Gomes, Andreas Goetz, Karin Kiewisch,
# Jetze Sikkema, and Lucas Visscher 
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
 Potential reconstruction with ADF.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""

from ADFSinglePoint import adfspjobdecorator
from PlotGrids      import adfgrid
from Errors         import PyAdfError

import os

class adfimportgridjob (adfspjobdecorator):
    '''
    ADF job using the IMPORTGRID option.
    '''
    
    def __init__ (self, wrappedjob, grid):
        adfspjobdecorator.__init__(self, wrappedjob)
        if not isinstance(grid, adfgrid) :
            raise PyAdfError("Grid to import must be an adfgrid.")
        self._grid = grid
    
    def get_other_blocks (self):
        block = adfspjobdecorator.get_other_blocks(self)
        block += " IMPORTGRID grid.t10 \n\n"
        if self._checksum_only :
            block += self._grid.get_grid_block(True)
        return block
    
    def before_run (self):
        adfspjobdecorator.before_run(self)
        self._grid.get_grid_tape10('grid.t10')
    
    def after_run (self):
        adfspjobdecorator.after_run(self)
        os.remove('grid.t10')

    def convert_results (self, results):
        res = adfspjobdecorator.convert_results(self, results)
        res._grid = self._grid
        return res

class adfpotentialjob (adfimportgridjob):
    '''
    ADF potential reconstruction job.
    '''
    
    def __init__ (self, wrappedjob, refdens, startpot=None, potoptions=None):
        if not isinstance(refdens.grid, adfgrid) :
            raise PyAdfError("Reference density must be on adfgrid.")
        adfimportgridjob.__init__(self, wrappedjob, refdens.grid)
        self._refdens = refdens
        self._startpot = startpot
        if potoptions is None:
            self._potoptions = {}
        else:
            self._potoptions = potoptions
    
    def get_other_blocks (self):
        block = adfimportgridjob.get_other_blocks(self)
        block += " CONSTRUCTPOT \n"
        block += "   CPBASIS\n"
        block += "   IMPORTDENS refdens.t41 \n"
        if self._checksum_only :
            block += self._refdens.get_checksum()
        if self._startpot is not None :
            block += "   STARTPOT startpot.t41 \n"
            if self._checksum_only :
                block += self._startpot.get_checksum()
        for opt, val in sorted(self._potoptions.items()) :
            block += "   %s %s \n" % (opt, str(val))
        block += " END \n\n"
        return block
    
    def before_run (self):
        import kf
        
        adfimportgridjob.before_run(self)
        f = kf.kffile('refdens.t41')
        if self._refdens.nspin == 2 :
            self._refdens.grid.write_grid_to_t41(f)
            values = self._refdens['alpha'].get_values()
            f.writereals('SCF', 'Density_A', values.reshape((values.size,), order='Fortran'))
            values = self._refdens['beta'].get_values()
            f.writereals('SCF', 'Density_B', values.reshape((values.size,), order='Fortran'))
        else:
            self._refdens.grid.write_grid_to_t41(f)
            values = self._refdens.get_values()
            f.writereals('SCF', 'Density', values.reshape((values.size,), order='Fortran'))
        f.close()

        if self._startpot is not None :
            f = kf.kffile('startpot.t41')
            self._startpot.grid.write_grid_to_t41(f)
            if self._refdens.nspin == 2 :
                if self._startpot.nspin == 2:
                    values = self._startpot['alpha'].get_values()
                    f.writereals('Potential', 'Total_A', values.reshape((values.size,), order='Fortran'))
                    values = self._startpot['beta'].get_values()
                    f.writereals('Potential', 'Total_B', values.reshape((values.size,), order='Fortran'))
                else :
                    values = self._startpot.get_values()
                    f.writereals('Potential', 'Total_A', values.reshape((values.size,), order='Fortran'))
                    f.writereals('Potential', 'Total_B', values.reshape((values.size,), order='Fortran'))
            else:
                values = self._startpot.get_values()
                f.writereals('Potential', 'Total', values.reshape((values.size,), order='Fortran'))
            f.close()
    
    def after_run (self):
        adfimportgridjob.after_run(self)
        os.remove('refdens.t41')
        if self._startpot is not None :
            os.remove('startpot.t41')
        
