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
 Potential reconstruction with ADF.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""

from .ADFSinglePoint import adfspjobdecorator
from pyadf.PyEmbed.Plot.Grids import adfgrid
from .Errors import PyAdfError

import os


class adfimportgridjob(adfspjobdecorator):
    """
    ADF job using the IMPORTGRID option.
    """

    def __init__(self, wrappedjob, grid):
        super().__init__(wrappedjob)
        if not isinstance(grid, adfgrid):
            raise PyAdfError("Grid to import must be an adfgrid.")
        self._grid = grid

    def get_other_blocks(self):
        block = super().get_other_blocks()
        block += " IMPORTGRID grid.t10 \n\n"
        if self._checksum_only:
            block += self._grid.get_grid_block(True)
        return block

    def before_run(self):
        super().before_run()
        self._grid.get_grid_tape10('grid.t10')

    def after_run(self):
        super().after_run()
        os.remove('grid.t10')


class adfpotentialjob(adfimportgridjob):
    """
    ADF potential reconstruction job.
    """

    def __init__(self, wrappedjob, refdens, startpot=None, potoptions=None):
        if not isinstance(refdens.grid, adfgrid):
            raise PyAdfError("Reference density must be on adfgrid.")
        super().__init__(wrappedjob, refdens.grid)
        self._refdens = refdens
        self._startpot = startpot
        if potoptions is None:
            self._potoptions = {}
        else:
            self._potoptions = potoptions

    def get_other_blocks(self):
        block = adfimportgridjob.get_other_blocks(self)
        block += " CONSTRUCTPOT \n"
        block += "   CPBASIS\n"
        block += "   IMPORTDENS refdens.t41 \n"
        if self._checksum_only:
            block += self._refdens.checksum
        if self._startpot is not None:
            block += "   STARTPOT startpot.t41 \n"
            if self._checksum_only:
                block += self._startpot.checksum
        for opt, val in sorted(self._potoptions.items()):
            block += f"   {opt} {str(val)} \n"
        block += " END \n\n"
        return block

    def before_run(self):
        from .kf import kf

        adfimportgridjob.before_run(self)
        f = kf.kffile('refdens.t41')
        if self._refdens.nspin == 2:
            self._refdens.grid.write_grid_to_t41(f)
            values = self._refdens['alpha'].get_values()
            f.writereals('SCF', 'Density_A', values.reshape((values.size,), order='F'))
            values = self._refdens['beta'].get_values()
            f.writereals('SCF', 'Density_B', values.reshape((values.size,), order='F'))
        else:
            self._refdens.grid.write_grid_to_t41(f)
            values = self._refdens.get_values()
            f.writereals('SCF', 'Density', values.reshape((values.size,), order='F'))
        f.close()

        if self._startpot is not None:
            f = kf.kffile('startpot.t41')
            self._startpot.grid.write_grid_to_t41(f)
            if self._refdens.nspin == 2:
                if self._startpot.nspin == 2:
                    values = self._startpot['alpha'].get_values()
                    f.writereals('Potential', 'Total_A', values.reshape((values.size,), order='F'))
                    values = self._startpot['beta'].get_values()
                    f.writereals('Potential', 'Total_B', values.reshape((values.size,), order='F'))
                else:
                    values = self._startpot.get_values()
                    f.writereals('Potential', 'Total_A', values.reshape((values.size,), order='F'))
                    f.writereals('Potential', 'Total_B', values.reshape((values.size,), order='F'))
            else:
                values = self._startpot.get_values()
                f.writereals('Potential', 'Total', values.reshape((values.size,), order='F'))
            f.close()

    def after_run(self):
        adfimportgridjob.after_run(self)
        os.remove('refdens.t41')
        if self._startpot is not None:
            os.remove('startpot.t41')


class adfimportembpotjob(adfimportgridjob):
    """
    ADF job importing an external embedding potential.

    FIXME: Only nspin=1 case implemented in ADF.
    """

    def __init__(self, wrappedjob, embpot):
        if not isinstance(embpot.grid, adfgrid):
            raise PyAdfError("Embedding potential must be on adfgrid.")
        super().__init__(wrappedjob, embpot.grid)
        self._embpot = embpot

    def get_other_blocks(self):
        block = adfimportgridjob.get_other_blocks(self)
        block += " IMPORTEMBPOT embpot.t41 \n"
        if self._checksum_only:
            block += self._embpot.checksum
        return block

    def before_run(self):
        from .kf import kf

        super().before_run()

        f = kf.kffile('embpot.t41')
        self._embpot.grid.write_grid_to_t41(f)
        if self._embpot.nspin == 2:
            raise PyAdfError("IMPORTEMBPOT not implemented for nspin=2")
        else:
            values = self._embpot.get_values()
            f.writereals('Potential', 'EmbeddingPot',
                         values.reshape((values.size,), order='F'))
        f.close()

    def after_run(self):
        super().after_run()
        os.remove('embpot.t41')
