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
 Job and results for ADF FDE calculations.
 This module is derived from ADF_3FDE.
 It implements parallel and non-parallel freeze-thaw.

 @author:       Andreas W. Goetz
 @author:       S. Maya Beyhan
 @organization: Vrije Universiteit Amsterdam (2008)

"""


from .Errors import PyAdfError

from .BaseJob import metajob, results
from .ADFFragments import fragmentlist, adffragmentsjob
from .Plot.Grids import cubegrid
from functools import reduce


class adffdesettings:
    """
    Class for settings of L{adffdejob}.
    """

    def __init__(self, occupations=None, vshift=None, packtape=False):
        self.occupations = None
        self.packtape = None
        self.vshift = None

        self.set_occupations(occupations)
        self.set_packtape(packtape)
        self.set_lshift(vshift)

    def set_occupations(self, occupations):
        self.occupations = occupations

    def set_packtape(self, packtape):
        if isinstance(packtape, str):
            if packtape == 'On':
                self.packtape = True
            elif packtape == 'Off':
                self.packtape = False
            else:
                raise PyAdfError('invalid packtape argument')
        else:
            self.packtape = packtape

    def set_lshift(self, vshift):
        self.vshift = vshift

    switch_occupations = set_occupations
    switch_packtape = set_packtape
    switch_lshift = set_lshift


class adffderesults(results):
    """
    Class for results of an ADF FDE calculation.

    @group Retrival of specific results:
        get_fragmentlist, get_density
    """

    def __init__(self, job, frags=None):
        super().__init__(job)
        self._frags = frags

    def set_fragmentlist(self, frags):
        self._frags = frags

    def get_fragmentlist(self):
        return self._frags

    def get_dipole_vector(self):
        import numpy
        dipole = numpy.zeros(3)
        for f in self._frags:
            dipole += f.results.get_dipole_vector()
        return dipole

    def get_density(self, grid=None, spacing=0.5, fit=False):
        if grid is None:
            grid = cubegrid(self.job.get_molecule(), spacing)

        dens = [f.results.get_nonfrozen_density(grid, fit=fit) for f in iter(self._frags)]
        dens = reduce(lambda x, y: x + y, dens)

        return dens


class adffdejob(metajob):
    """
    Class for an ADF FDE job.
    """

    def __init__(self, frags, basis, settings=None, core=None, pointcharges=None,
                 options=None, fde=None, fdeoptions=None, adffdesetts=None):
        """
        Initialize an FDE job.

        For most of the parameters see class L{adffragmentsjob}.

        @param fde: options for the FDE run;
                    In addition to the standard options Freeze-Thaw (FT) can be controlled;
                    By default a parallel FT will be done;
                    For normal FT use 'NORMALFT'
        @type  fde: dictionary

        """
        super().__init__()

        if isinstance(frags, list):
            self._frags = fragmentlist(frags)
        else:
            self._frags = frags

        self._basis = basis
        self._settings = settings
        self._core = core
        self._pc = pointcharges
        self._options = options
        self._adffdesettings = adffdesetts

        if fde is None:
            self._fde = {}
        else:
            import copy
            self._fde = copy.copy(fde)
        if 'RELAXCYCLES' in self._fde:
            self._cycles = self._fde['RELAXCYCLES'] + 1
            del self._fde['RELAXCYCLES']
        else:
            self._cycles = 1

        if 'NORMALFT' in self._fde:
            self._normalft = True
            del self._fde['NORMALFT']
        else:
            self._normalft = False

        if fdeoptions is None:
            self._fdeoptions = {}
        else:
            self._fdeoptions = fdeoptions

        # make all fragments frozen and apply fdeoptions
        for f in iter(self._frags):
            f.isfrozen = True
            f.set_fdeoptions(self._fdeoptions)

    def create_results_instance(self):
        return adffderesults(self)

    def get_molecule(self):
        return self._frags.get_total_molecule()

    def parallel_ft_run(self):
        """
        Perform parallel freeze-thaw cycles.

        Fragments are updated only after each full freeze-thaw cycle.
        """
        import copy

        frags_new = None
        frags_old = copy.deepcopy(self._frags)
        for i in range(self._cycles):

            print("-" * 50)
            print("Beginning FDE cycle (parallel FT)", i)

            frags_new = copy.deepcopy(frags_old)

            # frags_old: fragments of the previous cycle
            # frags_new: the updated fragments

            if self._adffdesettings is not None:
                if self._adffdesettings.occupations is not None:
                    self._settings.set_occupations(self._adffdesettings.occupations[0])
                else:
                    self._settings.set_occupations(None)

            for f_new, f_old in zip(iter(frags_new), iter(frags_old)):
                f_old.isfrozen = False

                job = adffragmentsjob(frags_old, self._basis, settings=self._settings,
                                      core=self._core, pointcharges=self._pc, options=self._options,
                                      fde=self._fde)
                f_new.results = job.run()
                if self._adffdesettings is not None:
                    if self._adffdesettings.packtape:
                        f_new.results.pack_tape()

                f_old.isfrozen = True
                self._settings.set_lshift(None)
                if self._adffdesettings is not None:
                    if self._adffdesettings.occupations is not None:
                        if len(self._adffdesettings.occupations) == 2:
                            self._settings.set_occupations(self._adffdesettings.occupations[1])
                        else:
                            self._settings.set_occupations(None)

            frags_old = copy.deepcopy(frags_new)

        r = self.create_results_instance()
        r.set_fragmentlist(frags_new)
        return r

    def normal_ft_run(self):
        """
        Normal freeze-thaw cycles: everything is updated immediately
        """
        import copy

        frags_new = copy.deepcopy(self._frags)
        for i in range(self._cycles):

            print("-" * 50)
            print("Beginning FDE cycle (normal FT)", i)

            if self._adffdesettings is not None:
                self._settings.set_lshift(self._adffdesettings.vshift)

            if self._adffdesettings is not None:
                if self._adffdesettings.occupations is not None:
                    self._settings.set_occupations(self._adffdesettings.occupations[0])
                else:
                    self._settings.set_occupations(None)

            for f_new in frags_new:
                f_new.isfrozen = False
                job = adffragmentsjob(frags_new, self._basis, settings=self._settings,
                                      core=self._core, pointcharges=self._pc, options=self._options,
                                      fde=self._fde)
                f_new.results = job.run()
                if self._adffdesettings is not None:
                    if self._adffdesettings.packtape:
                        f_new.results.pack_tape()

                f_new.isfrozen = True
                self._settings.set_lshift(None)

                if self._adffdesettings is not None:
                    if self._adffdesettings.occupations is not None:
                        if len(self._adffdesettings.occupations) == 2:
                            self._settings.set_occupations(self._adffdesettings.occupations[1])
                        else:
                            self._settings.set_occupations(None)

        r = self.create_results_instance()
        r.set_fragmentlist(frags_new)
        return r

    def metarun(self):
        if self._normalft:
            return self.normal_ft_run()
        else:
            return self.parallel_ft_run()
