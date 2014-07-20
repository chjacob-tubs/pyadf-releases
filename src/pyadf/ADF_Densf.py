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
 Defines plotting related classes.

#FIXME: This needs to be updated!

 The central class for plotting is L{densfresults}. Instances of
 L{densfresults} represent densities (or related quantities like
 the Laplacian of the density on a grid. The easiest way to
 obtain such a density is to use the L{adfsinglepointresults.get_density}
 method.

 >>> res = adfsinglepointjob(mol, basis='DZP').run()
 >>> dens = res.get_density()
 >>> dens.get_tape41('density.t41')   # save the density to a TAPE41 file
 >>> dens.get_cubfile('density.cub')  # save the density to a Gaussian-type cube file

 The C{get_density} method takes an optional argument C{grid}, which
 specifies which type of grid is used. This can be instances of either
 L{cubegrid}, for an evenly spaced cubic grid, and L{adfgrid}, for the
 integration grid used by ADF (imported from a TAPE10 file). See the
 documentation for L{cubegrid} and L{adfgrid}.

 Densities can be added and substracted.

     - Obtain the sum of the densities of fragments 1 and 2:

     >>> grid = cubegrid(mol1 + mol2) # create a common grid for both fragments
     >>>
     >>> # density of fragment 1
     >>> res_frag1 = frag1_job.run()
     >>> dens1 = res_frag1.get_density(grid=grid)
     >>>
     >>> # density of fragment 2
     >>> res_frag2 = frag2_job.run()
     >>> dens2 = res_frag2.get_density(grid=grid)
     >>>
     >>> # total density
     >>> tot_dens = dens1 + dens2
     >>> tot_dens.get_cubfile ('totdens.cub')

     - Obtain the difference density (e.g., between supermolecule and FDE):

     >>> res_supermol = supermol_job.run()
     >>> res_fde = fde_job.run()
     >>>
     >>> grid = adfgrid(res_supermol)
     >>>
     >>> dens_supermol = res_supermol.get_density(grid=grid)     # supermolecular density
     >>> dens_fde      = res_fde.get_density(grid=grid)          # total FDE density
     >>>
     >>> diffdens = dens_supermol - dens_fde
     >>> diffdens.get_cubfile('diffdens.cub')

 Finally, it is also possible to integrate densities and functions
 of the density. Not that this should be done using L{adfgrid} in
 order to obtain accurate results.

     - Number of electrons:

     >>> print dens.integral()

     - Integrated absolute and RMS error in the density:

     >>> print "integrated absolute error: ", diffdens.integral (lambda x: abs(x))
     >>> print "integrated RMS error: ", math.sqrt(diffdens.integal (lambda x: x*x))

 More examples on the use of the routines in the L{Plot} module can be
 found in the tests C{ADFPlot} and C{ADF3FDE_Dialanine}.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Density Results:
     densfresults
 @group Density Generation:
     densfjob
"""

import numpy

from ADFBase import adfjob, adfresults
from Errors import PyAdfError

from Plot.Grids import cubegrid
from Plot.Properties import PlotProperty
from Plot.GridFunctions import GridFunctionFactory


class densfresults(adfresults):

    """
    Class representing densities (and related quantities) as produced
    by the ADF utility program C{densf} (or its replacement C{cjdensf}).

    Instances of C{densfresults} are returned by the run method of
    L{densfjob}. Alternatively, the methods L{adfsinglepointresults.get_density}
    and L{adfsinglepointresults.get_laplacian} can be used to obtain the
    electron density or the Laplacian of the electron density directly from
    an L{adfsinglepointresults} object.
    The documentation of the respective methods provides more detailed
    information.

    Even though an instance of C{densfresults} can contain different
    quantities given on a grid, in the following it will mostly be
    refered to as "density".

    The most common tasks that can be done with C{densfresults} are:
      - obtain the numerical values using L{get_values}
      - get a TAPE41 file of the density with L{get_tape41}
      - get a Gaussian-type cube file of the density with L{get_cubfile}
      - adding and substracting of different densities (see L{__add__} and L{__sub__})
      - integration over (a function of) the density using L{integral}

    @group Access to density:
        get_values, valueiter, get_tape41, get_cubfile
    @group Manipulation of densities:
        __add__, __sub__
    @group Integration over density:
        integral
    @group Retrieval of properties derived from density/potential:
        get_efield_in_point_grid, get_electronic_dipole_moment_grid
    @group Analysis of densities per Voronoi cell:
        integral_voronoi, get_electronic_dipole_voronoi
    @undocumented:
        _add_with_factor, _delete_values, _read_values_from_tape41, _write_tape41
    """

    def __init__(self, j=None, grid=None):
        """
        Constructor for densfresults.
        """
        adfresults.__init__(self, j)

        if j is not None:
            if grid is not None:
                raise PyAdfError('grid must not be passed if adfres is present')
            self.grid = self.job.grid
            self.prop = self.job.prop
            self._values = None
        elif grid is not None:
            import numpy
            self.grid = grid
            self.prop = None
            self._values = numpy.zeros((self.grid.npoints,))
        else:
            self.job = None
            self.files = None
            self.grid = None
            self.prop = None
            self._values = None

    def _read_values_from_tape41(self, spin=None):

        section, variable = self.prop.get_tape41_section_variable()
        vl = self.prop.vector_length

        if section is None:
            raise PyAdfError('Unknown property requested')

        if spin == 'alpha':
            ss = '_A'
        elif spin == 'beta':
            ss = '_B'
        else:
            ss = ''

        if vl == 1:
            values = self.get_result_from_tape(section, variable + ss, tape=41)
            values = values[:self.grid.npoints]

            if len(self.grid.shape) > 1:
                values = values.reshape(self.grid.shape, order='Fortran')
                values = values.reshape((self.grid.npoints,))
        else:
            components = self.prop.components
            values = numpy.empty((self.grid.npoints, vl))

            for i, c in enumerate(components):
                values_comp = self.get_result_from_tape(section, variable + ' ' + c + ss, tape=41)
                values_comp = values_comp[:self.grid.npoints]

                if len(self.grid.shape) > 1:
                    values_comp = values_comp.reshape(self.grid.shape, order='Fortran')
                    values_comp = values_comp.reshape((self.grid.npoints,))

                values[:, i] = values_comp

        return values

    def get_gridfunction(self):
        """
        Returns a grid function containing the values.
        """
        if self.prop.is_density:
            gf_type = 'density'
        elif self.prop.is_potential:
            gf_type = 'potential'
        else:
            gf_type = None

        checksum = self.job.get_checksum()

        if self.job.nspin == 2 and self.prop.is_unrestricted:
            values_alpha = self._read_values_from_tape41(spin='alpha')
            values_beta = self._read_values_from_tape41(spin='beta')

            gf = GridFunctionFactory.newGridFunctionUnrestricted(self.grid,
                                                                 values_alpha, values_beta,
                                                                 checksum, gf_type)

        else:
            values = self._read_values_from_tape41()

            gf = GridFunctionFactory.newGridFunction(self.grid, values, checksum, gf_type)

        return gf


class densfjob(adfjob):

    """
    A class for densf jobs.

    This can be used to obtain the density (and related quantities)
    on a grid, e.g. for plotting or integration.
    """

    def __init__(self, adfres, prop, grid=None, spacing=0.5, frag=None):
        """
        Constructor for densfjob.

        @param adfres:
            The results of the ADF job for which the density
            (or related quantity) should be calculated.
        @type  adfres: L{adfsinglepointresults} or subclass.
        @param prop:
            The property to calculate.
        @type  prop: L{PlotProperty}
        @param grid:
            The grid to use. If None, a default L{cubegrid} is used.
        @type  grid: subclass of L{grid}
        @param frag:
            Which fragment to use. Default is 'Active'.
            This can be used to get the densities of specific frozen
            fragments.
        @type frag: str
        """
        # pylint: disable=W0621

        adfjob.__init__(self)

        self._adfresults = adfres
        if grid is None:
            self.grid = cubegrid(adfres.get_molecule(), spacing)
        else:
            self.grid = grid
        if frag is None:
            self._frag = 'Active'
        else:
            self._frag = frag

        if not isinstance(prop, PlotProperty):
            raise PyAdfError('densfjob needs to be initialized with PlotProperty')
        else:
            self.prop = prop

        # consistency checks for properties that are not implemented

        if 'orbs' in self.prop.opts:
            if ('Loc' not in self.prop.opts['orbs']) and \
                    not (self.prop.opts['orbs'].keys() == ['A']):
                raise PyAdfError('CJDENSF only working for NSYM=1 (irrep A) orbitals')

        if self.prop.pclass == 'potential':
            if 'func' in self.prop.opts:
                if not self.prop.ptype in ['kinpot', 'nadkin']:
                    raise PyAdfError("Functional cannot be selected in CJDENSF "
                                     "with this potential type")

        self._olddensf = False

    @property
    def nspin(self):
        return self._adfresults.nspin

    def create_results_instance(self):
        return densfresults(self)

    def get_runscript(self):
        """
        Return a runscript for CJDENSF.
        """
        if self._olddensf:
            runscript = adfjob.get_runscript(self, program='densf', serial=True)
        else:
            runscript = adfjob.get_runscript(self, program='cjdensf', serial=True)

        return runscript

    def get_input(self):
        """
        Return an input file for CJDENSF.
        """
        inp = ""
        inp += self.grid.get_grid_block(self._checksum_only)
        if not self._olddensf:
            if self._frag == "ALL":
                inp += 'ALLFRAGMENTS \n'
            else:
                inp += 'FRAGMENT ' + self._frag + '\n'

        if self.prop.needs_locorbdens:
            inp += "LOCORBDENS \n"
            for i in self.prop.opts['orbs']['Loc']:
                inp += "%i \n" % i
            inp += "END\n"
        elif self.prop.needs_orbdens:
            inp += "ORBDENS \n"
            for i in self.prop.opts['orbs']['A']:
                inp += "%i \n" % i
            inp += "END\n"

        if self.prop.pclass == 'density':
            if self.prop.ptype == 'dens':
                inp += 'DENSITY SCF'
            elif self.prop.ptype == 'sqrgrad':
                inp += 'GRADIENT'
            elif self.prop.ptype == 'grad':
                inp += 'GRADIENT ALL'
            elif self.prop.ptype == 'lapl':
                inp += 'LAPLACIAN'
            elif self.prop.ptype == 'hess':
                inp += 'LAPLACIAN ALL'

            if self.prop.opts['fit']:
                inp += ' FIT'
            inp += '\n'

        elif self.prop.pclass == 'potential':
            if self.prop.ptype == 'kinpot':
                inp += 'KINPOT %s \n' % self.prop.opts['func'].upper()
            elif self.prop.ptype == 'embpot':
                inp += 'EMBPOT\n'
            elif self.prop.ptype == 'embcoul':
                inp += 'EMBPOT\n'
            elif self.prop.ptype == 'embnuc':
                inp += 'EMBPOT\n'
            elif self.prop.ptype == 'nadxc':
                inp += 'EMBPOT\n'
            elif self.prop.ptype == 'nadkin':
                inp += 'EMBPOT\n'
                if 'func' in self.prop.opts:
                    inp += 'NADKIN %s \n' % self.prop.opts['func'].upper()
            else:
                inp += "POTENTIAL %s \n" % self.prop.ptype.upper()

        elif self.prop.pclass == 'orbital':
            if self.prop.opts['irrep'] == "LOC":
                inp += "LOCORBITALS \n"
            else:
                inp += "ORBITALS \n"

        else:
            raise PyAdfError('Invalid property class in densfjob')

        inp += "END INPUT\n"

        if self._checksum_only:
            inp += self._adfresults.get_checksum()

        return inp

    def print_jobtype(self):
        return "DENSF job"

    def print_jobinfo(self):
        print " " + 50 * "-"
        print " Running " + self.print_jobtype()
        print
        print "   SCF taken from ADF job ", self._adfresults.fileid, " (results id)"
        print
        print "   Fragment used: ", self._frag
        print
        print "   Calculated property : ", self.prop.str
        print

    def before_run(self):
        self._adfresults.link_tape(21)
        self.grid.before_densf_run()

    def after_run(self):
        import os
        self.grid.after_densf_run()
        os.remove('TAPE21')
