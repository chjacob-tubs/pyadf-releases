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
 Job and results for ADF FDE calculations.
 This module is derived from ADF_3FDE.
 It implements parallel and non-parallel freeze-thaw.

 @author:       Christoph Jacob
 @author:       Samuel Fux
 @organization: ETH Zurich and Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

"""

from BaseJob import metajob
from ADFSinglePoint import adfsinglepointjob
from ADFFragments import fragment, adffragmentsjob
from ADFPotential import adfpotentialjob
from Plot.Grids import adfgrid
from Utils import Bohr_in_Angstrom
from pyadf.Errors import PyAdfError

import os.path


class adfaccurateembeddingresults(object):

    def __init__(self):
        self.super = None
        self.frag1 = None
        self.frag2 = None

        self.locorb_set1 = None
        self.locorb_set2 = None

        self.pot = None
        self.dens_err = None

    def write_accurate_potential(self, grid, outdir):
        # construct initial guess for the potential and save it to a variable
        startpot = self.frag1.get_potential(grid=grid, pot='total')

        pot_nuc = self.frag2.get_potential(grid, pot='nuc')
        pot_coul = self.super.get_locorb_coulpot(grid, orbs=self.locorb_set2)

        pot_xc = self.super.get_potential(grid, pot='xc')
        pot_xc = pot_xc - self.pot.get_potential(grid=grid, pot='xc')

        sum_pot = pot_xc + pot_coul + pot_nuc
        sum_pot.get_xyzvfile(os.path.join(outdir, "sum_pot.xyzv"))

        rec_pot = self.pot.get_potential(grid, pot='reconstructed')
        final_pot = startpot + rec_pot
        super_pot = self.super.get_potential(grid, pot='total')

        # difference between the supermolecular and the reconstructed potential
        diff_pot = super_pot - final_pot
        diff_pot.get_xyzvfile(os.path.join(outdir, "diff_pot.xyzv"))

    def write_approximate_potentials(self, grid, kin_funcs, outdir):
        # FIXME: add NDSD and CJCORR here

        for func in kin_funcs:
            #  get kinetic potential for the supermolecular system
            kinpot_tot = self.super.get_kinetic_potential(grid, func=func)

            # calculate approximate kinetic-energy potential from the reconstructed density
            apots = self.pot.get_kinetic_potential(grid, func=func)

            # difference potential
            apot = kinpot_tot - apots

            # write potential to file
            apot.get_xyzvfile(os.path.join(outdir, "approx_" + func + ".xyzv"))


class adfaccurateembeddingjob(metajob):

    def __init__(self, frag1, frag2, basis, ghostbasis, settings, potoptions):
        metajob.__init__(self)
        self.frag1 = frag1
        self.frag2 = frag2

        self.supermol = frag1 + frag2

        # switch off symmetry (just to be sure)
        self.frag1.set_symmetry('NOSYM')
        self.frag2.set_symmetry('NOSYM')
        self.supermol.set_symmetry('NOSYM')

        # if requested, add ghost basis functions to the active fragment
        if ghostbasis:
            self.frag1 = self.frag1.add_as_ghosts(self.frag2)

        self.basis = basis
        self.settings = settings

        self.potoptions = potoptions

        # maximum number of iterations in potential reconstruction
        if 'CYCLES' in self.potoptions:
            self.cyc = self.potoptions['CYCLES']
            del self.potoptions['CYCLES']
        else:
            self.cycles = 500

    def assign_locorbs_to_fragments(self, results):

        # get number of electrons for each fragment and check that it is even
        # then, number of electrons divided by two is the number of localized orbitals

        number_of_locorbs = results.super.get_number_of_electrons()
        if number_of_locorbs % 2 == 1:
            raise PyAdfError('odd number of electrons for supermolecule')
        else:
            number_of_locorbs = number_of_locorbs / 2

        number_of_locorbs_1 = results.frag1.get_number_of_electrons()
        if number_of_locorbs_1 % 2 == 1:
            raise PyAdfError('odd number of electrons for subsystem 1')
        else:
            number_of_locorbs_1 = number_of_locorbs_1 / 2

        number_of_locorbs_2 = results.frag2.get_number_of_electrons()
        if number_of_locorbs_2 % 2 == 1:
            raise PyAdfError('odd number of electrons for supsystem 2')
        else:
            number_of_locorbs_2 = number_of_locorbs_2 / 2

        # now calculate the density for each localized orbital of the
        # supermolecular calculation and assign it to the subsystem
        # that is closest to its "center of density"

        def calc_center_of_density(dens):
            """
            Calculates the center of density of a given density and returns a list [x,y,z].
            """
            import numpy
            center = numpy.zeros(3)
            for w, c, d in zip(dens.grid.weightiter(), dens.grid.coorditer(), dens.valueiter()):
                center += w * d * c
            return Bohr_in_Angstrom * center

        adfGrid = adfgrid(results.super)

        results.locorb_set1 = []
        results.locorb_set2 = []
        for i in range(1, number_of_locorbs + 1):
            locorb_dens = results.super.get_locorb_density(grid=adfGrid, orbs=[i])
            center_of_density = calc_center_of_density(locorb_dens)
            print "center of density of localized orbital nr. ", i, "  : ", center_of_density

            dist1 = self.frag1.distance_to_point(center_of_density, ghosts=False)
            dist2 = self.frag2.distance_to_point(center_of_density, ghosts=False)
            print "distance to subsystem 1: ", dist1
            print "distance to subsystem 2: ", dist2
            if dist1 < dist2 - 1e-4:
                results.locorb_set1.append(i)
            elif dist1 > dist2 + 1e-4:
                results.locorb_set2.append(i)
            else:
                if results.frag1.get_charge() < results.frag2.get_charge():
                    results.locorb_set1.append(i)
                elif results.frag2.get_charge() < results.frag1.get_charge():
                    results.locorb_set2.append(i)
                else:
                    raise PyAdfError('localized orbital can not be assigned clearly to one of the subsystems')

        print "locorb_set1 :", results.locorb_set1
        print "locorb_set2 :", results.locorb_set2

        if not (len(results.locorb_set1) == number_of_locorbs_1):
            raise PyAdfError('wrong number of localized orbitals for subsystem 1')
        if not (len(results.locorb_set2) == number_of_locorbs_2):
            raise PyAdfError('wrong number of localized orbitals for subsystem 2')

    def metarun(self):

        results = adfaccurateembeddingresults()

        # run supermolecular calculation
        self.settings.set_lmo(True)
        self.settings.set_save_tapes([21, 10])
        self.settings.set_exactdensity(True)

        results.super = adfsinglepointjob(self.supermol, self.basis, settings=self.settings).run()

        # orbital localization is turned off for the other calculations
        self.settings.set_lmo(False)

        # fragment density is obtained from frag1 and frag2
        results.frag1 = adfsinglepointjob(self.frag1, self.basis, settings=self.settings).run()
        results.frag2 = adfsinglepointjob(self.frag2, self.basis, settings=self.settings).run()

        # get localized orbital densities for subsystems 1 and 2

        self.assign_locorbs_to_fragments(results)

        adfGrid = adfgrid(results.super)

        sys1_dens = results.super.get_locorb_density(grid=adfGrid, orbs=results.locorb_set1)
        sys1_dens.prop = 'density scf'

        # results.sys2_dens = results.super.get_locorb_density(grid=adfGrid, orbs=results.locorb_set2)
        # results.sys2_dens.prop = 'density scf'

        print 'reference calculations finished'

        #  Now comes the potential reconstruction:

        self.settings.set_ncycles(self.cyc)

        job = adffragmentsjob([fragment(results.frag1, [self.frag1])], self.basis, settings=self.settings)
        potjob = adfpotentialjob(job, sys1_dens, potoptions=self.potoptions)

        print 'Running potential reconstruction job ... '
        results.pot = potjob.run()

        # Calculate the error in the density

        # the reconstructed density
        rec_dens = results.pot.get_density(grid=adfGrid)

        # the difference density
        diff_dens = sys1_dens - rec_dens

        # calculate integral over the difference density
        results.dens_error = diff_dens.integral(func=lambda x: abs(x))

        return results
