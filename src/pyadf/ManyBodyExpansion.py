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
Implementation of many-body expansions, both energy-based and density-based.

 @author: Christoph Jacob
 @organization: TU Braunschweig
"""

import itertools
from Errors import PyAdfError
from BaseJob import metajob, results

import numpy as np
import xcfun


class MBEResults(results):
    """
    Class for MBE results.

    This store the results of all the fragment calculation up to the chosen
    order. From the results, many-body expanded energies and properties can
    be obtained.
    """

    def __init__(self, job):
        results.__init__(self, job)
        self.order = self.job.order
        self.res_by_comb = {}
        self.n_mol = self.job.nfrag

        self._toten_by_order_ = None

    def is_fde_job(self):
        return hasattr(self.res_by_comb[(0,)], 'get_nonfrozen_density')

    def mbe_interactions(self, res_to_en):
        int_energies_by_order = []
        int_energies_by_comb = {}

        for order in range(1, self.order+1):
            int_energies_by_order.append(0.0)

            for c in itertools.combinations(range(self.n_mol), order):
                int_energy = res_to_en(self.res_by_comb[c])
                for oo in range(1, order):
                    for cc in itertools.combinations(c, oo):
                        int_energy = int_energy - int_energies_by_comb[cc]
                int_energies_by_order[-1] += int_energy
                int_energies_by_comb[c] = int_energy

        return int_energies_by_order

    def mbe_interactions_lowmem(self, res_to_en):
        int_energies_by_order = []

        energies_order_1 = []
        for i in range(self.n_mol):
            energies_order_1.append(res_to_en(self.res_by_comb[(i,)]))

        def int_energies_by_comb(c):
            if len(c) == 1:
                return energies_order_1[c[0]]
            else:
                int_energy = res_to_en(self.res_by_comb[c])
                for oo in range(1, len(c)):
                    for cc in itertools.combinations(c, oo):
                        int_energy = int_energy - int_energies_by_comb(cc)
                return int_energy

        for order in range(1, self.order+1):
            int_energies_by_order.append(0.0)
            for comb in itertools.combinations(range(self.n_mol), order):
                int_energies_by_order[-1] += int_energies_by_comb(comb)

        return int_energies_by_order

    @property
    def _toten_by_order(self):
        if self._toten_by_order_ is None:
            self._toten_by_order_ = self.mbe_interactions(lambda res: res.get_total_energy())

        return self._toten_by_order_

    def get_total_energy(self, order=None):
        if order is None:
            order = self.order
        return sum(self._toten_by_order[:order])

    def get_interaction_energy(self, order=None):
        if order is None:
            order = self.order
        return sum(self._toten_by_order[1:order])

    def get_total_energies_by_order(self):
        return [self.get_total_energy(order=i+1) for i in range(self.order)]

    def get_interaction_energies_by_order(self):
        return [self.get_interaction_energy(order=i+1) for i in range(self.order)]

    def get_diffdens_by_order(self, grid, deriv=1):
        if self.is_fde_job():
            res_to_en = lambda res: res.get_nonfrozen_density(grid, order=deriv)
        else:
            res_to_en = lambda res: res.get_density(grid, order=deriv)
        return self.mbe_interactions_lowmem(res_to_en)

    def get_fitcorr_by_order(self, grid):
        if self.is_fde_job():
            res_to_en = lambda res: res.get_nonfrozen_density(grid) - res.get_nonfrozen_density(grid, fit=True)
        else:
            res_to_en = lambda res: res.get_density(grid) - res.get_density(grid, fit=True)
        return self.mbe_interactions_lowmem(res_to_en)

    def get_coulpot_by_order(self, grid):
        if self.is_fde_job():
            res_to_en = lambda res: res.get_nonfrozen_potential(grid, pot='coul')
        else:
            res_to_en = lambda res: res.get_potential(grid, pot='coul')
        return self.mbe_interactions_lowmem(res_to_en)


class MBEJob(metajob):
    """
    Class for an (energy-based) many-body expansion job.

    This job will perform all the fragment calculations up to a specified
    order and return a L{MBEResults} object, which stores the results of all
    the individual calculations and can provide MB-expanded energies and
    properties.
    """

    def __init__(self, mol_list, jobfunc, jobfunc_kwargs=None, order=2):
        """
        Constructor for MBEJob.

        @param mol_list: list of monomer molecules
        @type mol_list: list of L{molecule}s

        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type jobfunc: function with signature
                       calculate_mol(mol: molecule, frozen_mols: list of molecules, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type jobfunc_kwargs: dict or None

        @param order: many-body expansion order
        @type order: int
        """
        metajob.__init__(self)

        self.mol_list = mol_list
        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self.order = order

    @property
    def nfrag(self):
        return len(self.mol_list)

    def _assemble_active_mol(self, active_mol_nums):
        mol = self.mol_list[active_mol_nums[0]]
        for ii in active_mol_nums[1:]:
            mol = mol + self.mol_list[ii]
        return mol

    def _assemble_frozen_mols(self, active_mol_nums):
        frozen_mol_nums = []
        for i in range(self.nfrag):
            if i not in active_mol_nums:
                frozen_mol_nums.append(i)
        return [self.mol_list[i] for i in frozen_mol_nums]

    def create_results_instance(self):
        return MBEResults(self)

    def get_molecule(self):
        supermol = self.mol_list[0]
        for m in self.mol_list[1:]:
            supermol = supermol + m
        return supermol

    def metarun(self):
        mbe_results = self.create_results_instance()

        for order in range(1, self.order + 1):
            for c in itertools.combinations(range(self.nfrag), order):
                active_mol = self._assemble_active_mol(c)
                frozen_mols = self._assemble_frozen_mols(c)

                res = self.jobfunc(active_mol, frozen_mols, **self._jobfunc_kwargs)
                mbe_results.res_by_comb[c] = res

        return mbe_results


class DensityBasedMBEResults(results):
    """
    Class for density-based MBE results.

    This store the many-body expansion of the electron density as well as
    the relevant potentials and gives access to the required energy terms.
    """

    def __init__(self, job):
        results.__init__(self, job)
        self.order = self.job.order

        self.ebmbe_res = job.ebmbe_res

        self._diffdens_by_order = None
        self._fitcorr_by_order = None
        self._coulpot_by_order = None

        self._xc_by_order_ = None
        self._kin_by_order_ = None

    def get_density_by_order(self, order=None):
        if order is None:
            order = self.order
        return sum(self._diffdens_by_order[:order])

    def _calc_fun_by_order(self, func):

        def res_to_en(res, fun, isfde):
            if isfde:
                dens = res.get_nonfrozen_density(self.job.grid, order=1)
            else:
                dens = res.get_density(self.job.grid, order=1)
            endens = fun.eval_energy_n(density=dens[0].values, densgrad=dens[1].values)
            return np.dot(self.job.grid.get_weights(), endens)

        return self.ebmbe_res.mbe_interactions(lambda res: res_to_en(res, func, self.ebmbe_res.is_fde_job()))

    @property
    def _xc_by_order(self):
        if self._xc_by_order_ is None:
            self._xc_by_order_ = self._calc_fun_by_order(self.job.nadxc)
        return self._xc_by_order_

    @property
    def _kin_by_order(self):
        if self._kin_by_order_ is None:
            self._kin_by_order_ = self._calc_fun_by_order(self.job.nadkin)
        return self._kin_by_order_

    def get_coulomb_by_order(self):
        res_to_en = lambda res: res.get_result_from_tape('Total Energy', 'Coulomb energy') \
                                + res.get_result_from_tape('Total Energy', 'Nuclear repulsion energy')
        return self.ebmbe_res.mbe_interactions(res_to_en)

    def get_elint_1st_order(self):
        nntot = 0.0
        entot = 0.0
        eetot = 0.0
        for i in range(self.job.nfrag):
            resi = self.ebmbe_res.res_by_comb[(i,)]
            if self.ebmbe_res.is_fde_job():
                nucpot = resi.get_nonfrozen_potential(self.job.grid, pot='nuc')
                elpot = resi.get_nonfrozen_potential(self.job.grid, pot='coul')
            else:
                nucpot = resi.get_potential(self.job.grid, pot='nuc')
                elpot = resi.get_potential(self.job.grid, pot='coul')

            for j in range(self.job.nfrag):
                if not (i == j):
                    resj = self.ebmbe_res.res_by_comb[(j,)]
                    if self.ebmbe_res.is_fde_job():
                        dens = resj.get_nonfrozen_density(self.job.grid)
                        fitdens = resj.get_nonfrozen_density(self.job.grid, fit=True)
                    else:
                        dens = resj.get_density(self.job.grid)
                        fitdens = resj.get_density(self.job.grid, fit=True)

                    nnint = self.job.mol_list[i].get_nuclear_interaction_energy(self.job.mol_list[j])
                    enint = (nucpot * dens).integral()
                    eeint = (elpot * dens).integral()
                    eecorr = 0.5 * (elpot * fitdens).integral()

                    nntot = nntot + 0.5 * nnint
                    entot = entot + enint
                    eetot = eetot + eeint - eecorr
        return nntot + entot + eetot

    def get_elint(self, order):
        nucpot_tot = 0.0
        for i in range(self.job.nfrag):
            resi = self.ebmbe_res.res_by_comb[(i,)]
            if self.ebmbe_res.is_fde_job():
                nucpot_tot += resi.get_nonfrozen_potential(self.job.grid, pot='nuc')
            else:
                nucpot_tot += resi.get_potential(self.job.grid, pot='nuc')

        entot = (nucpot_tot * self._diffdens_by_order[order-1][0]).integral()
        entot += 0.5 * (sum(self._coulpot_by_order[:order-1]) * self._diffdens_by_order[order-1][0]).integral()
        entot += 0.5 * (sum(self._fitcorr_by_order[:order-1]) * self._coulpot_by_order[order-1]).integral()

        entot += 0.5 * (sum(self._diffdens_by_order[:order-1])[0] * self._coulpot_by_order[order-1]).integral()
        entot += 0.5 * (sum(self._coulpot_by_order[:order-1]) * self._fitcorr_by_order[order-1]).integral()

        eetot = 0.5 * (self._coulpot_by_order[order-1] * self._diffdens_by_order[order-1][0]).integral()
        eetot += 0.5 * (self._fitcorr_by_order[order-1] * self._coulpot_by_order[order-1]).integral()

        return entot + eetot

    def get_xccorr(self, order):
        totdens = sum(self._diffdens_by_order[:order])

        xc_endens = self.job.nadxc.eval_energy_n(density=totdens[0].values,
                                                 densgrad=totdens[1].values)
        xctot = np.dot(self.job.grid.get_weights(), xc_endens)

        return xctot - sum(self._xc_by_order[:order])

    def get_kincorr(self, order):
        totdens = sum(self._diffdens_by_order[:order])

        kin_endens = self.job.nadkin.eval_energy_n(density=totdens[0].values,
                                                   densgrad=totdens[1].values)
        kintot = np.dot(self.job.grid.get_weights(), kin_endens)

        return kintot - sum(self._kin_by_order[:order])

    def get_dbcorr(self, order):
        if order == 1:
            dbcorr = self.get_elint_1st_order() + self.get_xccorr(order=1) + self.get_kincorr(order=1)
        else:
            elint_by_order = [self.get_elint_1st_order()]
            coulomb_by_order = self.get_coulomb_by_order()

            for i in range(2, order+1):
                elint_by_order.append(self.get_elint(order=i))

            dbcorr = self.get_kincorr(order=order) + self.get_xccorr(order=order) \
                + sum(elint_by_order[0:order]) - sum(coulomb_by_order[1:order])

        return dbcorr

    def get_total_energy(self, order=None):
        if order is None:
            order = self.order
        return self.ebmbe_res.get_total_energy(order=order) + self.get_dbcorr(order=order)

    def get_interaction_energy(self, order=None):
        return self.get_total_energy(order=order) - self.ebmbe_res.get_total_energy(order=1)

    def get_total_energies_by_order(self):
        return [self.get_total_energy(order=i+1) for i in range(self.order)]

    def get_interaction_energies_by_order(self):
        return [self.get_interaction_energy(order=i+1) for i in range(self.order)]


class DensityBasedMBEJob(metajob):
    """
    Class for a density-based many-body expansion job.

    DensitybasedMBE jobs are based on a previous (energy-based) MBE job.
    Running this job will trigger the evaluation of the many-body expansion
    of the elctron density as well as the calculation of the different
    required energy terms.
    """

    def __init__(self, ebmbe_res, order=None, grid=None, nadkin=None, nadxc=None):
        """
        Constructor for DensityBasedMBEJob.

        @param ebmbe_res: results of a previous (energy-based) MBEJob
        @type ebmbe_res: L{MBEResults}

        @param order: many-body expansion order; if None, same as in mberes will be used
        @type order: int or None

        @param grid: the integration grid that will be used for the db expansion
        @type grid: subclass of L{Grid}

        @param nadkin: nonadditive kinetic-energy functional (default: PW91k)
        @type nadkin: str or XCFun Functional object

        @param nadxc: nonadditive xc functional
                      (default: XC functional from eb-MBE single points, if available)
        @type nadxc: str or XCFun Functional object
        """
        metajob.__init__(self)

        self.ebmbe_res = ebmbe_res
        if order is None:
            self.order = ebmbe_res.order
        else:
            self.order = order

        if grid is None:
            raise PyAdfError('DensityBasedMBEJob requires a supermolecular grid.')
        self.grid = grid

        if nadkin is None:
            self._nadkin = 'PW91k'
        elif isinstance(nadkin, xcfun.Functional):
            self._nadkin = nadkin
        elif isinstance(nadkin, str) and nadkin.upper() in ['TF', 'PW91K']:
            self._nadkin = nadkin
        else:
            raise PyAdfError('Invalid nonadditive kinetic-energy functional in db-MBE job')

        if nadxc is None:
            if hasattr(self.ebmbe_res.res_by_comb[(0,)].job, 'functional'):
                self._nadxc = self.ebmbe_res.res_by_comb[(0,)].job.functional
            elif hasattr(self.ebmbe_res.res_by_comb[(0,)].job, 'settings') and \
                    hasattr(self.ebmbe_res.res_by_comb[(0,)].job.settings, 'functional'):
                self._nadxc = self.ebmbe_res.res_by_comb[(0, )].job.settings.functional
            else:
                raise PyAdfError('No nonadditive xc functional specified in db-MBE job')
            if self._nadxc.startswith('GGA '):
                self._nadxc = self._nadxc[4:]
        elif isinstance(nadxc, xcfun.Functional):
            self._nadxc = nadxc
        elif isinstance(nadxc, str) and nadxc.upper() in ['LDA', 'BP', 'BP86', 'BLYP']:
            self._nadxc = nadxc
        else:
            raise PyAdfError('Invalid nonadditive xc functional in db-MBE job')

    @property
    def nfrag(self):
        return self.ebmbe_res.job.nfrag

    @property
    def mol_list(self):
        return self.ebmbe_res.job.mol_list

    @property
    def nadkin(self):
        if isinstance(self._nadkin, xcfun.Functional):
            return self._nadkin
        elif self._nadkin.upper() == 'TF':
            return xcfun.Functional({'tfk': 1.0})
        elif self._nadkin.upper() == 'PW91K':
            return xcfun.Functional({'pw91k': 1.0})
        else:
            raise PyAdfError('Unknown nonadditive kinetic energy functional '
                             + str(self._nadkin) + 'in db-MBE job')

    @property
    def nadxc(self):
        if isinstance(self._nadxc, xcfun.Functional):
            return self._nadxc
        elif self._nadxc.upper() == 'LDA':
            return xcfun.Functional({'lda': 1.0})
        elif self._nadxc.upper() in ['BP', 'BP86']:
            return xcfun.Functional({'BeckeX': 1.0, 'P86C': 1.0})
        elif self._nadxc.upper() in ['BLYP']:
            return xcfun.Functional({'BeckeX': 1.0, 'LYP': 1.0})
        else:
            raise PyAdfError('Unknown nonadditive xc functional '
                             + str(self._nadxc) + ' in db-MBE job')

    def create_results_instance(self):
        return DensityBasedMBEResults(self)

    def get_molecule(self):
        return self.ebmbe_res.job.get_molecule()

    def metarun(self):
        dbmbe_res = self.create_results_instance()

        dbmbe_res._diffdens_by_order = self.ebmbe_res.get_diffdens_by_order(grid=self.grid, deriv=1)
        dbmbe_res._coulpot_by_order = self.ebmbe_res.get_coulpot_by_order(grid=self.grid)

        dbmbe_res._fitcorr_by_order = self.ebmbe_res.get_fitcorr_by_order(grid=self.grid)

        return dbmbe_res
