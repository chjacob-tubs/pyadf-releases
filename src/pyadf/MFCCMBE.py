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
 Job and results for MFCC-MBE(2), MFCC-MBE(3), general MFCC and various
 interaction energy calculations of these methods.

 @author:       Johannes Vornweg
 @organization: TU Braunschweig

"""

import itertools
import numpy as np
from .Errors import PyAdfError
from .BaseJob import metajob, results
from .ADFSinglePoint import adfsinglepointresults
from .ADF_3FDE import cappedfragment, cappedfragmentlist
from .Turbomole import TurbomoleSinglePointResults
from .Orca import OrcaResults
from .Utils import Bohr_in_Angstrom

try:
    import xcfun
except ImportError:
    print('Warning: XcFun could not be imported. db-MBE will not be available.')
    xcfun = None

import time


class GeneralMFCCResults(results):

    def __init__(self, job, frags=None):
        super().__init__(job)
        self._frags = frags

    def set_fragmentlist(self, frags):
        self._frags = frags

    def get_fragmentlist(self):
        return self._frags

    def get_dipole_vector(self):
        """
        returns dipole vector of the total molecule
        """
        import numpy
        dipole = numpy.zeros(3)
        for f in self._frags.fragiter():
            dipole += f.results.get_dipole_vector()
        for c in self._frags.capiter():
            dipole -= c.results.get_dipole_vector()
        return dipole

    def get_total_energy(self):
        """
        returns total energy of the total molecule
        """
        frag_energies = [f.results.get_total_energy() for f in self._frags.fragiter()]
        cap_energies = [c.results.get_total_energy() for c in self._frags.capiter()]
        return sum(frag_energies) - sum(cap_energies)


class GeneralMFCCJob(metajob):

    def __init__(self, frags, jobfunc, jobfunc_kwargs=None):
        """
        Initialize a Turbomole, ORCA or ADF MFCC job.

        @param frags: the list of MFCC fragments
        @type  frags: L{cappedfragmentlist}
        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type  jobfunc: function with signature XXXsinglepointjob(mol: molecule, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type  jobfunc_kwargs: dict or None
        """
        super().__init__()

        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self._frags = frags

    def create_results_instance(self):
        return GeneralMFCCResults(self)

    def get_molecule(self):
        return self._frags.get_total_molecule()

    def metarun(self):
        import copy
        frags = copy.deepcopy(self._frags)

        frags.calculate_all(lambda mol: self.jobfunc(mol, **self._jobfunc_kwargs))
        r = self.create_results_instance()
        r.set_fragmentlist(frags)
        return r


class DensityBasedMFCCResults(results):
    """
    Class for db-MFCC results.
    """

    def __init__(self, job):
        super().__init__(job)
        self.energy = []

    def get_correction_energy(self):
        return self.energy[0][0]


class DensityBasedMFCCJob(metajob):

    def __init__(self, ebmfcc_res, totmol, grid, lowcost=True, nadkin=None, nadxc=None):
        """
        Initialize a db-MFCC job.

        @param ebmfcc_res: results of a previous (energy-based) MFCC Job
        @type  ebmfcc_res: L{mfccresults}
        @param totmol: total molecule
        @type  totmol: L{molecule}
        @param grid: grid of the total molecule
        @type  grid: adfgrid
        @param lowcost: optimized calculations
        @type  lowcost: boolean
        @param nadkin: nonadditive kinetic-energy functional (default: PW91k)
        @type  nadkin: str or XCFun Functional object
        @param nadxc: nonadditive xc functional (default: XC functional from eb-MBE single points, if available)
        @type  nadxc: str or XCFun Functional object
        """
        import xcfun

        super().__init__()

        self._grid = grid
        self.ebmfcc_res = ebmfcc_res
        self._totmol = totmol
        self._lowcost = lowcost

        if nadkin is None:
            self._nadkin = 'PW91k'
        elif isinstance(nadkin, xcfun.Functional):
            self._nadkin = nadkin
        elif isinstance(nadkin, str) and nadkin.upper() in ['TF', 'PW91K']:
            self._nadkin = nadkin
        else:
            raise PyAdfError('Invalid nonadditive kinetic-energy functional in db-MFCC-MBE job')

        ebmfccfragres = self.ebmfcc_res.get_fragmentlist().frags[0].results
        if nadxc is None:
            if hasattr(ebmfccfragres.job, 'functional'):
                self._nadxc = ebmfccfragres.job.functional
            elif hasattr(ebmfccfragres.job, 'settings') and \
                    hasattr(ebmfccfragres.job.settings, 'functional'):
                self._nadxc = ebmfccfragres.job.settings.functional
            else:
                raise PyAdfError('No nonadditive xc functional specified in db-MFCC-MBE job')
            if self._nadxc is None:
                raise PyAdfError('No nonadditive xc functional specified in db-MFCC-MBE job')
            if self._nadxc.startswith('GGA '):
                self._nadxc = self._nadxc[4:]
        elif isinstance(nadxc, xcfun.Functional):
            self._nadxc = nadxc
        elif isinstance(nadxc, str) and nadxc.upper() in ['LDA', 'BP', 'BP86', 'BLYP', 'PBE']:
            self._nadxc = nadxc
        else:
            raise PyAdfError('Invalid nonadditive xc functional in db-MFCC-MBE job')

    @property
    def fraglist(self):
        """
        returns the list of cappedfragments
        """
        fraglist = []
        for frag in self.ebmfcc_res.get_fragmentlist().fragiter():
            fraglist.append(frag)
        return fraglist

    @property
    def nfrag(self):
        """
        returns the number of cappedfragments
        """
        return len(self.fraglist)

    @property
    def caplist(self):
        """
        returns the list of caps
        """
        caplist = []
        for cap in self.ebmfcc_res.get_fragmentlist().capiter():
            caplist.append(cap)
        return caplist

    @property
    def ncap(self):
        """
        returns the number of caps
        """
        return len(self.caplist)

    @property
    def nadkin(self):
        import xcfun

        if isinstance(self._nadkin, xcfun.Functional):
            return self._nadkin
        elif self._nadkin.upper() == 'TF':
            return xcfun.Functional({'tfk': 1.0})
        elif self._nadkin.upper() == 'PW91K':
            return xcfun.Functional({'pw91k': 1.0})
        else:
            raise PyAdfError('Unknown nonadditive kinetic energy functional '
                             + str(self._nadkin) + 'in db-MFCC-MBE job')

    @property
    def nadxc(self):
        import xcfun

        if isinstance(self._nadxc, xcfun.Functional):
            return self._nadxc
        elif self._nadxc.upper() == 'LDA':
            return xcfun.Functional({'lda': 1.0})
        elif self._nadxc.upper() in ['BP', 'BP86']:
            return xcfun.Functional({'BeckeX': 1.0, 'P86C': 1.0})
        elif self._nadxc.upper() in ['BLYP']:
            return xcfun.Functional({'BeckeX': 1.0, 'LYP': 1.0})
        elif self._nadxc.upper() in ['PBE']:
            return xcfun.Functional({'pbex': 1.0, 'pbec': 1.0})
        else:
            raise PyAdfError('Unknown nonadditive xc functional ' + str(self._nadxc) + ' in db-MFCC-MBE job')

    def create_results_instance(self):
        return DensityBasedMFCCResults(self)

    def get_total_molecule(self):
        """
        Returns the total molecule.

        @rtype: L{molecule}
        """
        return self.ebmfcc_res.job.get_total_molecule()

    def get_nuclear_repulsion_energy(self, mol):
        """
        Return the electrostatic interaction energy between the nuclei of this molecule.
        """
        inten = 0.0
        for i in range(len(mol.get_coordinates())):
            for j in range(len(mol.get_coordinates())):
                coord1, coord2 = mol.get_coordinates()[i], mol.get_coordinates()[j]
                atomNum1, atomNum2 = mol.get_atomic_numbers()[i], mol.get_atomic_numbers()[j]
                if i < j:
                    dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)
                    dist = dist / Bohr_in_Angstrom
                    inten = inten + atomNum1 * atomNum2 / dist
        return inten

    def get_dbcorr(self):

        timelist = []
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("START ", current_time))

        totmol = self._totmol

        rhotot = 0.0
        rhototfit = 0.0
        rhototfitcorr = 0.0
        vnuctot = 0.0
        vcoultot = 0.0
        nnint = 0.0
        enint = 0.0
        eeint = 0.0
        eecorr = 0.0
        xcint = 0.0
        kinint = 0.0

        # iterate over all fragments
        for i in self.fraglist:
            # get densities and potentials
            rho = i.results.get_density(self._grid, order=1)
            rhofit = i.results.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.results.get_potential(self._grid, pot='nuc')
            vcoul = i.results.get_potential(self._grid, pot='coul')

            # sum up all fragment densities and potentials
            rhotot = rho + rhotot
            rhototfit = rhofit + rhototfit
            rhototfitcorr = rhofitcorr + rhototfitcorr
            vnuctot += vnuc + vnuctot
            vcoultot += vcoul + vcoultot

            # calculate properties for the fragment on supermolecular grid
            nnint -= i.results.get_nuclear_repulsion_energy()
            enint -= np.dot(self._grid.weights, (rho[0] * vnuc).get_values())
            eeint -= 0.5 * np.dot(self._grid.weights, (rho[0] * vcoul).get_values())
            eecorr -= 0.5 * np.dot(self._grid.weights, (rhofitcorr * vcoul).get_values())
            xcint -= np.dot(self._grid.weights,
                            self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))
            kinint -= np.dot(self._grid.weights,
                             self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))

        # iterate over all caps
        for i in self.caplist:
            # get densities and potentials
            rho = i.results.get_density(self._grid, order=1)
            rhofit = i.results.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.results.get_potential(self._grid, pot='nuc')
            vcoul = i.results.get_potential(self._grid, pot='coul')

            # subtract all cap densities and potentials
            rhotot -= rho
            rhototfit -= rhofit
            rhototfitcorr -= rhofitcorr
            vnuctot -= vnuc
            vcoultot -= vcoul

            # calculate properties for the cap on supermolecular grid
            nnint += i.results.get_nuclear_repulsion_energy()
            enint += np.dot(self._grid.weights, (rho[0] * vnuc).get_values())
            eeint += 0.5 * np.dot(self._grid.weights, (rho[0] * vcoul).get_values())
            eecorr += 0.5 * np.dot(self._grid.weights, (rhofitcorr * vcoul).get_values())
            xcint += np.dot(self._grid.weights,
                            self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))
            kinint += np.dot(self._grid.weights,
                             self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))

        # calculate properties for total molecule on supermolecular grid
        nnint += self.get_nuclear_repulsion_energy(totmol)
        enint += np.dot(self._grid.weights, (rhotot[0] * vnuctot).get_values())
        eeint += 0.5 * np.dot(self._grid.weights, (rhotot[0] * vcoultot).get_values())
        eecorr += 0.5 * np.dot(self._grid.weights, (rhototfitcorr * vcoultot).get_values())
        xcint += np.dot(self._grid.weights,
                        self.nadxc.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values))
        kinint += np.dot(self._grid.weights,
                         self.nadkin.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values))

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("END ", current_time))

        return nnint + enint + eeint + eecorr + xcint + kinint, timelist

    def get_dbcorr_lowcost(self):

        timelist = []
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("START ", current_time))

        totmol = self._totmol

        rhotot = 0.0
        rhototfit = 0.0
        rhototfitcorr = 0.0
        vnuctot = 0.0
        vcoultot = 0.0
        nnint = 0.0
        energy = 0.0

        # iterate over all fragments
        for i in self.fraglist:

            # get densities and potentials
            rho = i.results.get_density(self._grid, order=1)
            rhofit = i.results.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.results.get_potential(self._grid, pot='nuc')
            vcoul = i.results.get_potential(self._grid, pot='coul')

            # sum up all fragment densities and potentials
            rhotot = rho + rhotot
            rhototfit = rhofit + rhototfit
            rhototfitcorr = rhofitcorr + rhototfitcorr
            vnuctot = vnuc + vnuctot
            vcoultot = vcoul + vcoultot

            # subtract the fragment interactions from the final energy
            nnint -= i.results.get_nuclear_repulsion_energy()
            energy -= (rho[0] * vnuc).get_values()
            energy -= 0.5 * (rho[0] * vcoul).get_values()
            energy -= 0.5 * (rhofitcorr * vcoul).get_values()
            energy -= self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)
            energy -= self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)

        # iterate over all caps
        for i in self.caplist:

            # get densities and potentials
            rho = i.results.get_density(self._grid, order=1)
            rhofit = i.results.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.results.get_potential(self._grid, pot='nuc')
            vcoul = i.results.get_potential(self._grid, pot='coul')

            # subtract all cap densities and potentials
            rhotot -= rho
            rhototfit -= rhofit
            rhototfitcorr -= rhofitcorr
            vnuctot -= vnuc
            vcoultot -= vcoul

            # add the cap interactions to the final energy
            nnint += i.results.get_nuclear_repulsion_energy()
            energy += (rho[0] * vnuc).get_values()
            energy += 0.5 * (rho[0] * vcoul).get_values()
            energy += 0.5 * (rhofitcorr * vcoul).get_values()
            energy += self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)
            energy += self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)

        # add the interactions for the total molecule to the final energy
        nnint += self.get_nuclear_repulsion_energy(totmol)
        energy += (rhotot[0] * vnuctot).get_values()
        energy += 0.5 * (rhotot[0] * vcoultot).get_values()
        energy += 0.5 * (rhototfitcorr * vcoultot).get_values()
        energy += self.nadxc.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values)
        energy += self.nadkin.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values)

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("END ", current_time))

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("START ", current_time))

        # calculate the energy on the supermolecular grid
        correction = np.dot(self._grid.weights, energy)

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("END ", current_time))

        return correction + nnint, timelist

    def metarun(self):

        db_res = self.create_results_instance()

        if self._lowcost:
            energy1, times = self.get_dbcorr_lowcost()
            db_res.energy.append((energy1, times))
        else:
            energy1, times = self.get_dbcorr()
            db_res.energy.append((energy1, times))

        return db_res


class MFCCInteractionResults(results):

    def __init__(self, job):
        super().__init__(job)
        self.frag_res_by_comb = {}
        self.cap_res_by_comb = {}
        self.frag_elint_corr_by_comb = {}
        self.cap_elint_corr_by_comb = {}

    def get_frag_interaction_energy(self):
        """
        returns total interaction energy of all fragment-ligand combinations
        """
        fragenergy = 0.0
        for i in self.frag_res_by_comb:
            energy = self.frag_res_by_comb[i][0].get_total_energy() \
                     - (self.frag_res_by_comb[i][1].get_total_energy()
                        + self.frag_res_by_comb[i][2].get_total_energy())
            fragenergy += energy
        return fragenergy

    def get_cap_interaction_energy(self):
        """
        returns total interaction energy of all cap-ligand combinations
        """
        capenergy = 0.0
        for i in self.cap_res_by_comb:
            energy = self.cap_res_by_comb[i][0].get_total_energy() \
                     - (self.cap_res_by_comb[i][1].get_total_energy()
                        + self.cap_res_by_comb[i][2].get_total_energy())
            capenergy += energy
        return capenergy

    def get_elint_corr(self):
        """
        returns electrostatic interaction energy
        """
        frag_elint_corr = 0.0
        for i in self.frag_elint_corr_by_comb:
            frag_elint_corr += self.frag_elint_corr_by_comb[i]
        cap_elint_corr = 0.0
        for i in self.cap_elint_corr_by_comb:
            cap_elint_corr += self.cap_elint_corr_by_comb[i]
        return frag_elint_corr - cap_elint_corr

    def get_total_interaction_energy(self):
        """
        returns total interaction energy
        """
        if not self.frag_elint_corr_by_comb:
            return self.get_frag_interaction_energy() - self.get_cap_interaction_energy()
        else:
            return self.get_frag_interaction_energy() - self.get_cap_interaction_energy() + self.get_elint_corr()


class MFCCInteractionJob(metajob):

    def __init__(self, frags, ligand, jobfunc, jobfunc_kwargs=None, cutoff=None, elintcorr=False):
        """
        Initialize a Turbomole or ORCA MFCC Interaction job.

        @param frags: the list of MFCC fragment
        @type  frags: L{cappedfragmentlist}
        @param ligand: ligand molecule
        @type  ligand: molecule
        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type  jobfunc: function with signature adfsinglepointjob(mol: molecule, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type  jobfunc_kwargs: dict or None
        @param cutoff: distance cutoff in Angstrom for calculating combinations
        @type  cutoff: int or float
        @param elintcorr: adds electrostatic correction for combinations outside the cutoff range
        @type  elintcorr: boolean
        """
        super().__init__()

        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self._frags = frags
        self._ligand = ligand
        self._cutoff = cutoff
        self._elintcorr = elintcorr

    @property
    def fraglist(self):
        """
        returns list of all fragments
        """
        fraglist = []
        for frag in self._frags.fragiter():
            fraglist.append(frag.mol)
        return fraglist

    @property
    def nfrag(self):
        """
        returns number of all fragments
        """
        return len(self.fraglist)

    @property
    def caplist(self):
        """
        returns list of all caps
        """
        caplist = []
        for cap in self._frags.capiter():
            caplist.append(cap.mol)
        return caplist

    @property
    def ncap(self):
        """
        returns number of all fragments
        """
        return len(self.caplist)

    def get_electrostatic_interaction_energy(self, frag1, frag2, frag1_charges, frag2_charges):
        """
        Return the electrostatic interaction energy between charges of two fragments

        @param frag1: fragment 1
        @type  frag1: molecule
        @param frag2: fragment 2
        @type  frag2: molecule
        @param frag1_charges: list of charges (multipole derived charges MDC-q)
        @type  frag1_charges: list
        @param frag2_charges: list of charges (multipole derived charges MDC-q)
        @type  frag2_charges: list
        """
        import numpy

        inten = 0.0
        for coord1, charge1 in zip(frag1.get_coordinates(), frag1_charges):
            for coord2, charge2 in zip(frag2.get_coordinates(), frag2_charges):
                dist = numpy.sqrt(
                    (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)
                dist = dist / Bohr_in_Angstrom
                inten = inten + charge1 * charge2 / dist
        return inten

    def create_results_instance(self):
        return MFCCInteractionResults(self)

    def metarun(self):
        print('>  Starting MFCC Interaction Job')
        mfcc_inten_results = self.create_results_instance()
        lig_res = self.jobfunc(self._ligand, **self._jobfunc_kwargs)
        ligand_charges = lig_res.get_multipolederiv_charges(level='MDC-q')

        print('>  Starting Fragment-Ligand Calculations')
        for i, frag in enumerate(self.fraglist):
            print('> Calculating Fragment', i + 1, 'of', self.nfrag, 'with Ligand')

            # check if fragment-ligand distance is in cutoff range
            fragligdist = frag.distance(self._ligand)
            cutoffbool = False
            if (self._cutoff and fragligdist <= self._cutoff) or self._cutoff is None:
                cutoffbool = True

            # if fragment-ligand distance is in cutoff range calculate interaction
            if cutoffbool:
                totmol = frag + self._ligand
                tot_res = self.jobfunc(totmol, **self._jobfunc_kwargs)
                frag_res = self.jobfunc(frag, **self._jobfunc_kwargs)
                mfcc_inten_results.frag_res_by_comb[i] = [tot_res, frag_res, lig_res]

            # if fragment-ligand distance is not in cutoff range but electrostatic correction is turned on
            # calculate electrostatic interaction
            elif self._elintcorr and not cutoffbool:
                print('>  Distance between Fragment and Ligand greater than the cutoff of', self._cutoff, 'Angstrom')
                print('>  Calculating electrostatic interaction energy')
                frag_res = self.jobfunc(frag, **self._jobfunc_kwargs)
                frag_charges = frag_res.get_multipolederiv_charges(level='MDC-q')
                elintcorr = self.get_electrostatic_interaction_energy(frag, self._ligand, frag_charges, ligand_charges)
                mfcc_inten_results.frag_elint_corr_by_comb[i] = elintcorr

            else:
                print('>  Distance between Fragment and Ligand greater than the cutoff of', self._cutoff, 'Angstrom')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')

        print('>  Starting Cap-Ligand Calculations')
        for i, cap in enumerate(self.caplist):
            print('> Calculating Cap', i + 1, 'of', self.ncap, 'with Ligand')

            # check if cap-ligand distance is in cutoff range
            capligdist = cap.distance(self._ligand)
            cutoffbool = False
            if (self._cutoff and capligdist <= self._cutoff) or self._cutoff is None:
                cutoffbool = True

            # if cap-ligand distance is in cutoff range calculate interaction
            if cutoffbool:
                totmol = cap + self._ligand
                tot_res = self.jobfunc(totmol, **self._jobfunc_kwargs)
                cap_res = self.jobfunc(cap, **self._jobfunc_kwargs)
                mfcc_inten_results.cap_res_by_comb[i] = [tot_res, cap_res, lig_res]

            # if cap-ligand distance is not in cutoff range but electrostatic correction is turned on
            # calculate electrostatic interaction
            elif self._elintcorr and not cutoffbool:
                print('>  Distance between Cap and Ligand greater than the cutoff of', self._cutoff, 'Angstrom')
                print('>  Calculating electrostatic interaction energy')
                cap_res = self.jobfunc(cap, **self._jobfunc_kwargs)
                cap_charges = cap_res.get_multipolederiv_charges(level='MDC-q')
                elintcorr = self.get_electrostatic_interaction_energy(cap, self._ligand, cap_charges, ligand_charges)
                mfcc_inten_results.cap_elint_corr_by_comb[i] = elintcorr

            else:
                print('>  Distance between Cap and Ligand greater than the cutoff of', self._cutoff, 'Angstrom')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')

        return mfcc_inten_results


class MFCCMBE2Results(results):
    """
    Class for MFCCMBE(2) results.
    """

    def __init__(self, job):
        super().__init__(job)
        self.frag_res = []
        self.cap_res = []
        self.overlap_res_by_comb = {}
        self.nooverlap_res_by_comb = {}
        self.trimer_res_by_comb = {}
        self.fragcap_res_by_comb = {}
        self.capcap_res_by_comb = {}

    def get_energy_function(self, res):
        """
        returns the total energy function depending on the type of results

        @param res: results
        @type  res: results object
        """
        if isinstance(res, TurbomoleSinglePointResults):
            return res.get_energy()
        elif isinstance(res, OrcaResults):
            return res.get_total_energy()
        elif isinstance(res, adfsinglepointresults):
            return res.get_total_energy()

    def get_fragfrag_energy(self):
        """
        returns total interaction energy of all fragment-fragment combinations
        """
        # calculate interaction for joint 1-2 Dimers
        overlapenergy = 0.0
        for i in self.overlap_res_by_comb:
            # i is a tuple with the combination (0, 1)
            # overlap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [dimer_res, cap_res] with the dimer results and cap results
            # frag_res is a list of all fragment results
            energy = self.get_energy_function(self.overlap_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.frag_res[i[0]])
                        + self.get_energy_function(self.frag_res[i[1]])
                        - self.get_energy_function(self.overlap_res_by_comb[i][1]))
            overlapenergy += energy

        # calculate interaction for disjoint Dimers (> 1-3)
        nooverlapenergy = 0.0
        for i in self.nooverlap_res_by_comb:
            # i is a tuple with the combination (0, 1)
            # nooverlap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the dimer results
            # frag_res is a list of all fragment results
            energy = self.get_energy_function(self.nooverlap_res_by_comb[i]) \
                     - (self.get_energy_function(self.frag_res[i[0]])
                        + self.get_energy_function(self.frag_res[i[1]]))
            nooverlapenergy += energy

        # calculate interaction for 1-3 Dimers
        trimerenergy = 0.0
        for i in self.trimer_res_by_comb:
            # i is a tuple with the combination (0, 1)
            # trimer_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the trimer results
            # overlap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [dimer_res, cap_res] with the dimer results and cap results
            # frag_res is a list of all fragment results
            dimer12 = (i[0], i[0] + 1)
            dimer23 = (i[1] - 1, i[1])
            energy = (self.get_energy_function(self.trimer_res_by_comb[i])
                      - (self.get_energy_function(self.overlap_res_by_comb[dimer12][0])
                         + self.get_energy_function(self.overlap_res_by_comb[dimer23][0])
                         - self.get_energy_function(self.frag_res[i[0] + 1])))
            trimerenergy += energy

        return overlapenergy + nooverlapenergy + trimerenergy

    def get_fragcap_energy(self):
        """
        returns total interaction energy of all fragment-cap combinations
        """
        fragcapenergy = 0.0
        # calculate interaction for fragment-cap combinations
        for i in self.fragcap_res_by_comb:
            # i is a tuple with the combination (0, 1)
            # fragcap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the fragment-cap
            # dimer results
            # frag_res is a list of all fragment results
            # cap_res is a list of all cap results
            energy = self.get_energy_function(self.fragcap_res_by_comb[i]) \
                     - (self.get_energy_function(self.frag_res[i[0]])
                        + self.get_energy_function(self.cap_res[i[1]]))
            fragcapenergy += energy
        return fragcapenergy

    def get_capcap_energy(self):
        """
        returns total interaction energy of all cap-cap combinations
        """
        capcapenergy = 0.0
        # calculate interaction for cap-cap combinations
        for i in self.capcap_res_by_comb:
            # i is a tuple with the combination (0, 1)
            # capcap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the cap-cap
            # dimer results
            # cap_res is a list of all cap results
            energy = self.get_energy_function(self.capcap_res_by_comb[i]) \
                     - (self.get_energy_function(self.cap_res[i[0]])
                        + self.get_energy_function(self.cap_res[i[1]]))
            capcapenergy += energy
        return capcapenergy

    def get_total_interaction_energy(self):
        """
        returns total interaction energy
        """
        fragfragenergy = self.get_fragfrag_energy()
        fragcapenergy = self.get_fragcap_energy()
        capcapenergy = self.get_capcap_energy()
        totalintenergy = fragfragenergy - fragcapenergy + capcapenergy
        return totalintenergy

    def get_mfcc_energy(self):
        """
        returns mfcc total energy
        """
        mfccenergy = 0.0
        for res in self.frag_res:
            mfccenergy += self.get_energy_function(res)
        for res in self.cap_res:
            mfccenergy -= self.get_energy_function(res)
        return mfccenergy

    def get_total_energy(self):
        """
        returns total energy
        """
        if self.job._order == 1:
            return self.get_mfcc_energy()
        elif self.job._order == 2:
            return self.get_mfcc_energy() + self.get_total_interaction_energy()

    def print_number_of_calculations(self):
        """
        returns number of calculated combinations
        """
        self.overlap_res_by_comb = {}
        self.nooverlap_res_by_comb = {}
        self.trimer_res_by_comb = {}
        self.fragcap_res_by_comb = {}
        self.capcap_res_by_comb = {}

        sumoffrags = len(self.overlap_res_by_comb) + len(self.nooverlap_res_by_comb) + len(self.trimer_res_by_comb)

        print('> Number of calculated combinations: ')
        print('> n(Frag-Frag): %d' % sumoffrags)
        print('> n(Frag-Cap):  %d' % len(self.fragcap_res_by_comb))
        print('> n(Cap-Cap):   %d' % len(self.capcap_res_by_comb))
        print('>', 20 * '-')
        print('> Sum:          %d' % (sumoffrags + len(self.fragcap_res_by_comb) + len(self.capcap_res_by_comb)))


class MFCCMBE2Job(metajob):

    def __init__(self, frags, jobfunc, jobfunc_kwargs=None, caps='mfcc', onlyffterms=False, order=2,
                 cutoff=None, printchargeinfo=False):
        """
        Initialize a MFCC-MBE(2) job.

        @param frags: list of capped fragments
        @type  frags: L{cappedfragmentlist}
        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type  jobfunc: function with signature adfsinglepointjob(mol: molecule, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type  jobfunc_kwargs: dict or None
        @param caps: 'mfcc' or 'hydrogen'
        @type  caps: str
        @param onlyffterms: if True only frag-frag terms are calculated
        @type  onlyffterms: Boolean
        @param order: many-body expansion order
        @type  order: int
        @param cutoff: distance cutoff in Angstrom for calculating combinations
        @type  cutoff: int or float
        @param printchargeinfo: if True the charges of the dimers are printed
        @type  printchargeinfo: boolean
        """
        super().__init__()

        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self._cutoff = cutoff
        self._onlyffterms = onlyffterms
        self._order = order
        self._caps = caps
        self._frags = frags
        self._printchargeinfo = printchargeinfo

    @property
    def fraglist(self):
        """
        returns the list of cappedfragments
        """
        fraglist = []
        for frag in self._frags.fragiter():
            fraglist.append(frag)
        return fraglist

    @property
    def nfrag(self):
        """
        returns the number of cappedfragments
        """
        return len(self.fraglist)

    @property
    def caplist(self):
        """
        returns the list of caps
        """
        caplist = []
        for cap in self._frags.capiter():
            caplist.append(cap)
        return caplist

    @property
    def ncap(self):
        """
        returns the number of caps
        """
        return len(self.caplist)

    @property
    def nfragcombi(self):
        """
        returns theorethical number of fragment-fragment combinations
        """
        return len(list(itertools.combinations(list(range(self.nfrag)), self._order)))

    @property
    def ncapcombi(self):
        """
        returns theorethical number of cap-cap combinations
        """
        return len(list(itertools.combinations(list(range(self.ncap)), self._order)))

    def create_results_instance(self):
        return MFCCMBE2Results(self)

    def dimercalc(self, monomer1, monomer2, overlap=False):
        """
        returns results dimer or dimer and cap

        @param monomer1: frag/cap molecule 1
        @type  monomer1: cappedfragment
        @param monomer2: frag/cap molecule 2
        @type  monomer2: cappedfragment
        @param overlap: True when overlap of 1-2 dimers
        @type  overlap: boolean
        """
        dimer = monomer1.merge_fragments(monomer2)
        dimer_res = self.jobfunc(dimer.mol, **self._jobfunc_kwargs)
        # if 1-2 Dimer calculate Dimer and the cap between the monomers
        if overlap:
            cap = dimer._overlapping_caps[0]
            if len(dimer.get_overlapping_caps()) > 1:
                raise PyAdfError("Too many overlapping caps !")
            cap_res = self.jobfunc(cap.mol, **self._jobfunc_kwargs)
            return [dimer_res, cap_res]
        # if disjoint dimer calculate dimer
        else:
            return dimer_res

    def trimercalc(self, trimer):
        """
        returns results for trimer

        @param trimer: trimer
        @type  trimer: cappedfragment
        """
        trimer_res = self.jobfunc(trimer.mol, **self._jobfunc_kwargs)
        return trimer_res

    def metarun(self):

        mfccmbe_results = self.create_results_instance()

        # calculate all fragments
        for frag in self.fraglist:
            fragres = self.jobfunc(frag.mol, **self._jobfunc_kwargs)
            mfccmbe_results.frag_res.append(fragres)

        # calculate all caps
        for cap in self.caplist:
            capres = self.jobfunc(cap.mol, **self._jobfunc_kwargs)
            mfccmbe_results.cap_res.append(capres)

        if self._order == 2:
            # FRAG-FRAG INTERACTIONS
            counter = 1
            print('>  Starting Fragment-Fragment Calculations')
            for c in itertools.combinations(list(range(self.nfrag)), self._order):
                print('>  Fragment-Fragment Combination', counter, 'of', self.nfragcombi)
                print('>  Consisting of Fragments', c[0] + 1, 'and', c[1] + 1, 'of', self.nfrag, 'Fragments')
                monomer1 = self.fraglist[c[0]]
                monomer2 = self.fraglist[c[1]]
                dimer = monomer1.merge_fragments(monomer2)

                if self._printchargeinfo:
                    print('> Charge Fragment ' + str(c[0] + 1) + ':', monomer1.mol.get_charge())
                    print('> Charge Fragment ' + str(c[1] + 1) + ':', monomer2.mol.get_charge())
                    print('> Charge Dimer:', dimer.mol.get_charge())

                # check if fragment-fragment distance is in cutoff range
                fragfragdist = monomer1.mol.distance(monomer2.mol)
                cutoffbool = False
                if (self._cutoff and fragfragdist <= self._cutoff) or self._cutoff is None:
                    cutoffbool = True

                overlapping_caps = dimer.get_overlapping_caps()

                if len(overlapping_caps) > 1:
                    # TOO MANY CAPS
                    raise PyAdfError("Handeling more than one overlapping cap not implemented yet!")

                # if joint 1-2 dimer and distance is in cutoff range calculate combination
                elif len(overlapping_caps) == 1 and cutoffbool:
                    # joint frag-frag inten
                    dimerres = self.dimercalc(monomer1, monomer2, overlap=True)
                    mfccmbe_results.overlap_res_by_comb[c] = dimerres

                # if disjoint dimer (> 1-3) and distance is in cutoff range calculate combination
                elif len(overlapping_caps) == 0 and fragfragdist > 0.0 and cutoffbool:
                    # disjoint frag-frag inten
                    dimerres = self.dimercalc(monomer1, monomer2, overlap=False)
                    mfccmbe_results.nooverlap_res_by_comb[c] = dimerres

                # if 1-3 dimer and distance is in cutoff range and ACE-NME Caps are used calculate combination
                elif len(overlapping_caps) == 0 and fragfragdist == 0.0 and self._caps == 'mfcc' and cutoffbool:
                    # 1-3-Fragment handeling for ACE-NME Caps
                    midmonomer = self.fraglist[c[0] + 1]
                    dimer12 = monomer1.merge_fragments(midmonomer)
                    trimer = dimer12.merge_fragments(monomer2)
                    trimerres = self.trimercalc(trimer)
                    mfccmbe_results.trimer_res_by_comb[c] = trimerres

                else:
                    print('>  Distance between Fragments greater than the cutoff of', self._cutoff, 'Angstrom')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')
                counter += 1

            # FRAG-CAP INTERACTIONS
            if not self._onlyffterms:
                print('>  Starting Fragment-Cap Calculations')
                for i, frag in enumerate(self.fraglist):
                    for j, cap in enumerate(self.caplist):
                        print('>  Fragment', i + 1, 'of', self.nfrag, 'with Cap', j + 1, 'of', self.ncap)

                        # check if fragment-fragment distance is in cutoff range
                        fragcapdist = frag.mol.distance(cap.mol)
                        cutoffbool = False
                        if (self._cutoff and fragcapdist <= self._cutoff) or self._cutoff is None:
                            cutoffbool = True

                        # if fragment-cap distance is in cutoff range calculate combination
                        if fragcapdist > 0.0 and cutoffbool:
                            fragcapres = self.dimercalc(frag, cap, overlap=False)
                            mfccmbe_results.fragcap_res_by_comb[(i, j)] = fragcapres

                        elif fragcapdist == 0.0:
                            print('>  Fragment and Cap are too close')
                            print('>  Skipping Combination')
                            print(' ' + 50 * '-')

                        else:
                            print('>  Distance between Fragment and Cap greater than the cutoff of',
                                  self._cutoff, 'Angstrom')
                            print('>  Skipping Combination')
                            print(' ' + 50 * '-')

            else:
                print('>  Not calculating Fragment-Cap-Terms')
                print(' ' + 50 * '-')

            # CAP-CAP INTERACTIONS
            if not self._onlyffterms:
                capcounter = 1
                print('>  Starting Cap-Cap Calculations')
                for c in itertools.combinations(list(range(self.ncap)), self._order):
                    print('>  Cap-Cap-combination', capcounter, 'of', self.ncapcombi)
                    print('>  Consisting of Caps', c[0] + 1, 'and', c[1] + 1, 'of', self.ncap, 'Caps')
                    cap1 = self.caplist[c[0]]
                    cap2 = self.caplist[c[1]]

                    # check if fragment-fragment distance is in cutoff range
                    capcapdist = cap1.mol.distance(cap2.mol)
                    cutoffbool = False
                    if (self._cutoff and capcapdist <= self._cutoff) or self._cutoff is None:
                        cutoffbool = True

                    # if cap-cap distance is in cutoff range calculate combination
                    if capcapdist > 0.0 and cutoffbool:
                        capcapres = self.dimercalc(cap1, cap2, overlap=False)
                        mfccmbe_results.capcap_res_by_comb[c] = capcapres

                    elif capcapdist == 0.0:
                        print('>  Caps are too close')
                        print('>  Skipping Combination')
                        print(' ' + 50 * '-')

                    else:
                        print('>  Distance between Caps greater than the cutoff of', self._cutoff, 'Angstrom')
                        print('>  Skipping Combination')
                        print(' ' + 50 * '-')
                    capcounter += 1

            else:
                print('>  Not calculating Cap-Cap-Terms')
                print(' ' + 50 * '-')

        return mfccmbe_results


class DensityBasedMFCCMBE2Results(results):
    """
    Class for db-MFCCMBE(2) results.
    """

    def __init__(self, job):
        super().__init__(job)
        self.energy = []

    def get_correction_energy(self):
        return self.energy[0][0]


class DensityBasedMFCCMBE2Job(metajob):

    def __init__(self, ebmfccmbe_res, totmol, grid, order=2,
                 lowcost=True, caps=None, cutoff=None, nadkin=None, nadxc=None):
        """
        Initialize a db-MFCC-MBE(2) job.

        @param ebmfccmbe_res: results of a previous (energy-based) MFCC-MBE Job
        @type  ebmfccmbe_res: L{MFCCMBE2Results}
        @param totmol: total molecule
        @type  totmol: L{molecule}
        @param grid: grid of the total molecule
        @type  grid: adfgrid
        @param order: many-body expansion order
        @type  order: int
        @param lowcost: optimized calculations
        @type  lowcost: boolean
        @param caps: 'mfcc' or 'hydrogen'
        @type  caps: str
        @param cutoff:
        @type  cutoff: float
        @param nadkin: nonadditive kinetic-energy functional (default: PW91k)
        @type  nadkin: str or XCFun Functional object
        @param nadxc: nonadditive xc functional (default: XC functional from eb-MBE single points, if available)
        @type  nadxc: str or XCFun Functional object
        """
        import xcfun

        super().__init__()

        self._grid = grid
        self._ebmfccmbe_res = ebmfccmbe_res
        self._order = order
        self._totmol = totmol
        self._lowcost = lowcost

        if cutoff is None:
            self._cutoff = self._ebmfccmbe_res.job._cutoff
        else:
            self._cutoff = cutoff

        if caps is None:
            self._caps = self._ebmfccmbe_res.job._caps
        else:
            self._caps = caps

        if nadkin is None:
            self._nadkin = 'PW91k'
        elif isinstance(nadkin, xcfun.Functional):
            self._nadkin = nadkin
        elif isinstance(nadkin, str) and nadkin.upper() in ['TF', 'PW91K']:
            self._nadkin = nadkin
        else:
            raise PyAdfError('Invalid nonadditive kinetic-energy functional in db-MFCC-MBE job')

        if nadxc is None:
            if hasattr(self._ebmfccmbe_res.frag_res[0].job, 'functional'):
                self._nadxc = self._ebmfccmbe_res.frag_res[0].job.functional
            elif hasattr(self._ebmfccmbe_res.frag_res[0].job, 'settings') and \
                    hasattr(self._ebmfccmbe_res.frag_res[0].job.settings, 'functional'):
                self._nadxc = self._ebmfccmbe_res.frag_res[0].job.settings.functional
            else:
                raise PyAdfError('No nonadditive xc functional specified in db-MFCC-MBE job')
            if self._nadxc is None:
                raise PyAdfError('No nonadditive xc functional specified in db-MFCC-MBE job')
            if self._nadxc.startswith('GGA '):
                self._nadxc = self._nadxc[4:]
        elif isinstance(nadxc, xcfun.Functional):
            self._nadxc = nadxc
        elif isinstance(nadxc, str) and nadxc.upper() in ['LDA', 'BP', 'BP86', 'BLYP', 'PBE']:
            self._nadxc = nadxc
        else:
            raise PyAdfError('Invalid nonadditive xc functional in db-MFCC-MBE job')

    @property
    def fraglist(self):
        """
        returns the list of cappedfragments
        """
        return self._ebmfccmbe_res.job.fraglist

    @property
    def nfrag(self):
        """
        returns the number of cappedfragments
        """
        return len(self.fraglist)

    @property
    def caplist(self):
        """
        returns the list of caps
        """
        return self._ebmfccmbe_res.job.caplist

    @property
    def ncap(self):
        """
        returns the number of caps
        """
        return len(self.caplist)

    @property
    def nffcombi(self):
        """
        returns number of fragment-fragment combinations
        """
        n = len(self._ebmfccmbe_res.overlap_res_by_comb) + len(self._ebmfccmbe_res.nooverlap_res_by_comb) \
            + len(self._ebmfccmbe_res.trimer_res_by_comb)
        return n

    @property
    def nfccombi(self):
        """
        returns number of fragment-cap combinations
        """
        return len(self._ebmfccmbe_res.fragcap_res_by_comb)

    @property
    def ncccombi(self):
        """
        returns number of cap-cap combinations
        """
        return len(self._ebmfccmbe_res.capcap_res_by_comb)

    @property
    def nadkin(self):
        import xcfun

        if isinstance(self._nadkin, xcfun.Functional):
            return self._nadkin
        elif self._nadkin.upper() == 'TF':
            return xcfun.Functional({'tfk': 1.0})
        elif self._nadkin.upper() == 'PW91K':
            return xcfun.Functional({'pw91k': 1.0})
        else:
            raise PyAdfError('Unknown nonadditive kinetic energy functional '
                             + str(self._nadkin) + 'in db-MFCC-MBE job')

    @property
    def nadxc(self):
        import xcfun

        if isinstance(self._nadxc, xcfun.Functional):
            return self._nadxc
        elif self._nadxc.upper() == 'LDA':
            return xcfun.Functional({'lda': 1.0})
        elif self._nadxc.upper() in ['BP', 'BP86']:
            return xcfun.Functional({'BeckeX': 1.0, 'P86C': 1.0})
        elif self._nadxc.upper() in ['BLYP']:
            return xcfun.Functional({'BeckeX': 1.0, 'LYP': 1.0})
        elif self._nadxc.upper() in ['PBE']:
            return xcfun.Functional({'pbex': 1.0, 'pbec': 1.0})
        else:
            raise PyAdfError('Unknown nonadditive xc functional ' + str(self._nadxc) + ' in db-MFCC-MBE job')

    def create_results_instance(self):
        return DensityBasedMFCCMBE2Results(self)

    def get_total_molecule(self):
        """
        Returns the total molecule.

        @rtype: L{molecule}
        """
        return self._ebmfccmbe_res.job.get_total_molecule()

    def get_molecule_without_caps(self, mol):
        newmol = mol
        deletelist = []
        for atomnum, coords in enumerate(newmol.get_coordinates()):
            chainid, resname, resnum = newmol.get_atom_resinfo(atomnum + 1)
            if resname == 'CAP' or resname == 'SCP':
                deletelist.append(atomnum + 1)
        newmol.delete_atoms(deletelist)
        return newmol

    def get_nuclear_repulsion_energy(self, mol):
        """
        Return the electrostatic interaction energy between the nuclei of this molecule.
        """
        inten = 0.0
        for i in range(len(mol.get_coordinates())):
            for j in range(len(mol.get_coordinates())):
                coord1, coord2 = mol.get_coordinates()[i], mol.get_coordinates()[j]
                atomNum1, atomNum2 = mol.get_atomic_numbers()[i], mol.get_atomic_numbers()[j]
                if i < j:
                    dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)
                    dist = dist / Bohr_in_Angstrom
                    inten = inten + atomNum1 * atomNum2 / dist
        return inten

    def rhocalc(self, main_res, sub1_res, sub2_res, add_res=None, overlap=False, trimer=False, order=None, fit=False):
        """
        returns densities

        @param main_res: dimer/trimer results
        @type  main_res: L{adfsinglepointresults}
        @param sub1_res: frag/cap/dimer 1 results
        @type  sub1_res: L{adfsinglepointresults}
        @param sub2_res: frag/cap/dimer 2 results
        @type  sub2_res: L{adfsinglepointresults}
        @param add_res: cap/frag results
        @type  add_res: L{adfsinglepointresults}
        @param overlap: True when overlap of 1-2 dimers
        @type  overlap: boolean
        @param trimer: True when trimer shall be calculated
        @type  trimer: boolean
        @param order: order of derivatives of the density to calculate (1 and 2 possible)
        @type  order: int
        @param fit: if the fitted density shall be calculated
        @type  fit: boolean
        """

        # if joint 1-2 dimer or 1-3 trimer
        if overlap or trimer:
            maindens = main_res.get_density(self._grid, order=order, fit=fit)  # dimer/trimer
            sub1dens = sub1_res.get_density(self._grid, order=order, fit=fit)  # frag1/dimer12
            sub2dens = sub2_res.get_density(self._grid, order=order, fit=fit)  # frag2/dimer23
            adddens = add_res.get_density(self._grid, order=order, fit=fit)  # cap/frag2
            diffdens = maindens - sub1dens - sub2dens + adddens
            return diffdens, maindens, sub1dens, sub2dens, adddens

        # if disjoint dimer (frag-frag, frag-cap, cap-cap)
        else:
            maindens = main_res.get_density(self._grid, order=order, fit=fit)  # frag-frag/frag-cap/cap-cap
            sub1dens = sub1_res.get_density(self._grid, order=order, fit=fit)  # frag1/frag/cap1
            sub2dens = sub2_res.get_density(self._grid, order=order, fit=fit)  # frag2/cap/cap2
            diffdens = maindens - sub1dens - sub2dens
            return diffdens, maindens, sub1dens, sub2dens

    def potcalc(self, main_res, sub1_res, sub2_res, add_res=None, overlap=False, trimer=False, pot='nuc'):
        """
        returns potentials

        @param main_res: dimer/trimer results
        @type  main_res:
        @param sub1_res: frag/cap/dimer 1 results
        @type  sub1_res:
        @param sub2_res: frag/cap/dimer 2 results
        @type  sub2_res:
        @param add_res: cap/frag results
        @type  add_res:
        @param overlap: True when overlap of 1-2 dimers
        @type  overlap: boolean
        @param trimer: True when trimer shall be calculated
        @type  trimer: boolean
        @param pot: potential to be calculated, 'coul' or 'nuc'
        @type  pot: str
        """

        # if joint 1-2 dimer or 1-3 trimer
        if overlap or trimer:
            mainpot = main_res.get_potential(self._grid, pot=pot)  # dimer/trimer
            sub1pot = sub1_res.get_potential(self._grid, pot=pot)  # frag1/dimer12
            sub2pot = sub2_res.get_potential(self._grid, pot=pot)  # frag2/dimer23
            addpot = add_res.get_potential(self._grid, pot=pot)  # cap/frag2
            return mainpot, sub1pot, sub2pot, addpot

        # if disjoint dimer (frag-frag, frag-cap, cap-cap)
        else:
            mainpot = main_res.get_potential(self._grid, pot=pot)  # frag-frag/frag-cap/cap-cap
            sub1pot = sub1_res.get_potential(self._grid, pot=pot)  # frag1/frag/cap1
            sub2pot = sub2_res.get_potential(self._grid, pot=pot)  # frag2/cap/cap2
            return mainpot, sub1pot, sub2pot

    def eb_calc(self, inttype, *args, joint=False):
        """

        @param inttype: choose interaction type 'en', 'ee', 'eecorr', 'xc', 'kin'
        @type  inttype: str
        @param args: the densities and potentials needed
        @type  args: GridFunction
        @param joint: True if joint fragment (1-2 oder 1-3 combination)
        @type  joint: boolean
        @return: the interaction energy delta
        """
        def calc_inten(dens, pot):
            if inttype == 'en':  # dens , vnuc
                return np.dot(self._grid.weights, (dens[0] * pot).get_values())
            if inttype == 'ee':  # ens , vcoul
                return 0.5 * np.dot(self._grid.weights, (dens[0] * pot).get_values())
            if inttype == 'eecorr':  # dens - densfit , vcoul
                return 0.5 * np.dot(self._grid.weights, (dens * pot).get_values())
            if inttype == 'xc':  # dens , dens
                return np.dot(self._grid.weights,
                              self.nadxc.eval_energy_n(density=dens[0].values, densgrad=dens[1].values))
            if inttype == 'kin':  # dens , dens
                return np.dot(self._grid.weights,
                              self.nadkin.eval_energy_n(density=dens[0].values, densgrad=dens[1].values))

        # if joint 1-2 dimer or 1-3 trimer
        if joint:
            delta = calc_inten(args[0], args[4]) \
                    - calc_inten(args[1], args[5]) \
                    - calc_inten(args[2], args[6]) \
                    + calc_inten(args[3], args[7])
            return delta

        # if disjoint dimer (frag-frag, frag-cap, cap-cap)
        else:
            delta = calc_inten(args[0], args[3]) \
                    - calc_inten(args[1], args[4]) \
                    - calc_inten(args[2], args[5])
            return delta

    def eb_calc_lowcost(self, inttype, *args, joint=False):
        """

        @param inttype: choose interaction type 'en', 'ee', 'eecorr', 'xc', 'kin'
        @type  inttype: str
        @param args: the densities and potentials needed
        @type  args: GridFunction
        @param joint: True if joint fragment (1-2 oder 1-3 combination)
        @type  joint: boolean
        @return: the interaction energy delta
        """
        def calc_inten(dens, pot):
            if inttype == 'en':  # dens , vnuc
                return (dens[0] * pot).get_values()
            if inttype == 'ee':  # ens , vcoul
                return 0.5 * (dens[0] * pot).get_values()
            if inttype == 'eecorr':  # dens - densfit , vcoul
                return 0.5 * (dens * pot).get_values()
            if inttype == 'xc':  # dens , dens
                return self.nadxc.eval_energy_n(density=dens[0].values, densgrad=dens[1].values)
            if inttype == 'kin':  # dens , dens
                return self.nadkin.eval_energy_n(density=dens[0].values, densgrad=dens[1].values)

        # if joint 1-2 dimer or 1-3 trimer
        if joint:
            delta = calc_inten(args[0], args[4]) \
                    - calc_inten(args[1], args[5]) \
                    - calc_inten(args[2], args[6]) \
                    + calc_inten(args[3], args[7])
            return delta

        # if disjoint dimer (frag-frag, frag-cap, cap-cap)
        else:
            delta = calc_inten(args[0], args[3]) \
                    - calc_inten(args[1], args[4]) \
                    - calc_inten(args[2], args[5])
            return delta

    def get_dbcorr_1st_order(self):

        timelist = []
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("START ", current_time))

        totmol = self._totmol

        rhotot = 0.0
        rhototfit = 0.0
        rhototfitcorr = 0.0
        vnuctot = 0.0
        vcoultot = 0.0
        nnint = 0.0
        enint = 0.0
        eeint = 0.0
        eecorr = 0.0
        xcint = 0.0
        kinint = 0.0

        # iterate over all fragments
        for i in self._ebmfccmbe_res.frag_res:
            # get densities and potentials
            rho = i.get_density(self._grid, order=1)
            rhofit = i.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.get_potential(self._grid, pot='nuc')
            vcoul = i.get_potential(self._grid, pot='coul')

            # sum up all fragment densities and potentials
            rhotot = rho + rhotot
            rhototfit = rhofit + rhototfit
            rhototfitcorr = rhofitcorr + rhototfitcorr
            vnuctot = vnuc + vnuctot
            vcoultot = vcoul + vcoultot

            # calculate properties for the fragment on supermolecular grid
            nnint -= i.get_nuclear_repulsion_energy()
            enint -= np.dot(self._grid.weights, (rho[0] * vnuc).get_values())
            eeint -= 0.5 * np.dot(self._grid.weights, (rho[0] * vcoul).get_values())
            eecorr -= 0.5 * np.dot(self._grid.weights, (rhofitcorr * vcoul).get_values())
            xcint -= np.dot(self._grid.weights,
                            self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))
            kinint -= np.dot(self._grid.weights,
                             self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))

        # iterate over all caps
        for i in self._ebmfccmbe_res.cap_res:
            # get densities and potentials
            rho = i.get_density(self._grid, order=1)
            rhofit = i.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.get_potential(self._grid, pot='nuc')
            vcoul = i.get_potential(self._grid, pot='coul')

            # subtract all cap densities and potentials
            rhotot -= rho
            rhototfit -= rhofit
            rhototfitcorr -= rhofitcorr
            vnuctot -= vnuc
            vcoultot -= vcoul

            # calculate properties for the cap on supermolecular grid
            nnint += i.get_nuclear_repulsion_energy()
            enint += np.dot(self._grid.weights, (rho[0] * vnuc).get_values())
            eeint += 0.5 * np.dot(self._grid.weights, (rho[0] * vcoul).get_values())
            eecorr += 0.5 * np.dot(self._grid.weights, (rhofitcorr * vcoul).get_values())
            xcint += np.dot(self._grid.weights,
                            self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))
            kinint += np.dot(self._grid.weights,
                             self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))

        # calculate properties for total molecule on supermolecular grid
        nnint += self.get_nuclear_repulsion_energy(totmol)
        enint += np.dot(self._grid.weights, (rhotot[0] * vnuctot).get_values())
        eeint += 0.5 * np.dot(self._grid.weights, (rhotot[0] * vcoultot).get_values())
        eecorr += 0.5 * np.dot(self._grid.weights, (rhototfitcorr * vcoultot).get_values())
        xcint += np.dot(self._grid.weights,
                        self.nadxc.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values))
        kinint += np.dot(self._grid.weights,
                         self.nadkin.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values))

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("END ", current_time))

        return nnint + enint + eeint + eecorr + xcint + kinint, timelist

    def get_dbcorr_1st_order_lowcost(self):

        timelist = []
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("START ", current_time))

        totmol = self._totmol

        rhotot = 0.0
        rhototfit = 0.0
        rhototfitcorr = 0.0
        vnuctot = 0.0
        vcoultot = 0.0
        nnint = 0.0
        energy = 0.0

        # iterate over all fragments
        for i in self._ebmfccmbe_res.frag_res:
            # get densities and potentials
            rho = i.get_density(self._grid, order=1)
            rhofit = i.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.get_potential(self._grid, pot='nuc')
            vcoul = i.get_potential(self._grid, pot='coul')

            # sum up all fragment densities and potentials
            rhotot = rho + rhotot
            rhototfit = rhofit + rhototfit
            rhototfitcorr = rhofitcorr + rhototfitcorr
            vnuctot = vnuc + vnuctot
            vcoultot = vcoul + vcoultot

            # subtract the fragment interactions from the final energy
            nnint -= i.get_nuclear_repulsion_energy()
            energy -= (rho[0] * vnuc).get_values()
            energy -= 0.5 * (rho[0] * vcoul).get_values()
            energy -= 0.5 * (rhofitcorr * vcoul).get_values()
            energy -= self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)
            energy -= self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)

        # iterate over all caps
        for i in self._ebmfccmbe_res.cap_res:
            # get densities and potentials
            rho = i.get_density(self._grid, order=1)
            rhofit = i.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.get_potential(self._grid, pot='nuc')
            vcoul = i.get_potential(self._grid, pot='coul')

            # subtract all cap densities and potentials
            rhotot -= rho
            rhototfit -= rhofit
            rhototfitcorr -= rhofitcorr
            vnuctot -= vnuc
            vcoultot -= vcoul

            # add the cap interactions to the final energy
            nnint += i.get_nuclear_repulsion_energy()
            energy += (rho[0] * vnuc).get_values()
            energy += 0.5 * (rho[0] * vcoul).get_values()
            energy += 0.5 * (rhofitcorr * vcoul).get_values()
            energy += self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)
            energy += self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)

        # add the interactions for the total molecule to the final energy
        nnint += self.get_nuclear_repulsion_energy(totmol)
        energy += (rhotot[0] * vnuctot).get_values()
        energy += 0.5 * (rhotot[0] * vcoultot).get_values()
        energy += 0.5 * (rhototfitcorr * vcoultot).get_values()
        energy += self.nadxc.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values)
        energy += self.nadkin.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values)

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("END ", current_time))

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("START ", current_time))

        # calculate the energy on the supermolecular grid
        correction = np.dot(self._grid.weights, energy)

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("END ", current_time))

        return correction + nnint, timelist

    def get_dbcorr_2nd_order(self):

        timelist = []
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("START ", current_time))

        totmol = self._totmol

        rhotot = 0.0
        rhototfit = 0.0
        diffdenstot = 0.0
        vnuctot = 0.0
        vcoultot = 0.0
        dbentot = 0.0
        ebentot = 0.0
        dbeetot = 0.0
        ebeetot = 0.0
        dbeecorr = 0.0
        ebeecorr = 0.0
        dbxctot = 0.0
        ebxctot = 0.0
        dbkintot = 0.0
        ebkintot = 0.0
        ebnntot = 0.0
        dbnntot = 0.0

        print('>  \n>  Starting Fragment Calculations\n>  ')
        counter = 1
        # iterate over all fragments
        for i in self._ebmfccmbe_res.frag_res:
            print('>  \n>  Fragment', counter, 'of', self.nfrag, '\n>  ')

            # get densities and potentials
            rho = i.get_density(self._grid, order=1)
            rhofit = i.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.get_potential(self._grid, pot='nuc')
            vcoul = i.get_potential(self._grid, pot='coul')

            # sum up all fragment densities and potentials
            rhotot = rho + rhotot
            rhototfit = rhofit + rhototfit
            vnuctot = vnuc + vnuctot
            vcoultot = vcoul + vcoultot

            # calculate properties for the fragment on supermolecular grid
            ebentot += np.dot(self._grid.weights, (rho[0] * vnuc).get_values())
            ebeetot += 0.5 * np.dot(self._grid.weights, (rho[0] * vcoul).get_values())
            ebeecorr += 0.5 * np.dot(self._grid.weights, (rhofitcorr * vcoul).get_values())
            ebxctot += np.dot(self._grid.weights,
                              self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))
            ebkintot += np.dot(self._grid.weights,
                               self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))
            ebnntot += i.get_nuclear_repulsion_energy()
            counter += 1

        print('>  \n>  Starting Cap Calculations\n>  ')
        counter = 1
        # iterate over all caps
        for i in self._ebmfccmbe_res.cap_res:
            print('>  \n>  Cap', counter, 'of', self.ncap, '\n>  ')

            # get densities and potentials
            rho = i.get_density(self._grid, order=1)
            rhofit = i.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.get_potential(self._grid, pot='nuc')
            vcoul = i.get_potential(self._grid, pot='coul')

            # subtract all cap densities and potentials
            rhotot -= rho
            rhototfit -= rhofit
            vnuctot -= vnuc
            vcoultot -= vcoul

            # calculate properties for the cap on supermolecular grid
            ebentot -= np.dot(self._grid.weights, (rho[0] * vnuc).get_values())
            ebeetot -= 0.5 * np.dot(self._grid.weights, (rho[0] * vcoul).get_values())
            ebeecorr -= 0.5 * np.dot(self._grid.weights, (rhofitcorr * vcoul).get_values())
            ebxctot -= np.dot(self._grid.weights,
                              self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))
            ebkintot -= np.dot(self._grid.weights,
                               self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values))
            ebnntot -= i.get_nuclear_repulsion_energy()
            counter += 1

        print('>  \n>  Starting Fragment-Fragment Calculations\n>  ')
        counter = 1
        # iterate over all joint 1-2 fragment-fragment combinations
        for i in self._ebmfccmbe_res.overlap_res_by_comb:
            print('>  \n>  Fragment-Fragment Combination', counter, 'of', self.nffcombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # overlap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [dimer_res, cap_res] with the dimer results and cap results
            # frag_res is a list of all fragment results
            dimer = self._ebmfccmbe_res.overlap_res_by_comb[i][0]
            mono1 = self._ebmfccmbe_res.frag_res[i[0]]
            mono2 = self._ebmfccmbe_res.frag_res[i[1]]
            cap = self._ebmfccmbe_res.overlap_res_by_comb[i][1]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens, adddens = self.rhocalc(dimer, mono1, mono2, cap,
                                                                           overlap=True, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit, adddensfit = self.rhocalc(dimer, mono1, mono2, cap,
                                                                                          overlap=True, fit=True)
            vnucmain, vnucsub1, vnucsub2, vnucadd = self.potcalc(dimer, mono1, mono2, cap,
                                                                 overlap=True, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2, vcouladd = self.potcalc(dimer, mono1, mono2, cap,
                                                                     overlap=True, pot='coul')

            # add all fragment-fragment and potentials
            rhotot += diffdens
            rhototfit += diffdensfit
            diffdenstot += diffdens
            vcoultot += vcoulmain - vcoulsub1 - vcoulsub2 + vcouladd

            # calculate difference in electron-nuclei interaction for fragment-fragment combination
            # en(dimer) - en(frag1) - en(frag2) + en(cap)
            ebentot += self.eb_calc('en',
                                    maindens, sub1dens, sub2dens, adddens, vnucmain, vnucsub1, vnucsub2, vnucadd,
                                    joint=True)

            # calculate difference in electron-electron interaction for fragment-fragment combination
            # ee(dimer) - ee(frag1) - ee(frag2) + ee(cap)
            ebeetot += self.eb_calc('ee',
                                    maindens, sub1dens, sub2dens, adddens,
                                    vcoulmain, vcoulsub1, vcoulsub2, vcouladd,
                                    joint=True)

            # calculate fitcorrection for fragment-fragment combination
            # density - fitted density
            dimerfitcorr = maindens[0] - maindensfit
            mono1fitcorr = sub1dens[0] - sub1densfit
            mono2fitcorr = sub2dens[0] - sub2densfit
            capfitcorr = adddens[0] - adddensfit

            # calculate difference in electron-electron correction for fragment-fragment combination
            # eecorr(dimer) - eecorr(frag1) - eecorr(frag2) + eecorr(cap)
            ebeecorr += self.eb_calc('eecorr',
                                     dimerfitcorr, mono1fitcorr, mono2fitcorr, capfitcorr,
                                     vcoulmain, vcoulsub1, vcoulsub2, vcouladd,
                                     joint=True)

            # calculate difference in XC energy for fragment-fragment combination
            # xc(dimer) - xc(frag1) - xc(frag2) + xc(cap)
            ebxctot += self.eb_calc('xc',
                                    maindens, sub1dens, sub2dens, adddens,
                                    maindens, sub1dens, sub2dens, adddens,
                                    joint=True)

            # calculate difference in kinetic energy for fragment-fragment combination
            # kin(dimer) - kin(frag1) - kin(frag2) + kin(cap)
            ebkintot += self.eb_calc('kin',
                                     maindens, sub1dens, sub2dens, adddens,
                                     maindens, sub1dens, sub2dens, adddens,
                                     joint=True)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot += (dimer.get_nuclear_repulsion_energy() - mono1.get_nuclear_repulsion_energy()
                        - mono2.get_nuclear_repulsion_energy() + cap.get_nuclear_repulsion_energy())

            counter += 1

        # iterate over all disjoint (>1-3) fragment-fragment combinations
        for i in self._ebmfccmbe_res.nooverlap_res_by_comb:
            print('>  \n>  Fragment-Fragment Combination', counter, 'of', self.nffcombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # nooverlap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the dimer results
            # frag_res is a list of all fragment results
            dimer = self._ebmfccmbe_res.nooverlap_res_by_comb[i]
            mono1 = self._ebmfccmbe_res.frag_res[i[0]]
            mono2 = self._ebmfccmbe_res.frag_res[i[1]]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens = self.rhocalc(dimer, mono1, mono2, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit = self.rhocalc(dimer, mono1, mono2, fit=True)
            vnucmain, vnucsub1, vnucsub2 = self.potcalc(dimer, mono1, mono2, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2 = self.potcalc(dimer, mono1, mono2, pot='coul')

            # add all fragment-fragment and potentials
            rhotot += diffdens
            rhototfit += diffdensfit
            diffdenstot += diffdens
            vcoultot += vcoulmain - vcoulsub1 - vcoulsub2

            # calculate difference in electron-nuclei interaction for fragment-fragment combination
            # en(dimer) - en(frag1) - en(frag2)
            ebentot += self.eb_calc('en',
                                    maindens, sub1dens, sub2dens, vnucmain, vnucsub1, vnucsub2,
                                    joint=False)

            # calculate difference in electron-electron interaction for fragment-fragment combination
            # ee(dimer) - ee(frag1) - ee(frag2)
            ebeetot += self.eb_calc('ee',
                                    maindens, sub1dens, sub2dens,
                                    vcoulmain, vcoulsub1, vcoulsub2,
                                    joint=False)

            # calculate fitcorrection for fragment-fragment combination
            # density - fitted density
            dimerfitcorr = maindens[0] - maindensfit
            mono1fitcorr = sub1dens[0] - sub1densfit
            mono2fitcorr = sub2dens[0] - sub2densfit

            # calculate difference in electron-electron correction for fragment-fragment combination
            # eecorr(dimer) - eecorr(frag1) - eecorr(frag2)
            ebeecorr += self.eb_calc('eecorr',
                                     dimerfitcorr, mono1fitcorr, mono2fitcorr,
                                     vcoulmain, vcoulsub1, vcoulsub2,
                                     joint=False)

            # calculate difference in XC energy for fragment-fragment combination
            # xc(dimer) - xc(frag1) - xc(frag2)
            ebxctot += self.eb_calc('xc',
                                    maindens, sub1dens, sub2dens,
                                    maindens, sub1dens, sub2dens,
                                    joint=False)

            # calculate difference in kinetic energy for fragment-fragment combination
            # kin(dimer) - kin(frag1) - kin(frag2)
            ebkintot += self.eb_calc('kin',
                                     maindens, sub1dens, sub2dens,
                                     maindens, sub1dens, sub2dens,
                                     joint=False)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot += (dimer.get_nuclear_repulsion_energy() - mono1.get_nuclear_repulsion_energy()
                        - mono2.get_nuclear_repulsion_energy())

            counter += 1

        # iterate over all 1-3 fragment-fragment combinations
        for i in self._ebmfccmbe_res.trimer_res_by_comb:
            print('>  \n>  Fragment-Fragment Combination', counter, 'of', self.nffcombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # trimer_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the trimer results
            # overlap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [dimer_res, cap_res] with the dimer results and cap results
            # frag_res is a list of all fragment results
            d12 = (i[0], i[0] + 1)
            d23 = (i[1] - 1, i[1])
            trimer = self._ebmfccmbe_res.trimer_res_by_comb[i]
            dimer12 = self._ebmfccmbe_res.overlap_res_by_comb[d12][0]
            dimer23 = self._ebmfccmbe_res.overlap_res_by_comb[d23][0]
            mono2 = self._ebmfccmbe_res.frag_res[i[0] + 1]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens, adddens = self.rhocalc(trimer, dimer12, dimer23, mono2,
                                                                           trimer=True, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit, adddensfit = self.rhocalc(trimer,
                                                                                          dimer12, dimer23, mono2,
                                                                                          trimer=True, fit=True)
            vnucmain, vnucsub1, vnucsub2, vnucadd = self.potcalc(trimer, dimer12, dimer23, mono2,
                                                                 trimer=True, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2, vcouladd = self.potcalc(trimer, dimer12, dimer23, mono2,
                                                                     trimer=True, pot='coul')

            # add all fragment-fragment and potentials
            rhotot += diffdens
            rhototfit += diffdensfit
            diffdenstot += diffdens
            vcoultot += vcoulmain - vcoulsub1 - vcoulsub2 + vcouladd

            # calculate difference in electron-nuclei interaction for fragment-fragment combination
            # en(trimer) - en(dimer12) - en(dimer23) + en(frag2)
            ebentot += self.eb_calc('en',
                                    maindens, sub1dens, sub2dens, adddens, vnucmain, vnucsub1, vnucsub2, vnucadd,
                                    joint=True)

            # calculate difference in electron-electron interaction for fragment-fragment combination
            # ee(trimer) - ee(dimer12) - ee(dimer23) + ee(frag2)
            ebeetot += self.eb_calc('ee',
                                    maindens, sub1dens, sub2dens, adddens,
                                    vcoulmain, vcoulsub1, vcoulsub2, vcouladd,
                                    joint=True)

            # calculate fitcorrection for fragment-fragment combination
            # density - fitted density
            trimerfitcorr = maindens[0] - maindensfit
            dimer12fitcorr = sub1dens[0] - sub1densfit
            dimer23fitcorr = sub2dens[0] - sub2densfit
            mono2fitcorr = adddens[0] - adddensfit

            # calculate difference in electron-electron correction for fragment-fragment combination
            # eecorr(trimer) - eecorr(dimer12) - eecorr(dimer23) + eecorr(frag2)
            ebeecorr += self.eb_calc('eecorr',
                                     trimerfitcorr, dimer12fitcorr, dimer23fitcorr, mono2fitcorr,
                                     vcoulmain, vcoulsub1, vcoulsub2, vcouladd,
                                     joint=True)

            # calculate difference in XC energy for fragment-fragment combination
            # xc(trimer) - xc(dimer12) - xc(dimer23) + xc(frag2)
            ebxctot += self.eb_calc('xc',
                                    maindens, sub1dens, sub2dens, adddens,
                                    maindens, sub1dens, sub2dens, adddens,
                                    joint=True)

            # calculate difference in kinetic energy for fragment-fragment combination
            # kin(trimer) - kin(dimer12) - kin(dimer23) + kin(frag2)
            ebkintot += self.eb_calc('kin',
                                     maindens, sub1dens, sub2dens, adddens,
                                     maindens, sub1dens, sub2dens, adddens,
                                     joint=True)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot += (trimer.get_nuclear_repulsion_energy() - dimer12.get_nuclear_repulsion_energy()
                        - dimer23.get_nuclear_repulsion_energy() + mono2.get_nuclear_repulsion_energy())

            counter += 1

        print('>  \n>  Starting Fragment-Cap Calculations\n>  ')
        counter = 1
        # iterate over all fragment-cap combinations
        for i in self._ebmfccmbe_res.fragcap_res_by_comb:
            print('>  \n>  Fragment-Cap Combination', counter, 'of', self.nfccombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # fragcap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the fragment-cap
            # dimer results
            # frag_res is a list of all fragment results
            # cap_res is a list of all cap results
            fragcap = self._ebmfccmbe_res.fragcap_res_by_comb[i]
            frag = self._ebmfccmbe_res.frag_res[i[0]]
            cap = self._ebmfccmbe_res.cap_res[i[1]]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens = self.rhocalc(fragcap, frag, cap, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit = self.rhocalc(fragcap, frag, cap, fit=True)
            vnucmain, vnucsub1, vnucsub2 = self.potcalc(fragcap, frag, cap, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2 = self.potcalc(fragcap, frag, cap, pot='coul')

            # subtract all fragment-cap and potentials
            rhotot -= diffdens
            rhototfit -= diffdensfit
            diffdenstot -= diffdens
            vcoultot -= vcoulmain - vcoulsub1 - vcoulsub2

            # calculate difference in electron-nuclei interaction for fragment-cap combination
            # en(dimer) - en(frag) - en(cap)
            ebentot -= self.eb_calc('en',
                                    maindens, sub1dens, sub2dens, vnucmain, vnucsub1, vnucsub2,
                                    joint=False)

            # calculate difference in electron-electron interaction for fragment-cap combination
            # ee(dimer) - ee(frag) - ee(cap)
            ebeetot -= self.eb_calc('ee',
                                    maindens, sub1dens, sub2dens,
                                    vcoulmain, vcoulsub1, vcoulsub2,
                                    joint=False)

            # calculate fitcorrection for fragment-cap combination
            # density - fitted density
            dimerfitcorr = maindens[0] - maindensfit
            mono1fitcorr = sub1dens[0] - sub1densfit
            mono2fitcorr = sub2dens[0] - sub2densfit

            # calculate difference in electron-electron correction for fragment-cap combination
            # eecorr(dimer) - eecorr(frag) - eecorr(cap)
            ebeecorr -= self.eb_calc('eecorr',
                                     dimerfitcorr, mono1fitcorr, mono2fitcorr,
                                     vcoulmain, vcoulsub1, vcoulsub2,
                                     joint=False)

            # calculate difference in XC energy for fragment-cap combination
            # xc(dimer) - xc(frag) - xc(cap)
            ebxctot -= self.eb_calc('xc',
                                    maindens, sub1dens, sub2dens,
                                    maindens, sub1dens, sub2dens,
                                    joint=False)

            # calculate difference in kinetic energy for fragment-cap combination
            # kin(dimer) - kin(frag) - kin(cap)
            ebkintot -= self.eb_calc('kin',
                                     maindens, sub1dens, sub2dens,
                                     maindens, sub1dens, sub2dens,
                                     joint=False)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot -= (fragcap.get_nuclear_repulsion_energy() - frag.get_nuclear_repulsion_energy()
                        - cap.get_nuclear_repulsion_energy())

            counter += 1

        print('>  \n>  Starting Cap-Cap Calculations\n>  ')
        counter = 1
        # iterate over all cap-cap combinations
        for i in self._ebmfccmbe_res.capcap_res_by_comb:
            print('>  \n>  Cap-Cap Combination', counter, 'of', self.ncccombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # capcap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the cap-cap
            # dimer results
            # cap_res is a list of all cap results
            capcap = self._ebmfccmbe_res.capcap_res_by_comb[i]
            cap1 = self._ebmfccmbe_res.cap_res[i[0]]
            cap2 = self._ebmfccmbe_res.cap_res[i[1]]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens = self.rhocalc(capcap, cap1, cap2, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit = self.rhocalc(capcap, cap1, cap2, fit=True)
            vnucmain, vnucsub1, vnucsub2 = self.potcalc(capcap, cap1, cap2, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2 = self.potcalc(capcap, cap1, cap2, pot='coul')

            # add all cap-cap and potentials
            rhotot += diffdens
            rhototfit += diffdensfit
            diffdenstot += diffdens
            vcoultot += vcoulmain - vcoulsub1 - vcoulsub2

            # calculate difference in electron-nuclei interaction for cap-cap combination
            # en(dimer) - en(cap) - en(cap)
            ebentot += self.eb_calc('en',
                                    maindens, sub1dens, sub2dens, vnucmain, vnucsub1, vnucsub2,
                                    joint=False)

            # calculate difference in electron-electron interaction for cap-cap combination
            # ee(dimer) - ee(cap) - ee(cap)
            ebeetot += self.eb_calc('ee',
                                    maindens, sub1dens, sub2dens,
                                    vcoulmain, vcoulsub1, vcoulsub2,
                                    joint=False)

            # calculate fitcorrection for cap-cap combination
            # density - fitted density
            dimerfitcorr = maindens[0] - maindensfit
            mono1fitcorr = sub1dens[0] - sub1densfit
            mono2fitcorr = sub2dens[0] - sub2densfit

            # calculate difference in electron-electron correction for cap-cap combination
            # eecorr(dimer) - eecorr(cap) - eecorr(cap)
            ebeecorr += self.eb_calc('eecorr',
                                     dimerfitcorr, mono1fitcorr, mono2fitcorr,
                                     vcoulmain, vcoulsub1, vcoulsub2,
                                     joint=False)

            # calculate difference in XC energy for cap-cap combination
            # xc(dimer) - xc(cap) - xc(cap)
            ebxctot += self.eb_calc('xc',
                                    maindens, sub1dens, sub2dens,
                                    maindens, sub1dens, sub2dens,
                                    joint=False)

            # calculate difference in kinetic energy for cap-cap combination
            # kin(dimer) - kin(cap) - kin(cap)
            ebkintot += self.eb_calc('kin',
                                     maindens, sub1dens, sub2dens,
                                     maindens, sub1dens, sub2dens,
                                     joint=False)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot += (capcap.get_nuclear_repulsion_energy() - cap1.get_nuclear_repulsion_energy()
                        - cap2.get_nuclear_repulsion_energy())

            counter += 1

        # db-Calculations
        # calculate properties for the total molecule on supermolecular grid
        dbentot += np.dot(self._grid.weights, (rhotot[0] * vnuctot).get_values())
        dbeetot += 0.5 * np.dot(self._grid.weights, (rhotot[0] * vcoultot).get_values())
        rhototfitcorr = rhotot[0] - rhototfit
        dbeecorr += 0.5 * np.dot(self._grid.weights, (rhototfitcorr * vcoultot).get_values())
        dbxctot += np.dot(self._grid.weights,
                          self.nadxc.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values))
        dbkintot += np.dot(self._grid.weights,
                           self.nadkin.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values))
        dbnntot += self.get_nuclear_repulsion_energy(totmol)

        enint = dbentot - ebentot
        eeint = dbeetot - ebeetot
        eecorrint = dbeecorr - ebeecorr
        xcint = dbxctot - ebxctot
        kinint = dbkintot - ebkintot
        nnint = dbnntot - ebnntot

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("END ", current_time))

        return enint + eeint + eecorrint + xcint + kinint + nnint, timelist

    def get_dbcorr_2nd_order_lowcost(self):

        timelist = []
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("START ", current_time))

        totmol = self._totmol

        rhotot = 0.0
        rhototfit = 0.0
        diffdenstot = 0.0
        vnuctot = 0.0
        vcoultot = 0.0
        ebnntot = 0.0
        dbnntot = 0.0

        energy = 0.0

        print('>  \n>  Starting Fragment Calculations\n>  ')
        counter = 1
        # iterate over all fragments
        for i in self._ebmfccmbe_res.frag_res:
            print('>  \n>  Fragment', counter, 'of', self.nfrag, '\n>  ')

            # get densities and potentials
            rho = i.get_density(self._grid, order=1)
            rhofit = i.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.get_potential(self._grid, pot='nuc')
            vcoul = i.get_potential(self._grid, pot='coul')

            # sum up all fragment densities and potentials
            rhotot = rho + rhotot
            rhototfit = rhofit + rhototfit
            vnuctot = vnuc + vnuctot
            vcoultot = vcoul + vcoultot

            # subtract the fragment interactions from the final energy
            energy -= (rho[0] * vnuc).get_values()
            energy -= 0.5 * (rho[0] * vcoul).get_values()
            energy -= 0.5 * (rhofitcorr * vcoul).get_values()
            energy -= self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)
            energy -= self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)
            ebnntot += i.get_nuclear_repulsion_energy()
            counter += 1

        print('>  \n>  Starting Cap Calculations\n>  ')
        counter = 1
        # iterate over all caps
        for i in self._ebmfccmbe_res.cap_res:
            print('>  \n>  Cap', counter, 'of', self.ncap, '\n>  ')

            # get densities and potentials
            rho = i.get_density(self._grid, order=1)
            rhofit = i.get_density(self._grid, fit=True)
            rhofitcorr = rho[0] - rhofit
            vnuc = i.get_potential(self._grid, pot='nuc')
            vcoul = i.get_potential(self._grid, pot='coul')

            # subtract all cap densities and potentials
            rhotot -= rho
            rhototfit -= rhofit
            vnuctot -= vnuc
            vcoultot -= vcoul

            # add the cap interactions to the final energy
            energy += (rho[0] * vnuc).get_values()
            energy += 0.5 * (rho[0] * vcoul).get_values()
            energy += 0.5 * (rhofitcorr * vcoul).get_values()
            energy += self.nadxc.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)
            energy += self.nadkin.eval_energy_n(density=rho[0].values, densgrad=rho[1].values)
            ebnntot -= i.get_nuclear_repulsion_energy()
            counter += 1

        print('>  \n>  Starting Fragment-Fragment Calculations\n>  ')
        counter = 1
        # iterate over all joint 1-2 fragment-fragment combinations
        for i in self._ebmfccmbe_res.overlap_res_by_comb:
            print('>  \n>  Fragment-Fragment Combination', counter, 'of', self.nffcombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # overlap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [dimer_res, cap_res] with the dimer results and cap results
            # frag_res is a list of all fragment results
            dimer = self._ebmfccmbe_res.overlap_res_by_comb[i][0]
            mono1 = self._ebmfccmbe_res.frag_res[i[0]]
            mono2 = self._ebmfccmbe_res.frag_res[i[1]]
            cap = self._ebmfccmbe_res.overlap_res_by_comb[i][1]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens, adddens = self.rhocalc(dimer, mono1, mono2, cap,
                                                                           overlap=True, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit, adddensfit = self.rhocalc(dimer, mono1, mono2, cap,
                                                                                          overlap=True, fit=True)
            vnucmain, vnucsub1, vnucsub2, vnucadd = self.potcalc(dimer, mono1, mono2, cap,
                                                                 overlap=True, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2, vcouladd = self.potcalc(dimer, mono1, mono2, cap,
                                                                     overlap=True, pot='coul')

            # add all fragment-fragment and potentials
            rhotot += diffdens
            rhototfit += diffdensfit
            diffdenstot += diffdens
            vcoultot += vcoulmain - vcoulsub1 - vcoulsub2 + vcouladd

            # calculate difference in electron-nuclei interaction for fragment-fragment combination
            # en(dimer) - en(frag1) - en(frag2) + en(cap)
            energy -= self.eb_calc_lowcost('en', maindens, sub1dens, sub2dens, adddens,
                                           vnucmain, vnucsub1, vnucsub2, vnucadd, joint=True)

            # calculate difference in electron-electron interaction for fragment-fragment combination
            # ee(dimer) - ee(frag1) - ee(frag2) + ee(cap)
            energy -= self.eb_calc_lowcost('ee', maindens, sub1dens, sub2dens, adddens,
                                           vcoulmain, vcoulsub1, vcoulsub2, vcouladd, joint=True)

            # calculate fitcorrection for fragment-fragment combination
            # density - fitted density
            dimerfitcorr = maindens[0] - maindensfit
            mono1fitcorr = sub1dens[0] - sub1densfit
            mono2fitcorr = sub2dens[0] - sub2densfit
            capfitcorr = adddens[0] - adddensfit

            # calculate difference in electron-electron correction for fragment-fragment combination
            # eecorr(dimer) - eecorr(frag1) - eecorr(frag2) + eecorr(cap)
            energy -= self.eb_calc_lowcost('eecorr', dimerfitcorr, mono1fitcorr, mono2fitcorr, capfitcorr,
                                           vcoulmain, vcoulsub1, vcoulsub2, vcouladd, joint=True)

            # calculate difference in XC energy for fragment-fragment combination
            # xc(dimer) - xc(frag1) - xc(frag2) + xc(cap)
            energy -= self.eb_calc_lowcost('xc', maindens, sub1dens, sub2dens, adddens,
                                           maindens, sub1dens, sub2dens, adddens, joint=True)

            # calculate difference in kinetic energy for fragment-fragment combination
            # kin(dimer) - kin(frag1) - kin(frag2) + kin(cap)
            energy -= self.eb_calc_lowcost('kin', maindens, sub1dens, sub2dens, adddens,
                                           maindens, sub1dens, sub2dens, adddens, joint=True)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot += (dimer.get_nuclear_repulsion_energy() - mono1.get_nuclear_repulsion_energy()
                        - mono2.get_nuclear_repulsion_energy() + cap.get_nuclear_repulsion_energy())

            counter += 1

        # iterate over all disjoint (>1-3) fragment-fragment combinations
        for i in self._ebmfccmbe_res.nooverlap_res_by_comb:
            print('>  \n>  Fragment-Fragment Combination', counter, 'of', self.nffcombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # nooverlap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the dimer results
            # frag_res is a list of all fragment results
            dimer = self._ebmfccmbe_res.nooverlap_res_by_comb[i]
            mono1 = self._ebmfccmbe_res.frag_res[i[0]]
            mono2 = self._ebmfccmbe_res.frag_res[i[1]]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens = self.rhocalc(dimer, mono1, mono2, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit = self.rhocalc(dimer, mono1, mono2, fit=True)
            vnucmain, vnucsub1, vnucsub2 = self.potcalc(dimer, mono1, mono2, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2 = self.potcalc(dimer, mono1, mono2, pot='coul')

            # add all fragment-fragment and potentials
            rhotot += diffdens
            rhototfit += diffdensfit
            diffdenstot += diffdens
            vcoultot += vcoulmain - vcoulsub1 - vcoulsub2

            # calculate difference in electron-nuclei interaction for fragment-fragment combination
            # en(dimer) - en(frag1) - en(frag2)
            energy -= self.eb_calc_lowcost('en', maindens, sub1dens, sub2dens,
                                           vnucmain, vnucsub1, vnucsub2, joint=False)

            # calculate difference in electron-electron interaction for fragment-fragment combination
            # ee(dimer) - ee(frag1) - ee(frag2)
            energy -= self.eb_calc_lowcost('ee', maindens, sub1dens, sub2dens,
                                           vcoulmain, vcoulsub1, vcoulsub2, joint=False)

            # calculate fitcorrection for fragment-fragment combination
            # density - fitted density
            dimerfitcorr = maindens[0] - maindensfit
            mono1fitcorr = sub1dens[0] - sub1densfit
            mono2fitcorr = sub2dens[0] - sub2densfit

            # calculate difference in electron-electron correction for fragment-fragment combination
            # eecorr(dimer) - eecorr(frag1) - eecorr(frag2)
            energy -= self.eb_calc_lowcost('eecorr', dimerfitcorr, mono1fitcorr, mono2fitcorr,
                                           vcoulmain, vcoulsub1, vcoulsub2, joint=False)

            # calculate difference in XC energy for fragment-fragment combination
            # xc(dimer) - xc(frag1) - xc(frag2)
            energy -= self.eb_calc_lowcost('xc', maindens, sub1dens, sub2dens,
                                           maindens, sub1dens, sub2dens, joint=False)

            # calculate difference in kinetic energy for fragment-fragment combination
            # kin(dimer) - kin(frag1) - kin(frag2)
            energy -= self.eb_calc_lowcost('kin', maindens, sub1dens, sub2dens,
                                           maindens, sub1dens, sub2dens, joint=False)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot += (dimer.get_nuclear_repulsion_energy() - mono1.get_nuclear_repulsion_energy()
                        - mono2.get_nuclear_repulsion_energy())

            counter += 1

        # iterate over all 1-3 fragment-fragment combinations
        for i in self._ebmfccmbe_res.trimer_res_by_comb:
            print('>  \n>  Fragment-Fragment Combination', counter, 'of', self.nffcombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # trimer_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the trimer results
            # overlap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [dimer_res, cap_res] with the dimer results and cap results
            # frag_res is a list of all fragment results
            d12 = (i[0], i[0] + 1)
            d23 = (i[1] - 1, i[1])
            trimer = self._ebmfccmbe_res.trimer_res_by_comb[i]
            dimer12 = self._ebmfccmbe_res.overlap_res_by_comb[d12][0]
            dimer23 = self._ebmfccmbe_res.overlap_res_by_comb[d23][0]
            mono2 = self._ebmfccmbe_res.frag_res[i[0] + 1]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens, adddens = self.rhocalc(trimer, dimer12, dimer23, mono2,
                                                                           trimer=True, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit, adddensfit = self.rhocalc(trimer,
                                                                                          dimer12, dimer23, mono2,
                                                                                          trimer=True, fit=True)
            vnucmain, vnucsub1, vnucsub2, vnucadd = self.potcalc(trimer, dimer12, dimer23, mono2,
                                                                 trimer=True, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2, vcouladd = self.potcalc(trimer, dimer12, dimer23, mono2,
                                                                     trimer=True, pot='coul')

            # add all fragment-fragment and potentials
            rhotot += diffdens
            rhototfit += diffdensfit
            diffdenstot += diffdens
            vcoultot += vcoulmain - vcoulsub1 - vcoulsub2 + vcouladd

            # calculate difference in electron-nuclei interaction for fragment-fragment combination
            # en(trimer) - en(dimer12) - en(dimer23) + en(frag2)
            energy -= self.eb_calc_lowcost('en', maindens, sub1dens, sub2dens, adddens,
                                           vnucmain, vnucsub1, vnucsub2, vnucadd, joint=True)

            # calculate difference in electron-electron interaction for fragment-fragment combination
            # ee(trimer) - ee(dimer12) - ee(dimer23) + ee(frag2)
            energy -= self.eb_calc_lowcost('ee', maindens, sub1dens, sub2dens, adddens,
                                           vcoulmain, vcoulsub1, vcoulsub2, vcouladd, joint=True)

            # calculate fitcorrection for fragment-fragment combination
            # density - fitted density
            trimerfitcorr = maindens[0] - maindensfit
            dimer12fitcorr = sub1dens[0] - sub1densfit
            dimer23fitcorr = sub2dens[0] - sub2densfit
            mono2fitcorr = adddens[0] - adddensfit

            # calculate difference in electron-electron correction for fragment-fragment combination
            # eecorr(trimer) - eecorr(dimer12) - eecorr(dimer23) + eecorr(frag2)
            energy -= self.eb_calc_lowcost('eecorr', trimerfitcorr, dimer12fitcorr, dimer23fitcorr,
                                           mono2fitcorr, vcoulmain, vcoulsub1, vcoulsub2, vcouladd, joint=True)

            # calculate difference in XC energy for fragment-fragment combination
            # xc(trimer) - xc(dimer12) - xc(dimer23) + xc(frag2)
            energy -= self.eb_calc_lowcost('xc', maindens, sub1dens, sub2dens, adddens,
                                           maindens, sub1dens, sub2dens, adddens, joint=True)

            # calculate difference in kinetic energy for fragment-fragment combination
            # kin(trimer) - kin(dimer12) - kin(dimer23) + kin(frag2)
            energy -= self.eb_calc_lowcost('kin', maindens, sub1dens, sub2dens, adddens,
                                           maindens, sub1dens, sub2dens, adddens, joint=True)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot += (trimer.get_nuclear_repulsion_energy() - dimer12.get_nuclear_repulsion_energy()
                        - dimer23.get_nuclear_repulsion_energy() + mono2.get_nuclear_repulsion_energy())

            counter += 1

        print('>  \n>  Starting Fragment-Cap Calculations\n>  ')
        counter = 1
        # iterate over all fragment-cap combinations
        for i in self._ebmfccmbe_res.fragcap_res_by_comb:
            print('>  \n>  Fragment-Cap Combination', counter, 'of', self.nfccombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # fragcap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the fragment-cap
            # dimer results
            # frag_res is a list of all fragment results
            # cap_res is a list of all cap results
            fragcap = self._ebmfccmbe_res.fragcap_res_by_comb[i]
            frag = self._ebmfccmbe_res.frag_res[i[0]]
            cap = self._ebmfccmbe_res.cap_res[i[1]]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens = self.rhocalc(fragcap, frag, cap, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit = self.rhocalc(fragcap, frag, cap, fit=True)
            vnucmain, vnucsub1, vnucsub2 = self.potcalc(fragcap, frag, cap, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2 = self.potcalc(fragcap, frag, cap, pot='coul')

            # subtract all fragment-cap and potentials
            rhotot -= diffdens
            rhototfit -= diffdensfit
            diffdenstot -= diffdens
            vcoultot -= vcoulmain - vcoulsub1 - vcoulsub2

            # calculate difference in electron-nuclei interaction for fragment-cap combination
            # en(dimer) - en(frag) - en(cap)
            energy += self.eb_calc_lowcost('en', maindens, sub1dens, sub2dens,
                                           vnucmain, vnucsub1, vnucsub2, joint=False)

            # calculate difference in electron-electron interaction for fragment-cap combination
            # ee(dimer) - ee(frag) - ee(cap)
            energy += self.eb_calc_lowcost('ee', maindens, sub1dens, sub2dens,
                                           vcoulmain, vcoulsub1, vcoulsub2, joint=False)

            # calculate fitcorrection for fragment-cap combination
            # density - fitted density
            dimerfitcorr = maindens[0] - maindensfit
            mono1fitcorr = sub1dens[0] - sub1densfit
            mono2fitcorr = sub2dens[0] - sub2densfit

            # calculate difference in electron-electron correction for fragment-cap combination
            # eecorr(dimer) - eecorr(frag) - eecorr(cap)
            energy += self.eb_calc_lowcost('eecorr', dimerfitcorr, mono1fitcorr, mono2fitcorr,
                                           vcoulmain, vcoulsub1, vcoulsub2, joint=False)

            # calculate difference in XC energy for fragment-cap combination
            # xc(dimer) - xc(frag) - xc(cap)
            energy += self.eb_calc_lowcost('xc', maindens, sub1dens, sub2dens,
                                           maindens, sub1dens, sub2dens, joint=False)

            # calculate difference in kinetic energy for fragment-cap combination
            # kin(dimer) - kin(frag) - kin(cap)
            energy += self.eb_calc_lowcost('kin', maindens, sub1dens, sub2dens,
                                           maindens, sub1dens, sub2dens, joint=False)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot -= (fragcap.get_nuclear_repulsion_energy() - frag.get_nuclear_repulsion_energy()
                        - cap.get_nuclear_repulsion_energy())

            counter += 1

        print('>  \n>  Starting Cap-Cap Calculations\n>  ')
        counter = 1
        # iterate over all cap-cap combinations
        for i in self._ebmfccmbe_res.capcap_res_by_comb:
            print('>  \n>  Cap-Cap Combination', counter, 'of', self.ncccombi, '\n>  ')

            # i is a tuple with the combination (0, 1)
            # capcap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning the cap-cap
            # dimer results
            # cap_res is a list of all cap results
            capcap = self._ebmfccmbe_res.capcap_res_by_comb[i]
            cap1 = self._ebmfccmbe_res.cap_res[i[0]]
            cap2 = self._ebmfccmbe_res.cap_res[i[1]]

            # get densities and potentials
            diffdens, maindens, sub1dens, sub2dens = self.rhocalc(capcap, cap1, cap2, order=1)
            diffdensfit, maindensfit, sub1densfit, sub2densfit = self.rhocalc(capcap, cap1, cap2, fit=True)
            vnucmain, vnucsub1, vnucsub2 = self.potcalc(capcap, cap1, cap2, pot='nuc')
            vcoulmain, vcoulsub1, vcoulsub2 = self.potcalc(capcap, cap1, cap2, pot='coul')

            # add all cap-cap and potentials
            rhotot += diffdens
            rhototfit += diffdensfit
            diffdenstot += diffdens
            vcoultot += vcoulmain - vcoulsub1 - vcoulsub2

            # calculate difference in electron-nuclei interaction for cap-cap combination
            # en(dimer) - en(cap) - en(cap)
            energy -= self.eb_calc_lowcost('en', maindens, sub1dens, sub2dens,
                                           vnucmain, vnucsub1, vnucsub2, joint=False)

            # calculate difference in electron-electron interaction for cap-cap combination
            # ee(dimer) - ee(cap) - ee(cap)
            energy -= self.eb_calc_lowcost('ee', maindens, sub1dens, sub2dens,
                                           vcoulmain, vcoulsub1, vcoulsub2, joint=False)

            # calculate fitcorrection for cap-cap combination
            # density - fitted density
            dimerfitcorr = maindens[0] - maindensfit
            mono1fitcorr = sub1dens[0] - sub1densfit
            mono2fitcorr = sub2dens[0] - sub2densfit

            # calculate difference in electron-electron correction for cap-cap combination
            # eecorr(dimer) - eecorr(cap) - eecorr(cap)
            energy -= self.eb_calc_lowcost('eecorr', dimerfitcorr, mono1fitcorr, mono2fitcorr,
                                           vcoulmain, vcoulsub1, vcoulsub2, joint=False)

            # calculate difference in XC energy for cap-cap combination
            # xc(dimer) - xc(cap) - xc(cap)
            energy -= self.eb_calc_lowcost('xc', maindens, sub1dens, sub2dens,
                                           maindens, sub1dens, sub2dens, joint=False)

            # calculate difference in kinetic energy for cap-cap combination
            # kin(dimer) - kin(cap) - kin(cap)
            energy -= self.eb_calc_lowcost('kin', maindens, sub1dens, sub2dens,
                                           maindens, sub1dens, sub2dens, joint=False)

            # calculate difference in nuclear repulsion energy for fragment-fragment combination
            ebnntot += (capcap.get_nuclear_repulsion_energy() - cap1.get_nuclear_repulsion_energy()
                        - cap2.get_nuclear_repulsion_energy())

            counter += 1

        # add the interactions for the total molecule to the final energy
        energy += (rhotot[0] * vnuctot).get_values()
        energy += 0.5 * (rhotot[0] * vcoultot).get_values()
        rhototfitcorr = rhotot[0] - rhototfit
        energy += 0.5 * (rhototfitcorr * vcoultot).get_values()
        energy += self.nadxc.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values)
        energy += self.nadkin.eval_energy_n(density=rhotot[0].values, densgrad=rhotot[1].values)
        dbnntot += self.get_nuclear_repulsion_energy(totmol)
        nnint = dbnntot - ebnntot

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("END ", current_time))

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("START ", current_time))

        # calculate the energy on the supermolecular grid
        inten = np.dot(self._grid.weights, energy)

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        timelist.append(("END ", current_time))

        return inten + nnint, timelist

    def metarun(self):

        db_res = self.create_results_instance()

        if self._order == 1:
            if self._lowcost:
                energy1, times = self.get_dbcorr_1st_order_lowcost()
                db_res.energy.append((energy1, times))
            else:
                energy1, times = self.get_dbcorr_1st_order()
                db_res.energy.append((energy1, times))
        elif self._order == 2:
            if self._lowcost:
                energy2, times = self.get_dbcorr_2nd_order_lowcost()
                db_res.energy.append((energy2, times))
            else:
                energy2, times = self.get_dbcorr_2nd_order()
                db_res.energy.append((energy2, times))
        else:
            raise PyAdfError('Not implemented order "' + str(self._order) + '" for db-MFCC-MBE(2) Job')

        return db_res


class MFCCMBE2InteractionResults(results):
    """
    Class for MFCC-MBE(2) interaction results.

    """

    def __init__(self, job):
        super().__init__(job)
        self.fragfrag_res_by_comb = {}
        self.frag1cap2_res_by_comb = {}
        self.frag2cap1_res_by_comb = {}
        self.capcap_res_by_comb = {}

    def get_energy_function(self, res):
        """
        returns the total energy function depending on the type of results

        @param res: results
        @type  res: results object
        """
        if isinstance(res, TurbomoleSinglePointResults):
            return res.get_energy()
        elif isinstance(res, OrcaResults):
            return res.get_total_energy()
        elif isinstance(res, adfsinglepointresults):
            return res.get_total_energy()

    def get_fragfrag_energy(self):
        """
        returns total interaction energy of all fragment-fragment combinations
        """
        fragfragenergy = 0.0
        # iterate over all fragment-fragment combinations
        for i in self.fragfrag_res_by_comb:
            # i is a tuple with the combination (0, 1)
            # fragfrag_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [fragfragres, frag1res, frag2res] with the dimer results and the results for frag1 and frag2
            energy = self.get_energy_function(self.fragfrag_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.fragfrag_res_by_comb[i][1])
                        + self.get_energy_function(self.fragfrag_res_by_comb[i][2]))
            fragfragenergy += energy
        return fragfragenergy

    def get_fragcap_energy(self):
        """
        returns total interaction energy of all fragment-cap combinations
        """
        fragcapenergy = 0.0
        # iterate over all fragment1-cap2 combinations
        for i in self.frag1cap2_res_by_comb:
            # i is a tuple with the combination (0, 1)
            # frag1cap2_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [fragcapres, fragres, capres] with the dimer results and the results for frag and cap
            energy = self.get_energy_function(self.frag1cap2_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.frag1cap2_res_by_comb[i][1])
                        + self.get_energy_function(self.frag1cap2_res_by_comb[i][2]))
            fragcapenergy += energy

        # iterate over all fragment2-cap1 combinations
        for i in self.frag2cap1_res_by_comb:
            # i is a tuple with the combination (0, 1)
            # frag2cap1_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [fragcapres, fragres, capres] with the dimer results and the results for frag and cap
            energy = self.get_energy_function(self.frag2cap1_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.frag2cap1_res_by_comb[i][1])
                        + self.get_energy_function(self.frag2cap1_res_by_comb[i][2]))
            fragcapenergy += energy
        return fragcapenergy

    def get_capcap_energy(self):
        """
        returns total interaction energy of all cap-cap combinations
        """
        capcapenergy = 0.0
        # iterate over all cap-cap combinations
        for i in self.capcap_res_by_comb:
            # i is a tuple with the combination (0, 1)
            # capcap_res_by_comb is a dictionary with the combinations (0, 1) as keys returning a
            # list [capcapres, cap1res, cap2res] with the dimer results and the results for cap1 and cap2
            energy = self.get_energy_function(self.capcap_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.capcap_res_by_comb[i][1])
                        + self.get_energy_function(self.capcap_res_by_comb[i][2]))
            capcapenergy += energy
        return capcapenergy

    def get_total_interaction_energy(self):
        """
        returns total interaction energy
        """
        fragfragenergy = self.get_fragfrag_energy()
        fragcapenergy = self.get_fragcap_energy()
        capcapenergy = self.get_capcap_energy()
        totalenergy = fragfragenergy - fragcapenergy + capcapenergy
        return totalenergy

    def print_number_of_calculations(self):
        """
        returns number of calculated combinations
        """
        sumofall = (len(self.fragfrag_res_by_comb) + len(self.frag1cap2_res_by_comb)
                    + len(self.frag2cap1_res_by_comb) + len(self.capcap_res_by_comb))

        print('> Number of calculated combinations: ')
        print('> n(Frag-Frag): %d' % len(self.fragfrag_res_by_comb))
        print('> n(Frag-Cap):  %d' % (len(self.frag1cap2_res_by_comb) + len(self.frag2cap1_res_by_comb)))
        print('> n(Cap-Cap):   %d' % len(self.capcap_res_by_comb))
        print('>', 20 * '-')
        print('> Sum:          %d' % sumofall)


class MFCCMBE2InteractionJob(metajob):

    def __init__(self, frags1, frags2, jobfunc, jobfunc_kwargs=None, caps='mfcc', order=2, cutoff=None):
        """
        Initialize a MFCC-MBE(2) interaction job.
        This job can be run to calculate ineractions between two proteins.

        @param frags1: list of capped fragments
        @type  frags1: L{cappedfragmentlist}
        @param frags2: list of capped fragments
        @type  frags2: L{cappedfragmentlist}
        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type  jobfunc: function with signature adfsinglepointjob(mol: molecule, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type  jobfunc_kwargs: dict or None
        @param caps: 'mfcc' or 'hydrogen'
        @type  caps: str
        @param order: many-body expansion order
        @type  order: int
        @param cutoff: distance cutoff in Angstrom for calculating combinations
        @type  cutoff: int or float

        """
        super().__init__()

        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self._cutoff = cutoff
        self._order = order
        self._caps = caps
        self._frags1 = frags1
        self._frags2 = frags2

    def monomerlist(self, fraglist):
        """
        @param fraglist: list of capped fragments
        @type  fraglist: L{cappedfragmentlist}

        returns the list of cappedfragments
        """
        monolist = []
        for frag in fraglist.fragiter():
            monolist.append(frag)
        return monolist

    def caplist(self, fraglist):
        """
        @param fraglist: list of capped fragments
        @type  fraglist: L{cappedfragmentlist}

        returns the list of caps
        """
        caplist = []
        for cap in fraglist.capiter():
            caplist.append(cap)
        return caplist

    def create_results_instance(self):
        return MFCCMBE2InteractionResults(self)

    def intencalc(self, frag1, frag2):
        """
        returns the results for the dimer and the two monomers

        @param frag1: fragment/cap 1 of the dimer
        @type  frag1: cappedfragment
        @param frag2: fragment/cap 2 of the dimer
        @type  frag2: cappedfragment
        """
        dimer = frag1.mol + frag2.mol
        dimer_res = self.jobfunc(dimer, **self._jobfunc_kwargs)
        frag1_res = self.jobfunc(frag1.mol, **self._jobfunc_kwargs)
        frag2_res = self.jobfunc(frag2.mol, **self._jobfunc_kwargs)
        return dimer_res, frag1_res, frag2_res

    def fragcapintencalc(self, frags1, frags2):
        """
        returns adictionary with the combinations (0, 1) as keys returning a list [fragcapres, fragres, capres]
        with the dimer results and the results for the fragment and the cap

        @param frags1: fragmentlist of protein 1
        @type  frags1: L{cappedfragmentlist}
        @param frags2: fragmentlist of protein 2
        @type  frags2: L{cappedfragmentlist}
        """
        fragcap_res_by_comb = {}
        for i, frag in enumerate(self.monomerlist(frags1)):
            for j, cap in enumerate(self.caplist(frags2)):
                print('>  Fragment', i + 1, 'of', len(self.monomerlist(frags1)),
                      'with Cap', j + 1, 'of', len(self.caplist(frags2)))

                # check if fragment-fragment distance is in cutoff range
                fragcapdist = frag.mol.distance(cap.mol)
                cutoffbool = False
                if (self._cutoff and fragcapdist <= self._cutoff) or self._cutoff is None:
                    cutoffbool = True

                # if the fragment and cap are not overlapping and the distance is in cutoff range calculate combination
                if fragcapdist > 0.0 and cutoffbool:
                    fragcapres, fragres, capres = self.intencalc(frag, cap)
                    fragcap_res_by_comb[(i, j)] = [fragcapres, fragres, capres]

                elif fragcapdist == 0.0:
                    print('>  WARNING! Fragment and Cap are overlapping!')
                    print(' ' + 50 * '-')
                    raise PyAdfError

                else:
                    print('>  Distance between Fragment and Cap greater than the cutoff of',
                          self._cutoff, 'Angstrom')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')
        return fragcap_res_by_comb

    def metarun(self):
        mfccmbe_results = self.create_results_instance()

        # FRAGMENT-FRAGMENT INTERACTIONS
        print('>  Starting Fragment-Fragment Calculations')
        for i, frag1 in enumerate(self.monomerlist(self._frags1)):
            for j, frag2 in enumerate(self.monomerlist(self._frags2)):
                print('>  Fragment', i + 1, 'of', len(self.monomerlist(self._frags1)),
                      'with Fragment', j + 1, 'of', len(self.monomerlist(self._frags2)))

                # check if fragment-fragment distance is in cutoff range
                fragfragdist = frag1.mol.distance(frag2.mol)
                cutoffbool = False
                if (self._cutoff and fragfragdist <= self._cutoff) or self._cutoff is None:
                    cutoffbool = True

                # if the fragments are not overlapping and the distance is in cutoff range calculate combination
                if fragfragdist > 0.0 and cutoffbool:
                    fragfragres, frag1res, frag2res = self.intencalc(frag1, frag2)
                    mfccmbe_results.fragfrag_res_by_comb[(i, j)] = [fragfragres, frag1res, frag2res]

                elif fragfragdist == 0.0:
                    print('>  WARNING! Fragments are overlapping!')
                    print(' ' + 50 * '-')
                    raise PyAdfError

                else:
                    print('>  Distance between Fragments greater than the cutoff of',
                          self._cutoff, 'Angstrom')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')

        # FRAGMENT-CAP INTERACTIONS
        print('>  Starting Fragment(1)-Cap(2) Calculations')
        frag1cap2res = self.fragcapintencalc(self._frags1, self._frags2)
        print('>  Starting Fragment(2)-Cap(1) Calculations')
        frag2cap1res = self.fragcapintencalc(self._frags2, self._frags1)
        for i in frag1cap2res:
            mfccmbe_results.frag1cap2_res_by_comb[i] = frag1cap2res[i]
        for i in frag2cap1res:
            mfccmbe_results.frag2cap1_res_by_comb[i] = frag2cap1res[i]

        # CAP-CAP INTERACTIONS
        print('>  Starting Cap-Cap Calculations')
        for i, cap1 in enumerate(self.caplist(self._frags1)):
            for j, cap2 in enumerate(self.caplist(self._frags2)):
                print('>  Cap', i + 1, 'of', len(self.caplist(self._frags1)),
                      'with Cap', j + 1, 'of', len(self.caplist(self._frags2)))

                # check if fragment-fragment distance is in cutoff range
                capcapdist = cap1.mol.distance(cap2.mol)
                cutoffbool = False
                if (self._cutoff and capcapdist <= self._cutoff) or self._cutoff is None:
                    cutoffbool = True

                # if the caps are not overlapping and the distance is in cutoff range calculate combination
                if capcapdist > 0.0 and cutoffbool:
                    capcapres, cap1res, cap2res = self.intencalc(cap1, cap2)
                    mfccmbe_results.capcap_res_by_comb[(i, j)] = [capcapres, cap1res, cap2res]

                elif capcapdist == 0.0:
                    print('>  WARNING! Fragments are overlapping!')
                    print(' ' + 50 * '-')
                    raise PyAdfError

                else:
                    print('>  Distance between Fragments greater than the cutoff of',
                          self._cutoff, 'Angstrom')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')

        return mfccmbe_results


class MFCCMBE3InteractionResults(results):
    """
    Class for MFCCMBE(3) Interaction results.
    """

    def __init__(self, job):
        super().__init__(job)
        self.ligand_res = []
        self.frag_res = {}
        self.cap_res = {}
        self.fraglig_res = {}
        self.caplig_res = {}
        self.fragfrag_res_by_comb = {}
        self.fragcap_res_by_comb = {}
        self.capcap_res_by_comb = {}
        self.fragfraglig_res_by_comb = {}
        self.fragcaplig_res_by_comb = {}
        self.capcaplig_res_by_comb = {}
        self.frag_elint_corr_by_comb = {}
        self.cap_elint_corr_by_comb = {}

    def get_elint_corr(self):
        """
        returns electrostatic interaction energy
        """
        frag_elint_corr = 0.0
        for i in self.frag_elint_corr_by_comb:
            frag_elint_corr += self.frag_elint_corr_by_comb[i]
        cap_elint_corr = 0.0
        for i in self.cap_elint_corr_by_comb:
            cap_elint_corr += self.cap_elint_corr_by_comb[i]
        return frag_elint_corr - cap_elint_corr

    def get_fragfraglig_energy(self):
        """
        returns total interaction energy of all fragment-fragment-ligand combinations
        """
        frag_frag_lig_energy = 0.0
        for i in self.fragfraglig_res_by_comb:
            # i is a tuple with the combination (0, 'A', 1, 'B')
            # fragfraglig_res_by_comb is a dictionary with the combinations (0, 'A', 1, 'B') as keys returning a
            # list [dimerlig_res, caplig_res], [dimerlig_res], [trimerlig_res]

            # joint fragments
            if len(self.fragfraglig_res_by_comb[i]) == 2:
                # fragfrag_res_by_comb is a dictionary with the fragment number and chain id (0, 'A', 1, 'B') as keys
                # returning a list [dimer_res, cap_res], [dimer_res], , [trimer_res]
                #
                # fraglig_res is a dictionary with the fragment number and chain id (0, A) as keys returning
                # the fragment-ligand results
                #
                # frag_res is a dictionary with the fragment number and chain id (0, A) as keys
                # returning the fragment results

                fragfraglig_energy = self.fragfraglig_res_by_comb[i][0].get_total_energy()
                fragfrag_energy = self.fragfrag_res_by_comb[i][0].get_total_energy()
                frag1lig_energy = self.fraglig_res[(i[0], i[1])].get_total_energy()
                frag2lig_energy = self.fraglig_res[(i[2], i[3])].get_total_energy()
                caplig_energy = self.fragfraglig_res_by_comb[i][1].get_total_energy()
                frag1_energy = self.frag_res[(i[0], i[1])].get_total_energy()
                frag2_energy = self.frag_res[(i[2], i[3])].get_total_energy()
                cap_energy = self.fragfrag_res_by_comb[i][1].get_total_energy()

                energy = (fragfraglig_energy - fragfrag_energy - frag1lig_energy - frag2lig_energy
                          + caplig_energy + frag1_energy + frag2_energy - cap_energy)
                frag_frag_lig_energy += energy

            # 1-3 fragments in the same chain
            elif i[2] == i[0] + 2 and i[1] == i[3]:
                # fragfrag_res_by_comb is a dictionary with the fragment number and chain id (0, 'A', 1, 'B') as keys
                # returning a list [dimer_res, cap_res], [dimer_res] , [trimer_res]
                #
                # fraglig_res is a dictionary with the fragment number and chain id (0, A) as keys returning
                # the fragment-ligand results
                #
                # frag_res is a dictionary with the fragment number and chain id (0, A) as keys
                # returning the fragment results

                c = i[1]
                f1f2key = (i[0], c, i[0] + 1, c)
                f2f3key = (i[2] - 1, c, i[2], c)
                f2key = (i[0] + 1, c)
                fragfragfraglig_energy = self.fragfraglig_res_by_comb[i][0].get_total_energy()
                fragfragfrag_energy = self.fragfrag_res_by_comb[i][0].get_total_energy()
                frag1frag2lig_energy = self.fragfraglig_res_by_comb[f1f2key][0].get_total_energy()
                frag2frag3lig_energy = self.fragfraglig_res_by_comb[f2f3key][0].get_total_energy()
                frag2lig_energy = self.fraglig_res[f2key].get_total_energy()
                frag1frag2_energy = self.fragfrag_res_by_comb[f1f2key][0].get_total_energy()
                frag2frag3_energy = self.fragfrag_res_by_comb[f2f3key][0].get_total_energy()
                frag2_energy = self.frag_res[f2key].get_total_energy()
                energy = (fragfragfraglig_energy - fragfragfrag_energy - frag1frag2lig_energy - frag2frag3lig_energy
                          + frag2lig_energy + frag1frag2_energy + frag2frag3_energy - frag2_energy)
                frag_frag_lig_energy += energy

            # disjoint fragments
            else:
                # fragfrag_res_by_comb is a dictionary with the fragment number and chain id (0, 'A', 1, 'B') as keys
                # returning a list [dimer_res, cap_res], [dimer_res] , [trimer_res]
                #
                # fraglig_res is a dictionary with the fragment number and chain id (0, A) as keys returning
                # the fragment-ligand results
                #
                # frag_res is a dictionary with the fragment number and chain id (0, A) as keys
                # returning the fragment results

                fragfraglig_energy = self.fragfraglig_res_by_comb[i][0].get_total_energy()
                fragfrag_energy = self.fragfrag_res_by_comb[i][0].get_total_energy()
                frag1lig_energy = self.fraglig_res[(i[0], i[1])].get_total_energy()
                frag2lig_energy = self.fraglig_res[(i[2], i[3])].get_total_energy()
                frag1_energy = self.frag_res[(i[0], i[1])].get_total_energy()
                frag2_energy = self.frag_res[(i[2], i[3])].get_total_energy()
                lig_energy = self.ligand_res[0].get_total_energy()

                energy = (fragfraglig_energy - fragfrag_energy - frag1lig_energy - frag2lig_energy
                          + frag1_energy + frag2_energy + lig_energy)
                frag_frag_lig_energy += energy

        return frag_frag_lig_energy

    def get_fragcaplig_energy(self):
        """
        returns total interaction energy of all fragment-cap-ligand combinations
        """
        frag_cap_lig_energy = 0.0
        for i in self.fragcaplig_res_by_comb:
            # fragcaplig_res_by_comb is a dictionary with the fragment/cap number and chain id (0, 'A', 1, 'B') as keys
            # returning a list [fragcaplig_res]
            #
            # fragcap_res_by_comb is a dictionary with the fragment/cap number and chain id (0, 'A', 1, 'B') as keys
            # returning a list [fragcap_res]
            #
            # fraglig_res is a dictionary with the fragment number and chain id (0, A) as keys returning
            # the fragment-ligand results
            #
            # caplig_res is a dictionary with the cap number and chain id (0, A) as keys returning
            # the cap-ligand results
            #
            # frag_res is a dictionary with the fragment number and chain id (0, A) as keys
            # returning the fragment results
            #
            # cap_res is a dictionary with the cap number and chain id (0, A) as keys
            # returning the cap results

            fragcaplig_energy = self.fragcaplig_res_by_comb[i][0].get_total_energy()
            fragcap_energy = self.fragcap_res_by_comb[i][0].get_total_energy()
            fraglig_energy = self.fraglig_res[(i[0], i[1])].get_total_energy()
            caplig_energy = self.caplig_res[(i[2], i[3])].get_total_energy()
            frag_energy = self.frag_res[(i[0], i[1])].get_total_energy()
            cap_energy = self.cap_res[(i[2], i[3])].get_total_energy()
            lig_energy = self.ligand_res[0].get_total_energy()

            energy = (fragcaplig_energy - fragcap_energy - fraglig_energy - caplig_energy
                      + frag_energy + cap_energy + lig_energy)
            frag_cap_lig_energy += energy

        return frag_cap_lig_energy

    def get_capcaplig_energy(self):
        """
        returns total interaction energy of all fragment-cap-ligand combinations
        """
        cap_cap_lig_energy = 0.0
        for i in self.capcaplig_res_by_comb:
            # capcaplig_res_by_comb is a dictionary with the cap number and chain id (0, 'A', 1, 'B') as keys
            # returning a list [capcaplig_res]
            #
            # capcap_res_by_comb is a dictionary with the cap number and chain id (0, 'A', 1, 'B') as keys
            # returning a list [capcap_res]
            #
            # caplig_res is a dictionary with the cap number and chain id (0, A) as keys returning
            # the cap-ligand results
            #
            # cap_res is a dictionary with the cap number and chain id (0, A) as keys
            # returning the cap results

            capcaplig_energy = self.capcaplig_res_by_comb[i][0].get_total_energy()
            capcap_energy = self.capcap_res_by_comb[i][0].get_total_energy()
            cap1lig_energy = self.caplig_res[(i[0], i[1])].get_total_energy()
            cap2lig_energy = self.caplig_res[(i[2], i[3])].get_total_energy()
            cap1_energy = self.cap_res[(i[0], i[1])].get_total_energy()
            cap2_energy = self.cap_res[(i[2], i[3])].get_total_energy()
            lig_energy = self.ligand_res[0].get_total_energy()

            energy = (capcaplig_energy - capcap_energy - cap1lig_energy - cap2lig_energy
                      + cap1_energy + cap2_energy + lig_energy)
            cap_cap_lig_energy += energy

        return cap_cap_lig_energy

    def get_total_interaction_energy(self):
        """
        returns total interaction energy
        """
        fragligenergy = 0.0
        for i in self.fraglig_res:
            fraglig_energy = self.fraglig_res[i].get_total_energy()
            frag_energy = self.frag_res[i].get_total_energy()
            lig_energy = self.ligand_res[0].get_total_energy()
            fragligenergy += fraglig_energy - frag_energy - lig_energy

        capligenergy = 0.0
        for i in self.caplig_res:
            caplig_energy = self.caplig_res[i].get_total_energy()
            cap_energy = self.cap_res[i].get_total_energy()
            lig_energy = self.ligand_res[0].get_total_energy()
            capligenergy += caplig_energy - cap_energy - lig_energy

        fragfragligenergy = self.get_fragfraglig_energy()
        fragcapligenergy = self.get_fragcaplig_energy()
        capcapligenergy = self.get_capcaplig_energy()
        totalintenergy = fragligenergy - capligenergy + fragfragligenergy - fragcapligenergy + capcapligenergy

        if not self.frag_elint_corr_by_comb:
            return totalintenergy
        else:
            return totalintenergy + self.get_elint_corr()


class MFCCMBE3InteractionJob(metajob):

    def __init__(self, frags, ligand, jobfunc, jobfunc_kwargs=None, caps='mfcc', order=2, mbe2cutoff=None,
                 mbe3cutoff=None, elintcorr=False):
        """
        Initialize a MFCC-MBE(3) Interaction job.

        @param frags: list of capped fragments
        @type  frags: L{cappedfragmentlist}
        @param ligand: ligand molecule
        @type  ligand: molecule
        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type  jobfunc: function with signature adfsinglepointjob(mol: molecule, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type  jobfunc_kwargs: dict or None
        @param caps: 'mfcc' or 'hydrogen'
        @type  caps: str
        @param order: many-body expansion order
        @type  order: int
        @param mbe2cutoff: distance cutoff in Angstrom for MBE(2) calculations
        @type  mbe2cutoff: int or float
        @param mbe3cutoff: distance cutoff in Angstrom for MBE(3) calculations
        @type  mbe3cutoff: int or float
        @param elintcorr: adds electrostatic correction for combinations outside the cutoff range
        @type  elintcorr: boolean
        """
        super().__init__()

        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self._mbe2cutoff = mbe2cutoff
        self._mbe3cutoff = mbe3cutoff
        self._order = order
        self._caps = caps
        self._frags = frags
        self._ligand = ligand
        self._elintcorr = elintcorr

    @property
    def fraglist(self):
        """
        returns the list of cappedfragments
        """
        fraglist = []
        for frag in self._frags.fragiter():
            fraglist.append(frag)
        return fraglist

    @property
    def nfrag(self):
        """
        returns the number of cappedfragments
        """
        return len(self.fraglist)

    @property
    def caplist(self):
        """
        returns the list of caps
        """
        caplist = []
        for cap in self._frags.capiter():
            caplist.append(cap)
        return caplist

    @property
    def ncap(self):
        """
        returns the number of caps
        """
        return len(self.caplist)

    @property
    def nfragcombi(self):
        """
        returns theorethical number of fragment-fragment combinations
        """
        return len(list(itertools.combinations(list(range(self.nfrag)), self._order)))

    @property
    def ncapcombi(self):
        """
        returns theorethical number of cap-cap combinations
        """
        return len(list(itertools.combinations(list(range(self.ncap)), self._order)))

    def cutoffcheck(self, frag1, frag2):
        """
        returns if cutoff conditions are fulfilled

        @param frag1: frag/cap 1
        @type  frag1: cappedfragment()
        @param frag2: frag/cap 1
        @type  frag2: cappedfragment()
        """
        frag1frag2dist = frag1.mol.distance(frag2.mol)
        frag1ligdist = frag1.mol.distance(self._ligand)
        frag2ligdist = frag2.mol.distance(self._ligand)

        cutoffbool = False
        if (self._mbe3cutoff and ((frag1ligdist <= self._mbe3cutoff and frag2ligdist <= self._mbe3cutoff) or
                                  (frag1ligdist <= self._mbe3cutoff and frag1frag2dist <= self._mbe3cutoff) or
                                  (frag2ligdist <= self._mbe3cutoff and frag1frag2dist <= self._mbe3cutoff))):
            cutoffbool = True
        elif self._mbe3cutoff is None:
            cutoffbool = True
        return cutoffbool

    def keygen(self, combi, frag1, frag2):
        """
        returns an individual key like (1, 'A', 2, 'B') for the given combination.

        @param combi: combination
        @type  combi: tuple
        @param frag1: frag/cap 1
        @type  frag1: cappedfragment()
        @param frag2: frag/cap 2
        @type  frag2: cappedfragment()
        """
        chain_id1, resname1, resnum1 = frag1.mol.get_atom_resinfo(1)
        chain_id2, resname2, resnum2 = frag2.mol.get_atom_resinfo(1)
        key = (combi[0], chain_id1, combi[1], chain_id2)
        return key

    def keycheckcalc(self, key, resdict, resdictlig, mol):
        """
        Calculates missing fragments and fragment-ligand combinations.
        This is needed because if a cutoff is set for the first order / mfcc part some frag-lig combinations may
        not be calculated but needed for the second order

        @param key: dictionary key to check
        @type  key: tuple
        @param resdict: target dictionary frag/cap results
        @type  resdict: dict
        @param resdictlig: target dictionary frag/cap + lig results
        @type  resdictlig: dict
        @param mol: frag/cap molecule
        @type  mol: molecule
        """
        if key not in resdict:
            fragres = self.jobfunc(mol, **self._jobfunc_kwargs)
            resdict[key] = fragres
            fragligres = self.jobfunc(mol + self._ligand, **self._jobfunc_kwargs)
            resdictlig[key] = fragligres

    def get_electrostatic_interaction_energy(self, frag1, frag2, frag1_charges, frag2_charges):
        """
        Return the electrostatic interaction energy between charges of two fragments

        @param frag1: fragment 1
        @type  frag1: molecule
        @param frag2: fragment 2
        @type  frag2: molecule
        @param frag1_charges: list of charges (multipole derived charges MDC-q)
        @type  frag1_charges: list
        @param frag2_charges: list of charges (multipole derived charges MDC-q)
        @type  frag2_charges: list
        """
        import numpy

        inten = 0.0
        for coord1, charge1 in zip(frag1.get_coordinates(), frag1_charges):
            for coord2, charge2 in zip(frag2.get_coordinates(), frag2_charges):
                dist = numpy.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)
                dist = dist / Bohr_in_Angstrom
                inten += charge1 * charge2 / dist
        return inten

    def create_results_instance(self):
        return MFCCMBE3InteractionResults(self)

    def fragfragcalc(self, monomer1, monomer2, overlap=False, lig=False):
        """
        returns results for dimers or dimer-ligand combinations

        [frag-frag, cap], [frag-frag], [frag-cap], [cap-cap]
        [frag-frag-lig, cap-lig], [frag-frag-lig], [frag-cap-lig], [cap-cap-lig]

        @param monomer1: frag/cap molecule 1
        @type  monomer1: cappedfragment
        @param monomer2: frag/cap molecule 2
        @type  monomer2: cappedfragment
        @param overlap: True when overlap of 1-2 dimers
        @type  overlap: boolean
        @param lig: True if ligand shall be calculated with the dimer
        @type  lig: boolean
        """
        dimer = monomer1.merge_fragments(monomer2)

        if lig:
            dimer_res = self.jobfunc(dimer.mol + self._ligand, **self._jobfunc_kwargs)
        else:
            dimer_res = self.jobfunc(dimer.mol, **self._jobfunc_kwargs)

        if overlap:
            cap = dimer._overlapping_caps[0]
            if len(dimer.get_overlapping_caps()) > 1:
                raise PyAdfError("Too many overlapping caps !")
            if lig:
                cap_res = self.jobfunc(cap.mol + self._ligand, **self._jobfunc_kwargs)
            else:
                cap_res = self.jobfunc(cap.mol, **self._jobfunc_kwargs)
            return [dimer_res, cap_res]
        else:
            return [dimer_res]

    def fragfragfragcalc(self, trimer, lig=False):
        """
        returns results for trimer or trimer-ligand combination

        @param trimer: trimer
        @type  trimer: cappedfragment
        @param lig: True if ligand shall be calculated with the dimer
        @type  lig: boolean
        """
        if lig:
            trimer_res = self.jobfunc(trimer.mol + self._ligand, **self._jobfunc_kwargs)
        else:
            trimer_res = self.jobfunc(trimer.mol, **self._jobfunc_kwargs)
        return [trimer_res]

    def metarun(self):
        print('>  Starting MFCC-MBE(3) Interaction Job')
        mfccmbe_results = self.create_results_instance()

        # calculate ligand
        ligandres = self.jobfunc(self._ligand, **self._jobfunc_kwargs)
        mfccmbe_results.ligand_res.append(ligandres)
        ligand_charges = ligandres.get_multipolederiv_charges(level='MDC-q')

        print('>  Starting Fragment and Fragment-Ligand Calculations')
        for i, frag in enumerate(self.fraglist):
            print('>  Calculating Fragment', i + 1, 'of', self.nfrag)

            chain_id, resname, resnum = frag.mol.get_atom_resinfo(1)  # 'A', 'ALA', 1
            key = (i, chain_id)  # (0, 'A')

            # check if fragment-ligand distance is in cutoff range
            fragligdist = frag.mol.distance(self._ligand)
            cutoffbool = False
            if (self._mbe2cutoff and fragligdist <= self._mbe2cutoff) or self._mbe2cutoff is None:
                cutoffbool = True

            # if the fragment-ligand distance is in cutoff range calculate combination
            if cutoffbool:
                # calculate fragment
                fragres = self.jobfunc(frag.mol, **self._jobfunc_kwargs)
                mfccmbe_results.frag_res[key] = fragres

                print('>  Calculating Fragment', i + 1, 'of', self.nfrag, 'with Ligand')
                # calculate fragment-ligand
                fragligres = self.jobfunc(frag.mol + self._ligand, **self._jobfunc_kwargs)
                mfccmbe_results.fraglig_res[key] = fragligres

            # if fragment-ligand distance is not in cutoff range but electrostatic correction is turned on
            # calculate electrostatic interaction
            elif self._elintcorr and not cutoffbool:
                print('>  Distance between Fragment and Ligand greater than the cutoff of',
                      self._mbe2cutoff, 'Angstrom')
                print('>  Calculating electrostatic interaction energy')
                fragres = self.jobfunc(frag.mol, **self._jobfunc_kwargs)
                frag_charges = fragres.get_multipolederiv_charges(level='MDC-q')
                elintcorr = self.get_electrostatic_interaction_energy(frag.mol, self._ligand,
                                                                      frag_charges, ligand_charges)
                mfccmbe_results.cap_elint_corr_by_comb[i] = elintcorr

            else:
                print('>  Distance between Fragment and Ligand greater than the cutoff of',
                      self._mbe2cutoff, 'Angstrom')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')

        print('>  Starting Cap and Cap-Ligand Calculations')
        for i, cap in enumerate(self.caplist):
            print('>  Calculating Cap', i + 1, 'of', self.ncap)

            chain_id, resname, resnum = cap.mol.get_atom_resinfo(1)  # 'A', 'ALA', 1
            key = (i, chain_id)  # (0, 'A')

            # check if cap-ligand distance is in cutoff range
            capligdist = cap.mol.distance(self._ligand)
            cutoffbool = False
            if (self._mbe2cutoff and capligdist <= self._mbe2cutoff) or self._mbe2cutoff is None:
                cutoffbool = True

            # if the cap-ligand distance is in cutoff range calculate combination
            if cutoffbool:
                # calculate cap
                capres = self.jobfunc(cap.mol, **self._jobfunc_kwargs)
                mfccmbe_results.cap_res[key] = capres

                print('>  Calculating Cap', i + 1, 'of', self.ncap, 'with Ligand')
                # calculate cap-ligand
                capligres = self.jobfunc(cap.mol + self._ligand, **self._jobfunc_kwargs)
                mfccmbe_results.caplig_res[key] = capligres

            # if cap-ligand distance is not in cutoff range but electrostatic correction is turned on
            # calculate electrostatic interaction
            elif self._elintcorr and not cutoffbool:
                print('>  Distance between Cap and Ligand greater than the cutoff of', self._mbe2cutoff, 'Angstrom')
                print('>  Calculating electrostatic interaction energy')
                cap_res = self.jobfunc(cap.mol, **self._jobfunc_kwargs)
                cap_charges = cap_res.get_multipolederiv_charges(level='MDC-q')
                elintcorr = self.get_electrostatic_interaction_energy(cap.mol, self._ligand,
                                                                      cap_charges, ligand_charges)
                mfccmbe_results.cap_elint_corr_by_comb[i] = elintcorr

            else:
                print('>  Distance between Cap and Ligand greater than the cutoff of', self._mbe2cutoff, 'Angstrom')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')

        # FRAG-FRAG INTERACTIONS
        counter = 1
        print('>  Starting Fragment-Fragment-Ligand Calculations')
        for c in itertools.combinations(list(range(self.nfrag)), self._order):
            print('>  Fragment-Fragment Combination', counter, 'of', self.nfragcombi)
            print('>  Consisting of Fragments', c[0] + 1, 'and', c[1] + 1, 'of', self.nfrag, 'Fragments')

            monomer1 = self.fraglist[c[0]]
            monomer2 = self.fraglist[c[1]]
            dimer = monomer1.merge_fragments(monomer2)
            overlapping_caps = dimer.get_overlapping_caps()
            key = self.keygen(c, monomer1, monomer2)  # (0, 'A', 1, 'B')

            # check if cutoff conditions are fulfilled
            fragfragdist = monomer1.mol.distance(monomer2.mol)
            cutoffbool = self.cutoffcheck(monomer1, monomer2)

            if len(overlapping_caps) > 1:
                # TOO MANY CAPS
                raise PyAdfError("Handeling more than one overlapping cap not implemented yet!")

            # if cutoff conditions are fulfilled and joint 1-2 dimer
            elif len(overlapping_caps) == 1 and cutoffbool:
                # calculate dimer and dimer-ligand
                fragfragres = self.fragfragcalc(monomer1, monomer2, overlap=True, lig=False)
                fragfragligres = self.fragfragcalc(monomer1, monomer2, overlap=True, lig=True)
                mfccmbe_results.fragfrag_res_by_comb[key] = fragfragres
                mfccmbe_results.fragfraglig_res_by_comb[key] = fragfragligres

                # check if both fragments have been calculated, if not do so
                self.keycheckcalc((key[0], key[1]), mfccmbe_results.frag_res, mfccmbe_results.fraglig_res, monomer1.mol)
                self.keycheckcalc((key[2], key[3]), mfccmbe_results.frag_res, mfccmbe_results.fraglig_res, monomer2.mol)

            # if cutoff conditions are fulfilled and disjoint (>1-3) dimer
            elif len(overlapping_caps) == 0 and fragfragdist > 0.0 and cutoffbool:
                # calculate dimer and dimer-ligand
                fragfragres = self.fragfragcalc(monomer1, monomer2, overlap=False, lig=False)
                fragfragligres = self.fragfragcalc(monomer1, monomer2, overlap=False, lig=True)
                mfccmbe_results.fragfrag_res_by_comb[key] = fragfragres
                mfccmbe_results.fragfraglig_res_by_comb[key] = fragfragligres

                # check if both fragments have been calculated, if not do so
                self.keycheckcalc((key[0], key[1]), mfccmbe_results.frag_res, mfccmbe_results.fraglig_res, monomer1.mol)
                self.keycheckcalc((key[2], key[3]), mfccmbe_results.frag_res, mfccmbe_results.fraglig_res, monomer2.mol)

            # if cutoff conditions are fulfilled and 1-3 dimer and mfcc caps are used
            elif len(overlapping_caps) == 0 and fragfragdist == 0.0 and self._caps == 'mfcc' and cutoffbool:
                # 1-3-Fragment handeling for ACE-NME Caps
                midmonomer = self.fraglist[c[0] + 1]
                dimer12 = monomer1.merge_fragments(midmonomer)
                trimer = dimer12.merge_fragments(monomer2)

                if not key[1] == key[3]:
                    raise PyAdfError("This is not a 1-3 Combination, why is it treated like one?")

                key12 = (key[0], key[1], key[0] + 1, key[1])  # key for dimer12
                # check if dimer have been calculated, if not do so
                if key12 not in mfccmbe_results.fragfrag_res_by_comb:
                    # calculate dimer and dimer-ligand
                    dimer12res = self.fragfragcalc(monomer1, midmonomer, overlap=True, lig=False)
                    dimer12ligres = self.fragfragcalc(monomer1, midmonomer, overlap=True, lig=True)
                    mfccmbe_results.fragfrag_res_by_comb[key12] = dimer12res
                    mfccmbe_results.fragfraglig_res_by_comb[key12] = dimer12ligres

                key23 = (key[2] - 1, key[1], key[2], key[3])  # key for dimer23
                # check if dimer have been calculated, if not do so
                if key23 not in mfccmbe_results.fragfrag_res_by_comb:
                    # calculate dimer and dimer-ligand
                    dimer23res = self.fragfragcalc(midmonomer, monomer2, overlap=True, lig=False)
                    dimer23ligres = self.fragfragcalc(midmonomer, monomer2, overlap=True, lig=True)
                    mfccmbe_results.fragfrag_res_by_comb[key23] = dimer23res
                    mfccmbe_results.fragfraglig_res_by_comb[key23] = dimer23ligres

                # check if all frtagments have been calculated, if not do so
                self.keycheckcalc((key[0], key[1]), mfccmbe_results.frag_res, mfccmbe_results.fraglig_res,
                                  monomer1.mol)
                self.keycheckcalc((key[2], key[3]), mfccmbe_results.frag_res, mfccmbe_results.fraglig_res,
                                  monomer2.mol)
                self.keycheckcalc((key[0] + 1, key[1]), mfccmbe_results.frag_res, mfccmbe_results.fraglig_res,
                                  midmonomer.mol)

                # calculate trimer and trimer-ligand
                trimerres = self.fragfragfragcalc(trimer, lig=False)
                trimerligres = self.fragfragfragcalc(trimer, lig=True)
                mfccmbe_results.fragfrag_res_by_comb[key] = trimerres
                mfccmbe_results.fragfraglig_res_by_comb[key] = trimerligres

            else:
                print('>  Cutoff Conditions (' + str(self._mbe3cutoff) + ' Angstrom) not satisfied')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')
            counter += 1

        # FRAG-CAP INTERACTIONS
        print('>  Starting Fragment-Cap-Ligand Calculations')
        for i, frag in enumerate(self.fraglist):
            for j, cap in enumerate(self.caplist):
                print('>  Fragment', i + 1, 'of', self.nfrag, 'with Cap', j + 1, 'of', self.ncap)

                key = self.keygen((i, j), frag, cap)  # (0, 'A', 1, 'B')

                # check if cutoff conditions are fulfilled
                fragcapdist = frag.mol.distance(cap.mol)
                cutoffbool = self.cutoffcheck(frag, cap)

                # if cutoff conditions are fulfilled
                if fragcapdist > 0.0 and cutoffbool:
                    # calculate dimer and dimer-ligand
                    fragcapres = self.fragfragcalc(frag, cap, overlap=False, lig=False)
                    fragcapligres = self.fragfragcalc(frag, cap, overlap=False, lig=True)
                    mfccmbe_results.fragcap_res_by_comb[key] = fragcapres
                    mfccmbe_results.fragcaplig_res_by_comb[key] = fragcapligres

                    # check if both fragments have been calculated, if not do so
                    self.keycheckcalc((key[0], key[1]), mfccmbe_results.frag_res, mfccmbe_results.fraglig_res, frag.mol)
                    self.keycheckcalc((key[2], key[3]), mfccmbe_results.cap_res, mfccmbe_results.caplig_res, cap.mol)

                elif fragcapdist == 0.0:
                    print('>  Fragment and Cap are too close')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')

                else:
                    print('>  Cutoff Conditions (' + str(self._mbe3cutoff) + ' Angstrom) not satisfied')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')

        # CAP-CAP INTERACTIONS
        capcounter = 1
        print('>  Starting Cap-Cap-Ligand Calculations')
        for c in itertools.combinations(list(range(self.ncap)), self._order):
            print('>  Cap-Cap-Combination', capcounter, 'of', self.ncapcombi)
            print('>  Consisting of Caps', c[0] + 1, 'and', c[1] + 1, 'of', self.ncap, 'Caps')

            cap1 = self.caplist[c[0]]
            cap2 = self.caplist[c[1]]

            key = self.keygen(c, cap1, cap2)  # (0, 'A', 1, 'B')

            # check if cutoff conditions are fulfilled
            cap1cap2dist = cap1.mol.distance(cap2.mol)
            cutoffbool = self.cutoffcheck(cap1, cap2)

            # if cutoff conditions are fulfilled
            if cap1cap2dist > 0.0 and cutoffbool:
                # calculate dimer and dimer-ligand
                capcapres = self.fragfragcalc(cap1, cap2, overlap=False, lig=False)
                capcapligres = self.fragfragcalc(cap1, cap2, overlap=False, lig=True)
                mfccmbe_results.capcap_res_by_comb[key] = capcapres
                mfccmbe_results.capcaplig_res_by_comb[key] = capcapligres

                # check if both fragments have been calculated, if not do so
                self.keycheckcalc((key[0], key[1]), mfccmbe_results.cap_res, mfccmbe_results.caplig_res, cap1.mol)
                self.keycheckcalc((key[2], key[3]), mfccmbe_results.cap_res, mfccmbe_results.caplig_res, cap2.mol)

            elif cap1cap2dist == 0.0:
                print('>  Caps are too close')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')

            else:
                print('>  Cutoff Conditions (' + str(self._mbe3cutoff) + ' Angstrom) not satisfied')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')
            capcounter += 1

        return mfccmbe_results
