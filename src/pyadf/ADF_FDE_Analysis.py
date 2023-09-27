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
 Job and results of analysis of an ADF FDE calculation for a list of fragments.

 It defines and collects measures of accuracy of subsystem DFT:

     - absolute error in difference density
     - rms error in difference density
     - error in dipole moment

 @author: S. Maya Beyhan
 @author: Andreas W. Goetz
 @organization: Vrije Univeristeit Amsterdam (2008)

"""

from .Errors import PyAdfError
from .Molecule import molecule
from .BaseJob import metajob
from .ADFSinglePoint import adfsinglepointjob
from .ADFFragments import adffragmentsjob, fragment
from .ADF_FDE import adffdejob
from .Plot.Grids import adfgrid

import os
import glob
from functools import reduce


class adffdeanalysissettings:
    """
    Class for settings of  adffdeanalysisjob.
    """

    def __init__(self, tnad=None, runtype=None, usebasis='False', options=None, ncycle=None, cjcorr='False'):

        self.tnad = None
        if tnad is None:
            self.set_tnad(['COULOMB', 'THOMASFERMI', 'PW91K', 'PW91Kscaled', 'TW02', 'TF9W', 'PBE2', 'PBE3', 'PBE4'])
        else:
            self.set_tnad(tnad)

        self.runtype = None
        if runtype is None:
            self.set_runtype(['parallelFt', 'normalFt'])
        else:
            self.set_runtype(runtype)

        self.usebasis = None
        self.set_usebasis(usebasis)

        self.options = None
        self.set_options(options)

        self.cjcorr = None
        self.set_cjcorr(cjcorr)

        self.ncycle = 0
        if ncycle is None:
            self.set_ncycle(1)
        else:
            self.set_ncycle(ncycle)

    def set_tnad(self, tnad):
        if not isinstance(tnad, list):
            tnad = [tnad]
        self.tnad = tnad

    def set_runtype(self, runtype):
        if not isinstance(runtype, list):
            runtype = [runtype]
        self.runtype = runtype

    def set_usebasis(self, usebasis):
        self.usebasis = usebasis

    def set_options(self, options):
        self.options = options

    def set_ncycle(self, ncycle):
        self.ncycle = ncycle

    def set_cjcorr(self, cjcorr):
        self.cjcorr = cjcorr


class adffdeanalysisjob(metajob):
    """
    Class for ADF FDE analysis jobs.
    """

    def __init__(self, molecules, settings, adfsettings=None, basis=None, core=None,
                 fde=None, adffdesettings=None):
        """
        Constructor of adffdeanalysisjob.

        @param molecules: molecules for which to run the analysis
        @type molecules:  list of L{molecule} objects

        @param settings:
           settings for the adffdeanalysisjob
        @type settings: L{adffdeanalysissettings}

        @param adfsettings:
           settings for the ADF runs
        @type adfsettings: L{adfsettings}

        @param basis:
           basis set for the ADF runs
        @type basis: str

        @param core:
           frozen core for ADF runs
        @type core: str

        @param fde:
           settings for the fde runs
        @type fde: dictionary (see ADFFragments.py)

        """

        super().__init__()

        if not isinstance(molecules, list):
            raise PyAdfError('molecule should be an instance of list')
        else:
            self.molecules = molecules

        if settings is None:
            raise PyAdfError('no adffdeanalysissettings provided')
        else:
            self.settings = settings

        self.adfsettings = adfsettings

        self.adffdesettings = adffdesettings

        if basis is None:
            raise PyAdfError('No basis provided in adffdeanalysisjob')
        else:
            self.basis = basis

        self.core = core

        if fde is None:
            self.fde = {}
        else:
            self.fde = fde

    @staticmethod
    def norm(vec):
        from math import sqrt
        return sqrt(sum(x * x for x in vec))

    @staticmethod
    def calc_diff(res_ref, res_test, grid):
        """
        Calculate difference in dipole moment and integrals of the difference density between a reference and a test.

        @param  res_ref:
            The results of a reference calculation (should be a KS calculation)
        @type res_ref: adfsinglepointresults object

        @param res_test:
            The results of test calculation(s)
        @type res_test: either one of or list of adfsinglepointresults or adffragmentresults

        @param grid:
            Grid for integration (should be grid of reference)
        @type grid: grid

        @returns:
            integrated difference density, integrated absolute value of difference density,
            RMS of difference density
        @rtype: list of floats

        """
        from pyadf.Utils import au_in_Debye
        from math import sqrt

        if (res_ref is None) or (res_test is None):
            raise PyAdfError('Reference and Test results objects required in calc_den')
        if grid is None:
            raise PyAdfError('Grid required in calc_diff')

        if not isinstance(res_test, list):
            res_test = [res_test]

        dip_ref = res_ref.get_dipole_vector()
        dens_ref = res_ref.get_density(grid=grid)

        # test density and dipole moment
        dip_test = []
        dens_test = []
        for r in res_test:
            dip = r.get_dipole_vector()
            dip_test.append(dip)
            dens = r.get_density(grid=grid)
            dens_test.append(dens)
        dip_test = reduce(lambda x, y: x + y, dip_test)
        dens_test = reduce(lambda x, y: x + y, dens_test)

        # difference density and integrals
        dens_diff = dens_ref - dens_test
        numElectrons = dens_ref.integral()
        print('-' * 50)
        err_dens_diff = dens_diff.integral()
        abserr_dens_diff = dens_diff.integral(abs) / numElectrons
        rmserr_dens_diff = sqrt(dens_diff.integral(lambda x: x * x)) / numElectrons

        return [au_in_Debye * (dip_ref - dip_test), err_dens_diff, abserr_dens_diff, rmserr_dens_diff]

    def prepare_frags(self):
        """
        Prepare fragments (make single point runs for molecules).

        @note:
            we cannot use symmetry in the fragments if we are interested in the
            difference density -> otherwise they will be rotated

        @returns: The results of single point runs for a list of molecules
        @rtype: tuple of list of adfsinglepointresults and fragments
        """

        r_frags = []
        frags = []
        self.adfsettings.set_occupations(None)
        for mol in self.molecules:
            mol.set_symmetry('NOSYM')
            r = adfsinglepointjob(mol=mol, basis=self.basis, core=self.core,
                                  settings=self.adfsettings, options=['NOSYMFIT']).run()
            r_frags.append(r)
            if self.settings.usebasis == 'True':
                frags.append(fragment(r, mol, fdeoptions={"USEBASIS": "", "RELAX": "", "XC": " GGA PBE"}))
            else:
                frags.append(fragment(r, mol, fdeoptions={"XC": "GGA PBE"}))

        return frags, r_frags

    def metarun(self):
        """
        Run the analysis job

        @returns: The results of the analysis run
        @rtype: adffdeanalysisresults
        """

        # prepare fragments (list of results)
        (frags, r_frags) = self.prepare_frags()

        # make KS calculation (the reference)
        self.adfsettings.set_save_tapes([10, 21])

        jobKs = adffragmentsjob(frags, basis=self.basis, core=self.core,
                                settings=self.adfsettings).run()
        self.adfsettings.set_save_tapes([21])

        # run analysis of densities and return result
        grid_super = adfgrid(jobKs)
        calc_diffSoF = self.calc_diff(jobKs, r_frags, grid_super)
        calc_diffSoF[0] = self.norm(calc_diffSoF[0])
        # if FDE requested (not only sum-of-fragments) run FDE
        results = {}
        if self.settings.options == 'scaled':
            options = ['SCALEDKINFUNCTIONALS']
        elif self.settings.options == 'dependency':
            options = ['DEPENDENCY']
        else:
            options = None

        if self.settings.usebasis == 'True':
            print('-' * 50)
            print('USEBASIS: Basis functions of frozen fragments will be included in the non-frozen system')

        for t in self.settings.tnad:
            if self.settings.cjcorr == 'True':
                fde = {'TNAD': t, 'CJCORR': 0.1}
            else:
                fde = {'TNAD': t}
            if self.settings.runtype == ['parallelFt']:
                print('\n')
                print('Testing the functional ', t)
                print('\n')
                jobParallelFt = adffdejob(frags, basis=self.basis, core=self.core,
                                          settings=self.adfsettings, options=options, fde=fde,
                                          adffdesetts=self.adffdesettings).parallel_ft_run()
                calc_diffFde0 = self.calc_diff(jobKs, jobParallelFt, grid_super)
                results[t] = [calc_diffSoF, calc_diffFde0]
                for i in range(0, len(results[t]) - 1):
                    results[t][i + 1][0] = self.norm(results[t][i + 1][0])

            elif self.settings.runtype == ['normalFt']:
                print('\n')
                print('Testing the functional ', t)
                print('\n')

                fde['NORMALFT'] = ''

                results[t] = [calc_diffSoF]

                fde['RELAXCYCLES'] = self.settings.ncycle

                jobNormalFt = adffdejob(frags, basis=self.basis, core=self.core,
                                        settings=self.adfsettings, options=options, fde=fde,
                                        adffdesetts=self.adffdesettings).normal_ft_run()
                calc_diffFde = self.calc_diff(jobKs, jobNormalFt, grid_super)
                results[t].append(calc_diffFde)
                results[t][1][0] = self.norm(results[t][1][0])

            else:
                print('\n')
                print('Testing the functional ', t)
                print('\n')
                jobParallelFt = adffdejob(frags, basis=self.basis, core=self.core,
                                          settings=self.adfsettings, options=options, fde=fde,
                                          adffdesetts=self.adffdesettings).parallel_ft_run()
                calc_diffFde0 = self.calc_diff(jobKs, jobParallelFt, grid_super)
                calc_diffFde0[0] = self.norm(calc_diffFde0[0])

                fde['NORMALFT'] = ''
                results[t] = [calc_diffSoF, calc_diffFde0]

                for i in range(1, self.settings.ncycle + 1):
                    fde['RELAXCYCLES'] = i

                    jobNormalFt = adffdejob(frags, basis=self.basis, core=self.core,
                                            settings=self.adfsettings, options=options, fde=fde,
                                            adffdesetts=self.adffdesettings).normal_ft_run()
                    calc_diffFde = self.calc_diff(jobKs, jobNormalFt, grid_super)
                    calc_diffFde[0] = self.norm(calc_diffFde[0])
                    results[t].append(calc_diffFde)

        return results

    def print_results(self):

        results = self.run()

        print('\n')
        print('+' + 70 * '-' + '+')
        print('|' + 'FINAL RESULTS'.center(70) + '|')
        print('+' + 70 * '-' + '+')
        print(' Basis set: ' + self.basis)

        for kin in results:
            print(' NADKIN: ' + kin)
            string = ' ' * 20
            string += '|dmu|(D)'.center(14)
            string += 'dabs'.center(14)
            string += 'drms'.center(14)
            print(string)

            string = 'Sum of Fragments'.ljust(20)
            string += f'  {results[kin][0][0]:12.6f}'
            string += f'  {results[kin][0][2]:12.6f}'
            string += f'  {results[kin][0][3]:12.6f}'
            print(string)

            if self.settings.runtype == ['normalFt']:
                string = f'FDE({self.settings.ncycle:d})'
                string = string.ljust(20)
                string += f'  {results[kin][1][0]:12.6f}'
                string += f'  {results[kin][1][2]:12.6f}'
                string += f'  {results[kin][1][3]:12.6f}'
                print(string)

            if not self.settings.runtype == ['normalFt']:
                string = 'FDE(0)'.ljust(20)
                string += f'  {results[kin][1][0]:12.6f}'
                string += f'  {results[kin][1][2]:12.6f}'
                string += f'  {results[kin][1][3]:12.6f}'
                print(string)

            if self.settings.runtype == ['parallelFt', 'normalFt']:
                for i in range(1, self.settings.ncycle + 1):
                    string = f'FDE({i:d})'
                    string += f'  {results[kin][i + 1][0]:12.6f}'
                    string += f'  {results[kin][i + 1][2]:12.6f}'
                    string += f'  {results[kin][i + 1][3]:12.6f}'
                    print(string)


class datasetjob(metajob):
    """
    Runs L{adffdeanalysisjob}s for a data set and then calculates the average per
    nadkin for every molecule in the data set
    """

    def __init__(self, pathNames, settings, adfsettings=None, basis=None, core=None,
                 fde=None, adffdesettings=None):

        super().__init__()

        if pathNames is None:
            raise PyAdfError('pathName should be supplied')
        else:
            self.pathNames = pathNames

        if settings is None:
            raise PyAdfError('no adffdeanalysissettings provided')
        else:
            self.settings = settings

        self.adfsettings = adfsettings
        self.adffdesettings = adffdesettings

        if basis is None:
            raise PyAdfError('No basis provided in adffdeanalysisjob')
        else:
            self.basis = basis
            self.core = core

        if fde is None:
            self.fde = {}
        else:
            self.fde = fde

    @staticmethod
    def norm(vec=None):
        from math import sqrt
        return sqrt(sum(x * x for x in vec))

    @staticmethod
    def add_lists(list1, list2):
        newlist = []
        for i in range(0, len(list1)):
            tmplist = []
            for j in range(0, len(list1[i])):
                tmplist.append(list1[i][j] + list2[i][j])
            newlist.append(tmplist)
        return newlist

    @staticmethod
    def divide_by_number(list1, number):
        newlist = []
        for i in range(0, len(list1)):
            tmplist = []
            for j in range(0, len(list1[i])):
                tmplist.append(list1[i][j] / number)
            newlist.append(tmplist)
        return newlist

    def metarun(self):

        dataSetJobResults = []
        for p in self.pathNames:
            filelist = glob.glob(os.path.join(p, '*.xyz'))
            molecules = []
            for f in filelist:
                molecules.append(molecule(os.path.join(p, f), inputformat='xyz'))

            dataSetJob = adffdeanalysisjob(molecules=molecules, settings=self.settings,
                                           adfsettings=self.adfsettings, basis=self.basis, core=self.core,
                                           fde=self.fde, adffdesettings=self.adffdesettings).run()
            dataSetJobResults.append(dataSetJob)

        dictTmp = []
        for key, value1 in dataSetJobResults[0].items():
            value2 = dataSetJobResults[1][key]
            dictTmp.append((key, self.add_lists(value1, value2)))
        dictTmp = dict(dictTmp)

        tmpAvrg = []
        if len(dataSetJobResults) >= 3:
            for i in range(2, len(dataSetJobResults)):
                tmpAvrg = []
                for key, value1 in dictTmp.items():
                    value2 = dataSetJobResults[i][key]
                    tmpAvrg.append((key, self.add_lists(value1, value2)))
                tmpAvrg = dict(tmpAvrg)
                dictTmp.update(tmpAvrg)
            tmpAvrg = dict(tmpAvrg)
        else:
            tmpAvrg = dictTmp
        AvrgResults = []
        number = len(dataSetJobResults)
        for key, value in tmpAvrg.items():
            AvrgResults.append((key, self.divide_by_number(value, number)))
        AvrgResults = dict(AvrgResults)

        return AvrgResults, dataSetJobResults

    def print_results(self):

        (avrg, dataSetJobResults) = self.run()

        for results, p in zip(dataSetJobResults, self.pathNames):
            print('\n')
            print('Molecule Name: ', p)
            print('Basis Set: ', self.basis)
            print('+' + 70 * '-' + '+')
            print('|' + 'FINAL RESULTS'.center(70) + '|')
            print('+' + 70 * '-' + '+')

            for kin in results:
                print(' NADKIN: ' + kin)
                print(' ' * 20 + '|dmu|(D)'.center(14) + 'dabs'.center(14) + 'drms'.center(14))
                string = 'Sum of Fragments'.ljust(20)
                string += f'  {results[kin][0][0]:12.6f}'
                string += f'  {results[kin][0][2]:12.6f}'
                string += f'  {results[kin][0][3]:12.6f}'
                print(string)

                if self.settings.runtype == ['normalFt']:
                    string = f'FDE({self.settings.ncycle:d})'
                    string = string.ljust(20)
                    string += f'  {results[kin][1][0]:12.6f}'
                    string += f'  {results[kin][1][2]:12.6f}'
                    string += f'  {results[kin][1][3]:12.6f}'
                    print(string)

                if not self.settings.runtype == ['normalFt']:
                    string = 'FDE(0)'.ljust(20)
                    string += f'  {results[kin][1][0]:12.6f}'
                    string += f'  {results[kin][1][2]:12.6f}'
                    string += f'  {results[kin][1][3]:12.6f}'
                    print(string)

                if self.settings.runtype == ['parallelFt', 'normalFt']:
                    for i in range(1, self.settings.ncycle + 1):
                        string = f'FDE({i:d})'
                        string = string.ljust(20)
                        string += f'  {results[kin][i + 1][0]:12.6f}'
                        string += f'  {results[kin][i + 1][2]:12.6f}'
                        string += f'  {results[kin][i + 1][3]:12.6f}'
                        print(string)
        print('\n')
        print('\n')
        print('+' + 70 * '-' + '+')
        print('|' + 'AVERAGE OF FINAL RESULTS'.center(70) + '|')
        print('+' + 70 * '-' + '+')

        for kin in list(avrg.keys()):
            print(' NADKIN: ' + kin)
            print('|dmu|(D)'.rjust(19) + 'dabs'.rjust(13) + 'drms'.rjust(13))
            print('Sum of Fragments'.center(70))
            print(f'{avrg[kin][0][0]:12.6f}   {avrg[kin][0][2]:12.6f}   {avrg[kin][0][3]:12.6f}')
            if self.settings.runtype == ['normalFt']:
                print(f'FDE{self.settings.ncycle:d}'.center(70))
                print(f'{avrg[kin][1][0]:12.6f}   {avrg[kin][1][2]:12.6f}   {avrg[kin][1][3]:12.6f}')
            if not self.settings.runtype == ['normalFt']:
                print('FDE0'.center(70))
                print(f'{avrg[kin][1][0]:12.6f}   {avrg[kin][1][2]:12.6f}   {avrg[kin][1][3]:12.6f}')
            if self.settings.runtype == ['parallelFt', 'normalFt']:
                for i in range(1, self.settings.ncycle + 1):
                    print(f'FDE{i:d}'.center(70))
                    print(f'{avrg[kin][i + 1][0]:12.6f}   {avrg[kin][i + 1][2]:12.6f}   {avrg[kin][i + 1][3]:12.6f}')
