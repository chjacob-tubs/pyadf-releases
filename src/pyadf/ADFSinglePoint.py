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
 Job and results for ADF single point calculation.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    adfsinglepointjob, adfspjobdecorator
 @group Settings:
    adfsettings, adfscfsettings
 @group Results:
    adfsinglepointresults
"""
# pylint: disable=E1103

from ADFBase import adfjob, adfresults
from Plot.Grids import cubegrid, adfgrid
from ADF_Densf import densfjob

from Plot.Properties import PlotPropertyFactory
from Plot.GridFunctions import GridFunctionFactory, GridFunctionDensityWithDerivatives
from Plot.FileWriters import GridWriter

from Errors import PyAdfError
from Molecule import molecule
from Utils import au_in_eV
import os
import re
import shutil


class adfsettings(object):
    """
    Class that holds the settings for an ADF calculation.

    @group Initialization:
        __init__,
        set_functional, set_dispersion, set_integration,
        set_convergence, set_mixing, set_diis, set_lshift,
        set_unrestricted, set_occupations, set_cosmo, set_cosmosurf,
        set_lmo, set_basispath, set_printing, set_ncycles,
        set_dependency, set_ZORA, set_exactdensity,
        set_save_tapes, set_noncollinear
    @group Input Generation:
        get_settings_block
    @group Other Internals:
        __str__
    """

    def __init__(self, functional='LDA', accint=4.0, converge=1e-6, ncycles=100, cosmosurf='Delley',
                 dep=False, ZORA=False, SpinOrbit=False, mix=0.2, unrestricted=False, noncollinear=False,
                 occupations=None, cosmo=None, lmo=False, basispath=None, zlmfit=False,
                 printing=False):
        """
        Constructor for adfsettings.

        All arguments are optional, leaving out an argument will choose default settings.

        @param functional: exchange-correlation functional, see L{set_functional}
        @type  functional: str
        @param accint: integration accuracy, see L{set_integration}
        @type  accint: float
        @param converge: SCF convergence threshold, see L{set_convergence}
        @type  converge: float
        @param ncycles: maximum number of SCF cycles, see L{set_ncycles}
        @type  ncycles: int
        @param dep: dependency settings, see L{set_dependency}
        @type  dep: bool
        @param ZORA: ZORA relativistic option, see L{set_ZORA}
        @type  ZORA: bool
        @param SpinOrbit: ZORA SpinOrbit option, see L{set_ZORA}
        @type  SpinOrbit: bool
        @param mix: SCF mixing (damping) setting, see L{set_mixing}
        @type  mix: float
        @param unrestricted: unrestricted setting, see L{set_unrestricted}
        @type  unrestricted: bool
        @param noncollinear: noncollinear setting, see L{set_noncollinear}
        @type  noncollinear: bool
        @param occupations: orbital occupations, see L{set_occupations}
        @param cosmo: COSMO solvation setting, see L{set_cosmo}
        @param cosmosurf: COSMO solvation surface option, see L{set_cosmosurf}
        @param lmo: switch on calculation of localized orbitals
        @type  lmo: bool
        @param basispath: path to basis sets, see L{set_basispath}
        @type  basispath: str
        @param printing: extended output printing, see L{set_printing}
        @type  printing: bool
        """

        # first declare all instance variables here
        self.functional = None
        self.accint = None
        self.freeze_accmin = None
        self.acclist = None
        self.int_special = None
        self.becke = None
        self.converge = None
        self.mix = None
        self.diis = None
        self.adiis = None
        self.vshift = None
        self.unrestricted = None
        self.noncollinear = None
        self.occupations = None
        self.lmo = None
        self.cosmo = None
        self.cosmosurf = None
        self.basispath = None
        self.printing = None
        self.printcharge = None
        self.printfit = None
        self.printeig = {}
        self.createoutput = None
        self.lmo = None
        self.dependency = None
        self.dependency_bas = None
        self.dependency_fit = None
        self.ZORA = None
        self.SpinOrbit = None
        self.exactdens = None
        self.save_tapes = None
        self.tapelist = None
        self.dispersion = None
        self.zlmfit = zlmfit

        # and now initialize them using setter methods
        self.set_functional(functional)
        self.set_integration(accint)
        self.set_convergence(converge)
        self.set_mixing(mix)
        self.set_diis(None)
        self.set_adiis(False)
        self.set_unrestricted(unrestricted)
        self.set_noncollinear(noncollinear)
        self.set_occupations(occupations)
        self.set_cosmo(cosmo)
        self.set_cosmosurf(cosmosurf)
        self.set_basispath(basispath)
        self.set_lmo(lmo)
        self.set_printing(printing)
        self.set_createoutput(True)
        self.set_dependency(dep)
        self.set_ZORA(ZORA, SpinOrbit)
        self.set_exactdensity(False)
        self.set_save_tapes()
        self.set_lshift(None)
        self.ncycles = ncycles

    def __str__(self):
        """
        Returns a human-readable description of the settings.
        """
        s = "   Relativistic options: "
        if self.ZORA:
            if self.SpinOrbit:
                s += " ZORA SpinOrbit Relativistic \n\n"
            else:
                s += " ZORA Scalar Relativistic \n\n"
        else:
            s += " Non-Relativistic \n\n"
        s += "   XC functional   : " + self.functional + "\n"
        s += "   Integration     : accint %-6.1f \n" % self.accint
        for i in self.int_special:
            s += "                   : " + i + " %-6.1f \n" % self.int_special[i]
        s += "   SCF convergence : %-6.1e \n" % self.converge[0]
        s += "\n"

        if self.exactdens:
            s += "   Exact density will be used in SCF \n\n"

        if self.basispath is not None:
            s += "   Basis sets will be taken from " + self.basispath + "\n\n"

        if self.dependency:
            s += "   Dependent basis functions will be removed "
            s += "(bas=%.1e fit=%.1e) \n\n" % (self.dependency_bas, self.dependency_fit)

        if len(self.tapelist) > 0:
            for t in self.tapelist:
                s += "   TAPE" + str(t) + " will be saved\n"
        else:
            s += "   No TAPEs will be saved\n"

        return s

    def set_functional(self, functional):
        """
        Select exchange--correlation functional.

        @param functional: string specifying a functional, e.g., LDA, BP86, B3LYP, MO6-L
        @type functional: str
        """
        self.freeze_accmin = False

        if functional in ['HARTREEFOCK'] or functional.startswith('LDA'):
            self.functional = functional
        elif functional in ['SAOP', 'LB94']:
            self.functional = 'MODEL ' + functional
        elif functional in ['B3LYP', 'B3LYP*', 'B1LYP', 'KMLYP', 'O3LYP', 'X3LYP', 'BHandH',
                            'BHandHLYP', 'B1PW91', 'mPW1PW', 'mPW1K', 'PBE0', 'OPBE0']:
            self.functional = 'HYBRID ' + functional
        elif functional in ['oldPBE']:
            self.functional = 'LDA VWN\n   GGA PBE USEBURKEROUTINES'
        elif functional in ['M06-L', 'M06L', 'SSB-D', 'TPSS']:
            self.functional = 'MetaGGA ' + functional
            # m06l requires tighter integration accuracy, especially in geometry optimizations
            # so here integration is set to a lower limit of 8.0 all the way
            if functional == 'M06-L' or functional == 'M06L':
                self.set_integration(8.0, acclist=[8.0, 8.0])
                self.freeze_accmin = True
        elif functional in ['M06-HF', 'M06', 'M06-2X', 'TPSSH']:
            self.functional = 'MetaHybrid ' + functional
        elif functional in ['CAMYB3LYP']:
            self.functional = 'HYBRID ' + functional + "\n  xcfun\n  RANGESEP GAMMA=0.34"
        elif functional in ['CAMB3LYP']:
            self.functional = 'LibXC CAM-B3LYP\n'
        else:
            self.functional = 'GGA ' + functional

    def set_dispersion(self, dispersion):
        """
        Sets the dispersion parameter (Grimme dispersion correction).

        @param dispersion :
                Is either an empty string (DFT-D), or
                'Grimme3' (DFT-D3). In case no dispersion
                correction is required it should be set to None.
        @type dispersion : str or None
        """
        self.dispersion = dispersion

    def set_integration(self, accint, acclist=None, int_special=None, dishul=None, becke=None):
        """
        ADF numerical integration settings.

        For details, see INTEGRATION key in the ADF manual.

        @param accint: general integration accuracy parameter
        @type  accint: float
        @param acclist: FIXME ADD DOCUMENTATION
        @param dishul: DISHUL option, see ADF manual
        @type  dishul: None or float
        """
        if self.freeze_accmin:
            if accint < self.accint:
                return
            if acclist is not None:
                if acclist[0] < self.acclist[0] or acclist[1] < self.acclist[1]:
                    return

        self.accint = accint
        self.acclist = []
        if acclist is not None:
            self.acclist = acclist
        if int_special is None:
            self.int_special = {}
        else:
            self.int_special = int_special

        if dishul is not None:
            self.int_special['dishul'] = dishul

        self.becke = becke

    def set_convergence(self, converge=1.0e-6, convlist=None):
        """
        SCF convergence settings.

        @param converge: standard convergence threshold
        @type  converge: float
        @param convlist: FIXME ADD DOCUMENTATION

        """
        self.converge = [converge, converge]
        if convlist is not None:
            self.converge = convlist

    def set_mixing(self, mix):
        """
        Mixing/damping setting for SCF.

        For details, see mix option of the SCF key in the ADF manual.

        @param mix: the mixing value
        @type  mix: float
        """
        self.mix = mix

    def set_diis(self, diis):
        """
        DIIS settings for SCF.

        For details, see diis option of the SCF key in the ADF manual.

        @param diis: DIIS options, as key-value pairs
        @type  diis: None or dict
        """
        self.diis = diis

    def set_adiis(self, adiis):
        self.adiis = adiis

    def set_lshift(self, vshift):
        """
        Level shifting settings for SCF.

        For details, see lshift option of the SCF key in the ADF manual.

        @param vshift: level shifting parameter and additional options as a dictionary,
                       for example set_lshift(0.1) or set_lshift([0.1,{'Err':0.001}])
        @type vshift: float or list
        """
        self.vshift = vshift

    def set_unrestricted(self, unrestricted):
        """
        Unrestricted KS calculations.

        See UNRESTRICTED keyword in the ADF manual.

        @param unrestricted: switch on unrestricted calculation
        @type  unrestricted: bool
        """
        self.unrestricted = unrestricted

    def set_noncollinear(self, noncollinear):
        """
        Noncollinear KS calculations.

        See NONCOLLINEAR keyword in the ADF manual.

        @param noncollinear: switch on noncollinear calculation
        @type  noncollinear: bool
        """
        self.noncollinear = noncollinear

    def set_occupations(self, occupations):
        """
        Occupation number specification.

        See OCCUPATIONS block in the ADF manual.

        FIXME ADD DOCUMENTATION
        """
        self.occupations = occupations

    def set_cosmo(self, cosmo):
        """
        COSMO solvation options.

        See SOLVATION block in the ADF manual.

        @param cosmo: cosmo options,
                      e.g. set_cosmo(True) will use Water
                      e.g. set_cosmo('Hexane') will use Hexane
                      e.g. set_cosmo([12.4, 3.18]) (eps and rad of Pyridine)
        @type cosmo: boolean, str, or list
        """
        self.cosmo = cosmo
                
    def set_cosmosurf(self, cosmosurf):
        """
        COSMO surface type.
        By default Delley is used.

        See SOLVATION block in the ADF manual.

        @param cosmosurf: Wsurf, Asurf, Esurf, Klamt, or Delley
        @type cosmosurf: str
        """
        self.cosmosurf = cosmosurf

    def set_lmo(self, lmo):
        """
        Calculation of localized orbitals.

        @param lmo: switch on calculation of localized orbitals
        @type  lmo: bool
        """
        self.lmo = lmo

    def set_basispath(self, basispath):
        """
        Set the path to the basis sets.

        @param basispath:
           the path to the basis set
           (either absolute or as environment variable, for example '$MYBASIS')
        @type basispath: str

        """
        self.basispath = basispath

    def set_printing(self, printing, printcharge=False, printfit=False, printeig=None):
        """
        Extended output printing via the EPRINT keyword.

        note that at this point if self.printing flags is set to false, what will happen when setting up the
        input block (see get_printing_block) is that the ADF outputs are modified to supress certain parts
        of the default
        if set to false (which is the default), it will modify some of the defaults in ADF and
        e.g. suppress printing MO compositions etc.


        @param printing: switch on extended output printing (default: off)
        @type  printing: bool
        @param printcharge: switch on printing of mulliken population etc (default: off)
        @type  printing: bool
        @param printfit: switch on printing of fit coefficients (default: off)
        @type  printing: bool
        @param printeig: a dictionary of the form {"occ":[n_occ], "virt":[n_virt]} containing the
            number of occupied and virtual orbitals to print
        @type  printing: dict 
        """
        self.printing = printing
        self.printcharge = printcharge
        self.printfit = printfit

        self.printeig = {}
        if printeig is None:
            keys = []
        else:
            keys = printeig.keys()

        if 'occ' in keys:
            self.printeig['occ'] = printeig['occ']
        else:
            self.printeig['occ'] = -1

        if 'virt' in keys:
            self.printeig['virt'] = printeig['virt']
        else:
            self.printeig['virt'] = -1

    def set_createoutput(self, createoutput):
        """
        Create output printing.

        @param createoutput: switch on/off createoutput printing (default: on)
        @type  createoutput: bool
        """
        self.createoutput = createoutput

    def set_ncycles(self, ncycles):
        """
        Maximum number of SCF cycles.

        @param ncycles: set maximum number of SCF cycles.
        @type  ncycles: int
        """
        self.ncycles = ncycles

    def set_dependency(self, use_dep, bas=1e-3, fit=1e-8):
        """
        Remove linearly dependent basis functions.

        See DEPENDENCY key in the ADF manual for details.

        @param use_dep: switch on DEPENDENCY key
        @type use_dep: bool
        @param bas: threshold for removing basis functions
        @type bas: float
        @param fit: threshold for removing fit functions
        @type fit: float
        """
        self.dependency = use_dep
        self.dependency_bas = bas
        self.dependency_fit = fit

    def set_ZORA(self, ZORA, SpinOrbit=False):
        """
        Relativistic options.

        @param ZORA: switch on ZORA approximation
        @type  ZORA: bool
        @param SpinOrbit: switch on SpinOrbit coupling
        @type  SpinOrbit: bool
        """
        self.ZORA = ZORA
        if ZORA:
            self.SpinOrbit = SpinOrbit

    def set_exactdensity(self, ed):
        """
        Use exact (instead of fitted) density for xc potential.

        See EXACTDENSITY key in the ADF manual.

        @param ed: switch on EXACTDENSITY
        @type ed: bool
        """
        self.exactdens = ed

    def set_save_tapes(self, tapelist=(21,)):
        """
        Save TAPE files produced by ADF calculations.

        @param tapelist: numbers of TAPE files that should be saved.
        @type  tapelist: list of int
        """

        self.tapelist = list(tapelist)

        if len(tapelist) > 0:
            self.save_tapes = 'SAVE '
            for t in tapelist:
                self.save_tapes += 'TAPE%2i  ' % t
        else:
            self.save_tapes = ''

    def get_settings_block(self):
        """
        Generate the block in the ADF input related to the settings.
        """
        sblock = ""

        # ZORA relativistic options
        if self.ZORA:
            if self.SpinOrbit:
                sblock += " Relativistic SpinOrbit ZORA\n\n"
            else:
                sblock += " Relativistic Scalar ZORA\n\n"

        # xc functional
        sblock += " XC\n"
        sblock += "   " + self.functional + "\n"
        if self.dispersion is not None:
            sblock += "   Dispersion " + self.dispersion + "\n"
        sblock += " END\n\n"

        # integration
        if self.becke is None:
            sblock += " INTEGRATION \n"
            sblock += "  accint %4.1f " % self.accint
            for a in self.acclist:
                sblock += "%4.1f " % a
            sblock += " \n"
            for s in self.int_special:
                sblock += "  " + s + " %4.1f\n" % self.int_special[s]
            sblock += " END \n\n"
        else:
            sblock += " BeckeGrid \n"
            sblock += "  Quality %s \n" % self.becke
            sblock += " END \n"

        # convergence
        sblock += " SCF\n"
        sblock += "   iterations %4d \n" % self.ncycles
        sblock += "   converge %6.1e %6.1e \n" % (self.converge[0], self.converge[1])
        if self.mix != 0.2:
            sblock += "   mixing %4f \n" % self.mix
        if self.vshift is not None:
            if not isinstance(self.vshift, list):
                sblock += "   lshift %4f " % self.vshift
            else:
                for opt, val in sorted(self.vshift[1].items()):
                    sblock += "   lshift %4f %s=%4f" % (self.vshift[0], opt, val)
            sblock += "\n"
        if self.diis is not None:
            sblock += "   diis "
            for opt, val in sorted(self.diis.items()):
                if opt == "n":
                    sblock += "%s=%d " % (opt.lower(), val)
                elif opt == "cyc":
                    sblock += "%s=%d " % (opt.lower(), val)
                else:
                    sblock += "%s=%.6f " % (opt.lower(), val)
            sblock += "\n"
        if self.adiis:
            sblock += " ADIIS"
            sblock += "\n"
        sblock += " END\n\n"

        # unrestricted
        if self.unrestricted:
            sblock += " UNRESTRICTED\n\n"

        # noncollinear
        if self.noncollinear:
            sblock += " NONCOLLINEAR\n\n"

        # occupations
        if self.occupations is not None:
            if len(self.occupations[0]) > 1:
                occupations = self.occupations
            else:
                occupations = [self.occupations]
            occopts = []
            occs = []
            for occ in occupations:
                if re.search('KEEPORBITALS', occ.upper()):
                    occopts.append(occ)
                elif re.search('SMEARQ', occ.upper()):
                    occopts.append(occ)
                else:
                    occs.append(occ)
            if len(occopts) > 0:
                sblock += " Occupations "
                for occ in occopts:
                    sblock += " %s " % occ
                sblock += " \n\n"

            if len(occs) > 0:
                sblock += " IrrepOccupations \n"
                for occ in occs:
                    sblock += "  %s\n" % occ
                sblock += " END"
                sblock += "\n\n"

        # cosmo
        if self.cosmo is not None:
            if isinstance(self.cosmo, bool):
                # If the parameter is a boolean, an empty solvation block should be printed (or not).
                if self.cosmo:
                    sblock += " SOLVATION\n"
                    sblock += " surf %s\n" % self.cosmosurf
                    sblock += " END\n\n"
            elif isinstance(self.cosmo, str):
                # If the parameter is a string, it contains the name of the solvent
                sblock += " SOLVATION\n"
                if self.cosmo != 'True':
                    sblock += " solv name=%s\n" % self.cosmo
                    sblock += " surf %s\n" % self.cosmosurf
                sblock += " END\n\n"
            elif isinstance(self.cosmo, list):
                # If the parameter is a list, it should either contain only the name of the solvent,
                # or the parameters rad and eps.
                sblock += " SOLVATION\n"
                sblock += " solv "
                if len(self.cosmo) > 2:
                    raise PyAdfError(
                        "Cosmo input list should contain either the solvent name, or the eps and rad parameters!")
                if len(self.cosmo) == 1:
                    if not isinstance(self.cosmo[0], str):
                        raise PyAdfError("Cosmo input list should contain either the solvent name, "
                                         "or the eps and rad parameters!")
                elif len(self.cosmo) == 2:
                    for i, par in enumerate(self.cosmo):
                        try:
                            float(par)
                        except ValueError:
                            raise PyAdfError("Cosmo input list should contain either the solvent name, "
                                             "or the eps and rad parameters!")
                    sblock += " eps=%f rad=%f\n" % (self.cosmo[0], self.cosmo[1])
                    sblock += " surf %s\n" % self.cosmosurf
                sblock += " END\n\n"
   
        # exact density
        if self.exactdens:
            sblock += " EXACTDENSITY\n\n"

        # dependency
        if self.dependency:
            sblock += " DEPENDENCY bas=%.1e fit=%.1e \n\n" % \
                      (self.dependency_bas, self.dependency_fit)

        if not self.zlmfit:
            sblock += 'STOFIT\n\n'

        # save TAPEs
        sblock += " " + self.save_tapes + "\n\n"

        # localized MOs
        if self.lmo:
            sblock += " LocOrb Store\n End\n\n"

        return sblock


class adfscfsettings(object):
    """
    Class for all settings of an ADF single point run.

    This is a simple wrapper for adfsettings plus basis, core etc.
    """

    def __init__(self, basis, settings, core=None, pointcharges=None, options=None,
                 createproc=None, frozenfrags=None):
        """
        Initialize the settings for the SCF calculation.

        For a description of the parameters see
        L{adfsettings} and L{adfsinglepointjob}

        """

        if not isinstance(basis, str):
            raise PyAdfError("no basis specified in adfscfsettings")
        self.basis = basis

        if not isinstance(settings, adfsettings):
            raise PyAdfError("no adfsettings object provided in adfscfsettings")
        self.settings = settings

        self.core = core
        self.pointcharges = pointcharges
        self.options = options

        self.frozenfrags = frozenfrags

        self.create_job = createproc


def use_default_grid(func):
    """
    Decorator to ensure that get_* class method use the proper default grid.
    """

    def _wrapper(self, grid=None, spacing=0.5, *args, **kwargs):
        if grid is None:
            grid = self.get_default_grid(spacing)
        # pylint: disable=E1102
        return func(self, grid, *args, **kwargs)

    return _wrapper


class adfsinglepointresults(adfresults):
    """
    Class for results of an ADF single point calculation.

    No additional information is stored compared to L{adfresults},
    but there are a lot more methods to obtain certain results.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_molecule, get_atom_index,
        get_number_of_electrons, get_s2, get_nspin,
        get_orbital_energy, get_bond_energy,
        get_voronoi_charges, get_mulliken_charges,
        get_hirshfeld_charges, get_multipolederiv_charges
    @group Retrieval of results on a grid:
        get_density, get_nonfrozen_density, get_laplacian,
        get_orbital, get_orbital_density, get_locorb_density,
        get_sqrgradient, get_locorb_sqrgradient, get_potential,
        get_nonfrozen_potential, get_kinetic_potential,
        get_locorb_kinpot, get_locorb_xcpot,
        get_locorb_coulpot, get_locorb_coulpot_numint,
        get_final_potential
    @group Access to internal properties:
        is_fde_job
    @group Access to result files:
        export_grid
    """

    def __init__(self, j=None):
        """
        Constructor for adfsinglepointresults.
        """
        adfresults.__init__(self, j)
        self._grid = None
        self._default_grids = {}

    def get_molecule(self, InputOrder=True):
        """
        Return the molecular geometry after the ADF job.

        This can be changes with respect to the input geometry
        because (a) it was optimized in the calculation or
        because (b) the molecule was brought to a standard
        orientation.

        @param InputOrder: If True, the order of the atoms
            will be the same that was used in the input molecular
            geometry, otherwise the internal ADF order is used.
        @type  InputOrder: bool

        @returns: The molecular geometry.
        @rtype:   L{molecule}
        """

        nnuc = self.get_result_from_tape('Geometry', 'nnuc')
        ntyp = self.get_result_from_tape('Geometry', 'ntyp')

        nqptr = self.get_result_from_tape('Geometry', 'nqptr', always_array=True)
        inpatm = self.get_result_from_tape('Geometry', 'atom order index', always_array=True)
        qtch = self.get_result_from_tape('Geometry', 'atomtype total charge', always_array=True)

        charge = []

        # ROSA: The charges are not read here (sometimes?).
        # So I added an if-clause to the following loop.

        for ityp in range(ntyp):
            for iat in range(nqptr[ityp] - 1, nqptr[ityp + 1] - 1):
                if qtch is not None:
                    charge.append(int(qtch[ityp]))
                else:
                    charge.append(0)

        xyz = self.get_result_from_tape('Geometry', 'xyz')
        # pylint: disable-msg=E1103
        xyznuc = xyz.reshape(nnuc, 3)

        if InputOrder:
            charge_ordered = []
            xyznuc_ordered = []
            for iat in range(nnuc):
                charge_ordered.append(charge[inpatm[iat] - 1])
                xyznuc_ordered.append(xyznuc[inpatm[iat] - 1])
        else:
            charge_ordered = charge
            xyznuc_ordered = xyznuc

        m = molecule()
        m.add_atoms(charge_ordered, xyznuc_ordered, atomicunits=True)

        return m

    def get_atom_index(self, inputatoms):
        """
        Return a list of the internal ADF atom indices for the given atoms.

        @param inputatoms: A list of atoms, atom number given in input order
        @type  inputatoms: list of int

        @rtype: list in ints
        """

        inpatm = self.get_result_from_tape('Geometry', 'atom order index', 21)

        adforder = []
        for a in inputatoms:
            adforder.append(inpatm[a - 1])

        return adforder

    def get_number_of_electrons(self):
        """
        Return number of electrons.

        @rtype: int
        """
        return int(round(self.get_result_from_tape('General', 'electrons', tape=21)))

    def get_orbital_energy(self, orb, irrep='A'):
        """
        Return orbital energy of a specific orbital.

        @param orb: 'HOMO', 'LUMO' or symmetry label (e.g., '11a') of an orbital
        @type  orb: str

        @returns: the orbital energy, in eV
        @rtype: float
        """

        output = self.get_output()

        if orb == 'HOMO' or orb == 'LUMO':
            if orb == 'HOMO':
                exp = re.compile(r"\s*(HOMO\s:)\s+(\d+)\s+(\w+)\s+"
                                 r"(?P<orben>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)", re.VERBOSE)

            else:  # orb == 'LUMO'
                exp = re.compile(r"\s*(LUMO\s:)\s+(\d+)\s+(\w+)\s+"
                                 r"(?P<orben>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)", re.VERBOSE)

            for line in reversed(output):
                m = exp.match(line)
                if m:
                    return float(m.group('orben')) * au_in_eV

        else:
            start = re.compile('.*Orbital Energies, all Irreps')
            exp = re.compile(r"\s*(?P<irrep>\w+)\s+(?P<num>\d+)\s+(d+(\.\d*)?|\d*\.\d+)"
                             r"\s+(?P<orbenau>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)"
                             r"\s+(?P<orbeneV>[-+]?(\d+(\.\d*)?|\d*\.\d+))\s*", re.VERBOSE)

            for i, line in enumerate(output):
                m = start.match(line)
                if m:
                    startline = i

            for line in output[startline:]:
                m = exp.match(line)
                if m:
                    if m.group('irrep') == irrep and int(m.group('num')) == orb:
                        return float(m.group('orbeneV'))

    def get_dipole_vector(self):
        """
        Return the dipole moment vector.

        @returns: the dipole moment vector, in atomic units
        @rtype: float[3]
        """
        return self.get_result_from_tape('Properties', 'Dipole')

    def get_bond_energy(self):
        """
        Return the bond energy.

        @returns: the bond energy in atomic units
        @rtype: float
        """
        return self.get_result_from_tape('Energy', 'Bond Energy')

    def get_total_energy(self):
        """
        Return the total energy (requires TOTALENERGY option).

        @returns: the total energy in atomic units
        @rtype: float
        """
        return self.get_result_from_tape('Total Energy', 'Total energy')

    def get_voronoi_charges(self, vdd=False):
        """
        Returns the Voronoi charge at the end of the calculation

        @param vdd: whether to return the VDD instead of the Voronoi charge
        @type  vdd: bool

        @returns: the Voronoi charges at the end of the calculation or the VDD charges
        """
        if vdd:
            initial = self.get_result_from_tape('Properties', 'AtomCharge_initial Voronoi')
            scf = self.get_result_from_tape('Properties', 'AtomCharge_SCF Voronoi')
            vdd_charges = scf - initial
            return vdd_charges
        else:
            return self.get_result_from_tape('Properties', 'AtomCharge_SCF Voronoi')

    def get_mulliken_charges(self):
        """
        Returns the Mulliken charges

        @returns: the Mulliken charges
        """
        return self.get_result_from_tape('Properties', 'AtomCharge Mulliken')

    def get_hirshfeld_charges(self):
        """
        Returns the Hirshfeld charges

        @returns: the Hirshfeld charges
        """
        return self.get_result_from_tape('Properties', 'FragmentCharge Hirshfeld')

    def get_multipolederiv_charges(self, level='MDC-q'):
        """
        Returns the multipole derived charges

        @returns: the multipole derived charges
        """
        if level == 'MDC-q':
            return self.get_result_from_tape('Properties', 'MDC-q charges')
        elif level == 'MDC-m':
            return self.get_result_from_tape('Properties', 'MDC-m charges')
        elif level == 'MDC-d':
            return self.get_result_from_tape('Properties', 'MDC-d charges')

    def get_s2(self):
        """
        Return the S**2 expectation value.

        @rtype: float
        """

        output = self.get_output()

        exp = re.compile(r".*Total\sS2\s\(S\ssquared\)\s*"
                         r"[-+]?(\d+(\.\d*)?|\d*\.\d+)\s*"
                         r"(?P<s2>[-+]?(\d+(\.\d*)?|\d*\.\d+))", re.VERBOSE)

        for line in reversed(output):
            m = exp.match(line)
            if m:
                return float(m.group('s2'))

        raise PyAdfError('Error retrieving S2')

    @property
    def nspin(self):
        """
        Return the number of different spins in the calculation.

        This is 1 for restricted or closed-chell calculations
        and 2 for unrestricted calculations.
        """
        return self.get_result_from_tape('General', 'nspin')

    def is_fde_job(self):
        """
        Returns whether the original job was an FDE job.

        @rtype: bool
        """
        from ADFFragments import adffragmentsjob

        if isinstance(self.job, adffragmentsjob):
            return self.job.is_fde_job()
        else:
            return False

    @property
    def grid(self):
        """
        The grid used in the calculation.

        @type: L{adfgrid}
        """
        if self._grid is None:
            self._grid = adfgrid(self)
        return self._grid

    def get_default_grid(self, spacing=0.5):
        if spacing not in self._default_grids:
            self._default_grids[spacing] = cubegrid(self.get_molecule(), spacing)
        return self._default_grids[spacing]

    @use_default_grid
    def get_nonfrozen_density(self, grid=None, fit=False, orbs=None, order=None):
        """
        Return the electron density of the nonfrozen fragments.

        @rtype: L{GridFunctionDensity}
        """
        return self.get_density(grid=grid, fit=fit, orbs=orbs, order=order)

    @use_default_grid
    def get_density(self, grid=None, fit=False, orbs=None, order=None):
        """
        Returns the electron density.

        For details on the processing of the electron density,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the fit density is returned, otherwise
                    the exact density.
        @type  fit: bool
        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict
        @param order: order of derivatives of the density to calculate (1 and 2 possible)
        @type order: int

        @rtype: L{GridFunctionDensity}
        """
        if (orbs is not None) and ('Loc' not in orbs.keys()):
            if (order is not None) and (order > 1):
                raise PyAdfError("Derivatives not implemented for orbital densities.")
            return self.get_orbital_density(grid, orbs)
        elif (order is not None) and (order >= 1):
            # the density itself
            gfs = [self.get_density(grid, fit=fit, orbs=orbs, order=None)]

            # density gradient
            if order >= 1:
                dg = self.get_densgradient(grid, fit=fit, orbs=orbs)
                gfs.append(dg)

            if order >= 2:
                dh = self.get_density_hessian(grid, fit=fit, orbs=orbs)
                gfs.append(dh)

            return GridFunctionDensityWithDerivatives(gfs)

        else:
            prop = PlotPropertyFactory.newDensity('dens', fit, orbs)

            dres = densfjob(self, prop, grid=grid).run()
            return dres.get_gridfunction()

    @use_default_grid
    def get_orbital(self, grid=None, irrep=None, num=None):
        """
        Get a specific orbital on a grid.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param irrep:
            The symmetry label of the irrep of the requested orbital.
            To get localized orbitals, use "Loc" as irrep.
        @param num: The number of the requested orbital

        @rtype: L{GridFunctionDensity}
        """
        prop = PlotPropertyFactory.newOrbital(irrep, num)

        res = densfjob(self, prop, grid=grid).run()
        return res.get_gridfunction()

    @use_default_grid
    def get_orbital_laplacian(self, grid=None, irrep=None, num=None):
        """
        Get the Laplacian of a specific orbital on a grid.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param irrep:
            The symmetry label of the irrep of the requested orbital.
        @param num: The number of the requested orbital
        """
        prop = PlotPropertyFactory.newOrbital(irrep, num, lapl=True)

        res = densfjob(self, prop, grid=grid).run()
        return res.get_gridfunction()

    @use_default_grid
    def get_orbital_density(self, grid=None, orbs=None):
        """
        Get the density of a given set of orbitals.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include.
        @type  orbs: dict

        @rtype: L{GridFunctionDensity}
        """
        if orbs.keys() == ['A']:
            prop = PlotPropertyFactory.newDensity('dens', orbs=orbs)
            res = densfjob(self, prop, grid=grid).run()
            orbdens = res.get_gridfunction()
        else:
            orbdens = []
            for sym, nums in sorted(orbs.items()):
                if not isinstance(nums, list):
                    nums = [nums]
                for i in nums:
                    orbdens.append(self.get_orbital(grid=grid, irrep=sym, num=i))

            # square the orbitals and multiply by 2.0 (occupation number)
            orbdens = [(d**2) * 2.0 for d in orbdens]
            orbdens = reduce(lambda x, y: x + y, orbdens)

        return orbdens

    @use_default_grid
    def get_locorb_density(self, grid=None, orbs=None):
        """
        @deprecated: Use get_density(grid, orbs={'Loc': orbs}) instead
        """
        prop = PlotPropertyFactory.newDensity('dens', orbs={'Loc': orbs})

        res = densfjob(self, prop, grid=grid).run()
        return res.get_gridfunction()

    @use_default_grid
    def get_sqrgradient(self, grid=None, fit=False, orbs=None):
        """
        Returns the squared gradient of the electron density.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the gradient of the fit density is returned,
            otherwise of the exact density.
        @type  fit: bool
        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict

        @rtype: L{GridFunction1D}
        """
        prop = PlotPropertyFactory.newDensity('sqrgrad', fit=fit, orbs=orbs)

        res = densfjob(self, prop, grid=grid).run()
        return res.get_gridfunction()

    @use_default_grid
    def get_densgradient(self, grid=None, fit=False, orbs=None):
        """
        Returns the gradient of the electron density.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the gradient of the fit density is returned,
                    otherwise the exact density is used.
        @type  fit: bool
        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict

        @rtype: L{GridFunction2D}
        """
        prop = PlotPropertyFactory.newDensity('grad', fit=fit, orbs=orbs)

        res = densfjob(self, prop, grid=grid).run()
        return res.get_gridfunction()

    @use_default_grid
    def get_laplacian(self, grid=None, fit=False, orbs=None):
        """
        Returns the Laplacian of the electron density.

        For details on the processing of the Laplacian,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the Laplacian of the fit density is returned,
                    otherwise the exact density is used.
        @type  fit: bool
        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict

        @rtype: L{GridFunction1D}
        """
        prop = PlotPropertyFactory.newDensity('lapl', fit=fit, orbs=orbs)

        res = densfjob(self, prop, grid=grid).run()
        return res.get_gridfunction()

    @use_default_grid
    def get_locorb_laplacian(self, grid=None, orbs=None):
        """
        @deprecated: Use get_laplacian(grid, fit=False, orbs={'Loc': orbs}) instead
        """
        return self.get_laplacian(grid, fit=False, orbs={'Loc': orbs})

    @use_default_grid
    def get_orbdens_laplacian(self, grid=None, orbs=None):
        """
        @deprecated: Use get_laplacian(grid, fit=False, orbs=orbs) instead
        """
        return self.get_laplacian(grid, fit=False, orbs=orbs)

    @use_default_grid
    def get_density_hessian(self, grid=None, fit=False, orbs=None):
        """
        Returns the Hessian of the electron density.

        For details on the processing of the Laplacian,
        e.g., for plotting, see L{Plot.Grids}.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the Laplacian of the fit density is returned,
                    otherwise the exact density is used.
        @type  fit: bool
        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict

        @rtype: L{GridFunction2D}
        """
        prop = PlotPropertyFactory.newDensity('hess', fit=fit, orbs=orbs)

        res = densfjob(self, prop, grid=grid).run()
        return res.get_gridfunction()

    @use_default_grid
    def get_nonfrozen_potential(self, grid=None, pot='total'):
        """
        Return the potential of the nonfrozen fragments.

        @rtype: L{GridFunctionPotential}
        """
        return self.get_potential(grid=grid, pot=pot)

    @use_default_grid
    def get_potential(self, grid=None, pot='total', orbs=None):
        """
        Returns the total potential or one of its components.

        The total potential, nuclear potential, electronic Coulomb potential,
        and exchange-correlation potential are available.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param pot: Which potential to calculate. One of: total, nuc, coul, xc
        @type  pot: str
        @param orbs:
            Used for calculating the potential from the density of selected orbitals.
            A dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals.
        @type orbs: dict

        @rtype: L{GridFunctionPotential}
        """
        prop = PlotPropertyFactory.newPotential(pot.lower(), orbs=orbs)

        res = densfjob(self, prop, grid=grid).run()
        return res.get_gridfunction()

    @use_default_grid
    def get_kinetic_potential(self, grid=None, func='THOMASFERMI', orbs=None):
        """
        Returns the kinetic potential calculated using an approximate kinetic-energy functional.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param func: The kinetic-energy functional to use.
        @type  func: str
        @param orbs:
            Used for calculating the potential from the density of selected orbitals.
            A dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals.
        @type orbs: dict

        @rtype: L{GridFunctionPotential}
        """
        prop = PlotPropertyFactory.newPotential('KINPOT', func=func, orbs=orbs)

        res = densfjob(self, prop, grid=grid).run()
        return res.get_gridfunction()

    @use_default_grid
    def get_locorb_kinpot(self, grid=None, orbs=None, func='THOMASFERMI'):
        """
        @deprecated: Use get_kinetic_potential(grid, func=func, orbs={'Loc': orbs}) instead
        """
        return self.get_kinetic_potential(grid, func=func, orbs={'Loc': orbs})

    @use_default_grid
    def get_orbdens_kinpot(self, grid=None, orbs=None, func='THOMASFERMI'):
        """
        @deprecated: Use get_kinetic_potential(grid, func=func, orbs=orbs) instead
        """
        return self.get_kinetic_potential(grid, func=func, orbs=orbs)

    @use_default_grid
    def get_locorb_xcpot(self, grid=None, orbs=None):
        """
        @deprecated: Use get_potential(grid, pot='xc', orbs={'Loc':orbs}) instead
        """
        return self.get_potential(grid, pot='xc', orbs={'Loc': orbs})

    @use_default_grid
    def get_locorb_coulpot(self, grid=None, orbs=None):
        """
        @deprecated: Use get_potential(grid, pot='coul', orbs={'Loc':orbs}) instead
        """
        return self.get_potential(grid, pot='coul', orbs={'Loc': orbs})

    @use_default_grid
    def get_locorb_coulpot_numint(self, grid=None, orbs=None):
        """
        Get the xc potential for the density of localized orbitals.

        This routine uses numerical integration and is obsolete.
        Use L{get_locorb_coulpot} instead.

        @param grid: The grid to use. For details, see L{Plot.Grids}.
        @type  grid: subclass of L{grid}
        @param orbs: numbers of the localized orbitals to include
        @type  orbs: list

        @rtype: L{GridFunctionPotential}
        """

        import numpy
        from Utils import Bohr_in_Angstrom

        # get density on ADF grid
        dens = self.get_density(grid=self.grid, orbs={'Loc': orbs})

        # get ADF grid weights and coordinates and prepare arrays

        weights = numpy.zeros((self.grid.npoints,))
        coords = numpy.zeros((self.grid.npoints, 3))

        for i, (w, c) in enumerate(zip(self.grid.weightiter(), self.grid.coorditer())):
            weights[i] = w
            coords[i, :] = c / Bohr_in_Angstrom

        densval = dens.get_values().flat

        print "Densint: ", (weights * densval).sum()

        # now calculate Coulomb potential on requested grid
        coulpot = numpy.zeros(grid.npoints)
        p = numpy.empty_like(coords)

        print "Number of points to calculate: ", grid.npoints

        for i, point in enumerate(grid.coorditer()):
            if i % 500 == 0:
                print "Calculating potential for point %i of %i " % (i, grid.npoints)

            p[:, 0] = point[0] / Bohr_in_Angstrom
            p[:, 1] = point[1] / Bohr_in_Angstrom
            p[:, 2] = point[2] / Bohr_in_Angstrom
            dist = numpy.sqrt(((coords - p)**2).sum(axis=1))
            dist = numpy.where(dist < 1e-3, 1e50 * numpy.ones_like(dist), dist)
            dist = 1.0 / dist

            coulpot[i] = (weights * dist * densval).sum()

        import hashlib
        m = hashlib.md5()
        m.update("Numerically calculated Coulomb potential from :\n")
        m.update(self.get_checksum())
        m.update("using localized orbitals:")
        m.update(repr(orbs))
        m.update("on grid :\n")
        m.update(grid.get_grid_block(True))

        return GridFunctionFactory.newGridFunction(grid, coulpot, m.digest(), 'potential')

    def get_final_potential(self):
        """
        Get the final total potential from the jobs's TAPE10 file.

        This will use the potential from TAPE10, and does not recalculate
        the potential. This is mainly intended to be used wir L{adfpotentialjob}s.

        @rtype: L{GridFunctionPotential}
        """
        import hashlib
        m = hashlib.md5()
        m.update("Total potential from ADF job :\n")
        m.update(self.get_checksum())

        values = self.get_result_from_tape('Total Potential', 'vtot', tape=10)
        values = values[:self.grid.npoints]

        return GridFunctionFactory.newGridFunction(self.grid, values, m.digest(), 'density')

    def export_grid(self, filename):
        """
        Export the integration grid to a text file (for use with Dalton/Dirac)

        The integration grid and weights will be written to a text file with
        the following format:
        First line: number of points
        Following lines: x,y,z coordinates of the grid point, w in this point
        Last line: -42

        @param filename: name of the file where the potential will be written
        @type filename:  str
        """
        GridWriter.write_xyzw(self.grid, filename, bohr=True, add_comment=False, endmarker=True)


class adfsinglepointjob(adfjob):
    """
    A job class for ADF single point calculations.

    See the documentation of L{__init__} and L{adfsettings} for details
    on the available options.

    Corresponding results class: L{adfsinglepointresults}

    @group Initialization:
        set_restart
    @group Input Generation:
        needs_basis_block, get_atomtypes_without_fragfile,
        get_molecule, get_printing_block, get_atoms_block, get_units_block,
        get_charge_block, get_symmetry_block, get_fragments_block,
        get_basis_block, get_geometry_block, get_efield_block,
        get_restart_block, get_options_block, get_other_blocks
    @group Other Internals:
        print_molecule, print_settings, print_extras
    """

    def __init__(self, mol, basis, settings=None, core=None, pointcharges=None, fitbas=None, options=None):
        """
        Constructor for ADF single point jobs.

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

        @param core:
            A string specifying which frozen cores to use (C{None}, C{Small}, C{Medium}, or C{Large}).
            Alternatively, a dictionary can be given to specify explicitly which core to
            use for each atom type (e.g. C{core={O:'1s', H:'None'}}).
            Such a dictionary can contain an entry 'default' giving the frozen core to
            use with all other atoms (possible values are C{None}, C{Small}, C{Medium}, or C{Large}).
        @type core: None or str or dict

        @param pointcharges:
            Coordinates (x, y, z in Angstrom) and charges of point charges.
            If no charges are given, zero charges are used.
        @type pointcharges: float[3][n] or float[4][n]

        @param options:
            Additional options. These will each be included directly in the ADF input.
        @type options: list of str
        """
        import copy

        adfjob.__init__(self)

        self.mol = mol
        self.basis = basis
        self.fitbas = fitbas
        self.core = core
        if self.core == "None":
            self.core = None
        if self.mol and (self.basis is None):
            raise PyAdfError("Missing basis set in ADF single point job")

        if settings is None:
            self.settings = adfsettings()
        else:
            self.settings = settings

        self.pc = copy.deepcopy(pointcharges)
        if self.pc is not None:
            for pc in self.pc:
                if len(pc) == 3:
                    pc.append(0.0)

        self.restart = None
        self.restartfile = None
        self.restartoptions = None

        if options is None:
            self._options = []
        else:
            self._options = options

        if ('NOSYM' in self._options) and self.mol:
            self.mol.set_symmetry('NOSYM')

    def create_results_instance(self):
        return adfsinglepointresults(self)

    def needs_basis_block(self):
        return len(self.get_atomtypes_without_fragfile()) > 0

    def get_atomtypes_without_fragfile(self):
        return self.mol.get_atom_symbols(prefix_ghosts=True)

    def set_restart(self, restart=None, filename=None, options=None):
        """

        restart: results object
        """
        self.restart = restart
        self.restartfile = filename
        if options is None:
            self.restartoptions = []
        else:
            self.restartoptions = options

    def get_molecule(self):
        return self.mol

    def get_printing_block(self):
        block = ""
        if not self.settings.printing:
            if self.settings.printcharge:
                block += " EPRINT \n"
                block += "   SFO NOEIG NOOVL NOORBPOP \n"
                block += "   SCF ATOMPOP\n"
                block += "   ATOMPOP GROSS\n"
                block += "   BASPOP NONE\n"
                if self.settings.printfit:
                    block += "   FIT Coef charge comb \n"
                block += " END\n\n"
                block += " NOPRINT BAS FUNCTIONS QMPOT EPAULI\n\n"
            else:
                block += " EPRINT \n"
                block += "   SFO NOEIG NOOVL NOORBPOP \n"
                block += "   SCF NOPOP \n"
                if self.settings.printfit:
                    block += "   FIT Coef charge comb \n"
                block += " END\n\n"
                block += " NOPRINT BAS FUNCTIONS \n\n"
        else:

            block += " EPRINT \n"
            if self.settings.printeig['occ'] > 0:
                if self.settings.printeig['virt'] > 0:
                    block += "   Eigval " + str(self.settings.printeig['occ']) + " " + str(
                        self.settings.printeig['virt']) + "\n"
                else:
                    block += "   Eigval " + str(self.settings.printeig['occ']) + "\n"
            if self.settings.printfit:
                block += "   FIT Coef charge comb \n"
            block += " END\n\n"
        return block

    def get_atoms_block(self):
        block = " ATOMS\n"
        block += self.mol.print_coordinates(index=False)
        block += " END\n\n"
        return block

    @staticmethod
    def get_units_block():
        return " UNITS\n   Length Angstrom\n   Angle Degree\n END\n\n"

    def get_charge_block(self):
        block = ""
        if isinstance(self.get_molecule().get_charge(), int):
            block += " CHARGE %i" % self.get_molecule().get_charge()
        else:
            block += " CHARGE %f8.4" % self.get_molecule().get_charge()
        if self.get_molecule().get_spin() > 0:
            block += " %2i" % self.get_molecule().get_spin()
        block += " \n\n"
        return block

    def get_symmetry_block(self):
        tol = '1e-2'
        for o in self._options:
            o = str(o)
            if o.startswith('TOL'):
                tol = o.strip('TOL').strip()
        block = ""
        if 'NOSYM' in self._options:
            #    block += " SYMMETRY NOSYM\n"
            #    tol added again since necessary for some FDE calculations
            #    might cause other problems (see svn log r30673)
            block += " SYMMETRY NOSYM tol=" + tol + "\n\n"
        elif self.get_molecule().symmetry is None:
            block += " SYMMETRY tol=" + tol + "\n\n"
        else:
            block += " SYMMETRY " + self.get_molecule().symmetry + " tol=" + tol + "\n\n"
        return block

    def get_fragments_block(self):
        return ""

    def get_basis_block(self):
        """
        Assemble the BASIS block of the ADF input.

        The following parts are used:
           - Type: the basis set string given, or the default entry of a basis dictionary
           - Core: the frozen core identifier, or the default entry of a core dictionary
           - Path: the BASISPATH specified via L{adfsettings}
           - entries of individual atom types if basis and/or core ditionary is given
        """
        block = " BASIS\n"

        if isinstance(self.basis, dict):
            block += "  Type " + self.basis['default'] + "\n"
        else:
            block += "  Type " + self.basis + "\n"

        if self.fitbas is not None:
            block += "  FitType " + self.fitbas + "\n"

        if self.core is None:
            block += "  Core None \n"
        elif isinstance(self.core, dict):
            block += "  Core " + self.core.get('default', 'None') + "\n"
        else:
            block += "  Core " + self.core + "\n"

        if self.settings.basispath is not None:
            block += "  Path " + self.settings.basispath + "\n"
        if not self.settings.createoutput:
            block += "  CreateOutput None \n"

        if isinstance(self.basis, dict) or isinstance(self.core, dict):

            if isinstance(self.basis, dict):
                atoms = set(self.basis.keys())
                basisdict = self.basis
            else:
                atoms = set()
                basisdict = {'default': self.basis}

            if self.settings.ZORA and not basisdict['default'].startswith('ZORA'):
                basisdict['default'] = "ZORA/" + basisdict["default"]

            if isinstance(self.core, dict):
                atoms = atoms.union(self.core.keys())
                coredict = self.core
            else:
                coredict = {}

            atoms = atoms.difference(['default'])

            for at in atoms:
                basisfile = "$ADFRESOURCES/" + basisdict.get(at, basisdict['default']) + '/' + at
                if at in coredict:
                    if (coredict[at] is not None) and (coredict[at].lower() != "none"):
                        basisfile += '.' + coredict[at]

                block += "  " + at + " " + basisfile + "\n"

        block += " END\n\n"

        return block

    def get_geometry_block(self):
        block = " GEOMETRY \n"
        block += "   sp \n"
        block += " END\n\n"
        return block

    def get_efield_block(self):
        block = ""
        if self.pc is not None:
            block += " PointCharges \n"
            for i in self.pc:
                block += " %14.5f %14.5f %14.5f %14.5f\n" % tuple(i)
            block += " END\n\n"
        return block

    def get_restart_block(self):
        block = ""
        if self.restart is not None or self.restartfile is not None:
            block += " RESTART \n"
            block += "  File t21.restart \n"
            if len(self.restartoptions) > 0:
                for opt in self.restartoptions:
                    block += "  %s\n" % opt
            block += " END\n\n"
        return block

    def get_options_block(self):
        block = ""
        for opt in self._options:
            if not ((opt == 'NOSYM') or opt.startswith('TOL')):
                block += " " + opt + "\n\n"
        return block

    def get_other_blocks(self):
        return ""

    def get_input(self):
        adfinput = "Title Input generated by PyADF\n\n"
        adfinput += self.get_atoms_block()
        adfinput += self.get_units_block()
        adfinput += self.get_charge_block()
        adfinput += self.get_symmetry_block()
        adfinput += self.get_fragments_block()
        if self.needs_basis_block():
            adfinput += self.get_basis_block()
        adfinput += self.settings.get_settings_block()
        adfinput += self.get_printing_block()
        adfinput += self.get_geometry_block()
        adfinput += self.get_efield_block()

        adfinput += self.get_restart_block()
        adfinput += self.get_options_block()
        adfinput += self.get_other_blocks()

        return adfinput

    def print_jobtype(self):
        return "ADF single point job"

    def before_run(self):
        adfjob.before_run(self)

        if self.restart is not None:
            self.restart.copy_tape(tape=21, name="t21.restart")
        if self.restartfile is not None:
            shutil.copyfile(self.restartfile, "t21.restart")

    def after_run(self):
        adfjob.after_run(self)
        if self.restart is not None or self.restartfile is not None:
            os.remove('t21.restart')

    def print_molecule(self):

        print "   Molecule"
        print "   ========"
        print
        print self.get_molecule()
        print

    def print_settings(self):

        print "   Settings"
        print "   ========"
        print
        print self.settings
        print

    def print_extras(self):

        if self._options:
            print "   Options"
            print "   ======="
            print
            for opt in self._options:
                print "   " + opt
            print

        if self.restart is not None:
            print " Using restart file " + self.restart.get_tape_filename(21)

    def print_jobinfo(self):
        print " " + 50 * "-"
        print " Running " + self.print_jobtype()
        print

        self.print_molecule()

        self.print_settings()

        self.print_extras()


class adfspjobdecorator(adfsinglepointjob):
    """
    Abstract decorator base class for decorators around L{adfsinglepointjob}s.

    Decorators extend a given job by adding certain blocks to the input.
    Subclasses have to implement:
        - C{get_other_blocks} for extending the input
        - C{print_jobtype} for printing a job type
        - C{print_extras} for extending PyADF the output
    They can also implement:
        - C{before_run} and C{after_run}, e.g., for copying
          files before and after running.
        - C{create_results_instance} to create a results object of
          the correct type
    """

    def __init__(self, wrappedjob):
        adfsinglepointjob.__init__(self, wrappedjob.mol, wrappedjob.basis,
                                   wrappedjob.settings, wrappedjob.core)
        self._wrappedjob = wrappedjob

    def get_input(self):
        adfinput = self._wrappedjob.get_input()
        adfinput += self.get_other_blocks()
        return adfinput

    def get_other_blocks(self):
        block = ""
        return block

    def print_jobtype(self):
        return self._wrappedjob.print_jobtype()

    def before_run(self):
        self._wrappedjob.before_run()

    def after_run(self):
        self._wrappedjob.after_run()

    def print_jobinfo(self):
        self._wrappedjob.print_jobinfo()

    def print_extras(self):
        pass

    def create_results_instance(self):
        res = self._wrappedjob.create_results_instance()
        res.job = self
        return res

    # refer everything that is not defined here to the wrapped job
    # (__gettattr__ is only called when the normal lookup mechanisms failed)
    def __getattr__(self, name):
        return self._wrappedjob.__getattribute__(name)
