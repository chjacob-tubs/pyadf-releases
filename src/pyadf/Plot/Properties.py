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
Defines classes for functions defined on grids.
"""

from ..Errors import PyAdfError


class PlotPropertyFactory:
    """
    Factory for PlotProperty.
    """

    @classmethod
    def newDensity(cls, dens_type='dens', fit=False, orbs=None):
        """
        Create a new property for a density-related quantity.

        @param dens_type: Density-related quantity to create
            Possible values are:
            dens:    density itself
            sqrgrad: squared gradient of the density
            grad:    density gradient (x, y, z)
            lapl:    density Laplacian
            hess:    density Hessian (xx, xy, xz, yy, yz, zz)
        @type dens_type: str

        @param fit: Whether to use the fit density or the exact density
        @type fit: bool

        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict or None
        """
        options = {'fit': fit}
        if orbs is not None:
            options['orbs'] = orbs

        prop = PlotProperty("density", prop_type=dens_type, options=options)

        return prop

    @classmethod
    def newPotential(cls, pot_type='total', func=None, orbs=None):
        """
        Create a new property for a potential.

        @param pot_type: Potential to create
            Possible values are:
            total:   the total potential
            nuc:     the nuclear potential
            coul:    the Coulomb potential
            xc:      the xc potential
            kinpot:  kinetic potential, func needs to be specified
            recon:   the reconstructed potential
            embpot:  the embedding potential
            embcoul: the embedding potential, electrostatic Coulomb potential only
            embnuc:  the embedding potential, nuclear potential only
            nadxc:   the nonadditive xc potential
            nadkin:  the nonadditve kinetic potential, func needs to be specified
        @type pot_type: str

        @param func: the functional to use. Only applicable with 'kinpot' and 'nadkin'
        @type func: fit

        @param orbs:
            a dictionary of the form {"irrep":[nums]} containing the
            orbitals to include. Use irrep "Loc" for localized orbitals
        @type orbs: dict
        """
        pot_type = pot_type.lower()
        if pot_type.startswith('recon'):
            pot_type = 'recon'
        if pot_type.startswith('tot'):
            pot_type = 'total'

        if func is not None:
            # FIXME: this check should be moved to densfjob
            if pot_type not in ['kinpot', 'nadkin']:
                raise PyAdfError("Functional can only be selected with kinpot and nadkin")
            else:
                options = {'func': func}
        else:
            options = {}

        if orbs is not None:
            options['orbs'] = orbs

        prop = PlotProperty("potential", prop_type=pot_type, options=options)

        return prop

    @classmethod
    def newOrbital(cls, irrep='A', orbnum=1, lapl=False):
        """
        Create a new property for an orbital.

        @param irrep: which irrep to use, 'Loc' for localized orbitals
        @param irrep: str

        @param orbnum: number of the orbital in the choosen irrep
        @type orbnum: int

        @param lapl: Whether to choose the Laplacian instead of the orbital itself
        @type lapl: bool
        """
        if not lapl:
            orb_type = 'orbital'
        else:
            orb_type = 'orblapl'

        options = {'irrep': irrep, 'orbnum': orbnum}

        prop = PlotProperty("orbital", prop_type=orb_type, options=options)

        return prop


class PlotProperty:
    """
    Class representing properties that can be stored as GridFunctions.
    """

    def __init__(self, prop_class, prop_type=None, options=None):

        self.pclass = prop_class.lower()

        self.ptype = prop_type
        if self.ptype is not None:
            self.ptype = self.ptype.lower()

        if options is None:
            self.opts = {}
        else:
            self.opts = options

        if 'orbs' in self.opts:
            orbs = self.opts['orbs']
            orbs_temp = {}
            for k, v in orbs.items():
                if not isinstance(v, list):
                    orbs_temp[k] = [v]
                else:
                    orbs_temp[k] = v

            self.opts['orbs'] = orbs_temp

        self._check_consistency()

    def _check_consistency(self):

        if self.pclass not in ['density', 'potential', 'orbital']:
            raise PyAdfError("Invalid property class.")

        if self.pclass == 'density':
            if self.ptype not in ['dens', 'sqrgrad', 'grad', 'lapl', 'hess']:
                raise PyAdfError("Invalid type in for Density property.")

        elif self.pclass == 'potential':
            if self.ptype not in ['total', 'nuc', 'coul', 'xc', 'kinpot',
                                  'recon', 'embpot', 'embcoul', 'embnuc', 'nadxc', 'nadkin']:
                raise PyAdfError("Invalid type in for Potential property.")

            if 'func' in self.opts:
                if self.ptype not in ['total', 'xc', 'kinpot', 'nadkin']:
                    raise PyAdfError("Functional can only be selected with this potential type.")

        elif self.pclass == 'orbital':
            if self.ptype not in ['orbital', 'orblapl']:
                raise PyAdfError("Invalid type in for Orbital property")

            if not ('irrep' in self.opts and 'orbnum' in self.opts):
                raise PyAdfError("Irrep and orbnum required for orbitals")

        if 'orbs' in self.opts:
            if self.pclass == 'potential' and self.ptype in ['recon', 'embpot', 'nadkin']:
                raise PyAdfError("Orbitals cannot be selected with this potential type.")

            if 'Loc' in self.opts['orbs']:
                if len(self.opts['orbs']) > 1:
                    raise PyAdfError('Localized and canonical orbitals cannot be used together.')

        if not (self.is_density or self.is_density_derivative or self.is_potential or
                self.is_orbital or self.is_orbital_derivative):
            raise PyAdfError('Invalid property class / type. This should not happen.')

    @property
    def str(self):
        s = f"{self.pclass} {self.ptype} Options: {str(self.opts)}"
        return s

    def __str__(self):
        return self.str

    def get_tape41_section_variable(self):
        """
        Get the section and variable name of the property on TAPE41.
        """

        if self.pclass == 'density':
            section = 'SCF'

            if self.ptype == 'dens':
                if self.opts['fit']:
                    variable = 'Fitdensity'
                else:
                    variable = 'Density'
            elif self.ptype == 'sqrgrad':
                if self.opts['fit']:
                    variable = 'FitSGradient'
                else:
                    variable = 'SGradient'
            elif self.ptype == 'grad':
                if self.opts['fit']:
                    variable = 'FitGradient'
                else:
                    variable = 'Gradient'
            elif self.ptype == 'lapl':
                if self.opts['fit']:
                    variable = 'FitdensityLap'
                else:
                    variable = 'DensityLap'
            elif self.ptype == 'hess':
                if self.opts['fit']:
                    variable = 'FitDensityHessian'
                else:
                    variable = 'DensityHessian'
            else:
                raise PyAdfError("Invalid type for property class DENSITY.")

        elif self.pclass == 'potential':
            section = 'Potential'

            if self.ptype == 'total':
                variable = 'Total'
            elif self.ptype == 'nuc':
                variable = 'Nuclear'
            elif self.ptype == 'coul':
                variable = 'Coulomb'
            elif self.ptype == 'xc':
                variable = 'XC'
            elif self.ptype == 'recon':
                variable = 'Reconstructed'
            elif self.ptype == 'kinpot':
                variable = 'Kinetic'
            elif self.ptype == 'embpot':
                variable = 'EmbeddingPot'
            elif self.ptype == 'embnuc':
                variable = 'FrozenNucPot'
            elif self.ptype == 'embcoul':
                variable = 'FrozenCoulPot'
            elif self.ptype == 'nadxc':
                variable = 'nadXcFrozen'
            elif self.ptype == 'nadkin':
                variable = 'nadKinFrozen'
            else:
                raise PyAdfError("Invalid type for property class POTENTIAL.")

        elif self.pclass == 'orbital':
            if self.ptype == 'orbital':
                section = f"SCF_{self.opts['irrep'].replace('LOC', 'A')}"
                variable = f"{self.opts['orbnum']:d}"
            elif self.ptype == 'orblapl':
                section = f"OrbLapl SCF_{self.opts['irrep'].replace('LOC', 'A')}"
                variable = f"{self.opts['orbnum']:d}"
            else:
                raise PyAdfError("Invalid type for property class ORBITAL")

        else:
            raise PyAdfError('Invalid or unknown property class')

        return section, variable

    @property
    def is_density(self):
        """
        Returns True if the property is a density.
        """
        return self.pclass == 'density' and self.ptype == 'dens'

    @property
    def is_density_derivative(self):
        """
        Returns True if the property is a density derivative (gradient or Laplacian).
        """
        return self.pclass == 'density' and not self.ptype == 'dens'

    @property
    def is_potential(self):
        """
        Returns True if the property is a potential.
        """
        return self.pclass == 'potential'

    @property
    def is_orbital(self):
        """
        Returns True if the property is an orbital.
        """
        return self.pclass == 'orbital' and self.ptype == 'orbital'

    @property
    def is_orbital_derivative(self):
        """
        Returns True if the property is an orbital.
        """
        return self.pclass == 'orbital' and self.ptype == 'orblapl'

    @property
    def is_localized_orbital(self):
        """
        Returns True if the property is (derived from) localized orbitals.
        """
        return self.is_orbital and self.opts['irrep'] == 'A'

    @property
    def needs_locorbdens(self):
        return ('orbs' in self.opts) and ('Loc' in self.opts['orbs'])

    @property
    def needs_orbdens(self):
        return ('orbs' in self.opts) and not ('Loc' in self.opts['orbs'])

    @property
    def is_unrestricted(self):
        if self.needs_locorbdens or self.needs_orbdens:
            return False
        elif self.is_orbital:
            return False
        elif self.is_potential and (self.ptype == 'nuc' or self.ptype == 'coul'):
            return False
        else:
            return True

    @property
    def vector_length(self):
        if self.pclass == 'density' and self.ptype == 'grad':
            vl = 3
        elif self.pclass == 'density' and self.ptype == 'hess':
            vl = 6
        else:
            vl = 1
        return vl

    @property
    def components(self):
        c = None
        if self.vector_length == 1:
            c = None
        elif self.pclass == 'density' and self.ptype == 'grad':
            c = ['x', 'y', 'z']
        elif self.pclass == 'density' and self.ptype == 'hess':
            c = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
        return c
