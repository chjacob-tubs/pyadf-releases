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

import xcfun

from ..Errors import PyAdfError
from ..Plot.GridFunctions import GridFunctionFactory


class EmbedXCFunSettings(object):
    """
    Class that holds the settings for Embed calculations using xcfun.
    """

    def __init__(self, fun_nad_k=None, fun_nad_xc=None):
        """
        """
        self.nad_kin = None
        self.nad_xc = None

        if fun_nad_k is None:
            self.set_fun_nad_kin({'tfk': 1.0})
        else:
            self.set_fun_nad_kin(fun_nad_k)
        if fun_nad_xc is None:
            self.set_fun_nad_xc({'lda': 1.0})
        else:
            self.set_fun_nad_xc(fun_nad_xc)

        self.embed_xcfun_splash()

    def set_fun_nad_kin(self, fun):
        """
        fun is a dictionary of valid functional names and weights.
        """
        self.nad_kin = fun

    def set_fun_nad_xc(self, fun):
        """
        fun is a dictionary of valid functional names and weights.
        """
        self.nad_xc = fun

    @staticmethod
    def show_xcfun_info():
        print "Using the XCFun library, version ", xcfun.xcfun_version()
        print xcfun.xcfun_splash()

    @staticmethod
    def embed_xcfun_splash():
        print "\nEmbed_xcfun: a module to calculate FDE embedding potentials" \
              "with density functionals\n"
        EmbedXCFunSettings.show_xcfun_info()

    def show_functionals(self):

        print "\nCurrently defined kinetic energy functional:"
        for (name, weight) in self.nad_kin.iteritems():
            print "%15s with weight %4f" % (name, weight)

        print "\nCurrently defined exchange-correlation functional:"
        for (name, weight) in self.nad_xc.iteritems():
            print "%15s with weight %4f" % (name, weight)

        print "\n"


class EmbedXCFunEvaluator(object):
    """
    class to calculate the embedding potential over a grid of points with the functionals
    setup in the embed_xcfun_settings
    """

    def __init__(self, settings=None):
        """
        """
        self.fun_nad_kin = None
        self.fun_nad_xc = None

        if settings is None:
            self.settings = EmbedXCFunSettings()
        else:
            self.settings = settings

        self.set_fun_nad_kin()
        self.set_fun_nad_xc()

    def set_fun_nad_kin(self):
        self.fun_nad_kin = xcfun.Functional(self.settings.nad_kin)

    def set_fun_nad_xc(self):
        self.fun_nad_xc = xcfun.Functional(self.settings.nad_xc)

    @staticmethod
    def _get_ta_difference(fun, density_t, density_a, order=1):

        if not ((density_t.grid is density_a.grid) or
                (density_t.grid.checksum == density_a.grid.checksum)):
            raise PyAdfError('grids have to be the same for binary operation on gridfunctions')

        if fun.type == 0:
            diff_ta = fun.eval_potential_n(density=density_t[0].values)[:, order] \
                      - fun.eval_potential_n(density=density_a[0].values)[:, order]
        elif fun.type == 1:
            diff_ta = fun.eval_potential_n(density=density_t[0].values,
                                           densgrad=density_t[1].values,
                                           denshess=density_t[2].values)[:, order] \
                      - fun.eval_potential_n(density=density_a[0].values,
                                             densgrad=density_a[1].values,
                                             denshess=density_a[2].values)[:, order]
        else:
            raise NotImplementedError

        import numpy
        diff_ta = numpy.nan_to_num(diff_ta)

        import hashlib
        m = hashlib.md5()
        m.update("Potential calculated in EmbedXCFunEvaluator._get_TA_difference :\n")
        m.update(density_t.get_checksum())
        m.update(density_a.get_checksum())
        m.update("with functional:\n")
        m.update(repr(fun))
        m.update("order: %i\n" % order)

        gf = GridFunctionFactory.newGridFunction(density_t.grid, diff_ta, m.digest(), 'potential')

        return gf

    def get_nad_pot_xc(self, density_a, density_b):
        density_t = density_a + density_b
        return self._get_ta_difference(self.fun_nad_xc, density_t, density_a, order=1)

    def get_nad_pot_kin(self, density_a, density_b):
        density_t = density_a + density_b
        return self._get_ta_difference(self.fun_nad_kin, density_t, density_a, order=1)

    def get_nad_pot(self, density_a, density_b):
        density_t = density_a + density_b

        pot_nad_kin = self._get_ta_difference(self.fun_nad_kin, density_t, density_a, order=1)
        pot_nad_xc = self._get_ta_difference(self.fun_nad_xc, density_t, density_a, order=1)

        return pot_nad_xc + pot_nad_kin

    def get_emb_pot(self, density_a, density_b, elpot_b):
        nad_pot_a = self.get_nad_pot(density_a, density_b)
        emb_pot_a = elpot_b + nad_pot_a

        return emb_pot_a

    def get_nad_energy_density(self, density_a, density_b):
        density_t = density_a + density_b

        ene_nad_kin = self._get_ta_difference(self.fun_nad_kin, density_t, density_a, order=0)
        ene_nad_xc = self._get_ta_difference(self.fun_nad_xc, density_t, density_a, order=0)

        return ene_nad_xc + ene_nad_kin

    @staticmethod
    def _get_nad_energy(fun, density_a, density_b):
        density_t = density_a + density_b

        if fun.type == 0:
            en_dens_t = fun.eval_energy_n(density=density_t[0].values)
            en_dens_a = fun.eval_energy_n(density=density_a[0].values)
            en_dens_b = fun.eval_energy_n(density=density_b[0].values)
        elif fun.type == 1:
            en_dens_t = fun.eval_energy_n(density=density_t[0].values, densgrad=density_t[1].values)
            en_dens_a = fun.eval_energy_n(density=density_a[0].values, densgrad=density_a[1].values)
            en_dens_b = fun.eval_energy_n(density=density_b[0].values, densgrad=density_b[1].values)
        else:
            raise NotImplementedError

        import numpy
        grid = density_t.grid

        energy_nad = numpy.dot(grid.get_weights(), (en_dens_t[:] - en_dens_a[:] - en_dens_b[:]))

        return energy_nad

    def get_interaction_energy(self, density_a, nucpot_a, coulomb_a, density_b, nucpot_b, coulomb_b):

        nad_kin = self._get_nad_energy(self.fun_nad_kin, density_a, density_b)
        nad_xc = self._get_nad_energy(self.fun_nad_xc, density_a, density_b)

        grid = density_a.grid

        en_elet_a = grid.get_weights() * density_a[0].values * (nucpot_b.get_values() + 0.5 * coulomb_b.get_values())
        en_elet_b = grid.get_weights() * density_b[0].values * (nucpot_a.get_values() + 0.5 * coulomb_a.get_values())
        return nad_kin + nad_xc + en_elet_a.sum() + en_elet_b.sum()
