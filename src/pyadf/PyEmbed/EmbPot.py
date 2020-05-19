#!/usr/bin/env python

import xcfun

from ..Errors import PyAdfError
from ..Plot.GridFunctions import GridFunctionFactory


class EmbedXCFunSettings(object):
    """
    Class that holds the settings for Embed calculations using xcfun.
    """

    def __init__(self, fun_nad_k={'tfk': 1.0}, fun_nad_xc={'lda': 1.0}):
        """
        """
        self.nad_kin = None
        self.nad_xc = None

        self.set_fun_nad_kin(fun_nad_k)
        self.set_fun_nad_xc(fun_nad_xc)
        self.set_fun_class()

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

    def set_fun_class(self):
        """
        here we will verify if both functionals are lda, gga or metagga,
        and stop in the case of a mismatch
        """
        pass

    def show_xcfun_info(self):
        print "Using the XCFun library, version ", xcfun.xcfun_version()
        print xcfun.xcfun_splash()

    def embed_xcfun_splash(self):
        print "\nEmbed_xcfun: a module to calculate FDE embedding potentials" \
              "with density functionals\n"
        self.show_xcfun_info()

    def show_functionals(self):

        print "\nCurrently defined kinetic energy functional:"
        for (name, weight) in self.nad_kin.iteritems():
            print "%15s with weight %4f" % (name, weight)

        print "\nCurrently defined exchange-correlation functional:"
        for (name, weight) in self.nad_xc.iteritems():
            print "%15s with weight %4f" % (name, weight)

        print "\n"

    def show_available_functionals(self):
        pass


class EmbedXCFunEvaluator (object):

    """
    class to calculate the embedding potential over a grid of points with the functionals
    setup in the embed_xcfun_settings
    """

    def __init__(self, settings=None, options=None):
        """
        """
        self.fun_nad_kin = None
        self.fun_nad_xc = None
        self.islda = True
        self.isgga = False
        self.ismgga = False

        if settings is None:
            self.settings = EmbedXCFunSettings()
        else:
            self.settings = settings

        self.set_fun_nad_kin()
        self.set_fun_nad_xc()

    def check_fun_type(self, fun):
        fun_type = fun.type

        if fun_type == 0:
            fun_class = 'LDA'
            self.islda = True
            self.isgga = False
            self.ismgga = False
        elif fun_type == 1:
            fun_class = 'GGA'
            self.islda = False
            self.isgga = True
            self.ismgga = False
        elif fun_type == 2:
            fun_class = 'Meta-GGA'
            self.islda = False
            self.isgga = False
            self.ismgga = True

        print "functional is of ", fun_class, " class"

    def set_fun_nad_kin(self):
        self.fun_nad_kin = xcfun.Functional(self.settings.nad_kin)
        self.check_fun_type(self.fun_nad_kin)

    def set_fun_nad_xc(self):
        self.fun_nad_xc = xcfun.Functional(self.settings.nad_xc)
        self.check_fun_type(self.fun_nad_xc)

    def _get_TA_difference(self, fun, density_T, density_A, order=1):

        if not ((density_T.grid is density_A.grid) or
                (density_T.grid.checksum == density_A.grid.checksum)):
            raise PyAdfError('grids have to be the same for binary operation on gridfunctions')

        if self.ismgga:
            pass
        elif self.isgga:
            diff_ta = fun.eval_potential_n(density=density_T[0].values,
                                           densgrad=density_T[1].values,
                                           denshess=density_T[2].values)[:, order] \
                - fun.eval_potential_n(density=density_A[0].values,
                                       densgrad=density_A[1].values,
                                       denshess=density_A[2].values)[:, order]
        else:
            diff_ta = fun.eval_potential_n(density=density_T[0].values)[:, order] \
                - fun.eval_potential_n(density=density_A[0].values)[:, order]

        import numpy
        diff_ta = numpy.nan_to_num(diff_ta)

        import hashlib
        m = hashlib.md5()
        m.update("Potential calculated in EmbedXCFunEvaluator._get_TA_difference :\n")
        m.update(density_T.get_checksum())
        m.update(density_A.get_checksum())
        m.update("with functional:\n")
        m.update(repr(fun))
        m.update("order: %i\n" % order)

        gf = GridFunctionFactory.newGridFunction(density_T.grid, diff_ta, m.digest(), 'potential')

        return gf

    def get_nad_pot_xc(self, density_A, density_B):
        density_T = density_A + density_B
        return self._get_TA_difference(self.fun_nad_xc, density_T, density_A, order=1)

    def get_nad_pot_kin(self, density_A, density_B):
        density_T = density_A + density_B
        return self._get_TA_difference(self.fun_nad_kin, density_T, density_A, order=1)

    def get_nad_pot(self, density_A, density_B):
        density_T = density_A + density_B

        pot_nad_kin = self._get_TA_difference(self.fun_nad_kin, density_T, density_A, order=1)
        pot_nad_xc = self._get_TA_difference(self.fun_nad_xc, density_T, density_A, order=1)

        return pot_nad_xc + pot_nad_kin

    def get_emb_pot(self, density_A, density_B, elpot_B):
        nad_pot_A = self.get_nad_pot(density_A, density_B)
        emb_pot_A = elpot_B + nad_pot_A

        return emb_pot_A

    def get_nad_energy_density(self, density_A, density_B):
        density_T = density_A + density_B

        ene_nad_kin = self._get_TA_difference(self.fun_nad_kin, density_T, density_A, order=0)
        ene_nad_xc = self._get_TA_difference(self.fun_nad_xc, density_T, density_A, order=0)

        return ene_nad_xc + ene_nad_kin

    def get_interaction_energy(self, grid_A, density_A, nucpot_A, coulomb_A, grid_B, density_B, nucpot_B, coulomb_B):

        density_T = density_A + density_B
        en_dens_T = self.fun_nad_kin.eval_potential_n(density=density_T[0].values)[:,0] \
            + self.fun_nad_xc.eval_potential_n(density=density_T[0].values)[:,0]
        en_dens_A = self.fun_nad_kin.eval_potential_n(density=density_A[0].values)[:,0] \
            + self.fun_nad_xc.eval_potential_n(density=density_A[0].values)[:,0]
        en_dens_B = self.fun_nad_kin.eval_potential_n(density=density_B[0].values)[:,0] \
            + self.fun_nad_xc.eval_potential_n(density=density_B[0].values)[:,0]

        en_dens_nad = grid_A.get_weights()*(en_dens_T[:] - en_dens_A[:] - en_dens_B[:])
        en_elet_A   = grid_A.get_weights()*density_A[0].values*(nucpot_B.get_values() + 0.5*coulomb_B.get_values())
        en_elet_B   = grid_B.get_weights()*density_B[0].values*(nucpot_A.get_values() + 0.5*coulomb_A.get_values())
        return (en_dens_nad.sum() + en_elet_A.sum() + en_elet_B.sum())

