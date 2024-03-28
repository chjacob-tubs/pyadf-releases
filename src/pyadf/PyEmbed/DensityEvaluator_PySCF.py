# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2024 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Maria Chekmeneva,
# Thomas Dresselhaus, Kevin Focke, Andre S. P. Gomes, Andreas Goetz,
# Michal Handzlik, Karin Kiewisch, Moritz Klammler, Lars Ridder,
# Jetze Sikkema, Lucas Visscher, Johannes Vornweg, Michael Welzel,
# and Mario Wolter.
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
 Functionality for evaluating densities and potentials for calculations
 using GTOs. This is implemented via PySCF.

 @author: Kevin Focke
"""
import numpy as np

try:
    # noinspection PyPackageRequirements
    from pyscf import tools, dft, df, lib, gto
except ImportError:
    print('Could not import PySCF. No PyScf functionality available.')
    tools = dft = df = lib = gto = None

from pyadf.Errors import PyAdfError


class PyScfInterface:
    """
    Import info from a calculation to PySCF via the molden format.
    Calculate the electron density values for an array of coordinates.
    Other things that can be done with PySCF should be easy to implement,
    as far as they are possible with the info gathered from the molden
    file format.
    """

    def __init__(self, molden_file):
        """
        Initializes a pyscf interface object.

        @param molden_file:   The contents of the molden file.
        @type  molden_file:    L{str}
        """
        import tempfile

        if tools is None:
            raise PyAdfError("Cannot create PySCF/Molden interface, PySCF not available.")

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as temp_file:
            temp_file.file.write(molden_file)

            self.mol, self.mo_energy, self.mo_coeff, self.mo_occ, self.irrrep_labels, self.spins = \
                tools.molden.load(temp_file.name)

        self.mol.verbose = 0
        self.mol.max_memory = 10000  # MB

        self._ao_cache = None
        self._densval_cache = None
        self._cache_deriv = -1

    def calc_ao_values(self, coordinates, deriv=0, cache=False):
        """
        Calculates atomic orbital values from a pyscf object.
        The values are added to the pyscf object.

        Optionally, the calculated AO values can be cache. This
        requires special care, and the cache needs to be reset
        before using a different grid.

        @param coordinates:   A list of N coordinates for which the atomic
                              orbital values are to be calculated.
        @type coordinates:    L{array}
        @param deriv:         Number of AO derivatives to calculate
        @type deriv:          int
        @param cache:         Whether to cache the caclulated AO values internally.
        @type cache:          bool
        """
        if (self._ao_cache is not None) and (deriv <= self._cache_deriv):
            return self._ao_cache

        ao_values = dft.numint.eval_ao(self.mol, coordinates, deriv=deriv)

        if cache:
            self._ao_cache = ao_values
            self._cache_deriv = deriv

        return ao_values

    def calc_density_values(self, coordinates, deriv=0, cache=False):
        if (self._densval_cache is not None) and (deriv <= self._cache_deriv):
            if deriv == 0:
                if self._cache_deriv > 0:
                    return self._densval_cache[0]
                else:
                    return self._densval_cache
            elif deriv == 1:
                return self._densval_cache[0:4]
            else:
                return self._densval_cache

        ao_values = self.calc_ao_values(coordinates, deriv=deriv, cache=cache)

        if deriv == 0:
            densvals = dft.numint.eval_rho2(self.mol, ao_values, self.mo_coeff, self.mo_occ)
        elif deriv == 1:
            densvals = dft.numint.eval_rho2(self.mol, ao_values, self.mo_coeff, self.mo_occ,
                                            xctype='GGA')
        elif deriv == 2:
            densvals = dft.numint.eval_rho2(self.mol, ao_values, self.mo_coeff, self.mo_occ,
                                            xctype='mGGA')
        else:
            raise NotImplementedError('Density derivatives only implemented up to 2nd deriv.')

        if cache:
            self._densval_cache = densvals
            self._cache_deriv = deriv

        return densvals

    def reset_cache(self):
        self._ao_cache = None
        self._densval_cache = None
        self._cache_deriv = -1

    def density(self, coordinates, deriv=0):
        """
        Calculates density values from a pyscf object.
        The values are added to the pyscf object. Different lists
        of values and coordinates can be stored and later used again.

        @param coordinates:   A list of N coordinates for which the density
        values are to be calculated. shape (N,3)
        @type coordinates:    L{array}

        @param deriv: order of derivatives of the density to calculate (1 and 2 possible)
        @type deriv: int
        """
        ao_values = self.calc_ao_values(coordinates, deriv=deriv)
        density_values = self.calc_density_values(coordinates, deriv=deriv)

        if deriv == 2:
            density_values = np.delete(density_values, 5, 0)
            density_values = np.delete(density_values, 4, 0)

            # noinspection PyPackageRequirements
            from pyscf.dft.gen_grid import BLKSIZE
            # noinspection PyPackageRequirements,PyProtectedMember
            from pyscf.dft.numint import _dot_ao_dm as dot_ao_dm
            # noinspection PyPackageRequirements,PyProtectedMember
            from pyscf.dft.numint import _contract_rho as contract_rho

            ngrids, nao = ao_values[0].shape
            non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, self.mol.nbas), dtype=np.uint8)
            shls_slice = (0, self.mol.nbas)
            ao_loc = self.mol.ao_loc_nr()
            pos = self.mo_occ > dft.numint.OCCDROP
            cpos = np.einsum('ij,j->ij', self.mo_coeff[:, pos], np.sqrt(self.mo_occ[pos]))
            cs = [dot_ao_dm(self.mol, ao_values[0], cpos, non0tab, shls_slice, ao_loc)]
            for i in range(1, 4):
                cs.append(dot_ao_dm(self.mol, ao_values[i], cpos, non0tab, shls_slice, ao_loc))
            for j in range(4, 10):
                c2 = dot_ao_dm(self.mol, ao_values[j], cpos, non0tab, shls_slice, ao_loc)
                density_values = np.append(density_values, [2 * contract_rho(cs[0], c2)],
                                              axis=0)

            density_values[4] += 2 * contract_rho(cs[1], cs[1])
            density_values[5] += 2 * contract_rho(cs[1], cs[2])
            density_values[6] += 2 * contract_rho(cs[1], cs[3])
            density_values[7] += 2 * contract_rho(cs[2], cs[2])
            density_values[8] += 2 * contract_rho(cs[2], cs[3])
            density_values[9] += 2 * contract_rho(cs[3], cs[3])

        return density_values

    def laplacian(self, coordinates):
        """
        Calculates density values from a pyscf object.
        The values are added to the pyscf object. Different lists
        of values and coordinates can be stored and later used again.

        @param coordinates:   A list of N coordinates for which the density
        values are to be calculated. shape (N,3)
        @type coordinates:    L{array}

        """
        density_values = self.calc_density_values(coordinates, deriv=2)
        return density_values[4]

    def nuclear_potential(self, coordinates):
        Vnuc = np.zeros(coordinates.shape[0])
        for i in range(self.mol.natm):
            r = self.mol.atom_coord(i)
            Z = self.mol.atom_charge(i)
            rp = r - coordinates
            Vnuc += -Z / np.sqrt(np.einsum('xi,xi->x', rp, rp))
        return Vnuc

    def coulomb_potential(self, coordinates):
        Vele = np.empty(coordinates.shape[0])
        mocc = self.mo_coeff[:, self.mo_occ > 0]
        dm = np.dot(mocc * self.mo_occ[self.mo_occ > 0], mocc.conj().T)
        for p0, p1 in lib.prange(0, Vele.size, 600):
            fakemol = gto.fakemol_for_charges(coordinates[p0:p1])
            ints = df.incore.aux_e2(self.mol, fakemol)
            Vele[p0:p1] = np.einsum('ijp,ij->p', ints, dm)
        return Vele
