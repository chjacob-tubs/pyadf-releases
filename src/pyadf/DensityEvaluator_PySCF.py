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
 Functionality for evaluating densities and potentials for calculations
 using GTOs. This is implemented via PySCF.

 @author: Kevin Focke
"""
import os

if 'PYADF_NPROC' in os.environ:
    default_nproc = int(os.environ['PYADF_NPROC'])
elif 'TC_NUM_PROCESSES' in os.environ:
    default_nproc = int(os.environ['TC_NUM_PROCESSES'])
elif 'NSCM' in os.environ:
    default_nproc = int(os.environ['NSCM'])
else:
    import multiprocessing
    default_nproc = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(default_nproc)
os.environ['MKL_NUM_THREADS'] = str(default_nproc)
os.environ['OPENBLAS_NUM_THREADS'] = str(default_nproc)

import numpy
import hashlib
import itertools
import h5py
import hdf5plugin
import functools
try:
    # noinspection PyPackageRequirements
    from pyscf import tools, dft, df, lib, gto
except ImportError:
    print('Could not import PySCF. No PyScf functionality available.')
    tools = dft = df = lib = gto = None

from .Errors import PyAdfError
from .Utils import pse

def checksum(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coord_list = args[1]
        coord_array = numpy.array(coord_list)
        if numpy.array_equal(coord_array, args[0].coords):
            kwargs['checksum'] = self.checksum
        else:
            m = hashlib.md5()
            m.update(bytes(coord_array))
            m.update(func.__name__.encode('utf-8'))
            m.update(args[0].molden_file.encode('utf-8'))
            args[0].checksum = m.hexdigest()
            kwargs['checksum'] = args[0].checksum

        return func(*args, **kwargs)
    return wrapper

class PyScfInterface:
    """
    Import info from a calculation to PySCF via the molden format.
    Calculate the electron density values for an array of coordinates.
    Other things that can be done with PySCF should be easy to implement,
    as far as they are possible with the info gathered from the molden
    file format.
    """

    def __init__(self, result=None, molden_file=None):
        """
        Initializes a pyscf interface object.

        @param molden_file:   The contents of the molden file.
        @type  molden_file:    L{str}
        """
        import tempfile
        self.result = result
        if molden_file is None:
            self.molden_file = self.result.read_molden_file()
        else:
            self.molden_file = molden_file

        self.coords = numpy.array([])

        if tools is None:
            raise PyAdfError("Cannot create PySCF/Molden interface, PySCF not available.")

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as temp_file:
            temp_file.file.write(self.molden_file)

            self.mol, self.mo_energy, self.mo_coeff, self.mo_occ, self.irrrep_labels, self.spins = \
                tools.molden.load(temp_file.name)

        self.mol.verbose = 0
        self.mol.max_memory = 10000  # MB

        self._ao_cache = None
        self._densval_cache = None
        self._ao_cache_deriv = -1
        self._dens_cache_deriv = -1

    def _hdf5_saver(self, checksum, result_array, exists=None, file_exists=None):

        array = numpy.array(result_array)
        if not os.path.isdir('resultfiles'):
            raise PyAdfError("wrong directory " + os.getcwd())
        rel_filename = self._filename
        if os.path.isfile(rel_filename):
            file_exists = True
        with h5py.File(rel_filename, 'a') as hf:
            if exists:
                del hf[checksum]
            elif exists == None:
                try:
                    del hf[checksum]
                    exists = True
                except KeyError:
                    exists = False
            hf.create_dataset(checksum, data=array,
                              **hdf5plugin.Blosc(cname=self.cname, clevel=9,
                                                 shuffle=hdf5plugin.Blosc.BITSHUFFLE))
        if not os.path.isfile(rel_filename):
            raise PyAdfError('creation of h5- file ' + os.path.abspath(rel_filename) + ' failed')
        if not self.result.files.have_file(rel_filename):
            if file_exists:
                raise PyAdfError('file should have already existed')
            self.result.files.add_file(rel_filename)
            self.result.files._resultfiles[self.result.fileid].append(rel_filename)
            # having exactly the same molden file and settings should also lead to the same results
            # even when it is not coming from the same result object, file handling should be made
            # independent
            print('saved to file ' + os.path.abspath(rel_filename))
        else:
            if not file_exists:
                raise PyAdfError('file should not have existed')

    def _hdf5_loader(self, checksum):

        filename =  self._filename
        with h5py.File(filename, 'r') as hf:
            retrieved_array = numpy.array(hf[checksum])
        return retrieved_array

    @property
    def _filename(self):
        return f'resultfiles/t77.results{self.result.fileid:04d}'

    @property
    def cname(self):
        try:
            return self._cname
        except:
            try:
                self._cname = self.result.job.settings.cname
                return self._cname
            except:
                self._cname = 'blosclz'
                return self._cname

    @checksum
    def calc_ao_values(self, coordinates=None, /, deriv=0, cache=False, checksum=None):
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
        if (self._ao_cache is not None) and (deriv <= self._ao_cache_deriv):
            return self._ao_cache

        if self.result.files.have_file(self._filename):
            try:
                ao_values = self._hdf5_loader(checksum)
            except Exception as e:
                print(e)
                print('ao values probably not calculated yet')
            else:
                if deriv == 0:
                    if cache:
                        self._ao_cache = ao_values
                        self._ao_cache_deriv = deriv
                    return ao_values
                if len(ao_values.shape) == 3:
                    if deriv == 1:
                        if cache:
                            self._ao_cache = ao_values
                            self._ao_cache_deriv = deriv
                        return ao_values
                    if deriv == 2 and ao_values.shape[0] in [10,20]:
                        if cache:
                            self._ao_cache = ao_values
                            self._ao_cache_deriv = deriv
                        return ao_values

        ao_values = dft.numint.eval_ao(self.mol, coordinates, deriv=deriv)

        if cache:
            self._ao_cache = ao_values
            self._ao_cache_deriv = deriv

        self._hdf5_saver(checksum, ao_values)
        return ao_values

    @checksum
    def calc_density_values(self, coordinates=None, /, deriv=0, cache=False, checksum=None):
        if (self._densval_cache is not None) and (deriv <= self._dens_cache_deriv):
            if deriv == 0:
                if self._dens_cache_deriv > 0:
                    return self._densval_cache[0]
                else:
                    return self._densval_cache
            elif deriv == 1:
                return self._densval_cache[0:4]
            else:
                return self._densval_cache
        # not in cash
        if self.result.files.have_file(self._filename):
            try:
                densvals = self._hdf5_loader(checksum)
            except Exception as e:
                print(e)
                print('density values probably not calculated yet')
            else:
                if len(densvals.shape) == 1:
                    file_deriv = 0
                elif densvals.shape[0] == 4:
                    file_deriv = 1
                else:
                    file_deriv = 2

                if file_deriv >= deriv:
                    if cache:
                        self._densval_cache = densvals
                        self._dens_cache_deriv = file_deriv

                    if deriv == 0:
                        if file_deriv == 0:
                            return densvals
                        else:
                            return densvals[0]

                    if deriv == 1:
                        return densvals[0:4]

                    elif deriv == 2:
                        return densvals

        # not in file
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
            self._dens_cache_deriv = deriv

        self._hdf5_saver(checksum, densvals)
        return densvals

    def reset_cache(self):
        self._ao_cache = None
        self._densval_cache = None
        self._dens_cache_deriv = -1
        self._ao_cache_deriv = -1

    @checksum
    def density(self, coordinates=None, /, deriv=0, checksum=None):
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
        density_values = self.calc_density_values(coordinates, deriv=deriv)
        if deriv < 2:
            return density_values

        if deriv == 2:
            ao_values = self.calc_ao_values(coordinates, deriv=deriv)
            if self.result.files.have_file(self._filename):
                try:
                    density_values = self._hdf5_loader(checksum)
                    return density_values
                except Exception as e:
                    print(e)
                    print('density derivatives probably not calculated yet')

            density_values = numpy.delete(density_values, 5, 0)
            density_values = numpy.delete(density_values, 4, 0)

            # noinspection PyPackageRequirements
            from pyscf.dft.gen_grid import BLKSIZE
            # noinspection PyPackageRequirements,PyProtectedMember
            from pyscf.dft.numint import _dot_ao_dm as dot_ao_dm
            # noinspection PyPackageRequirements,PyProtectedMember
            from pyscf.dft.numint import _contract_rho as contract_rho

            ngrids, nao = ao_values[0].shape
            non0tab = numpy.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, self.mol.nbas), dtype=numpy.uint8)
            shls_slice = (0, self.mol.nbas)
            ao_loc = self.mol.ao_loc_nr()
            pos = self.mo_occ > dft.numint.OCCDROP
            cpos = numpy.einsum('ij,j->ij', self.mo_coeff[:, pos], numpy.sqrt(self.mo_occ[pos]))
            cs = [dot_ao_dm(self.mol, ao_values[0], cpos, non0tab, shls_slice, ao_loc)]
            for i in range(1, 4):
                cs.append(dot_ao_dm(self.mol, ao_values[i], cpos, non0tab, shls_slice, ao_loc))
            for j in range(4, 10):
                c2 = dot_ao_dm(self.mol, ao_values[j], cpos, non0tab, shls_slice, ao_loc)
                density_values = numpy.append(density_values, [2 * contract_rho(cs[0], c2)],
                                              axis=0)

            density_values[4] += 2 * contract_rho(cs[1], cs[1])
            density_values[5] += 2 * contract_rho(cs[1], cs[2])
            density_values[6] += 2 * contract_rho(cs[1], cs[3])
            density_values[7] += 2 * contract_rho(cs[2], cs[2])
            density_values[8] += 2 * contract_rho(cs[2], cs[3])
            density_values[9] += 2 * contract_rho(cs[3], cs[3])
            self._hdf5_saver(checksum, density_values)
            return density_values


    @checksum
    def laplacian(self, coordinates=None, /, checksum=None):
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

    @checksum
    def nuclear_potential(self, coordinates=None, /, checksum=None):
        if self.result.files.have_file(self._filename):
            try:
                Vnuc = self._hdf5_loader(checksum)
                return Vnuc
            except Exception as e:
                print(e)
                print('Nuclear potential probably not calculated yet')
        Vnuc = numpy.zeros(coordinates.shape[0])
        for i in range(self.mol.natm):
            r = self.mol.atom_coord(i)
            atom_symbol = self.mol.atom_symbol(i)
            atom_symbol = ''.join([i for i in atom_symbol if not i.isdigit()])
            Z = pse.get_atomic_number(atom_symbol)
            rp = r - coordinates
            Vnuc += -Z / numpy.sqrt(numpy.einsum('xi,xi->x', rp, rp))
        self._hdf5_saver(checksum, Vnuc)
        return Vnuc

    @checksum
    def coulomb_potential(self, coordinates=None, /, checksum=None):
        if self.result.files.have_file(self._filename):
            try:
                Vele = self._hdf5_loader(checksum)
                return Vele
            except Exception as e:
                print(e)
                print('Coulomb potential probably not calculated yet')
        Vele = numpy.empty(coordinates.shape[0])
        mocc = self.mo_coeff[:, self.mo_occ > 0]
        dm = numpy.dot(mocc * self.mo_occ[self.mo_occ > 0], mocc.conj().T)
        for p0, p1 in lib.prange(0, Vele.size, 600):
            fakemol = gto.fakemol_for_charges(coordinates[p0:p1])
            ints = df.incore.aux_e2(self.mol, fakemol)
            Vele[p0:p1] = numpy.einsum('ijp,ij->p', ints, dm)
        self._hdf5_saver(checksum, Vele)
        return Vele
