# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2026 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Maria Chekmeneva,
# Thomas Dresselhaus, Kevin Focke, Andre S. P. Gomes, Andreas Goetz,
# Michal Handzlik, Karin Kiewisch, Moritz Klammler, Toni M. Maier, 
# Malgorzata Olejniczak, Lars Ridder, Jetze Sikkema, Erik Tsvetaev, 
# Lucas Visscher, Johannes Vornweg, Michael Welzel, and Mario Wolter.
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
 Functionality for saving arrays in h5py files. The h5py library is
 not part of the standard library, but an implicit dependency via
 pyscf

 @author: Kevin Focke
"""

import os
import h5py
import numpy as np
import functools
import hashlib
from pyadf.Errors import PyAdfError
from pyadf.PatternsLib import Singleton


def cacheresults(calc_func):
    """
    A function decorator that modifies a function that takes a checksum keyword
    argument and calculates an array.
    If no checksum is provided, a checksum is made.

    Using the decorator leads to the creation of an hdf5 file and caching functionality

    Parameters
    ----------
    calc_func : function
        The function that is being decorated. It is assumed to calculate an array (ndarray)
        and take checksum as a possible keyword argument.

    Returns
    _______
    wrapper : function
        A wrapped version of the calc_func. This adds a checksum value if it is missing,
        tries to read a cached version of the array from the ArraySaver and and ensures
        that the resulting array is saved via the ArraySaver if it had not been calculated
        yet.
    """
    @functools.wraps(calc_func)
    def wrapper(*args, **kwargs):
        arrayvault = ArraySaver()
        checksum = kwargs.get('checksum', None)
        # ^^^^^^^^
        # this has to be passed explicitly while calling the function, implicit
        # definitions via default settings do not show up in the kwargs dictionary!
        if not checksum:
            hash_obj = hashlib.new('md5')
            hash_obj.update(args[0].checksum.encode('utf-8'))  # this  is the molden file checksum
            hash_obj.update(args[1].tobytes())  # these are the coordinates in question
            hash_obj.update(calc_func.__name__.encode('utf-8'))  # this is the name of the calc func
            checksum = hash_obj.hexdigest()
        array = arrayvault.read_from_file(checksum)
        if array is None:
            array = calc_func(*args, **kwargs)
            arrayvault.save_to_file(checksum, array)
        return array
    return wrapper


class ArraySaver(metaclass=Singleton):
    """
    A new singleton class that is used to save arrays.
    The singleton approach only makes sense if the file does not get too big.
    TODO: This should be re-viewed together with the filemanager.

    Parameters
    ----------
    filedir : path-like object or str
        The directory in which the hdf5 file is to be created.
    filename : str
        The name of the hdf5 file that is used to store the arrays.
    compression : str
        Is passed on to h5py as compression. Default is gzip.
    compression_opts : int
        Is passed on to h5py as compression_opts. Default is 9.
        Highest compression could even be faster than lower compression as long as
        the array is small enough to be compressed in a single step in memory.
    """
    def __init__(self, filedir=None, filename=None,
                 compression='gzip', compression_opts=9):

        if not filedir:
            filedir = os.getcwd()
        self.filedir = filedir

        if not filename:
            filename = 'grid_vault.hdf5'
        self.filename = filename

        self.filepath = os.path.join(filedir, filename)

        self.compression = compression
        self.compression_opts = compression_opts

        self._files = set()

    def read_from_file(self, checksum, filepath=None):
        """
        A method that is used to read arrays from hdf5 files.

        Parameters
        ----------
        checksum : str
            This is the label under which the array was hopefully saved. The method always
            needs a checksum. The checksum should uniquely identify the calculation.
        filepath : path-like object or str
            The path to the actual hdf5 file. Defaults to self.filepath.

        Returns
        -------
        array : ndarray or None
            Either a successfully retrieved array or None is returned. The returned value
            has to be handled externally.
        """
        if not filepath:
            filepath = self.filepath
        self._files.add(filepath)
        self._ensure_dir(filepath)
        with h5py.File(filepath, 'a') as f:
            if checksum in f:
                return np.array(f[checksum])
            else:
                return None

    def save_to_file(self, checksum, arr, filepath=None):
        """
        A method that is used to save arrays to  hdf5 files.

        Parameters
        ----------
        checksum : str
            This is the label under which the array is saved. Always needs a checksum.
            The checksum should uniquely identify the calculation.
        arr : ndarray
            The array that is saved in the hdf5 file.
        filepath : path-like object or str
            The path to the actual hdf5 file. Defaults to self.filepath.

        Raises
        ------
        PyAdfError
            The function does not overwrite old entries. They have to be deleted if this
            is the goal. Otherwise the function should only ever be called if the entry for
            checksum had not been found before.
        """
        if not filepath:
            filepath = self.filepath
        self._files.add(filepath)
        self._ensure_dir(filepath)
        with h5py.File(filepath, 'a') as f:
            if checksum in f:
                raise PyAdfError(f'the entry {checksum} should not yet have existed' +
                                 ' in the hdf5 file {filepath}')
            f.create_dataset(checksum, data=arr, compression=self.compression,
                             compression_opts=self.compression_opts)

    def _ensure_dir(self, filepath):
        """
        The file creation by h5py fails when the directory does not exist.
        Before creating the file, this function ensures that the folder actually exists.
        TODO:
        The folder structure of the calculations and the saved results all need to be
        re-viewed with the filemanager.

        Parameters
        ----------
        filepath : path-like object or str
            The path to the file that is about to be created

        Returns
        -------
        bool
            The value returned is always True: either the method raises an error, or it
            successfully ensures that the direcotery exists.

        Raises
        ------
        PyAdfError
            The function does not accept paths that directly correspond to existing dirs.
        """
        if os.path.isfile(filepath):
            pass
        elif os.path.isdir(filepath):
            raise PyAdfError(f'{filepath} already is a directory, _ensure_dir is meant ' +
                             'for the filepath')
        elif os.path.isdir(os.path.dirname(filepath)):
            pass
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self._files.add(os.path.dirname(filepath))
        return True

    def cleanup(self):
        """
        This function is currently not used.
        Files and directories that were created are removed and the set
        TODO:
        The cleanup functionality should be reviewed together with the filemanager.
        """
        dirs = []
        for filepath in self._files:
            try:
                os.remove(filepath)
            except IsADirectoryError:
                dirs.append(filepath)
        for dir_path in dirs:
            try:
                os.rmdir(dir_path)
            except OSError as err:
                print(f'{dir_path} probably not empty while trying to remove it:\n' +
                      str(err))
                # we only want to remove it if its empty anyways
        self._files = set()
