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
Classes for writing grids and grid functions to file.
"""

import re
import os.path
import numpy as np

from . import Grids
from .GridFunctions import GridFunctionFactory, GridFunctionContainer

from pyadf.Errors import PyAdfError


def read_xmldataset_to_numpy(filename, dataset_names):
    import xml.etree.ElementTree as ET
    from io import StringIO

    tree_root = ET.parse(filename).getroot()

    data_list = []
    for name in dataset_names:
        dataset_element = tree_root.find(f"./dataset[@name='{name}']")

        # noinspection PyTypeChecker
        data = np.loadtxt(StringIO(dataset_element.text))

        data_size = int(dataset_element.attrib['size'])
        data_width = int(dataset_element.attrib['width'])
        if data_width == 1:
            data_shape = (data_size,)
        else:
            data_shape = (data_size, data_width)
        if not (data.shape == data_shape):
            raise PyAdfError('Size mismatch when reading XML file')

        data_list.append(data)

    if len(data_list) == 1:
        data_list = data_list[0]
    else:
        data_list = tuple(data_list)
    return data_list


class GridReader:

    @staticmethod
    def read_xyzw(filename_full):

        npoints_re = re.compile(r"^\s*(\d+)\s*$", re.IGNORECASE)
        xyzw_re = re.compile(r"^\s*(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                             r"\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)", re.IGNORECASE)

        i = 0
        with open(filename_full, encoding='utf-8') as f:
            for line in f:
                if npoints_re.match(line):
                    npoints = int(npoints_re.match(line).group(1))
                    xyz = np.zeros((npoints, 3))
                    w = np.zeros((npoints,))

                if xyzw_re.match(line):
                    xyz[i, :] = [xyzw_re.match(line).group(1),
                                 xyzw_re.match(line).group(2),
                                 xyzw_re.match(line).group(3)]
                    w[i] = xyzw_re.match(line).group(4)

                    i = i + 1

        grid = Grids.customgrid(None, xyz, w)
        return grid

    @staticmethod
    def read_xml(filename_full):
        xyzw = read_xmldataset_to_numpy(filename_full, ['gridpoints'])

        grid = Grids.customgrid(None, xyzw[:, 0:3], xyzw[:, 3])
        return grid


class GridFunctionReader:

    @staticmethod
    def read_xyzwv(filename_full, gf_type=None):

        npoints_re = re.compile(r"^\s*(\d+)\s*$", re.IGNORECASE)
        xyzwv_re = re.compile(r"^\s*(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                              r"\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)",
                              re.IGNORECASE)

        i = 0
        with open(filename_full, encoding='utf-8') as f:
            for line in f:
                if npoints_re.match(line):
                    npoints = int(npoints_re.match(line).group(1))
                    v = np.zeros((npoints,))
                    xyz = np.zeros((npoints, 3))
                    w = np.zeros((npoints,))

                if xyzwv_re.match(line):
                    xyz[i, :] = [xyzwv_re.match(line).group(1),
                                 xyzwv_re.match(line).group(2),
                                 xyzwv_re.match(line).group(3)]
                    w[i] = xyzwv_re.match(line).group(4)
                    v[i] = xyzwv_re.match(line).group(5)

                    i = i + 1

        grid = Grids.customgrid(None, xyz, w)

        import hashlib
        m = hashlib.md5()
        m.update(b"Gridfunction read from file:")
        m.update(os.path.abspath(filename_full).encode('utf-8'))

        gf = GridFunctionFactory.newGridFunction(grid, v, checksum=m.hexdigest(), gf_type=gf_type)

        return gf

    @staticmethod
    def _make_gfs_from_numpy(filename_full, grid_xyzw, elpot, nucpot, rho, rhod, rhodd):
        grid = Grids.customgrid(None, np.ascontiguousarray(grid_xyzw[:, 0:3]),
                                np.ascontiguousarray(grid_xyzw[:, 3]))

        import hashlib
        m = hashlib.md5()
        m.update(b"Electrostatic potential read from file:")
        m.update(os.path.abspath(filename_full).encode('utf-8'))

        pot_gf = GridFunctionFactory.newGridFunction(grid, elpot, checksum=m.hexdigest(),
                                                     gf_type="potential")

        if nucpot is not None:
            m = hashlib.md5()
            m.update(b"Nuclear potential read from file:")
            m.update(os.path.abspath(filename_full).encode('utf-8'))

            nucpot_gf = GridFunctionFactory.newGridFunction(grid, nucpot, checksum=m.hexdigest(),
                                                            gf_type="potential")
        else:
            nucpot_gf = None

        m = hashlib.md5()
        m.update(b"Density read from file:")
        m.update(filename_full.encode('utf-8'))

        dens_gf = GridFunctionFactory.newGridFunction(grid, rho, checksum=m.hexdigest(), gf_type="density")

        m = hashlib.md5()
        m.update(b"Density gradient read from file:")
        m.update(filename_full.encode('utf-8'))

        densgrad = GridFunctionFactory.newGridFunction(grid, rhod, checksum=m.hexdigest())

        m = hashlib.md5()
        m.update(b"Density Hessian read from file:")
        m.update(filename_full.encode('utf-8'))

        denshess = GridFunctionFactory.newGridFunction(grid, rhodd, checksum=m.hexdigest())

        rho_gf = GridFunctionContainer([dens_gf, densgrad, denshess])

        return grid, pot_gf, nucpot_gf, rho_gf

    @staticmethod
    def read_density_elpot_xyzwv(filename_full):
        """
        FIXME: This needs a docstring explaining the file format!
        """

        npoints_re = re.compile(r"^\s*(\d+)\s+\d", re.IGNORECASE)
        density_re = re.compile(r"^\s*(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                                r"\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                                r"\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                                r"\s+(-?[0-9a-zA-Z+-.]+)", re.IGNORECASE)
        ngngnn_re = re.compile(r"^\s*(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                               r"\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                               r"\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                               r"\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                               r"\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" +
                               r"\s+(-?[0-9a-zA-Z+-.]+)", re.IGNORECASE)

        i = 0
        with open(filename_full, encoding='utf-8') as f:
            for line in f:
                if npoints_re.match(line):
                    npoints = int(npoints_re.match(line).group(1))
                    rho = np.zeros((npoints, 10))
                    elpot = np.zeros((npoints,))
                    grid_xyzw = np.zeros((npoints, 4))
                if density_re.match(line):
                    # leaving space for the hessian of the density, force zeros for those
                    rho[i, :] = [density_re.match(line).group(7),
                                 density_re.match(line).group(8),
                                 density_re.match(line).group(9),
                                 density_re.match(line).group(10),
                                 0, 0, 0, 0, 0, 0]
                    elpot[i] = density_re.match(line).group(5)
                    elpot[i] += float(density_re.match(line).group(6))
                    grid_xyzw[i, :] = [density_re.match(line).group(1),
                                       density_re.match(line).group(2),
                                       density_re.match(line).group(3),
                                       density_re.match(line).group(4)]
                    i = i + 1
                elif ngngnn_re.match(line):
                    rho[i, :] = [density_re.match(line).group(7),
                                 density_re.match(line).group(8),
                                 density_re.match(line).group(9),
                                 density_re.match(line).group(10),
                                 density_re.match(line).group(11),
                                 density_re.match(line).group(12),
                                 density_re.match(line).group(13),
                                 density_re.match(line).group(14),
                                 density_re.match(line).group(15),
                                 density_re.match(line).group(16)]
                    elpot[i] = density_re.match(line).group(5)
                    elpot[i] += float(density_re.match(line).group(6))
                    grid_xyzw[i, :] = [density_re.match(line).group(1),
                                       density_re.match(line).group(2),
                                       density_re.match(line).group(3),
                                       density_re.match(line).group(4)]
                    i = i + 1

        grid, pot_gf, nucpot_gf, rho_gf = \
            GridFunctionReader._make_gfs_from_numpy(filename_full, grid_xyzw, elpot, None,
                                                    np.ascontiguousarray(rho[:, 0]),
                                                    np.ascontiguousarray(rho[:, 1:4]),
                                                    np.ascontiguousarray(rho[:, 4:10]))

        return grid, pot_gf, rho_gf

    @staticmethod
    def read_density_elpot_xml(filename_full):

        grid_xyzw, elpot, nucpot, rho, rhod, rhodd = \
            read_xmldataset_to_numpy(filename_full, ['gridpoints', 'vc', 'nuc',
                                                     'density', 'gradient', 'hessian'])

        grid, pot_gf, nucpot_gf, rho_gf = \
            GridFunctionReader._make_gfs_from_numpy(filename_full, grid_xyzw, elpot, nucpot,
                                                    rho, rhod, rhodd)

        return grid, pot_gf, nucpot_gf, rho_gf
