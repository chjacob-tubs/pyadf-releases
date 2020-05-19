# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2014 by Christoph R. Jacob, S. Maya Beyhan,
# Rosa E. Bulo, Andre S. P. Gomes, Andreas Goetz, Michal Handzlik,
# Karin Kiewisch, Moritz Klammler, Jetze Sikkema, and Lucas Visscher
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
Classes for writing grids and grid functions to file.
"""

import kf
import re
import os.path
import numpy

from ..Errors import PyAdfError
from . import Grids
from .GridFunctions import GridFunctionFactory, GridFunctionContainer


class GridFunctionReader(object):

    @classmethod
    def read_xyzwv(cls, filename_full, gf_type=None):

        npoints_re = re.compile(r'' + "^\s*(\d+)\s*$" + '', re.IGNORECASE)
        xyzwv_re = re.compile(r'' + "^\s*(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" + '', re.IGNORECASE)

        i = 0
        with open(filename_full) as f:
            for line in f:
                if npoints_re.match(line):
                    npoints = numpy.int(npoints_re.match(line).group(1))
                    v = numpy.zeros((npoints,))
                    xyz = numpy.zeros((npoints, 3))
                    w = numpy.zeros((npoints,))

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
        m.update("Gridfunction read from file:".encode("utf-8"))
        m.update(os.path.abspath(filename_full).encode("utf-8"))

        gf = GridFunctionFactory.newGridFunction(grid, v, checksum=m.digest(), gf_type=gf_type)

        return gf

    @classmethod
    def read_density_elpot_xyzwv(cls, filename_full):
        """
        FIXME: This needs a docstring explaining the file format!
        """

        npoints_re = re.compile(r'' + "^\s*(\d+)\s+\d" + '', re.IGNORECASE)
        density_re = re.compile(r'' + "^\s*(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" + '', re.IGNORECASE)
        ngngnn_re = re.compile(r'' + "^\s*(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)\s+(-?[0-9a-zA-Z+-.]+)" + '', re.IGNORECASE)

        i = 0
        with open(filename_full) as f:
            for line in f:
                if npoints_re.match(line):
                    npoints = numpy.int(npoints_re.match(line).group(1))
                    rho = numpy.zeros((npoints, 10))
                    elpot = numpy.zeros((npoints))
                    grid_xyzw = numpy.zeros((npoints, 4))
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

        grid = Grids.customgrid(None, numpy.ascontiguousarray(grid_xyzw[:, 0:3]),
                                numpy.ascontiguousarray(grid_xyzw[:, 3]))

        import hashlib
        m = hashlib.md5()
        m.update("Electrostatic potential read from file:".encode("utf-8"))
        m.update(os.path.abspath(filename_full).encode("utf-8"))

        pot_gf = GridFunctionFactory.newGridFunction(grid, elpot, checksum=m.digest(),
                                                     gf_type="potential")

        m = hashlib.md5()
        m.update("Density read from file:".encode("utf-8"))
        m.update(filename_full.encode("utf-8"))

        dens_gf = GridFunctionFactory.newGridFunction(grid, numpy.ascontiguousarray(rho[:, 0]),
                                                      checksum=m.digest(), gf_type="density")

        m = hashlib.md5()
        m.update("Density gradient read from file:".encode("utf-8"))
        m.update(filename_full.encode("utf-8"))

        densgrad = GridFunctionFactory.newGridFunction(grid, numpy.ascontiguousarray(rho[:, 1:4]),
                                                       checksum=m.digest())

        m = hashlib.md5()
        m.update("Density Hessian read from file:".encode("utf-8"))
        m.update(filename_full.encode("utf-8"))

        denshess = GridFunctionFactory.newGridFunction(grid, numpy.ascontiguousarray(rho[:, 4:10]),
                                                       checksum=m.digest())

        rho_gf = GridFunctionContainer([dens_gf, densgrad, denshess])

        return grid, pot_gf, rho_gf
