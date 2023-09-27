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

from xml.sax.handler import ContentHandler
from xml.sax import make_parser

from kf import kffile

import numpy


class GridHandler(ContentHandler):

    def __init__(self, file_name):
        super().__init__()

        self.data_name = 'dataset'
        self.data_grid = 'grid'
        self.data_tape = file_name
        self.lblock = 128

        self.variables = []
        self.data = None
        self.idata = None
        self.outfile = None

        self.section = None
        self.var = None

        self.width = None
        self.chars = ""

        self.npoints = None
        self.nblock = None
        self.dummypoints = None
        self.npoints_total = None
        self.gridname = None

    def startDocument(self):
        self.variables = []
        self.outfile = kffile(self.data_tape)

    def endDocument(self):
        self.outfile.close()

    def dataset_Handler(self, attrs):
        name = attrs.get('name', '')
        if name == "gridpoints":
            self.section = "Points"
            self.var = "Data"
        elif name == "vc":
            self.section = "FrozenDensityElpot"
            self.var = "ElpotFD"
        elif name == "density":
            self.section = "FrozenDensity"
            self.var = "rhoffd"
        elif name == "gradient":
            self.section = "FrozenDensityFirstDer"
            self.var = "drhoffd"
        elif name == "hessian":
            self.section = "FrozenDensitySecondDer"
            self.var = "d2rhoffd"
        else:
            self.section = name
            self.var = name

        self.width = int(attrs.get('width', 1))

        self.chars = ""
        self.data = numpy.zeros((self.npoints_total * self.width,))

        self.idata = 0

    def dataset_Write(self):

        if not self.idata == self.npoints * self.width:
            raise Exception("Wrong number of points read")

        self.data = self.data.reshape((self.npoints_total, self.width))

        outdata = numpy.zeros((self.nblock, self.lblock * self.width))

        for iblock in range(self.nblock):
            ipoint = self.lblock * iblock

            # get one block of points
            block = self.data[ipoint:ipoint + self.lblock]
            # points have to be written in Fortran ordering !
            block = block.flatten(order='F')

            outdata[iblock, :] = block

        outdata = outdata.flatten()

        self.outfile.writereals(self.section, self.var, outdata)

        # reset data
        self.data = None

    def grid_Handler(self, attrs):
        self.npoints = int(attrs.get('size', None))
        if self.npoints % self.lblock == 0:
            self.dummypoints = 0
        else:
            self.dummypoints = self.lblock - (self.npoints % self.lblock)
        self.npoints_total = self.npoints + self.dummypoints
        self.nblock = self.npoints_total // self.lblock

        # FIXME: nspin = 1 hardcoded
        self.outfile.writeints('General', 'nspin', 1)

        self.outfile.writeints("Points", "nblock", self.nblock)
        self.outfile.writeints("Points", "lblock", self.lblock)
        self.outfile.writeints("Points", "nmax", self.npoints_total)
        self.outfile.writeints("Points", "Equivalent Blocks", 1)

        self.outfile.writeints("Points", "Length of Blocks", [self.lblock] * self.nblock)

        self.gridname = attrs.get('name', None)

    def startElement(self, tag, attrs):
        if tag == self.data_name:
            self.dataset_Handler(attrs)
        elif tag == self.data_grid:
            self.grid_Handler(attrs)

    def endElement(self, tag):
        if tag == 'dataset':
            for ch in self.chars.split():
                self.data[self.idata] = float(ch)
                self.idata += 1
            self.dataset_Write()

    def characters(self, content):
        if self.data is not None:
            self.chars += content
            split_chars = self.chars.split()
            if len(split_chars) > 0:
                for ch in split_chars[:-1]:

                    self.data[self.idata] = float(ch)
                    self.idata += 1

                    if self.chars[-1].isspace():
                        split_chars[-1] += " "
                    self.chars = split_chars[-1]


def xml2kf(xmlfile, outfile):
    parser = make_parser()
    parser.setContentHandler(GridHandler(outfile))
    parser.parse(xmlfile)


if __name__ == '__main__':

    import sys

    if not len(sys.argv) == 3:
        raise Exception("Wrong number of arguments")

    xml2kf(sys.argv[1], sys.argv[2])
