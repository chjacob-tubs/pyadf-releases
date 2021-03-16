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
 General utilities that are used internally.

 @author:       Christoph Jacob and Michal Hanzlik
"""

from Errors import PTError, UnitsError

import math

newjobmarker = '--- PyADF *** new job *** PyADF ---\n'


# noinspection PyTypeChecker
class PeriodicTable(object):
    data = ['XX', 0.0, 0.0, 0] * 150
    # [symbol, mass, radius, connectors]
    data[0] = ['Xx', 0.00000, 0.22, 0]
    data[1] = ['H', 1.00794, 0.30, 1]
    data[2] = ['He', 4.00260, 0.99, 0]
    data[3] = ['Li', 6.94100, 1.52, 8]
    data[4] = ['Be', 9.01218, 1.12, 8]
    data[5] = ['B', 10.81100, 0.88, 6]
    data[6] = ['C', 12.01070, 0.77, 4]
    data[7] = ['N', 14.00670, 0.70, 3]
    data[8] = ['O', 15.99940, 0.66, 2]
    data[9] = ['F', 18.99840, 0.64, 1]
    data[10] = ['Ne', 20.17970, 1.60, 0]
    data[11] = ['Na', 22.98977, 1.86, 8]
    data[12] = ['Mg', 24.30500, 1.60, 8]
    data[13] = ['Al', 26.98154, 1.43, 8]
    data[14] = ['Si', 28.08550, 1.17, 8]
    data[15] = ['P', 30.97376, 1.10, 8]
    data[16] = ['S', 32.06500, 1.04, 2]
    data[17] = ['Cl', 35.45300, 0.99, 1]
    data[18] = ['Ar', 39.94800, 1.92, 0]
    data[19] = ['K', 39.09830, 2.31, 8]
    data[20] = ['Ca', 40.07800, 1.97, 8]
    data[21] = ['Sc', 44.95591, 1.60, 8]
    data[22] = ['Ti', 47.86700, 1.46, 8]
    data[23] = ['V', 50.94150, 1.31, 8]
    data[24] = ['Cr', 51.99610, 1.25, 8]
    data[25] = ['Mn', 54.93805, 1.29, 8]
    data[26] = ['Fe', 55.84500, 1.26, 8]
    data[27] = ['Co', 58.93320, 1.25, 8]
    data[28] = ['Ni', 58.69340, 1.24, 8]
    data[29] = ['Cu', 63.54600, 1.28, 8]
    data[30] = ['Zn', 65.40900, 1.33, 8]
    data[31] = ['Ga', 69.72300, 1.41, 8]
    data[32] = ['Ge', 72.64000, 1.22, 8]
    data[33] = ['As', 74.92160, 1.21, 8]
    data[34] = ['Se', 78.96000, 1.17, 8]
    data[35] = ['Br', 79.90400, 1.14, 1]
    data[36] = ['Kr', 83.79800, 1.97, 0]
    data[37] = ['Rb', 85.46780, 2.44, 8]
    data[38] = ['Sr', 87.62000, 2.15, 8]
    data[39] = ['Y', 88.90585, 1.80, 8]
    data[40] = ['Zr', 91.22400, 1.57, 8]
    data[41] = ['Nb', 92.90638, 1.41, 8]
    data[42] = ['Mo', 95.94000, 1.36, 8]
    data[43] = ['Tc', 98.00000, 1.35, 8]
    data[44] = ['Ru', 101.07000, 1.33, 8]
    data[45] = ['Rh', 102.90550, 1.34, 8]
    data[46] = ['Pd', 106.42000, 1.38, 8]
    data[47] = ['Ag', 107.86820, 1.44, 8]
    data[48] = ['Cd', 112.41100, 1.49, 8]
    data[49] = ['In', 114.81800, 1.66, 8]
    data[50] = ['Sn', 118.71000, 1.62, 8]
    data[51] = ['Sb', 121.76000, 1.41, 8]
    data[52] = ['Te', 127.60000, 1.37, 8]
    data[53] = ['I', 126.90447, 1.33, 1]
    data[54] = ['Xe', 131.29300, 2.17, 0]
    data[55] = ['Cs', 132.90545, 2.62, 8]
    data[56] = ['Ba', 137.32700, 2.17, 8]
    data[57] = ['La', 138.90550, 1.88, 8]
    data[58] = ['Ce', 140.11600, 1.818, 8]
    data[59] = ['Pr', 140.90765, 1.824, 8]
    data[60] = ['Nd', 144.24000, 1.814, 8]
    data[61] = ['Pm', 145.00000, 1.834, 8]
    data[62] = ['Sm', 150.36000, 1.804, 8]
    data[63] = ['Eu', 151.96400, 2.084, 8]
    data[64] = ['Gd', 157.25000, 1.804, 8]
    data[65] = ['Tb', 158.92534, 1.773, 8]
    data[66] = ['Dy', 162.50000, 1.781, 8]
    data[67] = ['Ho', 164.93032, 1.762, 8]
    data[68] = ['Er', 167.25900, 1.761, 8]
    data[69] = ['Tm', 168.93421, 1.759, 8]
    data[70] = ['Yb', 173.04000, 1.922, 8]
    data[71] = ['Lu', 174.96700, 1.738, 8]
    data[72] = ['Hf', 178.49000, 1.57, 8]
    data[73] = ['Ta', 180.94790, 1.43, 8]
    data[74] = ['W', 183.84000, 1.37, 8]
    data[75] = ['Re', 186.20700, 1.37, 8]
    data[76] = ['Os', 190.23000, 1.34, 8]
    data[77] = ['Ir', 192.21700, 1.35, 8]
    data[78] = ['Pt', 195.07800, 1.38, 8]
    data[79] = ['Au', 196.96655, 1.44, 8]
    data[80] = ['Hg', 200.59000, 1.52, 8]
    data[81] = ['Tl', 204.38330, 1.71, 8]
    data[82] = ['Pb', 207.20000, 1.75, 8]
    data[83] = ['Bi', 208.98038, 1.70, 8]
    data[84] = ['Po', 209.00000, 1.40, 8]
    data[85] = ['At', 210.00000, 1.40, 1]
    data[86] = ['Rn', 222.00000, 2.40, 8]
    data[87] = ['Fr', 223.00000, 2.70, 8]
    data[88] = ['Ra', 226.00000, 2.20, 0]
    data[89] = ['Ac', 227.00000, 2.00, 8]
    data[90] = ['Th', 232.03810, 1.79, 8]
    data[91] = ['Pa', 231.03588, 1.63, 8]
    data[92] = ['U', 238.02891, 1.56, 8]
    data[93] = ['Np', 237.00000, 1.55, 8]
    data[94] = ['Pu', 244.00000, 1.59, 8]
    data[95] = ['Am', 243.00000, 1.73, 8]
    data[96] = ['Cm', 247.00000, 1.74, 8]
    data[97] = ['Bk', 247.00000, 1.70, 8]
    data[98] = ['Cf', 251.00000, 1.86, 8]
    data[99] = ['Es', 252.00000, 1.86, 8]
    data[100] = ['Fm', 257.00000, 2.00, 8]
    data[101] = ['Md', 258.00000, 2.00, 8]
    data[102] = ['No', 259.00000, 2.00, 8]
    data[103] = ['Lr', 262.00000, 2.00, 8]
    data[104] = ['Rf', 261.00000, 2.00, 8]
    data[105] = ['Db', 262.00000, 2.00, 8]
    data[106] = ['Sg', 266.00000, 2.00, 8]
    data[107] = ['Bh', 264.00000, 2.00, 8]
    data[108] = ['Hs', 277.00000, 2.00, 8]
    data[109] = ['Mt', 268.00000, 2.00, 8]
    data[110] = ['Ds', 281.00000, 2.00, 8]
    data[111] = ['Rg', 280.00000, 2.00, 8]
    data[112] = ['Cn', 285.00000, 2.00, 8]

    symtonum = {'Xx': 0, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11,
                'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21,
                'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
                'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41,
                'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
                'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
                'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
                'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
                'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
                'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
                'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
                'Rg': 111, 'Cn': 112}

    def __init__(self):
        pass

    @classmethod
    def get_atomic_number(cls, symbol):
        symb = symbol.split('.')[0]
        try:
            number = cls.symtonum[symb.capitalize()]
        except KeyError:
            raise PTError('trying to convert incorrect atomic symbol')
        return number

    @classmethod
    def get_symbol(cls, atnum):
        try:
            symbol = cls.data[atnum][0]
        except KeyError:
            raise PTError('trying to convert incorrect atomic number')
        return symbol

    @classmethod
    def _get_pr(cls, arg, prop):
        if isinstance(arg, str):
            pr = cls.data[cls.get_atomic_number(arg)][prop]
        elif isinstance(arg, int):
            try:
                pr = cls.data[arg][prop]
            except KeyError:
                raise PTError('trying to convert incorrect atomic number')
        else:
            raise PTError('trying to convert incorrect atomic number')
        return pr

    @classmethod
    def get_mass(cls, arg):
        return cls._get_pr(arg, 1)

    @classmethod
    def get_radius(cls, arg):
        return cls._get_pr(arg, 2)

    @classmethod
    def get_connectors(cls, arg):
        return cls._get_pr(arg, 3)


PT = PeriodicTable


class Units(object):
    # Bohr radius (in 10^-10 m) according to CODATA 2010
    # see http://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
    bohr_radius = 0.52917721092

    # Avogadro constant according to CODATA 2010
    # see http://physics.nist.gov/cgi-bin/cuu/Value?na
    avogadro_constant = 6.02214129e23

    # Speed of light (in m/s), exact value according to CODATA 2010
    # see http://physics.nist.gov/cgi-bin/cuu/Value?c
    speed_of_light = 299792458

    # Electron charge (in Coulomb), according to CODATA 2010
    # see http://physics.nist.gov/cgi-bin/cuu/Value?e
    electron_charge = 1.602176565e-19

    dicts = []

    distance = {'__': 'Distance', 'angstrom': 1.0, 'A': 1.0, 'nm': 0.1, 'pm': 100.0, 'bohr': 1.0 / bohr_radius}
    distance['a0'] = distance['bohr']
    distance['au'] = distance['bohr']
    dicts.append(distance)

    # Hartree energy in eV according to CODATA 2010
    # see http://physics.nist.gov/cgi-bin/cuu/Value?threv
    # Hartree energy in kJ according to CODATA 2010
    # see http://physics.nist.gov/cgi-bin/cuu/Value?threv
    energy = {'__': 'Energy', 'au': 1.0, 'hartree': 1.0, 'Hartree': 1.0, 'eV': 27.21138505,
              'kJ/mol': 4.35974434e-21 * avogadro_constant}
    energy['ev'] = energy['eV']
    # By definition (ISO 31-4): 1 kcal = 4.184 kJ
    # see https://en.wikipedia.org/wiki/Calorie
    energy['kcal/mol'] = energy['kJ/mol'] / 4.184
    dicts.append(energy)

    angle = {'__': 'Angle', 'degree': 1.0, 'deg': 1.0, 'rad': math.pi / 180.0, 'grad': 100.0 / 90.0,
             'circle': 1.0 / 360.0}
    angle['radian'] = angle['rad']

    dicts.append(angle)

    dipole = {'__': 'Dipole Moment', 'au': 1.0, 'Cm': electron_charge * bohr_radius * 1e-10}
    # conversion to Colomb*Meter
    # 1 Debye = 1/c * 1e-21 Cm
    # see https://en.wikipedia.org/wiki/Debye
    dipole['debye'] = speed_of_light * 1e21 * dipole['Cm']
    dipole['Debye'] = dipole['debye']
    dipole['D'] = dipole['debye']
    dicts.append(dipole)

    # TODO:    mass, charge, force, frequency, constants ?

    def __init__(self):
        pass

    @classmethod
    def conversion(cls, inp, out):
        for d in cls.dicts:
            if inp in d.keys() and out in d.keys():
                return d[out] / d[inp]
        raise UnitsError('Invalid conversion call: unsupported units')

    @classmethod
    def convert(cls, value, inp, out):
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise UnitsError('Invalid conversion call: non-numerical value')
        return val * cls.conversion(inp, out)

    @classmethod
    def printinfo(cls, values=False):
        print 'Units supported for conversion:'
        for i in cls.dicts:
            print i['__'] + ':',
            for j in i.keys():
                if j != '__':
                    if values:
                        print j + '=' + str(i[j]) + ',',
                    else:
                        print j + ',',
            print ''


def f2f(literal):
    """
    Converts a string output by a Fortran routine to a float.

    If the literal can't be understood as a float, a NaN is returned silently.

    @param literal: Literal Fortran output
    @type  literal: L{str}
    @returns:       Parsed value
    @rtype:         L{float}
    @author:        Moritz Klammler
    @date:          Aug. 2011

    """

    # First we make sure that the argument actually is a string and contains
    # only uppercase characters.

    literal = str(literal)
    literal = literal.upper()

    # The reason Fortran output can't be converted directly is that Fortran
    # knows two types of floating point numbers: `REAL' and `DOUBLE
    # PRECISION'. While this is nothing unusual and may be found in almost any
    # strong typing language, Fortran uses a different literal representation
    # of these. E.g. the literal representation of the speed of light as a
    # FLOAT reads `2.99792458E8' while the DOUBLE PRECISION's literal is
    # `2.99792458D8'. To put it short, we have to replace `D's by `E's and
    # everything will be fine.

    literal = literal.replace('D', 'E')
    try:
        f = float(literal)
    except ValueError:
        f = float('NaN')
    return f


Bohr_in_Angstrom = Units.conversion('bohr', 'angstrom')
au_in_Debye = Units.conversion('au', 'debye')
au_in_eV = Units.conversion('au', 'eV')
conversion = Units.conversion
pse = PeriodicTable
