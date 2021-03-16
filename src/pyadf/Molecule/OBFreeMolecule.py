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

from ..Utils import Units, PT
from pdbtools import PDBHandler, PDBRecord
from ..Errors import *
import copy

from BaseMolecule import BaseMolecule


class Atom(object):

    def __init__(self, atnum=0, coords=None, unit='angstrom', bonds=None, mol=None, ghost=False, **other):
        self.atnum = atnum
        self.mol = mol
        self.ghost = ghost
        self.bonds = bonds or []
        self.properties = other

        ratio = Units.conversion(unit, 'angstrom')

        # coordinates are always stores in Angstrom internally
        if coords is None:
            self.coords = (0.0, 0.0, 0.0)
        else:
            try:
                self.coords = (float(coords[0]) * ratio,
                               float(coords[1]) * ratio,
                               float(coords[2]) * ratio)
            except TypeError:
                raise AtomError('__init__: Invalid coords passed')

    def __str__(self):
        symbol = PT.get_symbol(self.atnum)
        if self.ghost:
            symbol = 'Gh.' + symbol
        return '%5s %14.5f %14.5f %14.5f' % (symbol, self.x, self.y, self.z)

    def _setx(self, value):
        self.coords = (float(value), self.coords[1], self.coords[2])

    def _getx(self):
        return self.coords[0]

    def _sety(self, value):
        self.coords = (self.coords[0], float(value), self.coords[2])

    def _gety(self):
        return self.coords[1]

    def _setz(self, value):
        self.coords = (self.coords[0], self.coords[1], float(value))

    def _getz(self):
        return self.coords[2]

    x = property(_getx, _setx)
    y = property(_gety, _sety)
    z = property(_getz, _setz)

    def _getsymbol(self):
        return PT.get_symbol(self.atnum)

    def _setsymbol(self, symbol):
        self.atnum = PT.get_atomic_number(symbol)

    symbol = property(_getsymbol, _setsymbol)

    def _getmass(self):
        return PT.get_mass(self.atnum)

    mass = property(_getmass)

    def _getradius(self):
        return PT.get_radius(self.atnum)

    radius = property(_getradius)

    def _getconnectors(self):
        return PT.get_connectors(self.atnum)

    connectors = property(_getconnectors)

    def move_by(self, vector, unit='angstrom'):
        ratio = Units.conversion(unit, 'angstrom')
        try:
            self.coords = tuple(i + j * ratio for i, j in zip(self.coords, vector))
        except:
            raise AtomError('moveBy: 3D vector needed as argument')

    def move_to(self, coords, unit='angstrom'):
        ratio = Units.conversion(unit, 'angstrom')
        try:
            self.coords = (ratio * float(coords[0]), ratio * float(coords[1]), ratio * float(coords[2]))
        except:
            raise AtomError('moveTo: Y U MOVE ME TO SUCH NASTY PLACE?')

    def distance_to(self, atom, unit='angstrom'):
        # in unit, default Angstrom
        res = (atom.x - self.x)**2 + (atom.y - self.y)**2 + (atom.z - self.z)**2
        return res**0.5 * Units.conversion('angstrom', unit)

    def vector_to(self, atom, unit='angstrom'):
        # in unit, default Angstrom
        ratio = Units.conversion('angstrom', unit)
        return (atom.x - self.x) * ratio, (atom.y - self.y) * ratio, (atom.z - self.z) * ratio

    def dist_sqr(self, atom):
        # fast but always in Angstrom; be careful
        return (self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2


class Bond(object):
    AR = 1.5

    def __init__(self, atom1, atom2, order=1, mol=None, **other):
        self.atom1 = atom1
        self.atom2 = atom2
        self.order = order
        self.mol = mol
        self.properties = other

    def __str__(self):
        return '(' + str(self.atom1) + ')-' + str(self.order) + '-(' + str(self.atom2) + ')'

    def is_aromatic(self):
        return self.order == Bond.AR

    def length(self, unit='angstrom'):
        # in units of atom1
        return self.atom1.distance_to(self.atom2, unit)

    def other_end(self, atom):
        # 'hard' identity required
        if atom is self.atom1:
            return self.atom2
        elif atom is self.atom2:
            return self.atom1
        else:
            raise BondError('other_end: invalid atom passed')

    def resize(self, atom, length, unit='angstrom'):
        ratio = 1.0 - Units.convert(length, unit, 'angstrom') / self.length()
        moving = self.other_end(atom)
        moving.move_by(tuple(i * ratio for i in moving.vector_to(atom)), moving.unit)


class OBFreeMolecule(BaseMolecule):

    def __init__(self, filename=None, inputformat='xyz'):
        self.atoms = []
        self.bonds = []
        self.charge = 0
        self.spin = 0
        self.symmetry = None
        self.properties = {}

        if filename is not None:
            self.read(filename, inputformat)

    def __str__(self):
        s = '  Cartesian coordinates: \n' + self.print_coordinates(index=True)
        s += '  Bonds: \n'
        for bond in self.bonds:
            s += str(bond) + '\n'
        return s

    def __iter__(self):
        return iter(self.atoms)

    def __add__(self, other):
        m = copy.deepcopy(self)
        m += other
        return m

    def __iadd__(self, other):
        othercopy = copy.deepcopy(other)
        self.atoms += othercopy.atoms
        self.bonds += othercopy.bonds
        for atom in self.atoms:
            atom.mol = self
        for bond in self.bonds:
            bond.mol = self
        self.charge += othercopy.charge
        if self.spin is not None and othercopy.spin is not None:
            self.spin += othercopy.spin
        # self.symmetry = None
        self.properties = dict(othercopy.properties)
        return self

    # OBFreeMolecule is pure Python, so __copy__ and __deepcopy__ are not required here

    def copy(self, other):
        self.atoms = other.atoms
        self.bonds = other.bonds
        self.symmetry = other.symmetry
        self.charge = other.charge
        self.spin = other.spin

    def add_bond(self, bond):
        bond.mol = self
        self.bonds.append(bond)
        bond.atom1.bonds.append(bond)
        bond.atom2.bonds.append(bond)

    def delete_bond(self, bond):
        bond.atom1.bonds.remove(bond)
        bond.atom2.bonds.remove(bond)
        self.bonds.remove(bond)

    def delete_all_bonds(self):
        while self.bonds:
            self.delete_bond(self.bonds[0])

    def add_atom(self, atom, adjacent=None, orders=None):
        self.atoms.append(atom)
        atom.mol = self
        if adjacent is not None:
            for i, adj in enumerate(adjacent):
                newbond = Bond(atom, adj)
                if orders is not None:
                    newbond.order = orders[i]
                self.add_bond(newbond)

    def delete_atom(self, atom):
        try:
            self.atoms.remove(atom)
        except:
            raise MoleculeError('delete_atom: invalid argument passed as atom')
        for b in atom.bonds:
            b.other_end(atom).bonds.remove(b)
            self.bonds.remove(b)

    def set_atoms_id(self):
        for i, at in enumerate(self.atoms):
            at.id = i + 1

    def unset_atoms_id(self):
        for at in self.atoms:
            try:
                del at.id
            except:
                pass

    def get_fragment(self, atoms, ghosts=True):
        atoms = self._get_atoms(atoms)
        atoms = [atom for atom in atoms if ghosts or not atom.ghost]
        m = OBFreeMolecule()
        for atom in atoms:
            newatom = Atom(atnum=atom.atnum, coords=atom.coords, mol=m, ghost=atom.ghost)
            newatom.properties = dict(atom.properties)
            m.atoms.append(newatom)
            atom.bro = newatom
        for bond in self.bonds:
            if hasattr(bond.atom1, 'bro') and hasattr(bond.atom2, 'bro'):
                m.add_bond(Bond(bond.atom1.bro, bond.atom2.bro, order=bond.order,
                                **bond.properties))
        for atom in atoms:
            del atom.bro
        return m

    def add_atoms(self, atoms, coords, atomicunits=False, ghosts=False, bond_to=None):
        if atomicunits:
            unit = 'bohr'
        else:
            unit = 'angstrom'
        for at, co in zip(atoms, coords):
            if isinstance(at, str):
                at = PT.get_atomic_number(at)
            newatom = Atom(atnum=at, coords=co, unit=unit, mol=self, ghost=ghosts)
            self.add_atom(newatom, adjacent=bond_to)

    def delete_atoms(self, atoms):
        atoms = self._get_atoms(atoms)
        for at in atoms:
            self.delete_atom(at)

    def add_as_ghosts(self, other):
        m = copy.deepcopy(self)
        n = len(self.atoms)
        m += other
        for at in m.atoms[n:]:
            at.ghost = True
        return m

    # I don't like this, should be done with property()
    def get_number_of_atoms(self):
        return len(self.atoms)

    def set_symmetry(self, symmetry):
        self.symmetry = symmetry

    def get_symmetry(self):
        return self.symmetry

    def set_charge(self, charge):
        self.charge = charge

    def get_charge(self):
        return self.charge

    def set_spin(self, spin):
        self.spin = spin

    def get_spin(self):
        return self.spin

    def print_coordinates(self, atoms=None, index=True, suffix=''):
        lines = ''
        atoms = self._get_atoms(atoms)
        for i, at in enumerate(atoms):
            symb = at.symbol
            if at.ghost:
                symb = 'Gh.' + symb
            if index:
                line = "  %3i) %8s %14.5f %14.5f %14.5f" % (i + 1, symb, at.x, at.y, at.z)
            else:
                line = "  %8s %14.5f %14.5f %14.5f" % (symb, at.x, at.y, at.z)

            line += "    " + suffix + "\n"
            lines += line
        return lines

    def readxyz(self, f, frame):
        fr = frame
        first = True
        n = 0
        i = 1
        for line in f:
            if fr != 0:
                try:
                    n = int(line.strip())
                    fr -= 1
                except ValueError:
                    continue
            else:
                if first:
                    first = False
                    i = 1
                    if line:
                        self.properties['comment'] = line.rstrip()
                else:
                    if i <= n:
                        lst = line.split()
                        shift = 0
                        if len(lst) > 4 and lst[0] == str(i):
                            shift = 1
                        num = lst[0 + shift]
                        if isinstance(num, str):
                            num = PT.get_atomic_number(num)
                        self.add_atom(Atom(atnum=num, coords=(lst[1 + shift], lst[2 + shift], lst[3 + shift])))
                        i += 1
                    else:
                        break
        if fr > 0:
            raise MoleculeError('readxyz: There are only %i frames in %s' % (frame - fr, f.name))
        f.close()

    def writexyz(self, f):
        f.write(self.get_xyz_file())

    def readmol(self, f, frame):
        if frame != 1:
            raise MoleculeError('readmol: .mol files do not support multiple geometries')

        comment = []
        for i in xrange(4):
            line = f.readline().rstrip()
            if line:
                spl = line.split()
                if spl[len(spl) - 1] == 'V2000':
                    natom = int(spl[0])
                    nbond = int(spl[1])
                    for j in xrange(natom):
                        atomline = f.readline().split()
                        crd = tuple(map(float, atomline[0:3]))
                        symb = atomline[3]
                        try:
                            num = PT.get_atomic_number(symb)
                        except PTError:
                            num = 0
                        self.add_atom(Atom(atnum=num, coords=crd))
                    for j in xrange(nbond):
                        bondline = f.readline().split()
                        at1 = self.atoms[int(bondline[0]) - 1]
                        at2 = self.atoms[int(bondline[1]) - 1]
                        ordr = int(bondline[2])
                        if ordr == 4:
                            ordr = Bond.AR
                        self.add_bond(Bond(atom1=at1, atom2=at2, order=ordr))
                    break
                elif spl[len(spl) - 1] == 'V3000':
                    raise MoleculeError('readmol: Molfile V3000 not supported. Please convert')
                else:
                    comment.append(line)
        if comment:
            self.properties['comment'] = comment
        f.close()

    def writemol(self, f):
        commentblock = ['\n'] * 3
        if 'comment' in self.properties:
            comment = self.properties['comment']
            if isinstance(comment, str):
                commentblock[0] = comment + '\n'
            elif isinstance(comment, list):
                comment = comment[0:3]
                while len(comment) < 3:
                    comment.append('')
                commentblock = [a + b for a, b in zip(comment, commentblock)]
        f.writelines(commentblock)

        self.set_atoms_id()

        f.write('%3i%3i  0  0  0  0  0  0  0  0999 V2000\n' % (len(self.atoms), len(self.bonds)))
        for at in self.atoms:
            f.write('%10.4f%10.4f%10.4f %-3s 0  0  0  0  0  0\n' % (at.x, at.y, at.z, at.symbol))
        for bo in self.bonds:
            order = bo.order
            if order == Bond.AR:
                order = 4
            f.write('%3i%3i%3i  0  0  0\n' % (bo.atom1.id, bo.atom2.id, order))
        self.unset_atoms_id()
        f.write('M  END\n')
        f.close()

    def readmol2(self, f, frame):
        if frame != 1:
            raise MoleculeError('readmol: .mol2 files do not support multiple geometries')

        bondorders = {'1': 1, '2': 2, '3': 3, 'am': 1, 'ar': Bond.AR, 'du': 0, 'un': 1, 'nc': 0}
        mode = ('', 0)
        for i, line in enumerate(f):
            line = line.rstrip()
            if not line:
                continue
            elif line[0] == '#':
                continue
            elif line[0] == '@':
                line = line.partition('>')[2]
                if not line:
                    raise MoleculeError('readmol2: Error in %s line %i: invalid @ record' % (f.name, i + 1))
                mode = (line, i)

            elif mode[0] == 'MOLECULE':
                pos = i - mode[1]
                if pos == 1:
                    self.properties['name'] = line
                elif pos == 3:
                    self.properties['type'] = line
                elif pos == 4:
                    self.properties['charge_type'] = line
                elif pos == 5:
                    self.properties['flags'] = line
                elif pos == 6:
                    self.properties['comment'] = line

            elif mode[0] == 'ATOM':
                spl = line.split()
                if len(spl) < 6:
                    raise MoleculeError(
                        'readmol2: Error in %s line %i: not enough values in line' % (f.name, i + 1))
                symb = spl[5].partition('.')[0]
                try:
                    num = PT.get_atomic_number(symb)
                except PTError:
                    num = 0
                crd = tuple(map(float, spl[2:5]))
                newatom = Atom(atnum=num, coords=crd, name=spl[1], type=spl[5])
                if len(spl) > 6:
                    newatom.properties['subst_id'] = spl[6]
                if len(spl) > 7:
                    newatom.properties['subst_name'] = spl[7]
                if len(spl) > 8:
                    newatom.properties['charge'] = float(spl[8])
                if len(spl) > 9:
                    newatom.properties['flags'] = spl[9]
                self.add_atom(newatom)

            elif mode[0] == 'BOND':
                spl = line.split()
                if len(spl) < 4:
                    raise MoleculeError(
                        'readmol2: Error in %s line %i: not enough values in line' % (f.name, i + 1))
                try:
                    atom1 = self.atoms[int(spl[1]) - 1]
                    atom2 = self.atoms[int(spl[2]) - 1]
                except IndexError:
                    raise MoleculeError('readmol2: Error in %s line %i: wrong atom ID' % (f.name, i + 1))
                newbond = Bond(atom1, atom2, order=bondorders[spl[3]])
                if len(spl) > 4:
                    for flag in spl[4].split('|'):
                        newbond.properties[flag] = True
                self.add_bond(newbond)
        f.close()

    def writemol2(self, f):
        bondorders = ['1', '2', '3', 'ar']

        def write_prop(name, obj, separator, space=0, replacement=None):
            form_str = '%-' + str(space) + 's'
            if name in obj.properties:
                f.write(form_str % str(obj.properties[name]))
            elif replacement is not None:
                f.write(form_str % str(replacement))
            f.write(separator)

        f.write('@<TRIPOS>MOLECULE\n')
        write_prop('name', self, '\n')
        f.write('%i %i\n' % (len(self.atoms), len(self.bonds)))
        write_prop('type', self, '\n')
        write_prop('charge_type', self, '\n')
        write_prop('flags', self, '\n')
        write_prop('comment', self, '\n')

        f.write('\n@<TRIPOS>ATOM\n')
        for i, at in enumerate(self.atoms):
            f.write('%5i ' % (i + 1))
            write_prop('name', at, ' ', 5, at.symbol + str(i + 1))
            f.write('%10.4f %10.4f %10.4f ' % at.coords)
            write_prop('type', at, ' ', 5, at.symbol)
            write_prop('subst_id', at, ' ', 5)
            write_prop('subst_name', at, ' ', 7)
            write_prop('charge', at, ' ', 6)
            write_prop('flags', at, '\n')
            at.id = i + 1

        f.write('\n@<TRIPOS>BOND\n')
        for i, bo in enumerate(self.bonds):
            f.write('%5i %5i %5i %4s' % (i + 1, bo.atom1.id, bo.atom2.id, bondorders[bo.order]))
            write_prop('flags', bo, '\n')

        self.unset_atoms_id()

    def readpdb(self, f, frame):
        pdb = PDBHandler(f)
        models = pdb.get_models()
        if frame > len(models):
            raise MoleculeError('readpdb: There are only %i frames in %s' % (len(models), f.name))

        for i in models[frame - 1]:
            if i.name in ['ATOM  ', 'HETATM']:
                x = float(i.value[0][24:32])
                y = float(i.value[0][32:40])
                z = float(i.value[0][40:48])
                atnum = PT.get_atomic_number(i.value[0][70:72].strip())
                self.add_atom(Atom(atnum=atnum, coords=(x, y, z)))

        return pdb

    def writepdb(self, f):
        pdb = PDBHandler()
        pdb.add_record(PDBRecord('HEADER'))
        model = []
        for i, at in enumerate(self.atoms):
            s = 'ATOM  %5i                   %8.3f%8.3f%8.3f                      %2s  ' % (
                i + 1, at.x, at.y, at.z, at.symbol.upper())
            model.append(PDBRecord(s))
        pdb.add_model(model)
        pdb.add_record(pdb.calc_master())
        pdb.add_record(PDBRecord('END'))
        pdb.write(f)

    def readtmol(self, f, frame=1):
        if frame > 1:
            raise MoleculeError('readtmol: There is only 1 frame in %s' % f.name)

        in_coord_block = False
        for line in f:
            if in_coord_block:
                if line.startswith('$'):
                    break

                lst = line.split()
                if len(lst) != 4:
                    raise MoleculeError('readtmol: error reading tmol file')
                num = PT.get_atomic_number(lst[3])
                self.add_atom(Atom(atnum=num, coords=(lst[0], lst[1], lst[2]), unit='bohr'))
            elif '$coord' in line:
                in_coord_block = True

    def writetmol(self, f):
        ratio = Units.conversion('angstrom', 'bohr')

        f.write('$coord\n')
        for at in self.atoms:
            f.write('%20.14f %20.14f %20.14f %-8s \n' % (at.x * ratio, at.y * ratio, at.z * ratio, at.symbol.lower()))
        f.write('$end\n')

    def read(self, filename, inputformat='xyz', frame=1):
        if inputformat in self._iodict:
            try:
                f = open(filename, 'rU')
            except IOError:
                raise FileError('read: Error reading file %s' % filename)
            ret = self._iodict[inputformat][0](self, f, frame)
            f.close()
            if len(self.bonds) == 0:
                self.guess_bonds()
            return ret
        else:
            raise MoleculeError('read: Unsupported file format')

    def write(self, filename, outputformat='xyz'):
        if outputformat in self._iodict:
            try:
                f = open(filename, 'w')
            except IOError:
                raise FileError('write: Error opening file %s' % filename)
            self._iodict[outputformat][1](self, f)
            f.close()
        else:
            raise MoleculeError('write: Unsupported file format')

    def get_checksum(self, representation='xyz'):
        """
        Get a hexadecimal 128-bit md5 hash of the molecule.

        This method writes a coordinate file, digests it and returns the md5
        checksum of that file. If you think that the representation of the
        molecule matters, you can specify it explicitly via the C{representation}
        flag. Needless to say that you have to use the same representation
        to compare two molecules.

        @param representation: Molecule file format understood by I{Open Babel}.
        @returns:              Hexadicimal hash
        @rtype:                L{str}
        @author:               Moritz Klammler
        @date:                 Aug. 2011

        """

        import os
        import tempfile
        import hashlib

        # First write the  coordinates to a file. The  format obviously doesn't
        # matter as long as it it unambigous and we always use the same. We use
        # Python's  `tempfile' module to  write the  coordinates. This  has the
        # advantage that the method will also succeed if we do not have writing
        # access to the CWD and we don't risk acidently overwriting an existing
        # file. The temporary file will be  unlinked from the OS at the time of
        # disposal of the `tempfile.NamedTemporaryFile' object.

        # We  detect  one source  of  errors by  comparing  the  hash with  the
        # empty-string hash. If it matches, something must have went wrong with
        # writing and re-reading the file.

        m = hashlib.md5()
        emptyhash = m.hexdigest()

        tmp = tempfile.NamedTemporaryFile()
        tmp.file.close()

        # The file is empty now. Note  that we only call the `file' attribute's
        # `close()' method.  Saying `tmp.close()' would  immediately unlink the
        # pysical file which is not what we want.

        self.write(tmp.name, outputformat=representation)

        with open(tmp.name, 'r') as infile:
            for line in infile:
                m.update(line)

        molhash = m.hexdigest()
        if molhash == emptyhash:
            raise PyAdfError("""Error while trying to compute the md5 hash of
            the molecule. Hash equals empty-string hash.""")

        return molhash

    def get_xyz_file(self):
        lines = str(len(self.atoms)) + '\n'
        if 'comment' in self.properties:
            comment = self.properties['comment']
            if isinstance(comment, list):
                comment = comment[0]
            lines += comment
        lines += '\n'
        for at in self.atoms:
            lines += str(at) + '\n'
        return lines

    def get_geovar_atoms_block(self, geovar):
        lines = ''
        geovar = self._get_atoms(geovar)
        for i, at in enumerate(self.atoms):
            if at in geovar:
                if at.ghost:
                    symb = 'Gh.' + at.symbol
                else:
                    symb = at.symbol
                lines += '%5s         atom%ix         atom%iy         atom%iz\n' % (symb, i + 1, i + 1, i + 1)
            else:
                lines += str(at) + '\n'
        return lines

    def get_geovar_block(self, geovar):
        lines = 'GEOVAR\n'
        geovar = zip(geovar, self._get_atoms(geovar))
        for i, at in geovar:
            lines += '  atom%ix  %14.5f\n  atom%iy  %14.5f\n  atom%iz  %14.5f\n' % (i, at.x, i, at.y, i, at.z)
        lines += 'END\n\n'
        return lines

    def get_cube_header(self):
        lines = ''

        # coordinates have to be in Bohr in cube files
        ratio = Units.conversion('angstrom', 'bohr')

        for at in self.atoms:
            if not at.ghost:
                lines += '%5d%12.6f%12.6f%12.6f%12.6f\n' % (at.atnum, 0.0, at.x * ratio, at.y * ratio, at.z * ratio)
        return lines

    def get_dalton_molfile(self, basis):
        lines = 'BASIS\n' + basis + '\nThis Dalton molecule file was generated by PyADF\n' + \
                ' Homepage: http://www.pyadf.org\n'
        types = set([at.atnum for at in self.atoms])
        lines += 'Angstrom Nosymmetry Atomtypes=%d\n' % len(types)
        for tp in types:
            atoms = [at for at in self.atoms if at.atnum == tp]
            lines += 'Charge=%.1f Atoms=%d\n' % (tp, len(atoms))
            for i, at in enumerate(atoms):
                lines += '%-4s %14.5f %14.5f %14.5f \n' % (at.symbol + str(i + 1), at.x, at.y, at.z)
        return lines

    def write_dalton_molfile(self, filename, basis):
        f = open(filename, 'w')
        f.write(self.get_dalton_molfile(basis))
        f.close()

    def get_coordinates(self, atoms=None, ghosts=True):
        atoms = self._get_atoms(atoms)
        # without list() maybe?
        return [list(at.coords) for at in atoms if ghosts or not at.ghost]

    def get_atom_symbols(self, atoms=None, ghosts=True, prefix_ghosts=False):
        atoms = self._get_atoms(atoms)

        def pref(arg):
            return 'Gh.' * int(prefix_ghosts and arg.ghost)

        return [pref(at) + at.symbol for at in atoms if ghosts or not at.ghost]

    def get_atomic_numbers(self, atoms=None, ghosts=True):
        atoms = self._get_atoms(atoms)
        return [at.atnum * int(not at.ghost) for at in atoms if ghosts or not at.ghost]

    def get_mass(self):
        return sum([at.mass for at in self.atoms])

    def get_formula(self):
        atnums = [at.atnum for at in self.atoms]
        s = set(atnums)
        formula = ''
        for i in s:
            formula += PT.get_symbol(i) + str(atnums.count(i))
        return formula

    def get_center_of_mass(self):
        center = [0.0, 0.0, 0.0]
        total_mass = 0.0
        for at in self.atoms:
            mass = at.mass
            total_mass += mass
            for i in range(3):
                center[i] += mass * at.coords[i]
        for i in range(3):
            center[i] /= total_mass
        return tuple(center)

    def distance(self, other):
        dist = 999999
        for at1 in self.atoms:
            for at2 in other.atoms:
                dist = min(dist, (at1.x - at2.x)**2 + (at1.y - at2.y)**2 + (at1.z - at2.z)**2)
        return dist**0.5

    def distance_to_point(self, point, ghosts=True):
        dist = 999999
        for at in self.atoms:
            if ghosts or not at.ghost:
                dist = min(dist, (at.x - point[0])**2 + (at.y - point[1])**2 + (at.z - point[2])**2)
        return dist**0.5

    def separate(self):
        # all returned fragments have default spin, charge, symmetry, props etc.
        frags = []
        clone = copy.deepcopy(self)
        for at in clone.atoms:
            at.visited = False

        def dfs(v, mol):
            v.visited = True
            v.mol = mol
            for e in v.bonds:
                e.mol = mol
                u = e.other_end(v)
                if not u.visited:
                    dfs(u, mol)

        for src in clone.atoms:
            if not src.visited:
                m = OBFreeMolecule()
                dfs(src, m)
                frags.append(m)

        for at in clone.atoms:
            del at.visited
            at.mol.atoms.append(at)
        for b in clone.bonds:
            b.mol.bonds.append(b)

        del clone
        return frags

    def translate(self, vec, unit='angstrom'):
        for at in self.atoms:
            at.move_by(vec, unit)

    def rotate(self, rotmat):
        import numpy
        rotmat = numpy.array(rotmat).reshape(3, 3)
        for at in self.atoms:
            at.coords = tuple(numpy.dot(rotmat, numpy.array(at.coords)))

    def align(self, other, atoms, atoms_other=None):
        import numpy

        def quaternion_fit(coords_r, coords_f):
            # this function is based on the algorithm described in
            # Molecular Simulation 7, 113-119 (1991)

            x = numpy.zeros((3, 3))
            for r, f in zip(coords_r, coords_f):
                x = x + numpy.outer(f, r)

            c = numpy.zeros((4, 4))

            c[0, 0] = x[0, 0] + x[1, 1] + x[2, 2]
            c[1, 1] = x[0, 0] - x[1, 1] - x[2, 2]
            c[2, 2] = x[1, 1] - x[2, 2] - x[0, 0]
            c[3, 3] = x[2, 2] - x[0, 0] - x[1, 1]

            c[1, 0] = x[2, 1] - x[1, 2]
            c[2, 0] = x[0, 2] - x[2, 0]
            c[3, 0] = x[1, 0] - x[0, 1]

            c[0, 1] = x[2, 1] - x[1, 2]
            c[2, 1] = x[0, 1] + x[1, 0]
            c[3, 1] = x[2, 0] + x[0, 2]

            c[0, 2] = x[0, 2] - x[2, 0]
            c[1, 2] = x[0, 1] + x[1, 0]
            c[3, 2] = x[1, 2] + x[2, 1]

            c[0, 3] = x[1, 0] - x[0, 1]
            c[1, 3] = x[2, 0] + x[0, 2]
            c[2, 3] = x[1, 2] + x[2, 1]

            # diagonalize c
            d, v = numpy.linalg.eig(c)

            # extract the desired quaternion
            q = v[:, d.argmax()]

            # generate the rotation matrix

            u = numpy.zeros((3, 3))
            u[0, 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
            u[1, 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
            u[2, 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]

            u[1, 0] = 2.0 * (q[1] * q[2] - q[0] * q[3])
            u[2, 0] = 2.0 * (q[1] * q[3] + q[0] * q[2])

            u[0, 1] = 2.0 * (q[2] * q[1] + q[0] * q[3])
            u[2, 1] = 2.0 * (q[2] * q[3] - q[0] * q[1])

            u[0, 2] = 2.0 * (q[3] * q[1] - q[0] * q[2])
            u[1, 2] = 2.0 * (q[3] * q[2] + q[0] * q[1])

            return u

        frag_mv = self.get_fragment(atoms)
        if atoms_other is None:
            frag_ref = other.get_fragment(atoms)
        else:
            frag_ref = other.get_fragment(atoms_other)

        com_mv = numpy.array(frag_mv.get_center_of_mass())
        com_ref = numpy.array(frag_ref.get_center_of_mass())

        # move both fragments to center of mass
        frag_ref.translate(-com_ref)
        frag_mv.translate(-com_mv)

        rotmat = quaternion_fit(frag_ref.get_coordinates(), frag_mv.get_coordinates())

        transvec = com_ref - numpy.dot(rotmat, com_mv)

        self.rotate(rotmat)
        self.translate(transvec)

        return rotmat, transvec

    def displace_atom(self, atom=None, coordinate=None, displacement=0.01, atomicunits=True):
        # modification of this method would be nice
        if atomicunits:
            unit = 'bohr'
        else:
            unit = 'angstrom'

        if atom is not None:
            mol = copy.deepcopy(self)
            atom = mol._get_atoms([atom])
            vec = (displacement * int(coordinate == 'x'),
                   displacement * int(coordinate == 'y'),
                   displacement * int(coordinate == 'z'))
            atom[0].move_by(vec, unit)
        else:
            raise MoleculeError('displace_atom: no atom given')
        return mol

    def find_adjacent_hydrogens(self, atoms):
        return self.find_adjacent_atoms(atoms, atnum=1)

    def find_adjacent_atoms(self, atoms, atnum=None):
        atoms = self._get_atoms(atoms)
        adjacent = []
        for at in atoms:
            for b in at.bonds:
                adj = b.other_end(at)
                if (atnum is None) or (adj.atnum == atnum):
                    adjacent.append(adj)
        return adjacent

    def get_hetero_hydrogen_list(self):
        hetero_hydrogen_list = []

        for i, at in enumerate(self.atoms):
            if at.atnum == 1:
                for b in at.bonds:
                    if not b.other_end(at).atnum == 6:
                        hetero_hydrogen_list.append(i+1)
                        break

        return hetero_hydrogen_list

    def guess_bonds(self, eff=1.15, addd=0.9):
        from math import floor
        import heapq

        def element(order, ratio, atom1, atom2):
            eford = order
            if order == Bond.AR:
                eford = eff
            if order == 1 and ((atom1.symbol == 'N' and atom2.symbol == 'C')
                               or (atom1.symbol == 'C' and atom2.symbol == 'N')):
                eford = 1.11
            return (eford + addd) * ratio, order, ratio, atom1, atom2

        self.delete_all_bonds()

        dmax = 1.28
        dmax2 = dmax**2
        cubesize = dmax * 2.1 * max([at.radius for at in self.atoms])

        cubes = {}
        for i, at in enumerate(self.atoms):
            at.id = i + 1
            at.free = at.connectors
            at.cube = tuple(map(lambda x: int(floor(x / cubesize)), at.coords))
            if at.cube in cubes:
                cubes[at.cube].append(at)
            else:
                cubes[at.cube] = [at]

        neighbors = {}
        for cube in cubes:
            neighbors[cube] = []
            for i in range(cube[0] - 1, cube[0] + 2):
                for j in range(cube[1] - 1, cube[1] + 2):
                    for k in range(cube[2] - 1, cube[2] + 2):
                        if (i, j, k) in cubes:
                            neighbors[cube] += cubes[(i, j, k)]

        heap = []
        for at1 in self.atoms:
            if at1.free > 0:
                for at2 in neighbors[at1.cube]:
                    if (at2.free > 0) and (at1.id < at2.id):
                        ratio = at1.dist_sqr(at2) / ((at1.radius + at2.radius)**2)
                        if ratio < dmax2:
                            heap.append(element(0, ratio, at1, at2))
                            # I hate to do this, but I guess there's no other way :/ [MH]
                            if at1.atnum == 16 and at2.atnum == 8:
                                at1.free = 6
                            elif at2.atnum == 16 and at1.atnum == 8:
                                at2.free = 6
                            elif at1.atnum == 7:
                                at1.free += 1
                            elif at2.atnum == 7:
                                at2.free += 1
        heapq.heapify(heap)

        for at in filter(lambda x: x.atnum == 7, self.atoms):
            if at.free > 6:
                at.free = 4
            else:
                at.free = 3

        while heap:
            val, o, r, at1, at2 = heapq.heappop(heap)
            step = 0.5
            if o % 2 == 0:
                step = 1
            if at1.free >= step and at2.free >= step:
                o += step
                at1.free -= step
                at2.free -= step
                if o < 3.0:
                    heapq.heappush(heap, element(o, r, at1, at2))
                else:
                    if o == 1.5:
                        o = Bond.AR
                    self.add_bond(Bond(at1, at2, o))
            elif o > 0:
                if o == 1.5:
                    o = Bond.AR
                self.add_bond(Bond(at1, at2, o))

        def dfs(atom, par):
            atom.arom += 1000
            for b in atom.bonds:
                oe = b.other_end(atom)
                if b.is_aromatic() and oe.arom < 1000:
                    if oe.arom > 2:
                        return False
                    if par and oe.arom == 1:
                        b.order = 2
                        return True
                    if dfs(oe, 1 - par):
                        b.order = 1 + par
                        return True

        for at in self.atoms:
            at.arom = len(filter(Bond.is_aromatic, at.bonds))

        for at in self.atoms:
            if at.arom == 1:
                dfs(at, 1)
                pass

        _ret = {}
        for b in self.bonds:
            if b.atom1.id < b.atom2.id:
                _ret[(b.atom1.id, b.atom2.id)] = float(b.order)
            else:
                _ret[(b.atom2.id, b.atom1.id)] = float(b.order)

        for at in self.atoms:
            del at.cube, at.free, at.id, at.arom

        return _ret

    def get_nuclear_dipole_moment(self, atoms=None):
        import numpy
        printsum = (atoms is None)
        atoms = self._get_atoms(atoms)
        nucdip = []
        for at in atoms:
            nucdip.append(numpy.array([at.atnum * at.x * Units.conversion('angstrom', 'bohr'),
                                       at.atnum * at.y * Units.conversion('angstrom', 'bohr'),
                                       at.atnum * at.z * Units.conversion('angstrom', 'bohr')]))
        if printsum:
            return sum(nucdip)
        return nucdip

    def get_nuclear_efield_in_point(self, pointcoord):
        import numpy
        e = [0.0] * 3
        dummy = Atom(coords=tuple(pointcoord))
        for at in self.atoms:
            dist = dummy.distance_to(at)
            vec = dummy.vector_to(at)
            e = [e + (at.atnum * c) / dist**3 for e, c in zip(e, vec)]
        return numpy.array(e) * (Units.conversion('bohr', 'angstrom')**2)

    def get_nuclear_interaction_energy(self, other):
        inten = 0.0
        for at1 in self.atoms:
            for at2 in other.atoms:
                dist = at1.distance_to(at2) * Units.conversion('angstrom', 'bohr')
                inten = inten + at1.atnum * at2.atnum / dist
        return inten

    # Backwards compatibility:
    def _get_atoms(self, atomlist):
        # translate index list into atom list
        if atomlist is None:
            return self.atoms
        else:
            if isinstance(atomlist, list):
                if len(atomlist) > 0 and isinstance(atomlist[0], int):
                    return [self.atoms[i - 1] for i in atomlist]
                else:
                    return atomlist
            else:
                raise MoleculeError('_get_atoms: passed argument is not a list')

    def set_OBMol(self, mol):
        raise MoleculeError('set_OBMol: Openbabel is no longer welcome here')

    def add_hydrogens(self, correctForPH=False, pH=7.4):
        raise MoleculeError("add_hydrogens: I'm not implemented. If you REALLY need me, use OBMolecule")

    # used in 3FDE only, TOGETRIDOF, dirty and slow
    def get_all_bonds(self):
        return [[self.atoms.index(bond.atom1) + 1, self.atoms.index(bond.atom2) + 1] for bond in self.bonds]

    _iodict = {'xyz': (readxyz, writexyz), 'mol': (readmol, writemol), 'mol2': (readmol2, writemol2),
               'pdb': (readpdb, writepdb), 'tmol': (readtmol, writetmol)}
