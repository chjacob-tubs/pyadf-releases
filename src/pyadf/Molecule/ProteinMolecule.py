# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2020 by Christoph R. Jacob, S. Maya Beyhan,
# Rosa E. Bulo, Andre S. P. Gomes, Andreas Goetz, Michal Handzlik,
# Karin Kiewisch, Moritz Klammler, Lars Ridder, Jetze Sikkema,
# Lucas Visscher, and Mario Wolter.
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
 Defines the L{ProteinMoleculeMixin} class
 that adds protein-related functionality to the Molecule classes.
"""

from ..Errors import PyAdfError

import math
import numpy as np


# TODO: This is very rudimentary and not nice yet - these
#       functions rely on methods that are not implemented
#       in BaseMolecule.
#       We need to revise the Molecule class hierachy
#       to get things organized logically

def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


class ProteinMoleculeMixin(object):

    def __init__(self):
        super(ProteinMoleculeMixin, self).__init__()

    def add_hydrogens_to_sp3_atom(self, atom, bonddist, valence=4):
        coords = np.array(self.get_coordinates([atom])[0])
        neighbors = self.find_adjacent_atoms([atom])

        if len(neighbors) == 3:
            if valence == 3:
                return
            neighbor_coords = np.array(self.get_coordinates(neighbors))
            n1 = normalize_vector(neighbor_coords[0] - coords)
            n2 = normalize_vector(neighbor_coords[1] - coords)
            n3 = normalize_vector(neighbor_coords[2] - coords)

            a = - normalize_vector(n1 + n2 + n3)
            h1 = coords + 1.070 * a

            self.add_atoms(['H'], [h1], bond_to=atom)

        elif len(neighbors) == 2:
            neighbor_atnums = self.get_atomic_numbers()
            if neighbor_atnums[0] > neighbor_atnums[1]:
                neighbors.reverse()
            neighbor_coords = np.array(self.get_coordinates(neighbors))
            n1 = normalize_vector(neighbor_coords[0] - coords)
            n2 = normalize_vector(neighbor_coords[1] - coords)

            a = - normalize_vector(n1 + n2)
            b = normalize_vector(np.cross(n1, n2))

            h1 = normalize_vector(a - math.tan((109.5 / 360.0) * math.pi) * b)
            h1 = coords + bonddist * h1
            self.add_atoms(['H'], [h1], bond_to=atom)

            if valence == 4:
                h2 = normalize_vector(a + math.tan((109.5 / 360.0) * math.pi) * b)
                h2 = coords + bonddist * h2
                self.add_atoms(['H'], [h2], bond_to=atom)
        else:
            raise PyAdfError('Not implemented: add_hydrogens_to_sp3_c requires 2 or 3 neighbors')

    def add_hydrogens_to_sp3_c(self, c_atom):
        return self.add_hydrogens_to_sp3_atom(c_atom, bonddist=1.070)

    def add_hydrogens_to_sp3_n(self, n_atom):
        return self.add_hydrogens_to_sp3_atom(n_atom, bonddist=1.020, valence=3)

    def add_hydrogens_to_sp2_atom(self, atom, bonddist):
        coords = np.array(self.get_coordinates([atom])[0])
        neighbors = self.find_adjacent_atoms([atom])

        if len(neighbors) == 3:
            pass
        elif len(neighbors) == 2:
            neighbor_coords = np.array(self.get_coordinates(neighbors))

            n1 = normalize_vector(neighbor_coords[0] - coords)
            n2 = normalize_vector(neighbor_coords[1] - coords)

            a = - normalize_vector(n1 + n2)
            h1 = coords + bonddist * a

            self.add_atoms(['H'], [h1], bond_to=atom)
        else:
            raise PyAdfError('Not implemented: add_hydrogens_to_sp2_atom requires 2 or 3 neighbors')

    def add_hydrogens_to_sp2_n(self, n_atom):
        return self.add_hydrogens_to_sp2_atom(n_atom, bonddist=0.9845)

    def add_hydrogens_to_sp2_c(self, c_atom):
        return self.add_hydrogens_to_sp2_atom(c_atom, bonddist=1.0320)

    def get_peptide_orientation(self, include_origin=False, align='co'):
        """
        Calculates the orientation of peptide bonds in the molecule. Calculates three
        orthogonal vectors for each petide bond that is found.

        @param include_origin: if True, also return the coordinate vector of the peptide
                               C atom that is the origin of the coordinate system spanned
                               by the three orthogonal vectors
        @type include_origin: bool

        @param align: if 'co', the first vector is aligned with the peptide CO bond,
                      if 'cn', the first vector is aligned with the peptide CN bond
        @type align: str 'co' or str 'cn'

        @returns: 3 or 4 vectors per peptide bond
                  if include_origin == True: for each peptide bond [origin, vec1, vec2, vec3]
                  if include_origin == False: for each peptide bond [vec1, vec2, vec3]
        @rtype:   List of numpy vectors
        """
        orientation = []

        maplist = self.get_smarts_matches('[C;X4;H1,H2][CX3](=O)[NX3][C;X4;H1,H2][CX3](=O)')
        for mp in maplist:
            c_coord = np.array(self.get_coordinates([mp[1]])[0])
            o_coord = np.array(self.get_coordinates([mp[2]])[0])
            n_coord = np.array(self.get_coordinates([mp[3]])[0])

            vec_co = normalize_vector(o_coord - c_coord)
            vec_cn = normalize_vector(n_coord - c_coord)

            if align == 'co':
                vec_align = vec_co
                vec_other = vec_cn
            elif align == 'cn':
                vec_align = vec_cn
                vec_other = vec_co
            else:
                raise PyAdfError("Unsupported align parameter given. Use 'co' or 'cn'")

            vec_c1 = normalize_vector(vec_other - np.dot(vec_cn, vec_co) * vec_align)
            vec_c2 = normalize_vector(np.cross(vec_c1, vec_align))

            if include_origin:
                orientation.append([c_coord, vec_align, vec_c1, vec_c2])
            else:
                orientation.append([vec_align, vec_c1, vec_c2])

        return orientation

    def guess_hydrogen_bonds(self, cut_off=2.2, include_donor_hetero=False,
                             peptide_only=False, include_peptide_carbon=False):
        """
        Tries to find possible hydrogen bonds within a molecule

        @param cut_off:                distance cut off for hydrogen bonds in Angstrom
        @param include_donor_hetero:   additionally returns the hetero atom connected to the hydrogen
        @param include_peptide_carbon: additionally returns the C atom of the peptide group that
                                       is the hydrogen bond donor
        @param peptide_only:           guess only hydrogen bonds between peptide units

        @returns: list of pairs of atom numbers: [donor hetero atom (if requested), peptide carbon atom
                  (if requested), donor hydrogen atom, hydrogen bond acceptor atom]
        @rtype:   list of lists

        """
        h_donors = []
        h_acceptors = []
        h_donors_coords = []
        h_acceptors_coords = []

        if peptide_only:
            maplist = self.get_smarts_matches('[CX3](=O)[NX3]([H])')  # peptide group
            for mp in maplist:
                h_donors.append([mp[2], mp[3], mp[0]])
                h_donors_coords.append(self.get_coordinates([mp[3]])[0])
                h_acceptors.append(mp[1])
                h_acceptors_coords.append(self.get_coordinates([mp[1]])[0])
        else:
            # Any nitrogen or oxygen with at least one attached hydrogen and the hydrogen(s)
            maplist = self.get_smarts_matches('[#7,#8;!H0][#1]')
            for mp in maplist:
                h_donors.append(mp)
                h_donors_coords.append(self.get_coordinates([mp[1]])[0])

            # Any nitrogen or oxygen without hydrogen
            maplist = self.get_smarts_matches('[#7,#8;H0]')
            for mp in maplist:
                h_acceptors.append(mp[0])
                h_acceptors_coords.append(self.get_coordinates(mp)[0])

        h_bonds = []
        for donor, c1 in zip(h_donors, h_donors_coords):
            for acceptor, c2 in zip(h_acceptors, h_acceptors_coords):
                if np.linalg.norm(np.array(c1) - np.array(c2)) < cut_off:
                    h_bond_atoms = []
                    if include_donor_hetero:
                        h_bond_atoms.append(donor[0])
                    if peptide_only and include_peptide_carbon:
                        h_bond_atoms.append(donor[2])
                    h_bonds.append(h_bond_atoms + [donor[1], acceptor])

        return h_bonds

    def get_hbond_orientation(self, peptide_only=False, include_origin=False):
        """
        Calculates the orientation of hydrogen bonds in the molecule. Returns three
        orthogonal vectors defined relative to the hygrogen bond H atom as follows:
        [vector pointing to the hydrogen bond acceptor atom, vector pointing in the direction
        of the bond donor heteoro atom (orthogonal to the first vector), vector perpendicular
        to the first three vectors]

        @param include_origin: if True, also return the coordinate vector of the hydrogen bond
                               H atom that is the origin of the coordinate system spanned
                               by the three orthogonal vectors
        @type include_origin: bool

        @param peptide_only: only include hydrogen bonds between peptide units
        @type peptide_only: bool

        @returns: 3 or 4 vectors per peptide bond
                  if include_origin == True: for each hydrogen bond [origin, vec1, vec2, vec3]
                  if include_origin == False: for each hydrogen bond [vec1, vec2, vec3]
        @rtype:   List of numpy vectors
        """
        orientation = []
        donor_coord = []
        donor_h_coord = []
        acceptor_coord = []

        if peptide_only:
            h_bonds = self.guess_hydrogen_bonds(include_peptide_carbon=True, peptide_only=True)
        else:
            h_bonds = self.guess_hydrogen_bonds(include_donor_hetero=True, peptide_only=False)

        for bond in h_bonds:
            donor_coord.append(np.array(self.get_coordinates([bond[0]])[0]))
            donor_h_coord.append(np.array(self.get_coordinates([bond[1]])[0]))
            acceptor_coord.append(np.array(self.get_coordinates([bond[2]])[0]))

        for i, bond in enumerate(donor_coord):
            vec_h_bond = normalize_vector(acceptor_coord[i] - donor_h_coord[i])
            vec_h_donor = normalize_vector(donor_coord[i] - donor_h_coord[i])

            vec_c1 = normalize_vector(vec_h_donor - np.dot(vec_h_donor, vec_h_bond) * vec_h_bond)
            vec_c2 = normalize_vector(np.cross(vec_c1, vec_h_bond))

            if include_origin:
                orientation.append([donor_h_coord[i], vec_h_bond, vec_c1, vec_c2])
            else:
                orientation.append([vec_h_bond, vec_c1, vec_c2])

        return orientation

    def get_backbone_torsions(self, format='turbomole'):
        """
        gives back a list oft phi and psi torsions of peptides and proteins

        @param format: if 'turbomole' (default) return list of strings in Turbomole internal coordinate
                       format otherwise return tuples of four atom numbers for each torsion
        @type format:  str

        @returns:      torsions in requested format
        @rtype:        list of strings or list of integers
        """
        maplist = self.get_smarts_matches('[NX3][C;X4;H1,H2][CX3](=O)[NX3][C;X4;H1,H2][CX3]')

        tors_list = []
        for mp in maplist:
            psi = (mp[0], mp[1], mp[2], mp[4])
            phi = (mp[2], mp[4], mp[5], mp[6])
            tors_list.append(psi)
            tors_list.append(phi)

        if format == 'turbomole':
            tors_list = ['f tors %i %i %i %i' % tors for tors in tors_list]

        return tors_list
