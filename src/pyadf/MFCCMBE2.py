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
 Job and results for MFCC-MBE(2) calculations.

 @author:       Johannes Vornweg
 @organization: TU Braunschweig

"""

import itertools
from .Errors import PyAdfError
from .BaseJob import metajob, results
from .ADFSinglePoint import adfsinglepointresults
from .ADF_3FDE import cappedfragment, cappedfragmentlist
from .Turbomole import TurbomoleSinglePointResults
from .Orca import OrcaResults


class mfccmbe2results(results):
    """
    Class for MFCCMBE(2) results.
    """

    def __init__(self, job):
        super().__init__(job)
        self.overlap_res_by_comb = {}
        self.nooverlap_res_by_comb = {}
        self.trimer_res_by_comb = {}
        self.fragcap_res_by_comb = {}
        self.capcap_res_by_comb = {}

    def get_energy_function(self, res):
        if isinstance(res, TurbomoleSinglePointResults) or isinstance(res, OrcaResults):
            return res.get_energy()
        elif isinstance(res, adfsinglepointresults):
            return res.get_total_energy()

    def get_fragfrag_energy(self):
        overlapenergy = 0
        for i in self.overlap_res_by_comb:
            energy = self.get_energy_function(self.overlap_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.overlap_res_by_comb[i][1])
                        + self.get_energy_function(self.overlap_res_by_comb[i][2])
                        - self.get_energy_function(self.overlap_res_by_comb[i][3]))
            overlapenergy += energy
        nooverlapenergy = 0
        for i in self.nooverlap_res_by_comb:
            energy = self.get_energy_function(self.nooverlap_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.nooverlap_res_by_comb[i][1])
                        + self.get_energy_function(self.nooverlap_res_by_comb[i][2]))
            nooverlapenergy += energy
        trimerenergy = 0
        for i in self.trimer_res_by_comb:
            energy = self.get_energy_function(self.trimer_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.trimer_res_by_comb[i][1])
                        + self.get_energy_function(self.trimer_res_by_comb[i][2])
                        - self.get_energy_function(self.trimer_res_by_comb[i][3]))
            trimerenergy += energy
        return overlapenergy + nooverlapenergy + trimerenergy

    def get_fragcap_energy(self):
        """
        returns total interaction energy of all fragment-cap combinations
        """
        fragcapenergy = 0
        for i in self.fragcap_res_by_comb:
            energy = self.get_energy_function(self.fragcap_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.fragcap_res_by_comb[i][1])
                        + self.get_energy_function(self.fragcap_res_by_comb[i][2]))
            fragcapenergy += energy
        return fragcapenergy

    def get_capcap_energy(self):
        """
        returns total interaction energy of all cap-cap combinations
        """
        capcapenergy = 0
        for i in self.capcap_res_by_comb:
            energy = self.get_energy_function(self.capcap_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.capcap_res_by_comb[i][1])
                        + self.get_energy_function(self.capcap_res_by_comb[i][2]))
            capcapenergy += energy
        return capcapenergy

    def get_total_interaction_energy(self):
        """
        returns total interaction energy
        """
        fragfragenergy = self.get_fragfrag_energy()
        fragcapenergy = self.get_fragcap_energy()
        capcapenergy = self.get_capcap_energy()
        totalenergy = fragfragenergy - fragcapenergy + capcapenergy
        return totalenergy


class mfccmbe2job(metajob):

    def __init__(self, frags, jobfunc, jobfunc_kwargs=None, caps='mfcc', onlyffterms=False, order=2,
                 cutoff=None):
        """
        Initialize a MFCC-MBE(2) job.

        @param frags: list of capped fragments
        @type  frags: L{cappedfragmentlist}
        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type  jobfunc: function with signature adfsinglepointjob(mol: molecule, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type  jobfunc_kwargs: dict or None
        @param caps: 'mfcc' or 'hydrogen'
        @type  caps: str
        @param onlyffterms: if True only frag-frag terms are calculated
        @type  onlyffterms: Boolean
        @param order: many-body expansion order
        @type  order: int
        @param cutoff: distance cutoff in Angstrom for calculating combinations
        @type  cutoff: int or float

        """
        super().__init__()

        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self._cutoff = cutoff
        self._onlyffterms = onlyffterms
        self._order = order
        self._caps = caps
        self._frags = frags

    @property
    def monomerlist(self):
        """
        returns the list of cappedfragments
        """
        monolist = []
        for frag in self._frags.fragiter():
            monolist.append(frag)
        return monolist

    @property
    def nfrag(self):
        """
        returns the number of cappedfragments
        """
        return len(self.monomerlist)

    @property
    def caplist(self):
        """
        returns the list of caps
        """
        caplist = []
        for cap in self._frags.capiter():
            caplist.append(cap)
        return caplist

    @property
    def ncap(self):
        """
        returns the number of caps
        """
        return len(self.caplist)

    @property
    def nfragcombi(self):
        """
        returns theorethical number of fragment-fragment combinations
        """
        return len(list(itertools.combinations(list(range(self.nfrag)), self._order)))

    @property
    def ncapcombi(self):
        """
        returns theorethical number of cap-cap combinations
        """
        return len(list(itertools.combinations(list(range(self.ncap)), self._order)))

    def create_results_instance(self):
        return mfccmbe2results(self)

    def overlapintencalc(self, dimer, frag1, frag2):
        """
        returns results for every needed part of a nondisjointed dimer (1-2)

        @param dimer: dimer molecule
        @type  dimer: cappedfragment
        @param frag1: fragment 1 of the dimer
        @type  frag1: cappedfragment
        @param frag2: fragment 2 of the dimer
        @type  frag2: cappedfragment
        """
        cap = dimer.get_overlapping_caps()[0]
        dimer_res = self.jobfunc(dimer.mol, **self._jobfunc_kwargs)
        frag1_res = self.jobfunc(frag1.mol, **self._jobfunc_kwargs)
        frag2_res = self.jobfunc(frag2.mol, **self._jobfunc_kwargs)
        cap_res = self.jobfunc(cap.mol, **self._jobfunc_kwargs)
        return dimer_res, frag1_res, frag2_res, cap_res

    def nooverlapintencalc(self, frag1, frag2):
        """
        returns results for every needed part of a disjointed dimer (> 1-3)

        @param frag1: fragment/cap 1 of the dimer
        @type  frag1: cappedfragment
        @param frag2: fragment/cap 2 of the dimer
        @type  frag2: cappedfragment
        """
        dimer = frag1.mol + frag2.mol
        dimer_res = self.jobfunc(dimer, **self._jobfunc_kwargs)
        frag1_res = self.jobfunc(frag1.mol, **self._jobfunc_kwargs)
        frag2_res = self.jobfunc(frag2.mol, **self._jobfunc_kwargs)
        return dimer_res, frag1_res, frag2_res

    def trimerintencalc(self, dimer12, dimer23, trimer, midmonomer):
        """
        returns results for every needed part of a disjointed dimer (1-3)

        @param dimer12: 1-2 dimer
        @type  dimer12: cappedfragment
        @param dimer23: 2-3 dimer
        @type  dimer23: cappedfragment
        @param trimer: trimer
        @type  trimer: cappedfragment
        @param midmonomer: 2 monomer
        @type  midmonomer: cappedfragment
        """
        trimer_res = self.jobfunc(trimer.mol, **self._jobfunc_kwargs)
        dimer12_res = self.jobfunc(dimer12.mol, **self._jobfunc_kwargs)
        dimer23_res = self.jobfunc(dimer23.mol, **self._jobfunc_kwargs)
        midmono_res = self.jobfunc(midmonomer.mol, **self._jobfunc_kwargs)
        return trimer_res, dimer12_res, dimer23_res, midmono_res

    def metarun(self):
        mfccmbe_results = self.create_results_instance()

        # FRAG-FRAG INTERACTIONS
        counter = 1
        print('>  Starting Fragment-Fragment Calculations')
        for c in itertools.combinations(list(range(self.nfrag)), self._order):
            print('>  Fragment-Fragment Combination', counter, 'of', self.nfragcombi)
            print('>  Consisting of Fragments', c[0] + 1, 'and', c[1] + 1, 'of', self.nfrag, 'Fragments')
            monomer1 = self.monomerlist[c[0]]
            monomer2 = self.monomerlist[c[1]]
            dimer = monomer1.merge_fragments(monomer2)
            print('> Charge Fragment ' + str(c[0] + 1) + ':', monomer1.mol.get_charge())
            print('> Charge Fragment ' + str(c[1] + 1) + ':', monomer2.mol.get_charge())
            print('> Charge Dimer:', dimer.mol.get_charge())
            fragfragdist = monomer1.mol.distance(monomer2.mol)
            cutoffbool = False
            if (self._cutoff and fragfragdist <= self._cutoff) or self._cutoff is None:
                cutoffbool = True

            overlapping_caps = dimer.get_overlapping_caps()

            # TOO MANY CAPS
            if len(overlapping_caps) > 1:
                raise PyAdfError("Handeling more than one overlapping cap not implemented yet!")

            # OVERLAP
            elif len(overlapping_caps) == 1 and cutoffbool:
                # frag-frag inten
                dimerres, frag1res, frag2res, capres = self.overlapintencalc(dimer, monomer1, monomer2)
                mfccmbe_results.overlap_res_by_comb[c] = [dimerres, frag1res, frag2res, capres]

            # NO OVERLAP
            elif len(overlapping_caps) == 0 and cutoffbool:
                # frag-frag inten
                if fragfragdist > 0.0:
                    dimerres, frag1res, frag2res = self.nooverlapintencalc(monomer1, monomer2)
                    mfccmbe_results.nooverlap_res_by_comb[c] = [dimerres, frag1res, frag2res]

                # 1-3-Fragment handeling for ACE-NME Caps
                elif fragfragdist == 0.0 and self._caps == 'mfcc':
                    midmonomer = self.monomerlist[c[0] + 1]
                    dimer12 = monomer1.merge_fragments(midmonomer)
                    dimer23 = midmonomer.merge_fragments(monomer2)
                    trimer = dimer12.merge_fragments(monomer2)
                    trimerres, dimer12res, dimer23res, midmonores = \
                        self.trimerintencalc(dimer12, dimer23, trimer, midmonomer)
                    mfccmbe_results.trimer_res_by_comb[c] = [trimerres, dimer12res, dimer23res, midmonores]
            else:
                print('>  Distance between Fragments greater than the cutoff of', self._cutoff, 'Angstrom')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')
            counter += 1

        # FRAG-CAP INTERACTIONS
        if not self._onlyffterms:
            print('>  Starting Fragment-Cap Calculations')
            for i, frag in enumerate(self.monomerlist):
                for j, cap in enumerate(self.caplist):
                    print('>  Fragment', i + 1, 'of', self.nfrag, 'with Cap', j + 1, 'of', self.ncap)
                    fragcapdist = frag.mol.distance(cap.mol)
                    cutoffbool = False
                    if (self._cutoff and fragcapdist <= self._cutoff) or self._cutoff is None:
                        cutoffbool = True

                    if fragcapdist > 0.0 and cutoffbool:
                        fragcapres, fragres, capres = self.nooverlapintencalc(frag, cap)
                        mfccmbe_results.fragcap_res_by_comb[(i, j)] = [fragcapres, fragres, capres]
                    elif fragcapdist == 0.0:
                        print('>  Fragment and Cap are too close')
                        print('>  Skipping Combination')
                        print(' ' + 50 * '-')
                    else:
                        print('>  Distance between Fragment and Cap greater than the cutoff of',
                              self._cutoff, 'Angstrom')
                        print('>  Skipping Combination')
                        print(' ' + 50 * '-')
        else:
            print('>  Not calculating Fragment-Cap-Terms')
            print(' ' + 50 * '-')

        # CAP-CAP INTERACTIONS
        if not self._onlyffterms:
            capcounter = 1
            print('>  Starting Cap-Cap Calculations')
            for c in itertools.combinations(list(range(self.ncap)), self._order):
                print('>  Cap-Cap-combination', capcounter, 'of', self.ncapcombi)
                print('>  Consisting of Caps', c[0] + 1, 'and', c[1] + 1, 'of', self.ncap, 'Caps')
                cap1 = self.caplist[c[0]]
                cap2 = self.caplist[c[1]]
                capcapdist = cap1.mol.distance(cap2.mol)
                cutoffbool = False
                if (self._cutoff and capcapdist <= self._cutoff) or self._cutoff is None:
                    cutoffbool = True

                if capcapdist > 0.0 and cutoffbool:
                    capcapres, cap1res, cap2res = self.nooverlapintencalc(cap1, cap2)
                    mfccmbe_results.capcap_res_by_comb[c] = [capcapres, cap1res, cap2res]
                elif capcapdist == 0.0:
                    print('>  Caps are too close')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')
                else:
                    print('>  Distance between Caps greater than the cutoff of', self._cutoff, 'Angstrom')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')
                capcounter += 1
        else:
            print('>  Not calculating Cap-Cap-Terms')
            print(' ' + 50 * '-')
        return mfccmbe_results


class mfccmbe2interactionresults(results):
    """
    Class for MFCCMBE(2) results.

    """

    def __init__(self, job):
        super().__init__(job)
        self.fragfrag_res_by_comb = {}
        self.frag1cap2_res_by_comb = {}
        self.frag2cap1_res_by_comb = {}
        self.capcap_res_by_comb = {}

    def get_energy_function(self, res):
        if isinstance(res, TurbomoleSinglePointResults) or isinstance(res, OrcaResults):
            return res.get_energy()
        elif isinstance(res, adfsinglepointresults):
            return res.get_total_energy()

    def get_fragfrag_energy(self):
        """
        returns total interaction energy of all fragment-fragment combinations
        """
        fragfragenergy = 0
        for i in self.fragfrag_res_by_comb:
            energy = self.get_energy_function(self.fragfrag_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.fragfrag_res_by_comb[i][1])
                        + self.get_energy_function(self.fragfrag_res_by_comb[i][2]))
            fragfragenergy += energy
        return fragfragenergy

    def get_fragcap_energy(self):
        """
        returns total interaction energy of all fragment-cap combinations
        """
        fragcapenergy = 0
        for i in self.frag1cap2_res_by_comb:
            energy = self.get_energy_function(self.frag1cap2_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.frag1cap2_res_by_comb[i][1])
                        + self.get_energy_function(self.frag1cap2_res_by_comb[i][2]))
            fragcapenergy += energy
        for i in self.frag2cap1_res_by_comb:
            energy = self.get_energy_function(self.frag2cap1_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.frag2cap1_res_by_comb[i][1])
                        + self.get_energy_function(self.frag2cap1_res_by_comb[i][2]))
            fragcapenergy += energy
        return fragcapenergy

    def get_capcap_energy(self):
        """
        returns total interaction energy of all cap-cap combinations
        """
        capcapenergy = 0
        for i in self.capcap_res_by_comb:
            energy = self.get_energy_function(self.capcap_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.capcap_res_by_comb[i][1])
                        + self.get_energy_function(self.capcap_res_by_comb[i][2]))
            capcapenergy += energy
        return capcapenergy

    def get_total_interaction_energy(self):
        """
        returns total interaction energy
        """
        fragfragenergy = self.get_fragfrag_energy()
        fragcapenergy = self.get_fragcap_energy()
        capcapenergy = self.get_capcap_energy()
        totalenergy = fragfragenergy - fragcapenergy + capcapenergy
        return totalenergy

    def print_number_of_calculations(self):
        """
        returns number of calculated combinations
        """
        sumofall = (len(self.fragfrag_res_by_comb) + len(self.frag1cap2_res_by_comb)
                    + len(self.frag2cap1_res_by_comb) + len(self.capcap_res_by_comb))

        print('> n(Frag-Frag): %d' % len(self.fragfrag_res_by_comb))
        print('> n(Frag-Cap):  %d' % (len(self.frag1cap2_res_by_comb) + len(self.frag2cap1_res_by_comb)))
        print('> n(Cap-Cap):   %d' % len(self.capcap_res_by_comb))
        print('>', 20 * '-')
        print('> Sum:          %d' % sumofall)


class mfccmbe2interactionjob(metajob):

    def __init__(self, frags1, frags2, jobfunc, jobfunc_kwargs=None, caps='mfcc', order=2, cutoff=None):
        """
        Initialize a MFCC-MBE(2) job.

        @param frags1: list of capped fragments
        @type  frags1: L{cappedfragmentlist}
        @param frags2: list of capped fragments
        @type  frags2: L{cappedfragmentlist}
        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type  jobfunc: function with signature adfsinglepointjob(mol: molecule, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type  jobfunc_kwargs: dict or None
        @param caps: 'mfcc' or 'hydrogen'
        @type  caps: str
        @param order: many-body expansion order
        @type  order: int
        @param cutoff: distance cutoff in Angstrom for calculating combinations
        @type  cutoff: int or float

        """
        super().__init__()

        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self._cutoff = cutoff
        self._order = order
        self._caps = caps
        self._frags1 = frags1
        self._frags2 = frags2

    def monomerlist(self, fraglist):
        """
        @param fraglist: list of capped fragments
        @type  fraglist: L{cappedfragmentlist}

        returns the list of cappedfragments
        """
        monolist = []
        for frag in fraglist.fragiter():
            monolist.append(frag)
        return monolist

    def caplist(self, fraglist):
        """
        @param fraglist: list of capped fragments
        @type  fraglist: L{cappedfragmentlist}

        returns the list of caps
        """
        caplist = []
        for cap in fraglist.capiter():
            caplist.append(cap)
        return caplist

    def create_results_instance(self):
        return mfccmbe2interactionresults(self)

    def intencalc(self, frag1, frag2):
        """
        @param frag1: fragment/cap 1 of the dimer
        @type  frag1: cappedfragment
        @param frag2: fragment/cap 2 of the dimer
        @type  frag2: cappedfragment
        """
        dimer = frag1.mol + frag2.mol
        dimer_res = self.jobfunc(dimer, **self._jobfunc_kwargs)
        frag1_res = self.jobfunc(frag1.mol, **self._jobfunc_kwargs)
        frag2_res = self.jobfunc(frag2.mol, **self._jobfunc_kwargs)
        return dimer_res, frag1_res, frag2_res

    def fragcapintencalc(self, frags1, frags2):
        """
        @param frags1: fragmentlist of protein 1
        @type  frags1: L{cappedfragmentlist}
        @param frags2: fragmentlist of protein 1
        @type  frags2: L{cappedfragmentlist}
        """
        fragcap_res_by_comb = {}
        for i, frag in enumerate(self.monomerlist(frags1)):
            for j, cap in enumerate(self.caplist(frags2)):
                print('>  Fragment', i + 1, 'of', len(self.monomerlist(frags1)),
                      'with Cap', j + 1, 'of', len(self.caplist(frags2)))
                fragcapdist = frag.mol.distance(cap.mol)
                cutoffbool = False
                if (self._cutoff and fragcapdist <= self._cutoff) or self._cutoff is None:
                    cutoffbool = True

                if fragcapdist > 0.0 and cutoffbool:
                    fragcapres, fragres, capres = self.intencalc(frag, cap)
                    fragcap_res_by_comb[(i, j)] = [fragcapres, fragres, capres]
                elif fragcapdist == 0.0:
                    print('>  WARNING! Fragment and Cap are overlapping!')
                    print(' ' + 50 * '-')
                    raise PyAdfError
                else:
                    print('>  Distance between Fragment and Cap greater than the cutoff of',
                          self._cutoff, 'Angstrom')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')
        return fragcap_res_by_comb

    def metarun(self):
        mfccmbe_results = self.create_results_instance()

        # FRAGMENT-FRAGMENT INTERACTIONS
        print('>  Starting Fragment-Fragment Calculations')
        for i, frag1 in enumerate(self.monomerlist(self._frags1)):
            for j, frag2 in enumerate(self.monomerlist(self._frags2)):
                print('>  Fragment', i + 1, 'of', len(self.monomerlist(self._frags1)),
                      'with Fragment', j + 1, 'of', len(self.monomerlist(self._frags2)))
                fragfragdist = frag1.mol.distance(frag2.mol)
                cutoffbool = False
                if (self._cutoff and fragfragdist <= self._cutoff) or self._cutoff is None:
                    cutoffbool = True

                if fragfragdist > 0.0 and cutoffbool:
                    fragfragres, frag1res, frag2res = self.intencalc(frag1, frag2)
                    mfccmbe_results.fragfrag_res_by_comb[(i, j)] = [fragfragres, frag1res, frag2res]
                elif fragfragdist == 0.0:
                    print('>  WARNING! Fragments are overlapping!')
                    print(' ' + 50 * '-')
                    raise PyAdfError
                else:
                    print('>  Distance between Fragments greater than the cutoff of',
                          self._cutoff, 'Angstrom')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')

        # FRAGMENT-CAP INTERACTIONS
        print('>  Starting Fragment(1)-Cap(2) Calculations')
        frag1cap2res = self.fragcapintencalc(self._frags1, self._frags2)
        print('>  Starting Fragment(2)-Cap(1) Calculations')
        frag2cap1res = self.fragcapintencalc(self._frags2, self._frags1)
        for i in frag1cap2res:
            mfccmbe_results.frag1cap2_res_by_comb[i] = frag1cap2res[i]
        for i in frag2cap1res:
            mfccmbe_results.frag2cap1_res_by_comb[i] = frag2cap1res[i]

        # CAP-CAP INTERACTIONS
        print('>  Starting Cap-Cap Calculations')
        for i, cap1 in enumerate(self.caplist(self._frags1)):
            for j, cap2 in enumerate(self.caplist(self._frags2)):
                print('>  Cap', i + 1, 'of', len(self.caplist(self._frags1)),
                      'with Cap', j + 1, 'of', len(self.caplist(self._frags2)))
                capcapdist = cap1.mol.distance(cap2.mol)
                cutoffbool = False
                if (self._cutoff and capcapdist <= self._cutoff) or self._cutoff is None:
                    cutoffbool = True

                if capcapdist > 0.0 and cutoffbool:
                    capcapres, cap1res, cap2res = self.intencalc(cap1, cap2)
                    mfccmbe_results.capcap_res_by_comb[(i, j)] = [capcapres, cap1res, cap2res]
                elif capcapdist == 0.0:
                    print('>  WARNING! Fragments are overlapping!')
                    print(' ' + 50 * '-')
                    raise PyAdfError
                else:
                    print('>  Distance between Fragments greater than the cutoff of',
                          self._cutoff, 'Angstrom')
                    print('>  Skipping Combination')
                    print(' ' + 50 * '-')

        return mfccmbe_results


class generalmfccresults(results):

    def __init__(self, job, frags=None):
        super().__init__(job)
        self._frags = frags

    def set_fragmentlist(self, frags):
        self._frags = frags

    def get_fragmentlist(self):
        return self._frags

    def get_dipole_vector(self):
        import numpy
        dipole = numpy.zeros(3)
        for f in self._frags.fragiter():
            dipole += f.results.get_dipole_vector()
        for c in self._frags.capiter():
            dipole -= c.results.get_dipole_vector()
        return dipole

    def get_total_energy(self):
        frag_energies = [f.results.get_energy() for f in self._frags.fragiter()]
        cap_energies = [c.results.get_energy() for c in self._frags.capiter()]
        return sum(frag_energies) - sum(cap_energies)


class generalmfccjob(metajob):

    def __init__(self, frags, jobfunc, jobfunc_kwargs=None):
        """
        Initialize a Turbomole or ORCA MFCC job.

        @param frags: the list of MFCC fragments
        @type  frags: L{cappedfragmentlist}
        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type  jobfunc: function with signature adfsinglepointjob(mol: molecule, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type  jobfunc_kwargs: dict or None
        """
        super().__init__()

        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self._frags = frags

    def create_results_instance(self):
        return generalmfccresults(self)

    def get_molecule(self):
        return self._frags.get_total_molecule()

    def metarun(self):
        import copy
        frags = copy.deepcopy(self._frags)

        frags.calculate_all(lambda mol: self.jobfunc(mol, **self._jobfunc_kwargs))
        r = self.create_results_instance()
        r.set_fragmentlist(frags)
        return r


class mfccinteractionresults(results):

    def __init__(self, job):
        super().__init__(job)
        self.frag_res_by_comb = {}
        self.cap_res_by_comb = {}

    def get_energy_function(self, res):
        if isinstance(res, TurbomoleSinglePointResults) or isinstance(res, OrcaResults):
            return res.get_energy()
        elif isinstance(res, adfsinglepointresults):
            return res.get_total_energy()

    def get_frag_interactionenergy(self):
        """
        returns total interaction energy of all fragment-ligand combinations
        """
        fragenergy = 0
        for i in self.frag_res_by_comb:
            energy = self.get_energy_function(self.frag_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.frag_res_by_comb[i][1])
                        + self.get_energy_function(self.frag_res_by_comb[i][2]))
            fragenergy += energy
        return fragenergy

    def get_cap_interactionenergy(self):
        """
        returns total interaction energy of all cap-ligand combinations
        """
        capenergy = 0
        for i in self.cap_res_by_comb:
            energy = self.get_energy_function(self.cap_res_by_comb[i][0]) \
                     - (self.get_energy_function(self.cap_res_by_comb[i][1])
                        + self.get_energy_function(self.cap_res_by_comb[i][2]))
            capenergy += energy
        return capenergy

    def get_total_interaction_energy(self):
        """
        returns total interaction energy
        """
        return self.get_frag_interactionenergy() - self.get_cap_interactionenergy()


class mfccinteractionjob(metajob):

    def __init__(self, frags, ligand, jobfunc, jobfunc_kwargs=None, cutoff=None):
        """
        Initialize a Turbomole or ORCA MFCC Interaction job.

        @param frags: the list of MFCC fragment
        @type  frags: L{cappedfragmentlist}
        @param ligand: ligand molecule
        @type  ligand: molecule
        @param jobfunc: function to perform calculation for one fragment, returning a results object
        @type  jobfunc: function with signature adfsinglepointjob(mol: molecule, **kwargs)
        @param jobfunc_kwargs: kwargs that will be passed to jobfunc
        @type  jobfunc_kwargs: dict or None
        @param cutoff: distance cutoff in Angstrom for calculating combinations
        @type  cutoff: int or float
        """
        super().__init__()

        self.jobfunc = jobfunc
        if jobfunc_kwargs is None:
            self._jobfunc_kwargs = {}
        else:
            self._jobfunc_kwargs = jobfunc_kwargs

        self._frags = frags
        self._ligand = ligand
        self._cutoff = cutoff

    @property
    def fraglist(self):
        """
        returns list of all fragments
        """
        fraglist = []
        for frag in self._frags.fragiter():
            fraglist.append(frag.mol)
        return fraglist

    @property
    def nfrag(self):
        """
        returns number of all fragments
        """
        return len(self.fraglist)

    @property
    def caplist(self):
        """
        returns list of all caps
        """
        caplist = []
        for cap in self._frags.capiter():
            caplist.append(cap.mol)
        return caplist

    @property
    def ncap(self):
        """
        returns number of all fragments
        """
        return len(self.caplist)

    def create_results_instance(self):
        return mfccinteractionresults(self)

    def metarun(self):
        mfcc_inten_results = self.create_results_instance()

        for i, frag in enumerate(self.fraglist):
            print('> Fragment', i + 1, 'of', self.nfrag)
            fragligdist = frag.distance(self._ligand)
            cutoffbool = False
            if (self._cutoff and fragligdist <= self._cutoff) or self._cutoff is None:
                cutoffbool = True

            if fragligdist > 0.0 and cutoffbool:
                totmol = frag + self._ligand
                tot_res = self.jobfunc(totmol, **self._jobfunc_kwargs)
                frag_res = self.jobfunc(frag, **self._jobfunc_kwargs)
                lig_res = self.jobfunc(self._ligand, **self._jobfunc_kwargs)
                mfcc_inten_results.frag_res_by_comb[i] = [tot_res, frag_res, lig_res]
            else:
                print('>  Distance between Fragment and Ligand greater than the cutoff of', self._cutoff, 'Angstrom')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')

        for i, cap in enumerate(self.caplist):
            print('> Cap', i + 1, 'of', self.ncap)
            capligdist = cap.distance(self._ligand)
            cutoffbool = False
            if (self._cutoff and capligdist <= self._cutoff) or self._cutoff is None:
                cutoffbool = True

            if capligdist > 0.0 and cutoffbool:
                totmol = cap + self._ligand
                tot_res = self.jobfunc(totmol, **self._jobfunc_kwargs)
                cap_res = self.jobfunc(cap, **self._jobfunc_kwargs)
                lig_res = self.jobfunc(self._ligand, **self._jobfunc_kwargs)
                mfcc_inten_results.cap_res_by_comb[i] = [tot_res, cap_res, lig_res]
            else:
                print('>  Distance between Cap and Ligand greater than the cutoff of', self._cutoff, 'Angstrom')
                print('>  Skipping Combination')
                print(' ' + 50 * '-')

        return mfcc_inten_results
