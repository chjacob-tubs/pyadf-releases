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
 Job and results for ADF NMR calculations.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    adfnmrjob
 @group Results:
   adfnmrresults
"""

import os
import re

from Errors import PyAdfError
from ADFBase import adfjob, adfresults


class adfnmrresults(adfresults):
    """
    Class for the results of a ADF NMR job.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_shielding, get_all_shieldings
    """

    def __init__(self, j=None):
        """
        Constructur for adfnmrresults.
        """
        adfresults.__init__(self, j)

    def get_all_shieldings(self):
        """
        Returns all calculated shieldings.

        @returns:
            A tuple of two lists for nuclei and ghost sites, respectively.
            Each list contains (for each nucleus or ghost site) a tuple of
            three values: total shielding, diamagnetic part and paramagnetic part
        @rtype:    tuple of lists

        @example:
            get_all_shieldings[0][0][0] is the total shielding of the first
            calculated nucleus
        @example:
            get_all_shieldings[0][2][1] is the diamagnetic shielding of the
            third calculated nucleus
        """

        s = []
        for i in self.job.nucs:
            s.append(self.get_shielding(nuc=i))
        g = []
        if self.job.ghosts is not None:
            for i in range(len(self.job.ghosts)):
                g.append(self.get_shielding(ghost=i + 1))

        return s, g

    def get_shielding(self, nuc=None, ghost=None):
        """
        Return the calculated shielding for one nucleus.

        Either the nuc or the ghost option have to be specified (but not both).

        @param nuc:   number of the requested nucleus (numbering in INPUT ORDER)
        @type  nuc:   int
        @param ghost: number of requested ghost site
        @type  ghost: int
        @returns:     tuple of total shielding, diamagnetic part, paramagnetic part
        @rtype:       floats
        """

        # obtain the output
        output = self.get_output()

        # re's needed later on
        totre = re.compile(r"\s*total isotropic shielding =\s*(?P<tot>[-+]?(\d+\.\d+))")
        parare = re.compile(r"\s*total paramagnetic( shielding)? =\s*(?P<para>[-+]?(\d+\.\d+))")
        diare = re.compile(r"\s*total diamagnetic( shielding)? =\s*(?P<dia>[-+]?(\d+\.\d+))")

        isore = re.compile(r"\s{24}isotropic shielding =\s*(?P<iso>[-+]?(\d+\.\d+))")

        if nuc is not None:
            startre = re.compile(r"\*{4}\s*N U C L E U S.*?\(\s*(?P<num>\d+)\)")
            startnum = self.job.adfresults.get_atom_index([nuc])[0]
        elif ghost is not None:
            startre = re.compile(r"\*{4}\s*G H O S T.*?\(\s*(?P<num>\d+)\)")
            startnum = ghost
        else:
            raise PyAdfError('wrong arguments in get_shielding')

        # determine the beginning and end of the relevant section
        start = -1
        end = -1
        counter = 0
        for i, line in enumerate(output):
            m = startre.match(line)
            if m:
                if (start == -1) and (int(m.group('num')) == startnum):
                    start = i
            if (start != -1) and totre.match(line):
                end = i + 1
                break
            elif (start != -1) and isore.match(line):
                counter += 1
            if counter == 3:
                end = i + 1
                break

        # now do the real matching
        counter = 0

        total = 0.0
        para = 0.0
        for line in output[start:end]:

            m = totre.match(line)
            if m:
                total = float(m.group("tot"))
            m = parare.match(line)
            if m:
                para = float(m.group("para"))
            m = diare.match(line)
            if m:
                dia = float(m.group("dia"))
            m = isore.match(line)
            if m:
                counter += 1
            if m and counter == 1:
                para = float(m.group("iso"))
            if m and counter == 2:
                dia = float(m.group("iso"))
            if m and counter == 3:
                total = float(m.group("iso"))

        return total, para, dia


class adfnmrjob(adfjob):
    """
    A job class for ADF NMR shielding calculations.

    See the documentation of L{__init__} for details.

    Corresponding results class: L{adfnmrresults}

    @group Initialization:
        __init__
    """

    def __init__(self, adfres, nucs, ghosts=None, u1k='best', out=None, calc=None, use=None, analysis=None):
        """
        Constructor of ADF NMR job.

        @param adfres: results of the corresponding ADF single point job
        @type  adfres: (subclass of) L{adfsinglepointresults}
        @param nucs:   the nuclei to calculate the shielding for (numbers in INPUT ORDER)
        @type  nucs:   list of int
        @param ghosts: list of coordinates for ghost sites (NICS)
        @type  ghosts: list of float[3]
        @param u1k:    U1K options options, see ADF-NMR documentation of U1K key. 
                       set here to 'best' which if missing gives rubbish when spin-orbit coupling is on
        @type  u1k:    str
        @param out:    output options, see ADF-NMR documentation of OUT key
                       Default in NMR is 'iso', which outputs the isotropic shieldings
        @type  out:    str
        """

        adfjob.__init__(self)

        self.adfresults = adfres
        self.nucs = nucs
        self.u1k = u1k
        self.use = use
        self.analysis = analysis
        self.nmrnucs = adfres.get_atom_index(nucs)

        self.ghosts = ghosts
        self.out = out
        self.calc = calc

    def create_results_instance(self):
        return adfnmrresults(self)

    def get_runscript(self, nproc=1):
        return adfjob.get_runscript(self, nproc=nproc, program='nmr')

    def get_input(self):
        nmrinput = "NMR \n"

        if self.u1k is not None:
            nmrinput += " U1K %s\n" % self.u1k

        if self.use is not None:
            nmrinput += " USE %s\n" % self.use

        if self.out is not None:
            nmrinput += " out %s\n" % self.out
        else:
            nmrinput += " out iso\n"

        if self.calc is not None:
            nmrinput += " calc %s\n" % self.calc
        else:
            nmrinput += " calc all\n"

        if self.analysis is not None:
            nmrinput += " Analysis\n %s\n End\n" % self.analysis
            if self.zora:
                nmrinput += " FakeSO\n"

        nuc = ''
        for n in self.nmrnucs:
            nuc += str(n) + " "
        nmrinput += " nuc " + nuc + "\n"
        if self.ghosts is not None:
            nmrinput += " GHOSTS\n"
            for g in self.ghosts:
                nmrinput += " %14.5f %14.5f %14.5f\n" % tuple(g)
            nmrinput += " SubEnd\n"
        nmrinput += "END \n"

        if self._checksum_only:
            nmrinput += self.adfresults.get_checksum()

        return nmrinput

    def print_jobtype(self):
        return "NMR job"

    def print_jobinfo(self):
        print " " + 50 * "-"
        print " Running " + self.print_jobtype()
        print
        print "   SCF taken from ADF job ", self.adfresults.fileid, " (results id)"
        print
        print "   Shielding will be calculated for nuclei : "
        print self.adfresults.get_molecule().print_coordinates(self.nucs)
        print

        if self.u1k is not None:
            print "   U1K  set to : ", self.u1k
            if self.u1k.lower() == 'all':
                print "     Note: setting U1K to All is not recommended if with ZORA."
        else:
            print "   U1K  set to :  none"
            print "     Note: verify whether this is recommended for the Hamiltonian in use."

        if self.out is not None:
            print "   OUT  set to : ", self.out
        else:
            print "   OUT  set to :  iso"

        if self.calc is not None:
            print "   Calc set to : ", self.out
        else:
            print "   Calc set to :  All"

        if self.ghosts is not None:
            print "   Shielding will be calculated for ghost sites : "
            for i, g in enumerate(self.ghosts):
                print "   %3i) %14.5f %14.5f %14.5f" % (i + 1, g[0], g[1], g[2])
            print

        if self.analysis is not None:
            print "   Analysis is set to : ", self.analysis

    def before_run(self):
        self.adfresults.get_tapes_copy()

        if not os.path.exists('TAPE10'):
            print "   WARNING: TAPE10 was not saved in SCF job"
