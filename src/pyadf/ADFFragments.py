# This file is part of 
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2012 by Christoph R. Jacob, S. Maya Beyhan,
# Rosa E. Bulo, Andre S. P. Gomes, Andreas Goetz, Karin Kiewisch,
# Jetze Sikkema, and Lucas Visscher 
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
 Job and results for ADF fragment analysis calculations, including NewFDE.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu
 
"""

from ADFSinglePoint import adfsinglepointjob, adfsinglepointresults
from Plot import densfjob
from Errors import PyAdfError

import copy
import kf
import os.path


class fragment (object):
    """
    A class representing a fragment (a fragment type, to be more precise).

    This is used to setup L{adffragmentsjob}s.

    @undocumented: __deepcopy__
    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__
    """

    def __init__ (self, frag_results, mols=None, subfrag=None, isfrozen=False, fdeoptions=None, occ=None, fragoptions=None) :
        """
        Initialize a fragment (or more precisely, a fragment type).

        @param frag_results:
            Results of a previous ADF singlepoint job. If C{None}, this
            fragment is considered to be active, which means that it consists
            of atomic fragments for which fragment files will be prepared
            automatically.
        @type  frag_results: L{adfsinglepointresults}

        @param mols:
            A list of molecules, giving the coordinates where this fragment
            is used. Fragments can be used multiple times, but for each
            molecule it must be possible to rotate the fragment to this
            position.
        @type mols: list of L{molecule}s

        @param subfrag: The name of the subfragment to be used.
        @type  subfrag: str

        @param isfrozen: Specify if this fragment is frozen (FDE)
        @type  isfrozen: bool

        @param fdeoptions:
            A dictionary of FDE options.
            Possible options are: 
            For frozen fragments: USEBASIS, RELAX, XC, DENSTYPE,
            For active fragments: LBDAMP, CAPRADIUS, SCFCONVTHRESH, NOCAPSEPCONV, LBMAXSTEP
            B{Options can be upper, lower or mixed case}
        @type  fdeoptions: dictionary

        @param occ: fragment occupation numbers
        @type occ: list of lists of three elements in the format
                   ['irrep', num_alpha, num_beta]

        @param fragoptions:
            For fragments, possibly a dictionary of adfsettings options
        @type fragoptions: dictionary
        """

        self._frag_results = copy.deepcopy(frag_results)
        if mols:
            self._mols = mols
        elif self._frag_results:
            self._mols = [self._frag_results.get_molecule()]

        if not isinstance(self._mols, list) :
            self._mols = [self._mols]

        self._subfrag      = subfrag

        self._isfrozen     = isfrozen

        if fdeoptions == None :
            self._fdeoptions = {}
        else :
            self._fdeoptions = {}
            for k, v in fdeoptions.iteritems() :
                if k.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE'] :
                    # these options should always be uppercase
                    self._fdeoptions[k.upper()] = v
                else :
                    # all other options are case-sensitive!
                    self._fdeoptions[k] = v
            
        if occ == None :
            self._occ = []
        else :
            self._occ = occ

        if fragoptions == None :
            self._fragoptions = {}
        else :
            self._fragoptions = {}
            for k, v in fragoptions.iteritems() :
                self._fragoptions[k.lower()] = v

        # the name use for this fragment. This is assigned by fragmentlist
        self.fragname = None


    def __deepcopy__ (self, memo) :
        #pylint: disable-msg=W0613

        # make sure that deepcopies are not deeper than needed
        # (this can otherwise use up an incredible amount of time)
        other = copy.copy(self)
        other._fdeoptions = copy.copy(self._fdeoptions)
        other._fragoptions = copy.copy(self._fragoptions)

        return other

    def _set_isfrozen (self, isfrozen) :
        self._isfrozen = isfrozen

    def _get_isfrozen (self) :
        return self._isfrozen

    def _set_results (self, frag_results) :
        self._frag_results = frag_results

    def _get_results (self) :
        return self._frag_results

    isfrozen = property(_get_isfrozen, _set_isfrozen, None,
                        """
                        Whether this is a frozen fragment.
                        """)

    results   = property(_get_results, _set_results, None,
                        """
                        The results associated with this fragment.
                        """)

    def has_frag_results (self):
        return not (self._frag_results == None)

    def get_molecules (self) :
        """
        Return the list of molecules.
        
        @returns: the list of molecules
        @rtype:   list of L{molecule}s
        """
        return self._mols

    def get_total_molecule (self) :
        """
        Return the total molecule.

        A molecule consisting of all the copies of this fragment is returned.

        @returns: the total molecule 
        @rtype:   \L{molecule}s 
        """
        return reduce(lambda x, y: x+y, self._mols)

    def get_fragment_filename (self) :
        """
        Return the name of the fragment file.

        @returns: name of fragment TAPE21 file
        @rtype:   str
        """
        if self._frag_results :
            filename = os.path.basename (self._frag_results.get_tape_filename(21))
        else:
            filename = None
        return filename

    def copy_fragment_file (self) :
        """
        Copies the fragment file to the working directory.
        """

        if self._frag_results :
            filename = os.path.basename (self._frag_results.get_tape_filename(21))
            if not os.path.exists(filename) :
                self._frag_results.link_tape (name=filename)

    def delete_fragment_file_copy (self) :
        """
        Delete the fragment file that was previously copied to the working directory.
        """

        if self._frag_results :
            filename = os.path.basename (self._frag_results.get_tape_filename(21))
            if os.path.exists(filename) :
                os.remove(filename)

    def has_occupations (self) :
        """
        Returns True if fragoccupation data is present.
        """
        return (len(self._occ) > 0)

    def get_num_frags (self) :
        """
        Returns whether this fragment type is only used once.
        """
        return len(self._mols)

    def is_fde_fragment (self) :
        
        if self._frag_results :
            return self._frag_results.is_fde_job()
        else:
            return False

    def set_fdeoptions (self, opts):
        for k, v in opts.iteritems() :
            if k.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE'] :
                self._fdeoptions[k.upper()] = v
            else :
                self._fdeoptions[k] = v

    def has_fdeoption (self, opt) :
        if opt.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE'] :
            return (opt.upper() in self._fdeoptions)
        else :
            return (opt in self._fdeoptions)

    def delete_fdeoption (self, opt) :
        if opt.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE'] :
            if opt.upper() in self._fdeoptions :
                del self._fdeoptions[opt.upper()]
        else :
            if opt in self._fdeoptions :
                del self._fdeoptions[opt]

    def add_fdeoption (self, opt, val) :
        if opt.upper() in ['REALX', 'USEBASIS', 'XC', 'DENSTYPE'] :
            self._fdeoptions[opt.upper()] = val
        else :
            self._fdeoptions[opt] = val

    def set_fragoptions(self, opts):
        for k, v in fragoptions.iteritems() :
            self._fragoptions[k.lower()] = v

    def has_fragoption(self, opt) :
        return (opt.lower() in self._fragoptions)

    def delete_fragoption(self, opt) :
        if opt.lower() in self._fragoptions:
            del self._fragoptions[opt.lower()]

    def add_fragoption(self, opt, val) :
        if opt.lower() in self._frags:
            self._fragoptions[opt.lower()] = val
        else :
            self._fragoptions[opt.lower()] = val
                
    def print_fragment_options (self) :
        """
        Print information about the options used for this fragment.
        """
        if self.isfrozen :
            print " type: frozen FDE fragment"

            FrozenFDEOptions = ""

            for opt, value in self._fdeoptions.iteritems() :
                if (opt not in ['LBdamp', 'CapRadius', 'ScfConvThresh','NoCapSepConv','LBmaxStep', 'FullGrid']) :
                    if isinstance(value, float) :
                        FrozenFDEOptions += "           "+opt+"    "+ "%.2f \n" % (value)
                    if isinstance(value, str) :
                        FrozenFDEOptions += "           "+opt+"    "+ "%s \n" % (value)
            if FrozenFDEOptions != "":
                print "        FDE options: "
                print FrozenFDEOptions
        else :
            print " type: nonfrozen fragment"

            NonFrozenFDEOptions = ""
            for fdeoption in ['LBdamp', 'CapRadius', 'ScfConvThresh','NoCapSepConv','LBmaxStep', 'FullGrid'] :
                if fdeoption in self._fdeoptions :
                    if isinstance(self._fdeoptions[fdeoption], float) :
                        NonFrozenFDEOptions += "           "+fdeoption+"    "+ "%.2f \n" % (self._fdeoptions[fdeoption])
                    if isinstance(self._fdeoptions[fdeoption], str) :
                        NonFrozenFDEOptions += "           "+fdeoption+"    "+ "%s   \n" % (self._fdeoptions[fdeoption])
            if NonFrozenFDEOptions != "":
                print "        FDE options: "
                print NonFrozenFDEOptions
                     
            if (len(self._fragoptions) > 0) :
                print "        Fragment settings : "
            for opt, value in self._fragoptions.iteritems() :
                print "           ", opt, "  ", value
        print

    def get_atoms_block (self) :
        """
        Return the list of atoms in this fragment,
        to be used in the ATOMS block of the ADF input.

        @returns: a string that can be used in the ATOMS block
        @rtype:   str
        """
                
        AtomsBlock = ""

        if self.has_frag_results():
            for num_frag, m in enumerate(self._mols) :
                if len(self._mols) > 1 :
                    suffix = "f="+self.fragname+"/"+str(num_frag+1)
                else :
                    suffix = "f="+self.fragname

                AtomsBlock += m.print_coordinates (index=False, suffix=suffix)
        else :
            for m in self._mols:
                AtomsBlock += m.print_coordinates (index=False)
                                
        return AtomsBlock

    def get_type (self) :
        ftype = ""
        if self.isfrozen :
            ftype = "type=FDE"
        return ftype

    def get_fragments_block (self, checksumonly) :
        """
        Return a line that can be used in the FRAGMENTS block
        of the ADF input for this fragment.
        
        @returns: a string that can be used in the FRAGMENTS block
        @rtype:   str
        """
        block = ""
        if self._frag_results :
            block += "  "+self.fragname+"  "

            if checksumonly :
                block += self._frag_results.get_checksum()
            else:
                block += self.get_fragment_filename()

            if (self.is_fde_fragment() == True) :
                if self._subfrag :
                    block += " subfrag="+self._subfrag
                else:
                    block += " subfrag=active"
            if self.isfrozen :
                block += " " + self.get_type()

                fdeoptions_block = self.get_fdeoptions_block()
                if len(fdeoptions_block) > 0 :
                    block += "  & \n"
                    block += fdeoptions_block
                    block += "  SubEnd"

            block += "\n"

        return block

    def get_fdeoptions_block (self):
        block = ""
        if ('USEBASIS' in self._fdeoptions) or ('RELAX' in self._fdeoptions) or \
               ('DENSTYPE' in self._fdeoptions) or ('XC' in self._fdeoptions) :
            options = ''
            if 'USEBASIS' in self._fdeoptions :
                options += 'USEBASIS '
            if 'RELAX' in self._fdeoptions :
                options += 'RELAX '
            if (len(options) > 0) :
                block += "     fdeoptions "+options+"\n"
            if 'DENSTYPE' in self._fdeoptions :
                block += "     fdedenstype "
                block += self._fdeoptions['DENSTYPE']+"\n"
            if 'XC' in self._fdeoptions :
                block += "     XC "+self._fdeoptions['XC']+"\n"
        return block

    def get_fragoccupations_block (self) :
        block = ''
        if self.has_occupations() :
            block +=  "  %s \n" % (self.fragname)
            for irrep in self._occ :
                block += "   %s %i // %i\n" % (irrep[0], irrep[1], irrep[2])
            block += "  SubEnd \n"

        return block

    def get_special_options_block (self) :
        return ""

    def calculate (self, func, context=None) :
        """
        Calulate the results for this fragment.
        
        @param func: The function that performs the actual calculation.
        @type  func: A function with the signature func(mol, **context), where
                     mol is a L{molecule} and context are additional keyword
                     arguments, returning an instance of L{adfsinglepointresults} 
                     or of a derived class.
        @param context: An optional dictionary of additional keyword arguments to func.
        @type  context: dict
        """
        
        if context == None :
            context = {}
        #pylint: disable-msg=W0142
        self.results = func (self._mols[0], **context)


class fragmentlist (object):
    """
    A class repesenting a list of fragments, as used in L{adffragmentsjob}s.

    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__
    """

    def __init__ (self, frags=None):
        """
        Create a fragmentlist.
        
        @param frags: the list of L{fragment}s or C{None}
        @type frags: list 
        """
        if frags==None :
            self._frags = []
        else :
            self._frags = frags
            for num_ftyp, frag in enumerate(self._frags) :
                frag.fragname = "frag"+str(num_ftyp+1)

    def append (self, f):
        """
        Append a fragment to the list.
        
        @param f: the fragment
        @type f:  L{fragment}
        """
        self._frags.append(copy.copy(f))
        self._frags[-1].fragname = "frag"+str(len(self._frags))
        
    def __iter__ (self):
        """
        Iterator for fragmentlist.
        
        Iteration loops over the list of fragments.
        """
        return self._frags.__iter__()

    def get_frozen_frags (self) :
        """
        Returns a list of the frozen fragments.
        """
        return [f for f in self._frags if (f.isfrozen == True)]

    def get_nonfrozen_frags (self) :
        """
        Returns a list of the nonfrozen fragments.
        """
        return [f for f in self._frags if (f.isfrozen == False)]

    def get_total_molecule (self) :
        """
        Returns the total molecule.
        
        @rtype: L{molecule}
        """
        mols = [frag.get_total_molecule() for frag in self._frags]
        return reduce(lambda x, y: x+y, mols)

    def get_frozen_molecule (self) :
        """
        Returns the frozen part as a molecule.
        
        @rtype: L{molecule}
        """
        mols = [frag.get_total_molecule() for frag in self.get_frozen_frags()]
        return reduce(lambda x, y: x+y, mols)

    def get_nonfrozen_molecule (self) :
        """
        Returns the nonfrozen part as a molecule.
        
        @rtype: L{molecule}
        """
        mols = [frag.get_total_molecule() for frag in self.get_nonfrozen_frags()]
        return reduce(lambda x, y: x+y, mols)

    def get_atomtypes_without_fragfile (self):
        """
        Return a list of the atomtypes for which no fragment file is available.
        
        For these atom types, atomic fragment files (or a BASIS block)
        have top be provided
        """
 
        mols = [frag.get_total_molecule() for frag in self._frags if not frag.has_frag_results()]
        mols = [m for m in mols if m]

        if len(mols) == 0 :
            atomtypes = []
        else:
            mol = reduce(lambda x, y: x+y, mols)
            atomtypes = mol.get_atom_symbols(prefix_ghosts=True)
       
        return atomtypes
 
    def has_fde_fragments (self) :
        """
        Whether there is at least one fragment obtained from an FDE calculation in the list.
        
        This FDE fragment does not have to be frozen. The distiction refers to the
        way the fragment was calculated previously.
        """
        return reduce(lambda x, y: x or y, [f.is_fde_fragment() for f in self._frags])

    def has_fragoccupations (self) :
        """
        Whether fragment occupations have been specified for at least one fragment in the list.
        """
        return reduce(lambda x, y: x or y, [f.has_occupations() for f in self._frags])

    def get_num_frozen_frags (self) :
        """
        Returns the number of frozen fragments.
        """
        numfrozenfrags = 0
        for f in self._frags :
            if f.isfrozen :
                numfrozenfrags += f.get_num_frags()
        return numfrozenfrags
 
    def has_frozen_fragment (self) :
        """
        Whether there is at least one frozen fragment in the list.
        """
        return reduce(lambda x, y: x or y, [f.isfrozen for f in self._frags])

    def has_relax_fragment (self) :
        """
        Whether there is at least one frozen fragment with RELAX option in the list.
        """
        fderelax = False
        for f in self._frags :
            fderelax = fderelax or (f.isfrozen and f.has_fdeoption('RELAX'))
        return fderelax

    def get_atoms_block (self) :
        AtomsBlock = ""
        for frag in self.__iter__() :
            AtomsBlock += frag.get_atoms_block()
        return AtomsBlock 

    def get_fragments_block (self, checksum_only) :
        
        block = ""
        for frag in self.__iter__() :
            if frag.has_frag_results() :
                block += frag.get_fragments_block(checksum_only)
        return block

    def get_fragoccupations_block (self) :
        block = ""        
        for frag in self.__iter__() :
            block += frag.get_fragoccupations_block()
        return block

    def get_special_options_block (self) :
        block = ""
        for frag in self.__iter__() :
            block += frag.get_special_options_block()
        return block

    def copy_fragment_files (self) :
        """
        Copy all fragment files to the current working directory.
        """
        for f in self.__iter__() :
            f.copy_fragment_file ()
            
    def delete_fragment_files_copy (self):
        """
        Delete copies of fragment files in the current working directory.
        """
        for f in self.__iter__() :
            f.delete_fragment_file_copy ()

    def calculate_all (self, func, context=None) :
        """
        Loop over all fragments and calculated the results for the fragments.
        
        @param func: The function that performs the actual calculation.
        @type  func: A function with the signature func(mol, **context), where
                     mol is a L{molecule} and context are additional keyword
                     arguments, returning an instance of L{adfsinglepointresults} 
                     or of a derived class.
        @param context: An optional dictionary of additional keyword arguments to func.
        @type  context: dict
        """
        for f in self.__iter__() :
            f.calculate(func, context)


class adffragmentsresults (adfsinglepointresults) :
    """
    Results of a L{adffragmentsjob}.
    
    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__

    """

    def __init__ (self, j=None) :
        #pylint: disable-msg=W0231
        adfsinglepointresults.__init__ (self, j)

    def get_subfragments (self) :
        """
        Returns a list of the subfragments of this calculation.

        @returns: A list with the subfragments used in this calculation
        @rtype:   list of L{fragment}s
        """

        frags = []
        frags.append(fragment(None, self.job.get_nonfrozen_molecule(), subfrag="active"))

        return frags

    def export_embedding_data (self, filename_potential='embpot_adfgrid', filename_density='frzdens_adfgrid'):
        """
        Export the embedding potential and optionally frozen densities and gradients to 
        a text file (for use with Dalton/Dirac)
        
        The embedding potentail will be, together with the integration grid
        and weights, written to a text file with the following format:
        First line: number of points
        Following lines: x,y,z coordinates of the grid point, w in this point, v_emb in this point
        Last line: -42

        The density and gradient will be given in a separate file, that also contains
        the integration grid and weights, in the following format 
        First line: number of points, number of properties exported (densities, gradients)
        Following lines: x,y,z coordinates of the grid point, w in this point, density(a+b), grd.grd (averaged) in this point
        TODO: in the case of unrestricted calculations, see whether to export 
        density_a, density_b, grd_a.grd_a, grd_b.grd_b, grd_a.grd_b in 
        this point separately (that is, without generating the total ones)
        Last line: -42
        
        @param filename_potential: name of the file where the potential will be written
        @type filename_potential:  str
        @param filename_density: name of the file where the potential will be written
        @type filename_density:  str
        """

        if (not self.job.is_fde_job()) :
            raise PyAdfError('export_embedding_data called, but calculation is no FDE job')

        t10  = kf.kffile(self.get_tape_filename(10))
        t21  = kf.kffile(self.get_tape_filename(21))
        nspin = t21.read('General', 'nspin')
        t21.close()

        vemb_outfile = open(filename_potential, 'w')
        dens_outfile = open(filename_density, 'w')

        nblock  = t10.read('Points', 'nblock')
        npoints = t10.read('Points', 'lblock')

        vemb_outfile.write("%d \n" % (nblock*npoints))

        exported_properties = 6 # exported_properties reflects the number of quantities, 
                                # apart from the grid and weights, we are exporting
                                #
                                # this number is related to the following at th moment 
                                # (columns mean the columns after the grid point coordinates 
                                # and weights):
                                #
                                # columns 5-6: electrostatic potentials (5:coulomb+nuclear, 
                                #              6:point charges) of frozen subsystems/surroundings
                                # columns 6-9: 6:frozen density and  7:x, 8:y, 9:z frozen density gradient components

        dens_outfile.write("%d     %d\n" % (nblock*npoints,exported_properties))  

        PointsData = t10.read('Points', 'Data')
        elpotFD    = t10.read('FrozenDensityElpot', 'ElpotFD') # andre: as far as i see, v_nuc is always together with v_H
        efieldpot  = t10.read('Efield', 'Efield')
        xckinpotFD = t10.read('FrozenDensityXcKinpot', 'XcKinPotFD')

        frzdens    = t10.read('FrozenDensity', 'rhoffd')
        frzdensgrd = t10.read('FrozenDensityFirstDer', 'drhoffd')

        vemb_points_written = 0
        dens_points_written = 0

        for iblock in range(1, nblock+1) :
            ipoint = npoints*(iblock-1)

            coords  = PointsData[ipoint*4:ipoint*4+3*npoints]
            coords  = coords.reshape((npoints, 3), order='Fortran')
            weights = PointsData[ipoint*4+3*npoints:ipoint*4+4*npoints]
            
            potel   = elpotFD[ipoint:ipoint+npoints]
            potefld = efieldpot[ipoint:ipoint+npoints]
            potxc   = xckinpotFD[ipoint*nspin:ipoint*nspin+npoints*nspin]
            
            dens    = frzdens[ipoint*nspin:(ipoint+npoints)*nspin]
            densgrd = frzdensgrd[ipoint*nspin*3:(ipoint+npoints)*nspin*3]
            densgrd = densgrd.reshape((npoints,3,nspin), order='Fortran')

            if nspin == 2 :
                potxc = potxc.reshape((npoints, nspin), order='Fortran')
                potxc = 0.5 * (potxc[:,0] + potxc[:,1])
                # we export the total density now, so we sum alpha and beta components
                dens  = dens.reshape((npoints, nspin), order='Fortran')
                dens  = dens[:,0] + dens[:,1] 
                # and we also average the gradients..? (todo)
                
            
            for c, w, e, ef, x in zip(coords, weights, potel, potefld, potxc) :
                vemb_outfile.write("%25.18e  %25.18e  %25.18e  %25.18e %25.18e \n" % (c[0],c[1],c[2],w,(e+ef+x)))
                vemb_points_written += 1

            for c, w, e, ef, n, dg in zip(coords, weights, potel, potefld, dens, densgrd) :
                dens_outfile.write(" %25.18e   %25.18e   %25.18e   %25.18e       %25.18e   %25.18e      %25.18e   %25.18e  %25.18e  %25.18e\n" % (c[0],c[1],c[2],w,e,ef,n,dg[0],dg[1],dg[2]))
                dens_points_written += 1


        vemb_outfile.write("-42\n")
        dens_outfile.write("-42\n")

        if not (vemb_points_written == nblock*npoints):
            raise PyAdfError('wrong number of points written for embedding potential export')
        if not (dens_points_written == nblock*npoints):
            raise PyAdfError('wrong number of points written for frozen density export')

        vemb_outfile.close()
        dens_outfile.close()
        t10.close()

    def get_nonfrozen_molecule (self) :
        return self.job.get_nonfrozen_molecule()

    def get_frozen_molecule (self) :
        return self.job.get_frozen_molecule()

    def get_nonfrozen_density (self, grid=None, spacing=0.5, fit=False) :
        """
        Returns the electron density of the nonfrozen (active) subsystem.
        
        For details on the processing of the electron density,
        e.g., for plotting, see L{Plot}.
        
        @param grid: The grid to use. For details, see L{Plot}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the fit density is returned, otherwise
                    the exact density.
        @type  fit: bool
        """
        if grid == None :
            from Plot import cubegrid
            grid = cubegrid(self.job.get_molecule(), spacing)
        
        if fit == False :
            return densfjob( self, 'Density SCF', grid=grid, frag='Active').run()
        else:
            return densfjob( self, 'Density Fit SCF', grid=grid, frag='Active').run()
    
    def get_frozen_density (self, grid=None, spacing=0.5, fit=False) :
        """
        Returns the frozen electron density.
        
        For details on the processing of the electron density,
        e.g., for plotting, see L{Plot}.
        
        @param grid: The grid to use. For details, see L{Plot}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the fit density is returned, otherwise
                    the exact density.
        @type  fit: bool
        """
        if fit == False :
            prop = 'Density SCF'
        else :
            prop = 'Density Fit SCF'
            
        if grid == None :
            from Plot import cubegrid
            grid = cubegrid(self.job.get_molecule(), spacing)
        
        frozendens = []
        for f in self.job.get_fragmentlist().get_frozen_frags() :
            dens = densfjob( self, prop, grid=grid, frag=f.fragname).run()
            frozendens.append(dens)

        frozendens = reduce(lambda x,y: x+y, frozendens)

        return frozendens
    
    def get_density (self, grid=None, spacing=0.5, fit=False):
        """
        Returns the electron density.
        
        For details on the processing of the electron density,
        e.g., for plotting, see L{Plot}.
        
        @param grid: The grid to use. For details, see L{Plot}.
        @type  grid: subclass of L{grid}
        @param fit: If True, the fit density is returned, otherwise
                    the exact density.
        @type  fit: bool
        """
        if grid == None :
            from Plot import cubegrid
            grid = cubegrid(self.job.get_molecule(), spacing)
        
        if not self.job.is_fde_job() :
            return self.get_nonfrozen_density(grid=grid, fit=fit) 
        else:
            return self.get_nonfrozen_density(grid=grid, fit=fit) + \
                   self.get_frozen_density(grid=grid, fit=fit)

    def get_embedding_potential (self, grid=None, spacing=0.5, pot="total") :

        if grid == None :
            from Plot import cubegrid
            grid = cubegrid(self.job.get_molecule(), spacing)
        
        if not self.job.is_fde_job() :
            raise PyAdfError("Can get embedding potenial only for FDE jobs")

        if pot.lower() == "total" :
            return densfjob(self, "embpot", grid=grid, frag="ALL").run()
        elif pot.lower().startswith("kinpot") :
            s1, s2 = pot.split(None, 1)
            return densfjob(self, "nadkin "+s2, grid=grid, frag="ALL").run()
            
    def get_frozen_potential(self, grid=None, spacing=0.5, pot="total") :
        """
        Returns the potential belonging to the frozen electron density.
        
        @param grid: The grid to use. For details, see L{Plot}.
        @type  grid: subclass of L{grid}
        """

        if grid == None :
            from Plot import cubegrid
            grid = cubegrid(self.job.get_molecule(), spacing)

        if not self.job.is_fde_job() :
            raise PyAdfError("Can get frozen potential only for FDE jobs")
        
        frozenpot = []
        for f in self.job.get_fragmentlist().get_frozen_frags() :
            afrozenpot = densfjob( self, 'Potential '+ pot.lower(), grid=grid, frag=f.fragname).run()
            frozenpot.append(afrozenpot)

        frozenpot = reduce(lambda x,y: x+y, frozenpot)

        return frozenpot

    def get_nonfrozen_potential(self, grid=None, spacing=0.5, pot="total" ) :
        """
        Returns the potential of the electron density of the nonfrozen (active) subsystem.
        
        @param grid: The grid to use. For details, see L{Plot}.
        @type  grid: subclass of L{grid}
        """
        if grid == None :
            from Plot import cubegrid
            grid = cubegrid(self.job.get_molecule(), spacing)
        
        return densfjob( self, 'Potential '+pot.lower(), grid=grid, frag='Active').run()


class adffragmentsjob (adfsinglepointjob) :
    """
    ADF fragments analysis job.

    @undocumented: __delattr__, __getattribute__, __hash__, __new__, __reduce__,
                   __reduce_ex__, __repr__, __str__, __setattr__
    """

    def __init__ (self, fragments, basis=None, settings=None, core=None, 
                  fde=None, pointcharges=None, fitbas=None, options=None) :
        """
        Initialize ADF fragments analysis / FDE job.

        @param fragments:
            the list of fragments to be used in this job.
        @type fragments: list of L{fragment}s or L{fragmentlist}
            
        @param fde:
            a dictionary with options for FDE. They will be added in the FDE
            block of the input.
            The functional for the non-additive kinetic energy can be specified
            under the key TNAD. If is is not specified, PW91k will be
            chosen as default.
        @type fde: dict
        
        """

        if isinstance(fragments, list) :
            self._fragments = fragmentlist(fragments)
        else:
            self._fragments = fragments

        if fde == None :
            self._fde = {}
        else :
            self._fde = {}
            for k, v in fde.iteritems() :
                self._fde[k.upper()] = v

        adfsinglepointjob.__init__ (self, None, basis, core=core, settings=settings, 
                                    pointcharges=pointcharges, fitbas=fitbas, options=options)

        if self._fragments.has_fde_fragments() :
            if not 'ALLOW PARTIALSUPERFRAGS' in self._options :
                self._options.append('ALLOW PARTIALSUPERFRAGS')

    def create_results_instance (self):
        return adffragmentsresults(self)

    def is_fde_job (self) :
        return self._fragments.has_frozen_fragment()

    def is_fde_relax_job (self) :
        return self._fragments.has_relax_fragment()

    def get_fragmentlist (self):
        return self._fragments

    def get_molecule (self):
        return self._fragments.get_total_molecule()
 
    def get_frozen_molecule (self):
        return self._fragments.get_frozen_molecule()
 
    def get_nonfrozen_molecule (self):
        return self._fragments.get_nonfrozen_molecule()

    def get_atomtypes_without_fragfile (self) :
        return self._fragments.get_atomtypes_without_fragfile()
 
    def print_jobtype (self) :
                
        if (self.is_fde_job()) :
            jobtype = "ADF NewFDE job"
        else:
            jobtype = "ADF fragments analysis job"
                        
        return jobtype

    def print_molecule (self) :

        print "   Total Molecule"
        print "   =============="
        print
        print self.get_molecule()
        print
        print "   Fragments "
        print "   =========="
        print
        for num_ftyp, frag in enumerate(self._fragments) :
            print "     Fragment Typ ", num_ftyp+1, "  ", 
            frag.print_fragment_options()
            for num_frag, m in enumerate(frag.get_molecules()) :
                print "     Fragment Typ ", num_ftyp+1, ", Fragment ", num_frag+1
                print m
        print

        print "   Fragment Files "
        print "   ============== "
        print
        for num_ftyp, frag in enumerate(self._fragments) :
            filename = frag.get_fragment_filename()
            if filename:
                print "     Fragment Typ ", num_ftyp+1, ": ", filename

        print

    def print_settings (self) :
        
        print "   Settings"
        print "   ========"
        print
        print self.settings
        print

        if self.is_fde_job() :
            print "   FDE settings"
            print "   ============"
            print
            if not 'TNAD' in self._fde :
                print '   TNAD  PW91k'
            for k, v in self._fde.iteritems() :
                print '   %s  %s' % (k, v)
            print

    def get_atoms_block (self) :
        """
        give back the atoms input key for ADF input files
        (plus UNITS and CHARGE keys)
        """
        AtomsBlock  = ""
        AtomsBlock += " ATOMS\n"
        AtomsBlock += self._fragments.get_atoms_block()
        AtomsBlock += " END\n\n"
        return AtomsBlock

    def get_charge_block (self):
        block = ""
        if self._fragments.get_nonfrozen_molecule().mol.HasSpinMultiplicityAssigned():
            block += " CHARGE %2i %2i \n\n" % (self._fragments.get_nonfrozen_molecule().get_charge(),
                        self._fragments.get_nonfrozen_molecule().get_spin())
        else:
            block += " CHARGE %2i \n\n" % self._fragments.get_nonfrozen_molecule().get_charge()
        return block

    def get_fragments_block (self) :
        block = " FRAGMENTS\n"
        block += self._fragments.get_fragments_block(self._checksum_only)
        block += " END\n\n"
        return block

    def get_fragoccupations_block (self) :
        block = ""        
        block += " FRAGOCCUPATIONS\n"
        block += self._fragments.get_fragoccupations_block()
        block += " END\n\n"        
        return block

    def get_other_blocks (self) :
        block =  adfsinglepointjob.get_other_blocks(self)
        block += self._fragments.get_special_options_block() 
        if self.is_fde_job() :
            block += self.get_fde_block()
        if self._fragments.has_fragoccupations() :
            block += self.get_fragoccupations_block()
        return block

    def get_fde_block (self) :
        block  = " FDE\n"
        if 'TNAD' in self._fde :
            block += '   ' + self._fde['TNAD'] + '\n'
        else :
            block += "   PW91k\n"
        for opt, val in self._fde.iteritems() :
            if opt == 'TNAD' :
                continue
            block += "   "+opt+" "+str(val)+"\n"
        block += " END\n\n"
        return block

    def before_run (self):
        adfsinglepointjob.before_run(self)
        self._fragments.copy_fragment_files()
        
    def after_run (self):
        adfsinglepointjob.after_run(self)
        self._fragments.delete_fragment_files_copy ()
