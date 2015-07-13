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
 NWChem CC2 excitation energy calculations 

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu
 
 @group Jobs:
    nwchemCC2job
 @group Settings:
    nwchemCC2settings 
 @group Results:
    nwchemCC2results
"""

from NWChem import nwchemsinglepointjob, nwchemsinglepointresults, nwchemsettings

import re

class nwchemCC2results (nwchemsinglepointresults) :
    """
    Class for results of an NWChem CC2 calculation.
    
    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_excitation_energies, get_oscillator_strengths
    """

    def __init__ (self, j=None) :
        """
        Constructor for nwchemCC2results.
        """
        nwchemsinglepointresults.__init__ (self, j)

    def get_excitation_energies (self):
        """
        Returns the CC2 excitation energies (in eV).
        
        @returns: list of the calculated excitation energies, in eV
        @rtype: list of float 
        """  
        exens = []
        output = self.get_output()

        #exen_hartree = re.compile("\s*Excitation energy / hartree\s*=\s*(?P<exenau>[-+]?(\d+(\.\d*)?|\d*\.\d+))")
        exen_eV = re.compile("\s*/ eV\s*=\s*(?P<exeneV>[-+]?(\d+(\.\d*)?|\d*\.\d+))")

        iexci = 1
        root = re.compile(r"\s*Excited state root\s*"+str(iexci))

        for i, l in enumerate(output) :
            m = root.match(l)
            if m:
                #m1 = exen_hartree.match(output[i+1])
                m2 = exen_eV.match(output[i+2])

                exens.append(float(m2.group("exeneV")))

                iexci = iexci + 1
                root = re.compile(r"\s*Excited state root\s*"+str(iexci))

        return exens

    def get_oscillator_strengths (self):
        """
        Returns the CC2 excitation energies (in eV).
        
        @returns: a list of the calculated oszillator strengths (FIZME: units?)
        @rtype:   list of float 
        """
        return None

class nwchemCC2settings (nwchemsettings):
    """
    Class that holds the settings for a NWChem CC2 calculation..
    
    @group Initialization:
        set_nexci, set_freeze
    @group Other Internals:
        __str__
    """
    
    def __init__ (self, nexci=10, freeze_occ=0, freeze_virt=0,):
        """
        Constructor for nwchemCC2settings.

        All arguments are optional, leaving out an argument will choose default settings.
 
        @param nexci: Number of excitations to calculate, see L{set_nexci}.
        @type  nexci: int
        
        @param freeze_occ: number of occupied orbitals to freeze, see L{set_freeze}.
        @type  freeze_occ: int  

        @param freeze_virt: number of virtual orbitals to freeze, see L{set_freeze}.  
        @type  freeze_virt: int  
        """
        nwchemsettings.__init__ (self, method='HF')
       
        self.cc_method = 'CC2' 
        self.nexci = None
        self.freeze_occ = None
        self.freeze_virt = None
        
        self.set_nexci(nexci)
        self.set_freeze(freeze_occ, freeze_virt)
        
    def set_nexci(self, nexci):
        """
        Set the number of excitations to calculated.
        
        @param nexci: the number of excitations to calculate
        @type  nexci: int
        """
        self.nexci = nexci
        
    def set_freeze(self, freeze_occ, freeze_virt):
        """
        Set the number of orbitals to freeze in the CC2 calculation.
        
        @param freeze_occ: number of frozen occupied orbitals
        @type  freeze_occ: int
        @param freeze_virt: number of frozen virtual orbitals
        @type  freeze_virt: int
        """
        self.freeze_occ = freeze_occ
        self.freeze_virt = freeze_virt
        
    def __str__ (self):
        """
        Returns a human-readable description of the settings.
        """
        s =  "  Method: CC2 \n\n"
        s += "  Number of excitations: %i \n" % self.nexci
        s += "  Number of frozen occupied orbitals: %i \n" % self.freeze_occ 
        s += "  Number of frozen virtual orbitals:  %i \n" % self.freeze_virt
        return s


class nwchemCC2job (nwchemsinglepointjob):
    """
    A class for NWChem CC2 excitation energy calculations.
     
    See the documentation of L{__init__} and L{nwchemCC2settings } 
    for details on the available options.
    
    Corresponding results class: L{nwchemCC2results}

    @group Initialization:
        __init__
    @group Input Generation:
        get_cc_block
    @undocumented: _get_nexci
    """

    def __init__ (self, mol, basis, settings=None, fdein=None, options=None) :
        """
        Constructor for NWChem CC2 jobs.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}
        
        @param basis: 
            A string specifying the basis set to use (e.g. C{basis='cc-pVDZ'}).
        @type basis: str
        
        @param settings: The settings for the NWChem CC2 job. 
        @type  settings: L{nwchemCC2settings}
                
        @param fdein: 
            Results of an ADF FDE calculation. The embedding potential from this
            calculation will be imported into NWChem (requires modified NWChem version).        
        @type  fdein: L{adffragmentsresults}        
                
        @param options: 
            Additional options. 
            These will each be included directly in the NWChem input file.
        @type options: list of str
        """

        if settings == None :
            self.settings = nwchemCC2settings()
        else:
            self.settings = settings

        nwchemsinglepointjob.__init__ (self, mol, basis, fdein=fdein,
                                       settings=self.settings, options=options)

    def _get_nexci(self):
        return self.settings.nexci
    nexci = property(_get_nexci, None, None, """
    The number of excitations that were calculated.
    
    @type: int
    """)

    def create_results_instance (self):
        return nwchemCC2results(self)

    def get_tce_block (self):
        block  = "tce\n"
        block += " HF\n" 
        block += " CC2\n" 
        block += " dipole\n" 
        block += " nofock\n" 
        block += " nroots %i \n" % self.settings.nexci
        block += " freeze core %i \n" % self.settings.freeze_occ
        block += " freeze virtual %i \n" % self.settings.freeze_virt
        block += "end\n"
        return block
   
    def get_scftask_block (self):
        return ""

    def get_tcetask_block (self):
        block = "task tce energy\n"
        return block
 
    def get_other_blocks (self):
        blocks = nwchemsinglepointjob.get_other_blocks(self)
        blocks += self.get_tce_block()
        blocks += self.get_tcetask_block()
        return blocks

    def print_jobtype (self):
        return "NWChem Excitations (CC2) job"
