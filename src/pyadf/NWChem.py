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
 The basics needed for NWChem calculations 

 @author:       Lucas Visscher and others
 @organization: VU University Amsterdam 

 @group Jobs:
    nwchemjob, nwchemsinglepointjob
 @group Settings:
    nwchemsettings
 @group Results:
    nwchemresults, nwchemsinglepointresults  
"""

from Errors import PyAdfError
from BaseJob import results, job
import os, re

class nwchemresults (results) :
    """
    Class for results of an NWChem calculation.
    """

    def __init__ (self, j=None) :
        """
        Constructor for nwchemresults.
        """
        results.__init__ (self, j)

class nwchemsinglepointresults (nwchemresults) :
    """
    Class for results of an NWChem single point calculation.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_molecule, get_energy
    @undocumented: _get_fdein
    """

    def __init__ (self, j=None) :
        """
        Constructor for nwchemsinglepointresults.
        """
        nwchemresults.__init__ (self, j)

    def get_molecule (self) :
        """
        Return the molecular geometry after the NWChem job.
        
        @returns: The molecular geometry.
        @rtype:   L{molecule}

        @note: currently not implemented
        """
        pass

    def get_dipole_vector (self) :
        """
        Return the dipole moment vector.

        @returns: the dipole moment vector, in atomic units
        @rtype: float[3]
        """
        
        dipole = [0.0, 0.0, 0.0]
        
        output = self.get_output()
        
        start = re.compile("\s*Dipole Moment")
        for i, l in enumerate(output) :
            m = start.match(l)
            if m:
                startline = i

        for i, c in enumerate(['X', 'Y', 'Z']) :
            dip = re.compile("\s*DM"+c+"\s*(?P<dip>[-+]?(\d+(\.\d*)?|\d*\.\d+))\s+DM"+c+"EFC")
            m = dip.match(output[startline+7+i])
            dipole[i] = float(m.group('dip'))
       
        return dipole

    def get_energy(self):
        """
        Return the total energy

        @returns: the total energy in atomic units
        @rtype: float
        """

        energy = 0.0

        if not (self.job.settings.scf_method.upper() in ['HF', 'DFT']):
            raise PyAdfError('Energy only implemented for HF and DFT')

        output = self.get_output()
        en_re = re.compile("^ +Total (SCF|DFT) energy = *(?P<energy>-?\d+\.\d+)")
        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group("energy"))
                break
        return energy

    def _get_fdein (self):
        return self.job.fdein
    fdein = property(_get_fdein, None, None, """
    The results of the ADF FDE calculation from that the embedding potential was imported.
    
    @type: L{adffragmentsresults}
    """)


class nwchemjob (job) :
    """
    An abstract base class for NWChem jobs.
    
    Corresponding results class: L{nwchemresults}

    @group Initialization:
        __init__
    @group Running Internals:
        get_nwchemfile
    """

    def __init__ (self) :
        """
        Constructor for NWChem jobs.
        """
        job.__init__ (self)
        self._checksum_only = False

    def create_results_instance (self):
        return nwchemresults(self)

    def print_jobtype (self) :
        pass

    def get_nwchemfile (self) :
        """
        Abstract method. Should be overwritten to return the NWChem input file.
        """
        return ""

    def get_checksum (self) :
        import hashlib
        m = hashlib.md5()
        
        self._checksum_only = True
        m.update(self.get_nwchemfile())
        self._checksum_only = False

        return m.digest()

    def get_runscript (self) :
        runscript  = "#!/bin/bash \n\n"

        runscript += "cat <<eor >NWCHEM.INP\n"
        runscript += self.get_nwchemfile()  
        runscript += "eor\n"
        runscript += "cat NWCHEM.INP \n"
        
        runscript += "$NWCHEMBIN/nwchem NWCHEM.INP >NWCHEM.OUT \n"
        runscript += "retcode=$?\n"

        runscript += "if [[ -f NWCHEM.OUT ]]; then \n"
        runscript += "  cat NWCHEM.OUT \n"
        runscript += "else \n"
        runscript += "  cat NWCHEM.out \n"
        runscript += "fi \n"

        runscript += "rm NWCHEM.INP \n"
        runscript += "exit $retcode \n"
        
        print runscript
        
        return runscript

    def check_success (self, outfile, errfile):
        # check that NWChem terminated normally
        if not (os.path.exists('NWCHEM.OUT') or os.path.exists('NWCHEM.out')):
            raise PyAdfError('NWChem output file does not exist')
        return True
    
class nwchemsettings (object):
    """
    Settings for a NWChem calculation.
    
    @Note: 
        Currently, this class is just a placeholder. It should be
        extended once more NWChem functionality is added.
        
    @group Initialization:
        __init__
    @group Other Internals:
        __str__
    """
    
    def __init__ (self, method='DFT', functional='LDA', dftgrid=None, properties=None):
        """
        Constructor for nwchemsettings.

        All arguments are optional, leaving out an argument will choose default settings.
        
        @param method: the computational method, see L{set_scf_method}
        @type method: str
        @param functional: 
            exchange-correlation functional for DFT calculations, see L{set_functional}
        @type  functional: str
        @param dftgrid: the numerical integration grid for the xc part in DFT, see L{set_dftgrid}
        @type  dftgrid: None or str
        @param properties: a list of properties to calculate (e.g. 'dipole')
        @type  properties: list of str
        """
        self.scf_method = None
        self.functional = None
        self.dftgrid = None
        self.properties = None

        self.set_scf_method(method)
        self.set_functional(functional)
        self.set_dftgrid(dftgrid)
        self.set_properties(properties)
        
    def set_scf_method (self, method):
        """
        Select the computational method.
        
        Available options are: C{'HF'}, C{'DFT'}
        
        @param method: string identifying the selected method
        @type  method: str
        """
        self.scf_method = method

    def set_functional(self, functional):
        """
        Select the exchange-correlation functional for DFT.
        
        @param functional: 
            A string identifying the functional. 
            See Dalton manual for available options.
        @type functional: str
        """
        self.functional = functional
 
        if self.functional.upper() == 'LDA':
            self.functional = 'slater vwn_5'
   
    def set_dftgrid(self, dftgrid) :
        """
        Select the numerical integration grid.
        """
        self.dftgrid = dftgrid

    def set_properties(self, properties):
        """
        Select which properties to calculate.

        @param properties: A list of properties to calculate
        @type  properties: list of str
        """
        self.properties = properties       

    def get_scftask_block (self):
        if self.scf_method.upper() == 'HF' :
            block = "task scf"
        elif self.scf_method.upper() == 'DFT':
            block = "task dft"
        else :
            raise PyAdfError ('Unknown method in NWChem job')
        block += ' energy'
        if self.properties is not None:
            block += ' property'
        block += '\n'
        return block

    def get_properties_block (self):
        if self.properties is not None:
            block = 'property\n'
            for p in self.properties :
                block += p+'\n'
            block += 'end\n'
        else:
            block = ''
        return block

    def get_dft_block (self):
        block  = ""
        block += "   xc "+self.functional+"\n"
        if self.dftgrid is not None:
            block += "   grid "+self.dftgrid+"\n"
        return block

    def __str__ (self):
        """
        Returns a human-readable description of the settings.
        """
        return "Default NWChem settings"


class nwchemsinglepointjob (nwchemjob) :
    """
    A class for NWChem single point runs.

    See the documentation of L{__init__} and L{nwchemsettings} for details
    on the available options.
    
    Corresponding results class: L{nwchemsinglepointresults}

    @Note: Right now, HF, DFT, and CC jobs are supported.

    @Note: Importing of embedding potential requires a modified NWChem version.

    @group Initialization:
        set_restart
    @group Input Generation:
        get_molecule, get_nwchem_block, get_molecule_block,
        get_basis_block, get_scftask_block, get_dft_block,
        get_options_block, get_other_blocks
    @group Other Internals:
        print_extras, print_molecule, print_settings

    """

    def __init__ (self, mol, basis, settings=None, fdein=None, options=None) :
        """
        Constructor for NWChem single point jobs.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}
        
        @param basis: 
            A string specifying the basis set to use (e.g. C{basis='cc-pVDZ'}).
        @type basis: str
        
        @param settings: The settings for the Dalton job. Currently not used.
        @type  settings: L{nwchemsettings}
                
        @param fdein: 
            Results of an ADF FDE calculation. The embedding potential from this
            calculation will be imported into NWChem (requires modified NWChem version).        
        @type  fdein: L{adffragmentsresults}        
                
        @param options: 
            Additional options. 
            These will each be included directly in the NWChem input file.
        @type options: list of str
        """
        nwchemjob.__init__ (self)

        self.mol      = mol
        self.basis    = basis
        
        if settings == None :
            self.settings = nwchemsettings()
        else:
            self.settings = settings

        self.set_restart(None)

        self.fdein = fdein

        # FIXME: Symmetry hardcoded
        if self.mol:
            self.mol.set_symmetry('NOSYM')

        if options is None :
            self._options = []
        else :
            self._options = options

    def create_results_instance (self):
        return nwchemsinglepointresults(self)

    # FIXME: restart with NWChem not implemented
    def set_restart (self, restart) :
        """
        Set restart file. (NOT IMPLEMENTED)
        
        @param restart: results object of previous Dalton calculation
        @type  restart: L{nwchemsinglepointresults}
        
        @Note: restarts with NWChem are not implemented!
        """
        self.restart = restart

    def get_molecule (self) :
        return self.mol

    def get_nwchem_block (self):
        block  = "start NWCHEM\n"
        return block

    def get_molecule_block (self):
        block  = "geometry units angstrom noautoz nocenter noautosym\n"
        block += self.get_molecule().print_coordinates(index=False)
        block += 'end\n'
        return block

    def get_basis_block (self):
        block  = "basis spherical\n"
        block += "   * library "+self.basis+"\n"
        block += 'end\n'
        return block

    def get_dft_block (self):
        block  = "dft\n"
        block += self.settings.get_dft_block()
        if not self.fdein == None :
            block += '   frozemb\n'
        block += 'end\n'
        return block

    def get_scf_block (self):
        block  = "scf\n"
        if not self.fdein == None :
            block += '   frozemb\n'
        block += 'end\n'
        return block

    def get_options_block (self) :
        block = ""
        for opt in self._options :
            block += opt + "\n"
        return block

    def get_scftask_block (self) :
        return self.settings.get_scftask_block()

    def get_other_blocks (self) :
        return ""

    def get_nwchemfile (self) :
        # The following blocks should always be present: define the geometry and basis
        nwchemfile = "echo\n" 
        nwchemfile += self.get_nwchem_block() 
        nwchemfile += self.get_molecule_block() 
        nwchemfile += self.get_basis_block() 
        if self.settings.scf_method.upper() == 'DFT' :
            nwchemfile += self.get_dft_block() 
        else:
            nwchemfile += self.get_scf_block() 

        nwchemfile += self.settings.get_properties_block()

        nwchemfile += self.get_options_block ()
        
        nwchemfile += self.get_scftask_block() 

        # Here we make room for optional blocks
        nwchemfile += self.get_other_blocks ()

        return nwchemfile

    def print_jobtype (self) :
        return "NWChem single point job"

    def before_run (self) :
        nwchemjob.before_run (self)
        if not self.fdein == None :
            self.fdein.export_embedding_data('EMBPOT','FRZDNS')

    def after_run (self) :
        nwchemjob.after_run (self)
        if not self.fdein == None :
            os.remove('EMBPOT')
            os.remove('FRZDNS')

    def print_molecule (self) :

        print "   Molecule"
        print "   ========"
        print
        print self.get_molecule()
        print

    def print_settings (self) :
        
        print "   Settings"
        print "   ========"
        print
        print self.settings
        print

    def print_extras (self) :
        pass

    def print_jobinfo (self):
        print " "+50*"-"
        print " Running "+ self.print_jobtype()
        print

        self.print_molecule ()

        self.print_settings ()

        self.print_extras ()
 
