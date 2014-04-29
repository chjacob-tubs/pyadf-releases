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
 Job and results for ADF CPL calculations. 

 @author:       Andreas W. Goetz
 @organization: Vrije Universiteit Amsterdam (2008) 

 @group Jobs:
    adfcpljob
 @group Settings:
    cplsettings
 @group Results:
   adfcplresults, couplings
 @undocumented: operatorname
"""

from Errors  import PyAdfError
from ADFBase import adfjob, adfresults


class cplsettings (object) :
    """
    Class for the settings for an ADF CPL job.

    @group Initialisation:
       __init__,
       set_nuclei, set_cplnuclei, set_atompert, set_atomresp, set_operators,
       set_iterations, set_converge, set_contributions

    @group Internal:
       check_settings

    @group Printing:
       print_settings, get_settings_block

    """

    def __init__ (self, nuclei=None, atompert=None, atomresp=None, operators=None,
                  iterations=15, converge=1e-05, contributions=None) :
        """
        Create settings object for a CPL run.

        @param nuclei:
           list of pairs of the nuclei to calculate the nuclear spin-spin couplings for
           (numbers in INPUT ORDER, set_cplnuclei takes care of reordering for correct 
           use of nuclei key). Also see L{set_nuclei}.
        @type nuclei: list of list of int
    
        @param atompert:
           list of perturbing nuclei (numbers in INPUT ORDER). Also see L{set_atompert}.
        @type atompert: list of int

        @param atomresp:
           list of responding nuclei (numbers in INPUT ORDER). Also see L{set_atomresp}.
        @type atomresp: list of int
           
        @param operators:
           list of operators to be included in the Hamiltonian ('fc', 'sd', 'dso', 'pso').
           Also see L{set_operators}.
        @type operators: list of str

        @param iterations:
           number of CP-KS iterations. Also see L{set_iterations}.
        @type iterations: int
        
        @param converge:
           convergence criterion for CP-KS iterations. Also see L{set_converge}
        @type converge: float
        
        @param contributions:
           If orbital contributions to J-couplings shall be computed, a string has to be passed.
           It can contain the threshold and loc and sfo (see ADF-CPL documentation).
           Also see L{set_contributions}.
        @type contributions: str
        
        @note:
           You can *either* use nuclei *or* atompert/atomresp
           See also the ADF-CPL documentation
        
        @example:
           nuclei = [ [1, 2], [2, 4, 5] ] computes the couplings between
           nuclei 1 and 2,
           nuclei 2 and 4,
           nuclei 2 and 5

        @example:
           atompert = [1, 2] and atomresp = [4, 5] computes the couplings between
           nuclei 1 and 4,
           nuclei 1 and 5,
           nuclei 2 and 4,
           nuclei 2 and 5

        """
        self.nuclei        = nuclei
        self.atompert      = atompert
        self.atomresp      = atomresp
        self.operators     = None
        if operators is None:
            self.set_operators(['fc'])
        else:
            self.set_operators (operators)
        self.iterations    = iterations
        self.converge      = converge
        self.contributions = contributions

        self.cplnuclei     = None

    def set_nuclei(self, nuclei) :
        """
        Determine for which atoms the J-couplings calculations shall be done.

        @param nuclei:
           List if list of nuclei in ADF INPUT ORDER NUMBERING
        @type nuclei: list of list of int

        @note: should not be used together with set_atompert/set_atomresp
        """
        self.nuclei = nuclei

    def set_cplnuclei(self, adfres=None) :
        """
        Determines the ADF INTERNAL NUMBERING for the Nuclei block in the CPL input.
        
        This requires that the nuclei are already set in the constructor or with L{set_nuclei}.
        
        @param adfres: Results from an ADF single point run
        @type adfres:  L{adfsinglepointresults}
        """

        if adfres == None :
            raise PyAdfError ('No ADF results object provided in set_cplnuclei')
        
        if self.nuclei == None :
            self.cplnuclei = None
        else :
            self.cplnuclei = []
            for nucblock in self.nuclei :
                self.cplnuclei.append ( adfres.get_atom_index (nucblock) )

    def set_atompert(self, atompert) :
        """
        Determine the perturbing atoms for the J-couplings calculations.

        @param atompert:
           List of nuclei in ADF INPUT ORDER NUMBERING
        @type atompert: list of int

        @note: should not be used together with set_nuclei, needs set_atomresp
        """
        self.atompert = atompert

    def set_atomresp(self, atomresp) :
        """
        Determine the responding atoms for the J-couplings calculations.

        @param atomresp:
           List of nuclei in ADF INPUT ORDER NUMBERING
        @type atomresp: list of int

        @note: should not be used together with set_nuclei, needs set_atompert
        """
        self.atomresp = atomresp

    def set_operators(self, operators) :
        """
        Determine the operators used in the J-couplings calculations.

        @param operators:
           List of operators used (one ore more of 'fc', 'sd', 'dso', 'pso')
        @type operators: list of str

        @note: should not be used together with set_nuclei, needs set_atompert
        """
        self.operators = [ op.lower() for op in operators ]

    def set_iterations(self, iterations) :
        """
        Set the number of CP-KS iterations.
        
        @param iterations: maximum number of CP-KS iterations
        @type  iterations: int
        """
        self.iterations = iterations

    def set_converge(self, converge) :
        """
        Set the convergence criterion for the CP-KS iterations.
        
        @param converge: CP-KS convergence criterion
        @type  converge: float
        """
        self.converge = converge

    def set_contributions(self, contributions) :
        """
        Set computation of orbital contributions.
        
        @param contributions: 
            see ADF-CPL documentation for CONTRIBUTIONS key in NMRCOUPLING block
        @type contributions: str    
        """
        self.contributions = contributions

    def check_settings(self) :
        """
        Check consistency of settings for CPL.

        @note: set_cplnuclei should be called before!
        """
        if self.cplnuclei != None and ( self.atompert != None or self.atomresp != None ):
            raise PyAdfError ('You cannot use nuclei and atompert/atomresp')
        if self.cplnuclei == None:
            if self.atomresp == None or self.atomresp == None:
                raise PyAdfError ('You have to specify atompert and atomresp')

    def print_settings(self, mol=None) :
        """
        Print the settings for the CPL run.

        @param mol:
           Molecule, needed for coordinate printing
        @type mol:  L{molecule}

        """

        if mol == None :
            raise PyAdfError ('Molecule object required in print_settings')

        if self.nuclei == None:
            pnuc = self.atompert
            rnuc = self.atomresp
        else:
            pnuc = [ n[0] for n in self.nuclei ]
            pnuc = [ n for n in set(pnuc) ]
            rnuc = []
            for nblock in [ n[1:len(n)] for n in self.nuclei ]:
                rnuc += nblock
            rnuc = [ n for n in set(rnuc) ]

        operators = {'fc':'Fermi Contact',
                     'sd':'Spin Dipole',
                     'dso':'Diamagnetic Spin-Orbit',
                     'pso':'Paramagnetic Spin-Orbit'}

        print '   Nuclear spin-spin coupling constants calculation:'
        print '   (nuclei in INPUT ORDER)\n'
        print '   >> Perturbing nuclei <<'
        print mol.print_coordinates (pnuc)
        print '   >> Responding nuclei <<'
        print mol.print_coordinates (rnuc)
        print '   Operators included in the calculation:'
        for op in self.operators:
            print '   - '+operators[op]

    def get_settings_block(self) :
        """
        Obtain block for a CPL input which contains all settings

        @returns: all CPL settings prepared for the input file
        @rtype: str
        
        """
        settings_block = ''
        for op in self.operators:
            settings_block += ' '+op+'\n'
        if not 'fc' in self.operators:
            settings_block += ' nofc\n'
        # CPL keys have a weird behavior:
        #     sd is *not* computed by default for non-relativistic calculations
        #     sd is       computed by default for relativistic calculations *if* the fc term is computed
        #   nosd will switch off the sd term in relativistic calculations
        #   nosd will instruct CPL to use the spin-orbit routines for the fc term also in non-relativistic calculations
        #   nosd together with nofc will lead to a crash of CPL
        #   therefore I decided not to set 'nosd'
        ### if not 'sd' in self.operators:
        ###     settings_block += ' nosd\n'
        if self.nuclei == None:
            line = ' AtomPert'
            for nuc in self.atompert:
                line += ' '+str(nuc)
            settings_block += line+'\n'
            line = ' AtomResp'
            for nuc in self.atomresp:
                line += ' '+str(nuc)
            settings_block += line+'\n'
        else:
            for nucblock in self.cplnuclei:
                line = ' Nuclei'
                for nuc in nucblock:
                    line += ' '+str(nuc)
                settings_block += line+'\n'
        settings_block += ' scf iterations '+str(self.iterations)+' converge %.1e\n' % self.converge
        if self.contributions != None :
            settings_block += ' Contributions '+self.contributions+'\n'
            
        return settings_block


class adfcplresults (adfresults) :
    """
    Class for the results of an ADF CPL job.

    @group Initialization:
       __init__

    @group Retrieval of specific results:
       read_couplings, get_coupling, get_all_couplings
    
    """

    def __init__ (self, j=None) :
        """
        Consructor for adfcplresults.
        """
        adfresults.__init__ (self, j)

    def read_couplings (self, unit=None) :
        """
        Read all nuclear spin-spin coupling constants from tape21.

        @param unit:
           Unit of the nuclear spin-spin coupling constant ('J' or 'K')
        @type unit: str

        @returns: square matrix of nuclear spin-spin coupling constants
        @rtype: float[n][n] 

        """

        if unit == None:
            raise PyAdfError ( 'No unit specified for reading nuclear spin-spin couplings in read_couplings' )
        elif unit not in ('J', 'K'):
            raise PyAdfError ( 'Wrong unit specified for reading nuclear spin-spin couplings in read_couplings' )

        section = 'NMR Coupling %s const InputOrder' % unit
        clist = self.get_result_from_tape('Properties', section, 21)

        # Alternatively we can use adfreport to extract data
        # Note: Then we need to convert the return value to 
        #alternative: from os import popen
        #alternative: command = 'adfreport %s nmr-j-coupling-constant' % self.get_tape_filename(21)
        #alternative: f = popen(command)
        #alternative: clist = f.read().replace('\n','').split(' ')
        #alternative: f.close()

        # put couplings now in an list of lists (store as array)
        natom = self.job.mol.get_number_of_atoms()
        coupls = []
        for i in range(natom) :
            start = i*natom
            coupls.append( clist[start:start+natom] )

        return coupls

    def get_coupling (self, nucs=None, unit='J') :
        """
        Get nuclear spin-spin coupling for an atom pair.
        
        @param nucs:
           pair of nuclei (in ADF INPUT ORDER)
        @type nucs: list of int

        @param unit:
           Unit of the nuclear spin-spin coupling constant ('J' or 'K').
           Default is J coupling constants
        @type unit: str

        @returns: nuclear spin-spin coupling constant
        @rtype:   float
        """

        small = 1.e-02
        
        if nucs == None :
            raise PyAdfError ( 'No list of pairs of nuclei provided in get_coupling' )

        natom = self.job.mol.get_number_of_atoms()
        if ( nucs[0] > natom ) or ( nucs[1] > natom ) :
            raise PyAdfError ( 'Atom number too large in get_coupling' )

        unit = unit.upper()
        coupls = self.read_couplings(unit)
        na = nucs[0] - 1
        nb = nucs[1] - 1
        if abs ( coupls[na][nb] ) > small :
            return coupls[na][nb]
        elif abs ( coupls[nb][na] ) > small :
            return coupls[nb][na]
        else:
            return float(0)

    def get_all_couplings (self, unit='J') :
        """
        Get all non-zero nuclear spin-spin coupling constants in canonical order.
        
        That means all couplings for which a calculation has been done

        @param unit:
            Unit of the nuclear spin-spin coupling constant ('J' or 'K').
            Default is J coupling constants.
        @type unit: str

        @returns: 
            nuclear spin-spin coupling constants in a list of tuples.
            Each tuple has the form (atom 1, atom 2, coupling constant)
        @rtype: list of tuples 
        """

        small = 1.e-02

        unit = unit.upper()
        all_couplings = self.read_couplings(unit)

        natom = self.job.mol.get_number_of_atoms()

        coupls = []
        for i in range(natom):
            for j in range (i):
                if abs ( all_couplings[i][j] ) > small:
                    coupls.append( (i+1, j+1, all_couplings[i][j]) )
                elif abs ( all_couplings[j][i] ) > small:
                    coupls.append( (i+1, j+1, all_couplings[j][i]) )

        return coupls


class adfcpljob (adfjob) :
    """
    Class for ADF spin-spin coupling jobs (using the CPL program).

    See the documentation of L{__init__} and L{cplsettings} for details
    on the available options.

    Corresponding results class: L{adfcplresults}
    
    @group Initialization:
        __init__
    """

    def __init__ (self, adfres=None, settings=None, options=None) :
        """
        Constructor of ADF spin-spin coupling jobs.

        @param adfres:
           results from an ADF single point run
        @type adfres: (subclass of) L{adfsinglepointresults}
        
        @param settings:
           settings for this CPL run
        @type settings: L{cplsettings}

        @param options:
           optional settings for this CPL run (e.g. GGA).
           These options are included as given in the CPL input.
           See ADF-CPL documentation for details.
        @type options: list of str    
        """
        
        if adfres == None:
            raise PyAdfError('No ADF singe point results provided in adfcpljob')
        elif settings == None:
            raise PyAdfError('No settings (cplsettings) provided in adfcpljob')

        adfjob.__init__ (self)

        self.adfresults = adfres

        self.settings = settings
        self.settings.set_cplnuclei(self.adfresults)
        self.settings.check_settings()

        self.mol = self.adfresults.get_molecule()

        if options == None:
            self.options = []
        else:
            if isinstance(options, list):
                self.options = options
            else:
                self.options = [options]

    def create_results_instance (self):
        return adfcplresults(self)

    def get_runscript (self) :
        #pylint: disable-msg=W0221
        return adfjob.get_runscript (self, program='cpl')

    def get_input (self) :
        cplinput = ''
        for opt in self.options:
            cplinput += opt+'\n'
        cplinput += 'NMRCoupling \n'
        cplinput += self.settings.get_settings_block()
        cplinput += 'End \n'

        if self._checksum_only :
            cplinput += self.adfresults.get_checksum()
        
        return cplinput
        
    def print_jobtype (self):
        return "CPL job"
        
    def print_jobinfo(self) :
        print ' '+50*'-'
        print ' Running '+self.print_jobtype()
        print
        print '   SCF taken from ADF job ', self.adfresults.fileid,' (results id)'
        print
        self.settings.print_settings(self.adfresults.get_molecule())
        print
        print '   Options :'
        for opt in self.options:
            print opt
        print

    def before_run (self):
        self.adfresults.get_tapes_copy ()


class couplings(object):
    """
    Class for storing results of ADF spin-spin calculations.
    
    This class can be abused...
    ...it should only be used to store couplings constants of one 
    of the following combinations of operators:
    fc, fc+sd, sd, dso, pso, fc+sd+dso+pso (total)

    @group Initialisation:
       __init__

    @group Manipulation:
       set_coupling, compute_sd, compute_total

    @group Retrieval of couplings:
       get_coupling,
       set_fc, set_fcsd, set_sd, set_dso, set_pso, set_total,
       get_fc, get_fcsd, get_sd, get_dso, get_pso, get_total

    @exampleuse:
    
    >>> settings = cplsettings()
    >>> nuclei = [2,3]
    >>> settings.set_nuclei(nuclei)
    >>> operatorlist = [ ['fc'], ['fc', 'sd'], ['dso'], ['pso'] ]
    >>> jvalues = couplings()
    >>> for operators in operatorlist :
    >>>     settings.set_operators(operators)
    >>>     cpl_results = adfcpljob( adfres, settings).run()
    >>>     jvalues.set_coupling(cpl_results.get_coupling(nuclei), operators)
    >>> jvalues.compute_sd()
    >>> jvalues.compute_total()
    >>> operatorlist = [ ['fc'], ['sd'], ['dso'], ['pso'], ['fc', 'sd', 'dso', 'pso'] ]
    >>> for operators in operatorlist :
    >>>     print 'J = ', jvalues.get_coupling(operators)
    """

    def __init__(self, fcsd=None, fc=None, sd=None, dso=None, pso=None, total=None):
        self.fcsd  = fcsd
        self.fc    = fc
        self.sd    = sd
        self.dso   = dso
        self.pso   = pso
        self.total = total

    def compute_sd(self):
        if (self.fcsd == None) or (self.fc == None):
            print 'Cannot compute sd: fcsd or fc missing!'
        else:
            self.sd = self.fcsd - self.fc

    def compute_total(self):
        """
        Compute the total coupling constant
        (Requires that fc+sd, dso and pso terms have been computed before)
        """
        if (self.fcsd == None) or (self.dso == None) or (self.pso == None):
            print 'Cannot compute total: fcsd or dso or pso missing!'
        else:
            self.total = self.fcsd + self.dso + self.pso

    def set_coupling(self, coupling=None, operators=None):
        """
        Set value for a coupling constant.
        
        This method (as this class) can be abused...
        ...it should only be used to store one of the following combinations of operators:
        fc, fc+sd, sd, dso, pso, fc+sd+dso+pso (total)

        @param coupling:
           Coupling constant
        @type coupling: float
        
        @param operators:
           Operator list which determines the type of coupling constant
           (one of 'fc', 'sd', 'dso', 'pso'; or 'fc', 'sd'; or 'fc', 'sd', 'dso', 'pso')
        @type operators: list of str
        """
        if (operators is None) or (operators == []):
            print 'Cannot set coupling, operator list empty'
        if ('fc' in operators) and ('sd' in operators) and ('dso' in operators) and ('pso' in operators):
            self.set_total(coupling)
        elif ('fc' in operators) and ('sd' in operators):
            self.set_fcsd(coupling)
        elif ('fc' in operators):
            self.set_fc(coupling)
        elif ('sd' in operators):
            self.set_sd(coupling)
        elif ('dso' in operators):
            self.set_dso(coupling)
        elif ('pso' in operators):
            self.set_pso(coupling)
        else:
            print 'Unsupported operator list in set_coupling!'

    def get_coupling(self, operators=None):
        """
        Get value of a coupling constant.
        
        This method (as this class) can be abused...
        ...it should only be used to store one of the following combinations of operators:
        fc, fc+sd, sd, dso, pso, fc+sd+dso+pso (total)

        @param operators:
           Operator list which determines the type of coupling constant (FC, FC+SD, DSO, PSO, Total)
        @type operators: list of str
        """
        if (operators is None) or (operators == []):
            print 'Cannot get coupling, operator list empty'
        if ('fc' in operators) and ('sd' in operators) and ('dso' in operators) and ('pso' in operators):
            return self.get_total()
        elif ('fc' in operators) and ('sd' in operators):
            return self.get_fcsd()
        elif ('fc' in operators):
            return self.get_fc()
        elif ('sd' in operators):
            return self.get_sd()
        elif ('dso' in operators):
            return self.get_dso()
        elif ('pso' in operators):
            return self.get_pso()
        else:
            print 'Unsupported operator list in get_coupling!'

    def set_fcsd(self, fcsd=None):
        self.fcsd = fcsd

    def get_fcsd(self):
        return self.fcsd

    def set_fc(self, fc=None):
        self.fc = fc

    def get_fc(self):
        return self.fc

    def set_sd (self, sd=None) :
        self.sd = sd

    def get_sd(self):
        return self.sd

    def set_dso(self, dso=None):
        self.dso = dso

    def get_dso(self):
        return self.dso

    def set_pso(self, pso=None):
        self.pso = pso

    def get_pso(self):
        return self.pso

    def set_total(self, total=None):
        self.total = total

    def get_total(self):
        return self.total


def operatorname(operators=None):
    """
    Function returns list of operators concatenated with a plus symbol.
    
    ( 'fc+sd' for ['fc', 'sd'] )
    """
    if operators == None:
        return ''
    else:
        name = ''
        for op in operators:
            name += op + '+'
        return name[0:-1]

