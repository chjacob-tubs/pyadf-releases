# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2024 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Maria Chekmeneva,
# Thomas Dresselhaus, Kevin Focke, Andre S. P. Gomes, Andreas Goetz,
# Michal Handzlik, Karin Kiewisch, Moritz Klammler, Lars Ridder,
# Jetze Sikkema, Lucas Visscher, Johannes Vornweg, Michael Welzel,
# and Mario Wolter.
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
 The basics needed for Dirac calculations

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    diracjob, diracsinglepointjob
 @group Settings:
    diracsettings
 @group Results:
    diracresults, diracsinglepointresults
"""
from .Errors import PyAdfError
from .BaseJob import results, job
from pyadf.PyEmbed.Plot.GridFunctions import GridFunction1D
from .Utils import newjobmarker
import os
import re


class diracresults(results):
    """
    Class for results of an Dirac calculation.

    @group Initialization:
        __init__
    @group Access to results files:
        get_dfcoef_filename, get_gridout_filename,
        copy_dfcoef, copy_gridout
    """

    def __init__(self, j=None):
        """
        Constructor for diracresults.
        """
        super().__init__(j)

    def get_dfcoef_filename(self):
        """
        Return the file name of the DFCOEF file belonging to the results.
        """
        try:
            fn = self.files.get_results_filename(self.fileid, 22)
        except PyAdfError:
            try:
                fn = self.files.get_results_filename(self.fileid, 23)
            except PyAdfError:
                fn = self.files.get_results_filename(self.fileid, 21)
        return fn

    def get_gridout_filename(self):
        """
        Return the file name of the GRIDOUT file belonging to the results.

        The GRIDOUT file contains the electron density (and its gradient)
        as well as the Coulomb potential on the ADF grid given with the
        C{fdeout} option.
        """
        return self.files.get_results_filename(self.fileid, 10)

    def get_xml_filename(self):
        """
        Return the file name of the dirac.xml file belonging to the results.

        The dirac.xml file has the output in machine-readable form
        """
        return self.files.get_results_filename(self.fileid, 66)

    def copy_dfcoef(self):
        """
        Copy result DFCOEF file to the working directory.
        """
        try:
            self.files.copy_result_file(self.fileid, 21, "DFCOEF")
        except PyAdfError:
            pass

        try:
            self.files.copy_result_file(self.fileid, 22, "CHECKPOINT.h5")
        except PyAdfError:
            try:
                self.files.copy_result_file(self.fileid, 23, "CHECKPOINT.noh5.tar.gz")
            except PyAdfError:
                pass

    def copy_gridout(self, name="GRIDOUT"):
        """
        Copy result GRIDOUT file to the working directory.

        The GRIDOUT file contains the electron density (and its gradient)
        as well as the Coulomb potential on the ADF grid given with the
        C{fdeout} option.

        @param name: The name of the copied file
        @type  name: str
        """
        self.files.copy_result_file(self.fileid, 10, name)


class diracsinglepointresults(diracresults):
    """
    Class for results of a Dirac single point calculation.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_molecule, get_energy
    @group Access to results files:
        export_dirac_tape10
    @undocumented: _get_fdein, _get_fdeout
    """

    def __init__(self, j=None):
        """
        Constructor for diracsinglepointresults.
        """
        super().__init__(j)
        self._ccenergies = None

    def get_molecule(self):
        """
        Return the molecular geometry after the Dirac job.

        @returns: The molecular geometry.
        @rtype:   L{molecule}

        @note: currently not implemented
        """
        pass

    def get_ccenergies(self):
        """
        Returns energies calculated by the coupled cluster module.

        This uses the xml output that is still rather limited in DIRAC but is to be extended.

        @rtype: dict
        """
        if self._ccenergies is not None:
            ccenergies = self._ccenergies
        else:
            from xml.dom.minidom import parse
            dom = parse(self.get_xml_filename())

            ccenergies = {}
            for quantity in dom.getElementsByTagName('quantity'):
                label = quantity.getAttribute('label')
                if label.find('energy') != -1:
                    ccenergies[label] = float(quantity.firstChild.data)

        return ccenergies

    def get_scf_energy(self):
        """
        Return the total energy.

        @returns: the total energy in atomic units
        @rtype: float
        """
        energy = 0.0
        output = self.get_output()

        headerline = 1
        start0 = re.compile(r"\s+TOTAL ENERGY")

        for i, l in enumerate(output):
            if start0.match(l):
                headerline = i
        if headerline == 1:
            raise PyAdfError('Total energy not found')

        en = re.compile(r"\s+Total energy (\(active subsystem\))?\s+:\s+(?P<energy>[-+]?(\d+(\.\d*)?|\d*\.\d+))")
        for line in output[headerline:]:
            m = en.match(line)
            if m:
                energy = float(m.group('energy'))

        return energy

    def get_energy(self, level='SCF'):
        if level in ['SCF', 'HF', 'DHF', 'DFT', 'DKS']:
            energy = self.get_scf_energy()
        elif level in ['MP2', 'CCSD', 'CCSD(T)', 'CCSD-T', 'CCSD+T']:
            ccenergies = self.get_ccenergies()
            try:
                energy = ccenergies[level+' energy']
            except KeyError:
                raise PyAdfError('Unknown level specified for get_energy')
        else:
            raise PyAdfError('Unknown level specified for get_energy')

        return energy

    def get_total_energy(self):
        return self.get_energy(self.job.settings.method)

    def get_dipole_vector(self, level='DHF'):
        """
        Return the dipole moment vector.

        @returns: the dipole moment vector, in atomic units
        @rtype: float[3]
        """
        from .Utils import au_in_Debye
        import numpy as np

        dipole = np.zeros((3,))

        output = self.get_output()

        print("\n\nNote: extracting dipole vector for " + level + " level of theory\n")

        headerline = 1
        start0 = re.compile(r".+Properties\s+for\s+([A-Z0-9]+)")

        for i, l in enumerate(output):
            if start0.match(l):
                if start0.match(l).group(1) == level:
                    headerline = i

        startline = None
        start1 = re.compile(r"\s*\*\s*Dipole moment:")
        start2 = re.compile(r"\s*Electronic\s*Nuclear\s*Total")
        start3 = re.compile(r"\s*contribution\s*contribution\s*contribution")
        for i, l in enumerate(output):
            if start1.match(l) and start2.match(output[i + 2]) and start3.match(output[i + 3]) and (i > headerline):
                startline = i
                break

        if startline is None:
            raise PyAdfError('Dirac dipole moment not found in output')

        for i, c in enumerate(['x', 'y', 'z']):
            dip = re.compile(r"\s*" + c + r".*Debye\s*(?P<dip>[-+]?(\d+(\.\d*)?|\d*\.\d+)) Debye")
            m = dip.match(output[startline + 5 + i])
            dipole[i] = float(m.group('dip'))

        return dipole / au_in_Debye

    @property
    def fdeout(self):
        """
        The results of the ADF FDE calculation from which the grid is used for exporting.

        @type: L{adffragmentsresults}
        """
        return self.job.fdeout

    @property
    def fdein(self):
        """
        The results of the ADF FDE calculation from that the embedding potential was imported.

        @type: L{adffragmentsresults}
        """
        return self.job.fdein

    def export_dirac_tape10(self, outfile):
        """
        Export the density and potential (GRIDOUT) to an ADF TAPE10-like file.
        """
        from .kf import kf
        import subprocess
        from .kf.xml2kf import xml2kf

        xml2kf(self.get_gridout_filename(), outfile)

        if kf.kffile.env is not None:
            env = kf.kffile.env
        else:
            env = os.environ

        cpkfCmd = [os.path.join(env.setdefault('AMSBIN', ''), 'cpkf'),
                   self.fdeout.get_tape_filename(tape=10), outfile, 'Num Int Params']

        DEVNULL = open(os.devnull, 'wb')
        subprocess.Popen(cpkfCmd, stdout=DEVNULL, stderr=DEVNULL, env=env)


class diracjob(job):
    """
    An abstract base class for Dirac jobs.

    Corresponding results class: L{diracresults}

    @group Initialization:
        __init__
    @group Input Generation:
        get_molecule
    @group Running Internals:
        get_diracfile
    """

    def __init__(self):
        """
        Constructor for Dirac jobs.
        """
        super().__init__()

    def create_results_instance(self):
        return diracresults(self)

    def print_jobtype(self):
        raise NotImplementedError

    def get_molecule(self):
        """
        Abstract method. Should be overwritten to return the L{molecule}.
        """
        raise NotImplementedError

    def get_diracfile(self):
        """
        Abstract method. Should be overwritten to return the Dirac input file.
        """
        raise NotImplementedError

    @property
    def checksum(self):
        import hashlib
        m = hashlib.md5()

        m.update(self.get_diracfile().encode('utf-8'))
        m.update(str(self.get_molecule()).encode('utf-8'))

        return m.hexdigest()

    def get_runscript(self, nproc=1):

        put_files = ['FRZDNS', 'EMBPOT', 'GRIDOUT']
        get_files = ['GRIDOUT', 'dirac.xml']

        runscript = ""

        runscript += "cat <<eor >DIRAC.inp \n"
        runscript += self.get_diracfile()
        runscript += "eor\n"
        runscript += "cat DIRAC.inp \n"

        runscript += "cat <<eor >MOLECULE.xyz \n"
        runscript += self.get_molecule().get_xyz_file()
        runscript += "eor\n"
        runscript += "cat MOLECULE.xyz \n"

        runscript += "$DIRACBIN/pam "
        if nproc > 1:
            runscript += f'--mpi={nproc:d} '
        runscript += ' --put="' + " ".join([pf for pf in put_files if os.path.exists(pf)]) + '"'
        if os.path.exists('DFCOEF') or os.path.exists('CHECKPOINT.h5') or os.path.exists('CHECKPOINT.noh5.tar.gz'):
            runscript += ' --incmo'
        runscript += ' --get="' + " ".join([gf for gf in get_files]) + '"'
        runscript += ' --outcmo'
        if 'DIRMAX_GB' in os.environ:
            runscript += ' --ag=' + os.environ['DIRMAX_GB']
        runscript += " --mol=MOLECULE.xyz --inp=DIRAC.inp \n"
        runscript += " retcode=$? \n"

        runscript += "if [[ -f DIRAC_MOLECULE.OUT ]]; then \n"
        runscript += "  cat DIRAC_MOLECULE.OUT \n"
        runscript += "elif [[ -f DIRAC_MOLECULE.out ]]; then \n"
        runscript += "  cat DIRAC_MOLECULE.out \n"
        runscript += "else \n"
        runscript += "  echo 'Dirac output file'\n"
        runscript += "  exit 1\n"
        runscript += "fi \n"

        runscript += "rm DIRAC.inp \n"
        runscript += "exit $retcode\n"

        return runscript

    def result_filenames(self):
        return ['DFCOEF', 'CHECKPOINT.h5', 'CHECKPOINT.noh5.tar.gz', 'GRIDOUT', 'dirac.xml']

    def check_success(self, outfile, errfile):
        # check that Dirac terminated normally
        if not (os.path.exists('DIRAC_MOLECULE.OUT') or os.path.exists('DIRAC_MOLECULE.out')):
            raise PyAdfError('Dirac output file does not exist')

        f = open(errfile)
        err = f.readlines()
        for line in reversed(err):
            if "SEVERE ERROR" in line:
                raise PyAdfError("Error running Dirac job")
            if "dirac.x returned non-zero exit code" in line:
                raise PyAdfError("Error running Dirac job")
            if line == newjobmarker:
                break
        f.close()
        return True


class diracsettings:
    """
    Class that holds the settings for a Dirac calculation..

    @group Initialization:
        __init__,
        set_method, set_hamiltonian, set_functional,
        set_properties, set_transform, set_exportfde
    @group Input Generation:
        get_hamiltonian_block, get_fdeexportlevel_block, get_relccsd_block,
        get_dirproperties_block, get_properties_block, get_moltraactive_block
    @group Other Internals:
        __str__, setup_ccnamelist
    """

    def __init__(self, method='DFT', hamiltonian='DC', functional='LDA', dftgrid=None,
                 properties=None, uncontracted=False, transform=None, nucmod=None,
                 scf_subblock=None, wf_options=None, nosym=False, dossss=False):
        """
        Constructor for diracsettings.

        All arguments are optional, leaving out an argument will choose default settings.

        @param method: the computational method, see L{set_method}
        @type method: str
        @param hamiltonian: the Hamiltonian, see L{set_hamiltonian}
        @type  hamiltonian: str
        @param functional:
            exchange-correlation functional for DFT calculations, see L{set_functional}
        @type  functional: str
        @param dftgrid: the numerical integration grid for the xc part in DFT, see L{set_dftgrid}
        @type  dftgrid: None or str
        @param properties: properties to calculate, see L{set_properties}
        @type  properties: list
        @param transform: options for 4-index transformation, see L{set_transform}
        @type  transform: list
        @param nucmod: nuclear model used in the calculation, see L{set_nucmod}
        @type  nucmod: str
        """
        self.method = None

        self.ccmain = None
        self.ccener = None
        self.ccfopr = None
        self.ccsort = None

        self.hamiltonian = None
        self.dossss = dossss
        self.dossc = True
        if self.dossss:
            self.dossc = False

        self.uncontracted = uncontracted

        # scf_subblock is a dictionary with the keywords and values;
        # in the case the key has multiple values, like the definition
        # of open shells, each line is passed as a member of a list
        self.scf_subblock_options = scf_subblock
        self.wf_options = wf_options

        self.domoltra = False
        self.moltra_active = None

        self.exportfde = False
        self.exportfde_level = None

        self.doprop = False
        self.proplist = None

        self.fun_xc = None
        self.dftgrid = None
        self.nucmod = None

        self.nosym = nosym

        self.set_method(method)
        self.set_hamiltonian(hamiltonian)
        if self.domoltra or (transform is not None):
            self.set_transform(transform)
        self.set_properties(properties)
        if self.method == 'DFT':
            self.set_functional(functional)
        self.set_dftgrid(dftgrid)
        self.set_nucmod(nucmod)

        self.setup_ccnamelist()

    def set_method(self, method):
        """
        Select the computational method.

        Available options are: HF, DFT, MP2, CCSD, CCSDt [=CCSD(T)], FSCC, IHFSCC

        @param method: string identifying the selected method
        @type  method: str
        """
        self.method = method

        if self.method in ('HF', 'DFT'):
            self.exportfde_level = 'DHF'
        elif self.method == 'MP2':
            self.exportfde_level = 'MP2'
            self.domoltra = True
        elif self.method in ('CCSD', 'CCSDt', 'CCSD(T)', 'FSCC', 'IHFSCC'):
            self.exportfde_level = 'CCSD'
            self.domoltra = True

    def set_hamiltonian(self, hamiltonian):
        """
        Select the Hamiltonian.

        Possible choices are:
         - DC:    Dirac-Coulomb
         - DCG:   Dirac-Coulomb-Gaunt (available for DFT, HF)
         - MMF:   Molecular Mean Field (based on DCG Hamiltonian)
         - SFDC:  Spin-free Dirac-Coulomb
         - X2C:   eXact 2-Component
         - ECP:   Relativistic (1- or 2-component) ECP
         - Levy:  Levy-Leblond (NR limit of Dirac equation)
         - Nonr:  Non-relativistic (true 1 component) Hamiltonian

        @param hamiltonian: A string identifying the Hamiltonian.
        @type  hamiltonian: str
        """
        if hamiltonian is None:
            self.hamiltonian = 'DC'

        supported_hamiltonians = {'DC': 'Dirac-Coulomb',
                                  'DCG': 'Dirac-Coulomb-Gaunt (available for DFT, HF)',
                                  'SFDC': 'Spin-free Dirac-Coulomb',
                                  'MMF': 'DCG-based Molecular Mean Field',
                                  'X2C': 'eXact 2-Component',
                                  'ECP': 'Relativistic ECP',
                                  'Levy': 'Levy-Leblond (NR limit of Dirac equation)',
                                  'Nonr': 'Non-relativistic (true 1 component) Hamiltonian'}

        valid_input = list(supported_hamiltonians.keys())

        if hamiltonian not in valid_input:
            print("Invalid choice of Hamiltonian. Choose among:\n")
            for k in valid_input:
                print(k, "  ", supported_hamiltonians[k])
            raise PyAdfError("Invalid Hamiltonian in Dirac settings")
        else:
            self.hamiltonian = hamiltonian
            if self.hamiltonian == 'Nonr' or self.hamiltonian == 'Levy':
                self.dossc = False
            else:
                self.dossc = True

    def set_properties(self, properties):
        """
        Select properties to calculate.

        @param properties:
            A list of the properties to calculate. Possible choices
            are 'dipole', 'efg', 'nqcc'. See Dirac documentation of
            **PROPERTIES for more.
        @type  properties: list
        """
        self.proplist = properties
        self.doprop = (self.proplist is not None)

    @property
    def functional(self):
        if self.method == 'DFT':
            return self.fun_xc
        else:
            return None

    def set_functional(self, functional):
        """
        Select the exchange-correlation functional for DFT.

        @param functional:
            A string identifying the functional.
            See Dirac manual for available options.
        @type functional: str
        """
        if self.method == 'DFT':
            self.fun_xc = functional
        else:
            raise PyAdfError('Functional can only be set for DFT calculations')

    def set_dftgrid(self, dftgrid):
        """
        Select the numerical integration grid.
        """
        self.dftgrid = dftgrid

    def set_transform(self, moltra):
        """
        Switch on and set options for 4-index transformation.

        @param moltra:
            either ['all'] or a list giving minimum and maximum orbital energy and
            the degeneracy treshold (three numbers)
        @type moltra: list
        """
        self.domoltra = True
        if moltra is None:
            self.moltra_active = [-5.0, 10.0, 0.1]
        else:
            self.moltra_active = moltra

    def set_nucmod(self, nucmod):
        """
        Choose the nuclear model used in the Dirac calculation.

        @param nucmod: 'finite' or None for finite nuclei (default) or 'point' for point nuclei
        """
        if nucmod is None:
            self.nucmod = None
        elif nucmod.lower() == 'finite':
            self.nucmod = None
        elif nucmod.lower() == 'point':
            self.nucmod = 'point'
        else:
            raise PyAdfError('Ivalid choice for nuclear model')

    def set_exportfde(self, export, level=None):
        """
        Switch on/off export of density and Coulomb potential for FDE.

        @param export: Whether to export the density and Coulomb potential.
        @type  export: bool
        @param level:
            Which density to export. Possible choices: DHF, MP2, CCSD
        @type  level: str
        """
        self.exportfde = export
        if level is not None:
            self.exportfde_level = level
        self.setup_ccnamelist()

    def setup_ccnamelist(self):
        """
        Initialize CC namelist options.
        """
        # setting up the cc namelist options
        self.ccmain = {'TIMING': 'T', 'IPRNT': '2', 'DOSORT': 'T', 'DOENER': 'T', 'DOFOPR': 'F', 'DOSOPR': 'F'}
        self.ccener = {'DOMP2': 'T', 'DOCCSD': 'F', 'DOCCSDT': 'F', 'MAXIT': '100'}
        self.ccfopr = {'DOMP2G': 'T', 'DOCCSDG': 'T'}
        self.ccsort = {}

        if self.method in ('HF', 'DFT'):
            pass
        elif self.method in ('MP2', 'CCSD'):
            if self.exportfde or self.doprop:
                self.ccmain['DOFOPR'] = 'T'
        elif self.method == 'CCSD':
            self.ccener['DOCCSD'] = 'T'
        elif self.method in ['CCSDt', 'CCSD(T)']:
            self.ccener['DOCCSD'] = 'T'
            self.ccener['DOCCSDT'] = 'T'
        else:
            raise PyAdfError("Unsupported method for Dirac single point runs")

    @staticmethod
    def get_option_block_from_dict(options):
        block = ""
        for k, v in options.items():
            block += k + "\n"
            if isinstance(v, list):
                for e in v:
                    if e != '':
                        block += e + "\n"
            elif v != '':
                block += v + "\n"
        return block

    def get_relccsd_block(self):
        block = "**RELCCSD\n"
        if self.exportfde or self.doprop:
            block += '.GRADIENT\n'
            block += '*CCFOPR\n'
            if self.method == 'MP2':
                block += '.RELAXED\n'
                block += '.MP2G\n'
            elif self.method == 'CCSD':
                block += '.CCSDG\n'
        # at the moment, the ccsd and ccsd(T) energies are always switched on
        #        elif self.method == 'CCSD' :
        #            block += '.DOCCSD\n'
        #        elif self.method == 'CCSDt' :
        #            block += '.DOCCSD\n'
        #            block += '.DOCCSDT\n'
        return block

    def get_hamiltonian_block(self):
        block = ""
        if self.dossc:
            # if we had set ssss and ssc, we override the latter
            if self.dossss:
                block += '.DOSSSS\n'
            else:
                block += '.LVCORR\n'
        if self.hamiltonian == 'MMF':
            block += '.X2Cmmf\n'
        if self.hamiltonian == 'DCG' or self.hamiltonian == 'MMF':
            block += '.GAUNT\n'
        if self.hamiltonian == 'SFDC':
            block += '.SPINFREE\n'
        if self.hamiltonian == 'X2C':
            block += '.X2C\n'
        if self.hamiltonian == 'ECP':
            block += '.ECP\n'
        if self.hamiltonian == 'Levy':
            block += '.LEVY-LEBLOND\n'
        if self.hamiltonian == 'Nonr':
            block += '.NONREL\n'

        if self.method == 'DFT':
            block += ".DFT\n"
            if str(self.fun_xc) == "SAOP!":
                block += "GLLBhole\n*DFT\n.SAOP!\n"
            else:
                block += " " + str(self.fun_xc) + "\n"
        return block

    def get_integrals_block(self):
        block = ""
        if self.nucmod is not None or self.uncontracted:
            block += "**INTEGRALS\n"
        if self.nucmod is not None:
            block += ".NUCMOD\n"
            if self.nucmod.lower() == 'point':
                block += '1\n'
            elif self.nucmod.lower() == 'finite':
                block += '2\n'
            else:
                raise PyAdfError('Invalid nuclear model chosen')
        if self.uncontracted:
            block += "*READIN\n.UNCONTRACT\n"
        return block

    def get_fdeexportlevel_block(self):
        block = ""
        if self.exportfde_level is not None:
            block += ".LEVEL\n" + self.exportfde_level + "\n"
        else:
            block += "# unrecognized export option. will enable SCF-based one\n"
            block += ".LEVEL\nDHF\n"
        return block

    def get_dirproperties_block(self):
        block = ""
        if self.proplist is not None:
            block += ".PROPERTIES\n"
        if self.domoltra:
            block += ".4INDEX\n"
        return block

    def get_properties_block(self):
        block = ""
        if self.proplist is not None:
            block = "**PROPERTIES\n"
            for k in self.proplist:
                block += "." + str(k).upper() + "\n"
        return block

    def get_moltraactive_block(self):
        block = ".ACTIVE\n"

        # TODO: setting up active should be done also by orbital strings
        if str(self.moltra_active[0]).lower() == 'all':
            block += "all\n"
        else:
            block += "energy  " + str(self.moltra_active[0]) + "  " + \
                     str(self.moltra_active[1]) + "  " + str(self.moltra_active[2]) + "\n"
        return block

    def get_grid_block(self):
        block = ""
        if self.dftgrid is not None:
            block += "**GRID\n"
            block += "." + self.dftgrid.upper() + "\n"
        return block

    def __str__(self):
        """
        Returns a human-readable description of the settings.
        """
        s = ""
        s += f"  Method: {self.method} \n"
        s += f"  Hamiltonian: {self.hamiltonian} \n"
        if self.method == 'DFT':
            s += f"  Exchange-correlation functional: {self.fun_xc} \n"
        if self.domoltra:
            s += "  Moltra active: " + str(self.moltra_active) + "\n"
        if self.doprop:
            s += "  Calculated properties: " + str(self.proplist) + "\n"
        return s


class diracsinglepointjob(diracjob):
    """
    A class for Dirac single point runs

    See the documentation of L{__init__} and L{diracsettings} for details
    on the available options.

    Corresponding results class: L{diracsinglepointresults}

    @group Initialization:
        set_restart
    @group Input Generation:
        get_dirac_title, get_dirac_hamiltonian, get_relccsd_namelist,
        get_dirac_block, get_integral_block, get_wavefunction_block,
        get_properties_block, get_molecule_block, get_moltra_block,
        get_other_blocks, get_options_block
    @group Other Internals:
        print_extras, print_molecule, print_settings
    """

    def __init__(self, mol, basis, ecp=None, settings=None,
                 fdein=None, fdeout=None, options=None):
        """
        Constructor for Dirac single point jobs.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}

        @param basis:
            A string specifying the basis set to use (e.g. C{basis='cc-pVDZ'}).
        @type basis: str

        @param ecp:
            A string specifying the ecp file to use (e.g. C{ecp='ECPDS78MDFSO'}).
        @type ecp: str

        @param settings: The settings for the Dirac job, see L{diracsettings}.
        @type  settings: L{diracsettings}

        @param fdein:
            Results of an ADF FDE calculation. The embedding potential from this
            calculation will be imported into Dirac
        @type  fdein: L{adffragmentsresults}

        @param fdeout:
            Results of an ADF FDE calculation. The density and Coulomb potential will be
            exported on the grid from this calculation (requires Dirac development version).
        @type  fdeout: L{adffragmentsresults}

        @param options:
            Additional options.
            These will each be included directly in the Dirac input file.
        @type options: list of str

        """
        super().__init__()

        self.mol = mol
        self.basis = basis
        self.ecp = ecp

        # FXIME: functional should be moved to diracsettings, to be consistent with ADF
        if settings is None:
            self.settings = diracsettings()
        else:
            self.settings = settings

        if self.mol and (self.basis is None):
            raise PyAdfError("Missing basis set in Dirac single point job")

        self.restart = None
        self.fdein = fdein
        self.fdeout = fdeout
        if self.fdeout is not None:
            self.settings.set_exportfde(True)

        # FIXME: Symmetry in Dirac hardcoded
        if self.fdein or self.settings.nosym:
            self.mol.set_symmetry('NOSYM')

        if options is None:
            self._options = []
        else:
            self._options = options

    def create_results_instance(self):
        return diracsinglepointresults(self)

    @property
    def checksum(self):
        import hashlib
        m = hashlib.md5()

        m.update(self.get_diracfile().encode('utf-8'))
        m.update(str(self.get_molecule()).encode('utf-8'))

        if self.restart is not None:
            m.update(b'Restarted from Dirac job \n')
            m.update(self.restart.checksum.encode('utf-8'))

        if self.fdein is not None:
            m.update(b'Embedding potential imported from ADF job \n')
            m.update(self.fdein.checksum.encode('utf-8'))
        if self.fdeout is not None:
            m.update(b'Embedding potential exported for ADF job \n')
            m.update(self.fdeout.checksum.encode('utf-8'))

        return m.hexdigest()

    def get_runscript(self, nproc=1):
        return super().get_runscript(nproc=nproc)

    # FIXME: restart with Dirac not implemented
    def set_restart(self, restart):
        """
        Set restart file. (NOT IMPLEMENTED)

        @param restart: results object of previous Dirac calculation
        @type  restart: L{diracsinglepointresults}

        @Note: restarts with Dirac are not implemented!
        """
        self.restart = restart

    def get_molecule(self):
        return self.mol

    @staticmethod
    def get_dirac_title():
        block = ".TITLE\n"
        block += "Input file generated by pyadf.\n"
        return block

    def get_dirac_hamiltonian(self):
        block = "**HAMILTONIAN\n"

        block += self.settings.get_hamiltonian_block()
        if self.fdein is not None or self.fdeout is not None:
            block += ".FDE\n"
            block += "*FDE\n"
        if self.fdein is not None:
            block += ".EMBPOT\n"
            block += "EMBPOT\n"
        if self.fdeout is not None:
            if self.fdein is None:
                block += '.EXONLY\n'
                block += "GRIDOUT\n"
                block += "other\n"
            else:
                block += ".GRIDOUT\n"
                block += "GRIDOUT\n"
            block += ".OLDESP\n"
            block += self.settings.get_fdeexportlevel_block()

        return block

    def get_dirac_block(self):
        block = ".WAVE FUNCTION\n"
        block += self.settings.get_dirproperties_block()
        return block

    @staticmethod
    def get_xml_block():
        block = ".XMLOUT\n"
        return block

    def get_wavefunction_block(self):
        block = "**WAVE FUNCTION\n"
        block += ".SCF\n"
        if self.settings.method in ('MP2', 'CCSD', 'CCSDt', 'CCSD(T)', 'FSCC', 'IHFSCC'):
            block += ".RELCCSD\n"
        if self.settings.wf_options is not None:
            block += self.settings.get_option_block_from_dict(self.settings.wf_options)
        if self.settings.scf_subblock_options is not None:
            block += "*SCF\n"
            block += self.settings.get_option_block_from_dict(self.settings.scf_subblock_options)
        return block

    def get_molecule_block(self):
        block = "*BASIS\n"
        block += ".DEFAULT\n"
        block += self.basis + "\n"
        if self.mol.symmetry:
            block += "*SYMMETRY\n"
            if self.mol.symmetry.upper() == 'NOSYM':
                symm = '.NOSYM'
            else:
                symm = self.mol.symmetry
            block += symm + "\n"
        if self.mol.get_charge() != 0:
            block += "*CHARGE\n"
            block += ".CHARGE\n"
            block += str(self.mol.get_charge()) + "\n"
        return block

    def get_moltra_block(self):
        block = "**MOLTRA\n"
        block += self.settings.get_moltraactive_block()
        return block

    def get_options_block(self):
        block = ""
        for opt in self._options:
            block += opt + "\n"
        return block

    @staticmethod
    def get_other_blocks():
        return ""

    def get_diracfile(self):
        diracfile = "**DIRAC\n"
        diracfile += self.get_dirac_title()
        diracfile += self.get_dirac_block()
        diracfile += self.get_xml_block()
        diracfile += self.get_dirac_hamiltonian()
        diracfile += self.get_wavefunction_block()
        diracfile += self.settings.get_grid_block()
        diracfile += self.settings.get_properties_block()
        diracfile += self.settings.get_integrals_block()
        diracfile += self.get_options_block()
        diracfile += self.get_other_blocks()
        if self.settings.domoltra:
            diracfile += self.get_moltra_block()
        if self.settings.method in ('MP2', 'CCSD', 'CCSDt', 'CCSD(T)', 'FSCC', 'IHFSCC'):
            diracfile += self.settings.get_relccsd_block()

        diracfile += "*END OF\n \n"

        diracfile += "**MOLECULE\n"
        diracfile += self.get_molecule_block()
        diracfile += "*END OF\n"

        return diracfile

    def print_jobtype(self):
        return "Dirac single point job"

    def before_run(self):
        super().before_run()

        if self.restart is not None:
            self.restart.copy_dfcoef()
        if self.fdein is not None:
            if isinstance(self.fdein, GridFunction1D):
                self.fdein.get_xyzwvfile('EMBPOT', add_comment=False, endmarker=True)
            else:  # adffragmentsresults
                self.fdein.export_embedding_data('EMBPOT')

        if self.fdeout is not None:
            self.fdeout.export_grid('GRIDOUT')

    def after_run(self):
        if self.fdein is not None:
            os.remove('EMBPOT')

        super().after_run()

    def print_molecule(self):

        print("   Molecule")
        print("   ========")
        print()
        print(self.get_molecule())
        print()

    def print_settings(self):

        print("   Settings")
        print("   ========")
        print()
        print(self.settings)
        print()

    def print_extras(self):
        pass

    def print_jobinfo(self):
        print(" " + 50 * "-")
        print(" Running " + self.print_jobtype())
        print()

        self.print_molecule()

        self.print_settings()

        self.print_extras()
