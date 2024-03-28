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
 Molcas wrapper. It was created analog to the Dirac wrapper.
 This wrapper is capable of providing an embedding potential
 to Molcas calculations and in that case reads in the resulting
 density (and its derivatives) and electrostatic potential of
 the Molcas calculation. Only single point calculations are possible.

 @author:       Thomas Dresselhaus
 @organization: Westfaelische Wilhelmsuniversitaet Muenster (WWU)
 @contact:      t.dresselhaus@wwu.de
"""
from .BaseJob import job
from .Errors import PyAdfError
from pyadf.PyEmbed.DensityEvaluator import GTODensityEvaluatorMixin

import os


class MolcasResults(GTODensityEvaluatorMixin):
    """
    @group Initialization:
        __init__
    """

    def __init__(self, j=None):
        super().__init__(j)
        # TODO make this according to the structure of PyADF
        self.tape10_filename = 'molcasjob.tape10'

    def read_molden_file(self):
        """
        Returns Molden results file as a string.
        """

        try:
            molden_filename = self.files.get_results_filename(self.fileid, tape=41)
        except PyAdfError:
            raise PyAdfError("Molcas Molden file not found")

        with open(molden_filename, encoding='utf-8') as f:
            content = f.read()

        return content

    def get_scforb_filename(self):
        return self.files.get_results_filename(self.fileid, tape=21)

    def get_xml_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=66)
        except PyAdfError:
            raise PyAdfError("Molcas XML file not found")
        return fn

    def get_energy(self):
        import xml.etree.ElementTree as ET

        with open(self.get_xml_filename()) as f:
            xmlfile = f.read()

        root = ET.fromstring("<molcas>\n"+xmlfile+"\n</molcas>")
        energy = root.find("./module[@value='scf']/energy").text

        return float(energy)

    def get_dipole_vector(self):
        import xml.etree.ElementTree as ET

        with open(self.get_xml_filename()) as f:
            xmlfile = f.read()

        root = ET.fromstring("<molcas>\n"+xmlfile+"\n</molcas>")
        dipole_tag = root.find("./module[@value='scf']/dipole")
        dipole = [float(v.text) for v in dipole_tag]

        return dipole

    def get_molecule(self):
        return self.job.mol

    def copy_scforb(self, name="ScfOrb"):
        """
        Copy result ScfOrb file to the working directory.

        @param name: The name of the copied file
        @type  name: str
        """
        self.files.copy_result_file(self.fileid, 21, name)


class MolcasSettings:
    def __init__(self, basis='aug-cc-PVDZ', cholesky=True, functional='LDA', grid=None):
        """
        @param basis:      the basis set for the calculation
        @type  basis:      str
        @param cholesky:   Whether Cholesky decomposition shall be used. In
                           practice necessary for large systems
        @type  cholesky:   bool
        @param functional: exchange-correlation functional for DFT calculations
        @type  functional: str or None
        @param grid:       integration grid for DFT (available: COARSE, SG1GRID, FINE, ULTRAFINE)
        @type  grid:       str or None
        """
        self.method = None
        self.basis = None
        self.cholesky = None
        self._functional = None
        self.grid = None

        self.set_basis(basis)
        self.set_functional(functional)
        self.set_cholesky(cholesky)
        self.set_grid(grid)

    def set_method(self, method):
        """
        Select the computational method.

        @param method: string identifying the selected method,
                       available: Currently supported: HF, DFT, CASSCF, DMRG
        @type  method: str
        """
        if method not in ("HF", "DFT", "CASSCF", "DMRG"):
            raise PyAdfError("The theory " + method + " is not known for a molcas calculation in PyADF.")

        self.method = method.upper()

    def set_basis(self, basisset):
        """
        Select a basis set to be used for all atoms.

        @param basisset: Name of a basis set (e.g. C{cc-pVDZ}).
        @type  basisset: str
        """
        self.basis = basisset

    @property
    def functional(self):
        if self.method == 'DFT':
            return self._functional
        else:
            return None

    def set_functional(self, functional):
        """
        Select the exchange-correlation functional for DFT.

        @param functional:
            A string identifying the functional.
            See Orca manual for available options.
        @type functional: str
        """
        self._functional = functional

    def set_cholesky(self, value):
        """
        Switch Cholesky approximation on or off.

        @param value:  C{True} to switch on Cholesky approximation, C{False} to switch off.
        @type  value:  L{bool}
        """
        self.cholesky = value

    def set_grid(self, value):
        """
        Set integration grid for DFT calculations.

        @param value: grid accuracy (available: COARSE, SG1GRID, FINE, ULTRAFINE)
        @type  value: str
        """
        if value is None:
            self.grid = value
        else:
            self.grid = value.upper()

    def __str__(self):
        """
        Get a nicely formatted text block summarizing the settings.

        @returns: Text block
        @rtype:   L{str}
        """
        s = f'   Method: {self.method} '
        if self.cholesky:
            s += '(Cholesky: ON) \n'
        else:
            s += '(Cholesky: OFF) \n'
        s += f'   Basis Set: {self.basis} \n'
        if self.method == 'DFT':
            s += f'   Exchange-correlation functional: {self.functional} \n'
            s += f'   DFT integration grid: {self.grid} \n'
        return s


class MolcasRasSettings(MolcasSettings):
    def __init__(self, basis='aug-cc-PVDZ', cholesky=True, nActEl=None, nActOrbs=None,
                 mDMRG=1000, nInact=None, symmetry='NoSym', occupation=None):
        """
        @param basis:      the basis set for the calculation
        @type  basis:      str
        @param cholesky:   Whether Cholesky decomposition shall be used. In
                           practice necessary for large systems
        @type  cholesky:   bool
        @param nActEl:     Only relevant for CASSCF/DMRG calculations
        @param nActOrbs:   Only relevant for CASSCF/DMRG calculations, one number
                           for each symmetry group
        @type  nActOrbs:   [int]
        @param mDMRG:      The accuracy parameter of DMRG, typically denoted as 'm'
        @type  mDMRG:      int
        @param nInact:     Only relevant for CASSCF/DMRG calculations, number of
                           inactive orbitals for each symmetry group
        @type  nInact:     [int]
        @param symmetry:   The symmetry generators for the molcas input
        @type  symmetry:   string
        @param occupation: ...of the active orbitals, e.g. [4,1,4,1] means the first
                           and third active orbital are initially doubly occupied,
                           while the second and fourth are initially empty
        @type  occupation: [int]
        """
        super().__init__(basis, cholesky, functional=None)

        if nActEl is None:
            self.nActEl = 2
        else:
            self.nActEl = nActEl
        if nActOrbs is None:
            self.nActOrbs = [2]
        else:
            self.nActOrbs = nActOrbs
        self.mDMRG = mDMRG
        if nInact is None:
            self.nInact = [0]
        else:
            self.nInact = nInact
        self.symmetry = symmetry
        if occupation is None:
            self.occupation = [0]
        else:
            self.occupation = occupation

    def __str__(self):
        """
        Get a nicely formatted text block summarizing the settings.

        @returns: Text block
        @rtype:   L{str}
        """
        s = super().__init__()
        s += f'   symmetry: {self.symmetry} \n'
        if self.method in ("CASSCF", "DMRG"):
            s += '   CASSCF / DMRG settings: \n'
            s += '      number of active electrons    : ' + str(self.nActEl) + '\n'
            s += '      number of active orbitals     : ' + str(self.nActOrbs) + '\n'
            s += '      number of inactive electrons  : ' + str(self.nInact) + '\n'
            s += '      occupation of active orbitals : ' + str(self.occupation) + '\n'
        if self.method == "DMRG":
            s += '   DMRG settings: \n'
            s += f'      number of DMRG states: {self.mDMRG:d} '
        return s


class MolcasJob(job):
    def __init__(self, mol, method='HF', settings=None, fdein=None):
        """
        @param method:     Currently supported: HF, DFT, CASSCF, DMRG
        @type  method:     str
        @type  settings:   MolcasSettings or MolcasRasSettings
        @param fdein:      If an embedding potential from an ADF fragment
                           calculation shall be used it is taken from this job.
        @type  fdein:      L{adffragmentsresults}
        """
        super().__init__()
        self.mol = mol

        if method.upper() not in ("HF", "DFT", "CASSCF", "DMRG"):
            raise PyAdfError("The theory " + method + " is not known for a molcas calculation in PyADF.")
        else:
            self.method = method.upper()
            if method in ("CASSCF", "DMRG"):
                print("WARNING: CASSCF and DMRG support for Molcas is not tested and probably not working")

        if settings is None:
            if self.method in ("CASSCF", "DMRG"):
                self.settings = MolcasRasSettings()
            else:
                self.settings = MolcasSettings()
        else:
            if self.method in ("CASSCF", "DMRG") and not isinstance(settings, MolcasRasSettings):
                raise PyAdfError("Molcas CAS/RAS or DMRG calculation requires MolcasRasSettings")
            self.settings = settings

        self.settings.set_method(self.method)

        self.restart = None
        self.fdein = fdein

    def set_restart(self, res):
        """
        Use a previous Molcas job for obtaining the initial orbitals.
        """
        self.restart = res

    def before_run(self):
        if self.restart is not None:
            self.restart.copy_scforb(name='restart.ScfOrb')

        if self.fdein is not None:
            self.fdein.export_embedding_data('EMBPOT')

    def get_runscript(self, nproc=1):
        runscript = ""
        runscript += "cat <<eor >molcasjob.xyz\n"
        runscript += self.mol.get_xyz_file()
        runscript += "eor\n"
        runscript += "cat molcasjob.xyz\n"
        runscript += "\n"
        runscript += "cat <<eor >molcasjob.input\n"
        runscript += self.get_inputfile()
        runscript += "eor\n"
        runscript += "cat molcasjob.input\n"

        runscript += "export MOLCASMEM=4000\n"
        runscript += "export MOLCASDISK=10000\n"
        runscript += "export WorkDir=`pwd`\n"

        if nproc > 1:
            runscript += f"export MOLCAS_NPROCS={nproc:d}\n"

        # execute pymolcas (the main Molcas driver)
        runscript += "pymolcas molcasjob.input\n"

        return runscript

    @staticmethod
    def get_restart_block():
        block = " LUMORB\n"
        block += " FILEORB=restart.ScfOrb\n"
        return block

    def get_charge_spin_block(self):
        block = " Charge= " + str(self.mol.get_charge()) + "\n"
        if self.mol.get_spin() > 0:
            block = " Spin= " + str(self.mol.get_spin()) + "\n"
        return block

    def get_gateway_block(self):
        block = "&Gateway\n"
        block += " coord=molcasjob.xyz\n"
        block += " basis=" + self.settings.basis + "\n"

        # we are hardcoding NoSym here, but allow this to be overridden by
        # the RASSCF settings; Usually, symmetry should be a property of the
        # molecule, but for RAS, the input is very symmetry-dependent
        if isinstance(self.settings, MolcasRasSettings):
            sym = self.settings.symmetry
        else:
            sym = 'NoSym'
        block += " group=" + sym + "\n"

        return block

    def get_seward_block(self):
        block = "&Seward\n"
        if self.settings.cholesky:
            block += " CHOLesky\n"
        if (self.method == "DFT") and (self.settings.grid is not None):
            block += " GRID Input\n"
            block += f"  GRID={self.settings.grid}\n"
            block += " END Of Grid Input\n"
        if self.fdein is not None:
            block += " EMBEdding\n"
            block += "  EMBInput=$PWD/EMBPOT\n"
            block += " ENDEmbedding\n"

        return block

    def get_scf_block(self):
        block = "&SCF\n"
        block += self.get_charge_spin_block()
        if self.method == 'DFT':
            block += " KSDFT=" + self.settings.functional + "\n"
        if self.restart is not None:
            block += self.get_restart_block()

        return block

    def get_rasscf_block(self):
        block = "&RASSCF\n"
        if self.method == "DMRG":
            block += " DMRG\n"
            block += " PREServed=" + str(self.settings.mDMRG) + "\n"
            block += " RGInput\n"
            block += "  CPUS=" + os.getenv('OMP_NUM_THREADS', 1) + "\n"
            block += "  NSWEEPS=6\n"
            block += "  NWARMUP=2\n"
            block += "  NMAIN=4\n"
            if self.settings.occupation != [0]:
                block += "  OCCUPATION= "
                for occThisOrb in self.settings.occupation:
                    block += str(occThisOrb) + " "
                block += "\n"
            block += " ENDRG\n"
            block += " LUMOrb\n"
        block += " nActEl= " + str(self.settings.nActEl) + " 0 0\n"
        block += " Ras2= "
        for nActOrbThisSym in self.settings.nActOrbs:
            block += str(nActOrbThisSym) + " "
        block += "\n"
        if self.settings.nInact != [0]:
            block += " Inactive= "
            for nInactThisSym in self.settings.nInact:
                block += str(nInactThisSym) + " "
            block += "\n"
            self.get_charge_spin_block()
        return block

    def get_inputfile(self):
        infile_string = self.get_gateway_block()
        infile_string += self.get_seward_block()
        if self.method in ("HF", "DFT"):
            infile_string += self.get_scf_block()
        elif self.method in ("CASSCF", "DMRG"):
            infile_string += self.get_scf_block()
            infile_string += self.get_rasscf_block()

        return infile_string

    def result_filenames(self):
        return ['molcasjob.ScfOrb', 'molcasjob.RasOrb', 'molcasjob.ScfOrb',
                'molcasjob.scf.molden', 'xmldump']

    def check_success(self, outfile, errfile):
        f = open(outfile, encoding='utf-8')
        success = False
        for ll in f.readlines()[-10:]:
            if '.# Happy landing! #.' in ll:
                success = True
            if 'error' in ll:
                raise PyAdfError("Error termination in Molcas")
        return success

    @property
    def checksum(self):
        import hashlib

        m = hashlib.md5()
        m.update(self.mol.get_xyz_file().encode('utf-8'))
        m.update(self.get_inputfile().encode('utf-8'))

        if self.restart is not None:
            m.update(b'Restarted from Molcas job \n')
            m.update(self.restart.checksum.encode('utf-8'))

        if self.fdein is not None:
            m.update(b'Embedding potential imported from ADF job \n')
            m.update(self.fdein.checksum.encode('utf-8'))

        return m.hexdigest()

    def create_results_instance(self):
        return MolcasResults(self)

    def print_jobtype(self):
        return "Molcas job"

    def get_molecule(self):
        return self.mol

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
        if self.restart is not None:
            print(" Using restart file " + self.restart.get_scforb_filename())

    def print_jobinfo(self):
        print(" " + 50 * "-")
        print(" Running " + self.print_jobtype())
        print()

        self.print_molecule()

        self.print_settings()

        self.print_extras()


class MolcasSinglePointJob(MolcasJob):
    pass
