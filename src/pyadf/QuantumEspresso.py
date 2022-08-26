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
 The basics needed for Quantum Espresso calculations

 @author:       Andre Gomes and others
 @organization: CNRS

 @group Jobs:
    QEJob, QESinglePointJob
 @group Settings:
    QESettings
 @group Results:
    QEResults, QESinglePointResults
"""


from abc import abstractmethod

from .Errors import PyAdfError
from .BaseJob import results, job
import re


class QEResults(results):
    """
    Class for results of an Quantum Espresso calculation.
    """

    def __init__(self, j=None):
        """
        Constructor for QEResults.
        """
        super().__init__(j)

    def get_data_filename(self):
        """
        Return the file name of the data file (pwscf.save TAR archive)
        belonging to the results.
        """
        return self.files.get_results_filename(self.fileid, 21)

    def get_xml_filename(self):
        """
        Return the file name of the data-file.xml file belonging to the results.

        The data-file.xml file has the output in machine-readable form
        """
        return self.files.get_results_filename(self.fileid, 66)

    def copy_data(self, name="pwscf.save.tar"):
        """
        Copy result data file to the working directory.

        @param name: The name of the copied file
        @type  name: str
        """
        self.files.copy_result_file(self.fileid, 21, name)

    def copy_xml(self, name="data-file.xml"):
        """
        Copy result XML file to the working directory.

        @param name: The name of the copied file
        @type  name: str
        """
        self.files.copy_result_file(self.fileid, 66, name)


class QESinglePointResults(QEResults):
    """
    Class for results of an Quantum Espresso single point calculation.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_molecule, get_energy
    """

    def __init__(self, j=None):
        """
        Constructor for QESinglePointResults.
        """
        super().__init__(j)

    def get_molecule(self):
        """
        Return the molecular geometry after the Quantum Espresso job.

        @returns: The molecular geometry.
        @rtype:   L{molecule}

        @note: currently not implemented
        """
        pass

    def get_dipole_vector(self):
        """
        Return the dipole moment vector.

        @returns: the dipole moment vector, in atomic units
        @rtype: float[3]
        """
        pass

    def get_energy(self):
        """
        Return the total energy

        @returns: the total energy in atomic units
        @rtype: float
        """
        energy = 0.0

        if not (self.job.settings.control['calculation'] in ['scf']):
            raise PyAdfError('Energy only implemented for pwscf runs at the moment')

        output = self.get_output()
        en_re = re.compile(r"^!\s{4}total energy\s{14}= *(?P<energy>-?\d+\.\d+) Ry")
        for line in reversed(output):
            m = en_re.match(line)
            if m:
                energy = float(m.group("energy"))
                break
        return energy


class QESettings:
    """
    Settings for a Quantum Espresso calculation.
    """

    def __init__(self, runtype='pw', control=None, system=None, electrons=None, ions=None, kpoints=None, cell=None):
        """
        Constructor for QESettings.

        All arguments are optional, leaving out an argument will choose default settings.
        """

        self.control = {}
        self.system = {}
        self.electrons = {}
        self.ions = {}
        self.cell = {}
        self.kpoints = {}

        self.runtype = runtype

        self.functional = None

        if self.runtype in ('pw', 'fdepw'):
            self.set_pw_defaults()

            if control is not None:
                self.set_control(control)
            if system is not None:
                self.set_system(system)
            if electrons is not None:
                self.set_electrons(electrons)
            if ions is not None:
                self.set_ions(ions)
            if cell is not None:
                self.set_cell(cell)
            if kpoints is not None:
                self.set_kpoints(kpoints)

        else:
            raise PyAdfError('Quantum Espresso runs other than PW not yet supported')

    def set_pw_defaults(self):
        # for the dictionaries with options below, the value None will be used to flag
        # which variables will remain with the quantum espresso defaults.
        """
        Default input options for QE. Here None is used to flag optional keywords
        """
        self.control = {
            'calculation': 'scf',
            'title': 'Quantum Espresso calculation via PyADF',
            'verbosity': None,
            'restart_mode': None,
            'wf_collect': None,
            'nstep': None,
            'pseudo': None,
            'pseudo_dir': '$PSEUDO_DIR'
            }

        self.system = {
            'ibrav': -42,
            'cell_dim': [0, 0, 0, 0, 0, 0],
            'cell_abc': [0, 0, 0, 0, 0, 0],
            'nat': 0,
            'ntyp': 0,
            'ecutwfc': 0,
            'nbnd': None,
            'tot_charge': None,
            'tot_magnetization': None,
            'ecutrho': None,
            'ecutfock': None,
            'nosym': None,
            'nosym_evc': None,
            'noinv': None,
            'no_t_rev': None,
            'occupations': None,
            'nspin': None,
            'noncolin': None,
            'input_dft': None,
            'exx_fraction': None,
            'screening_parameter': None,
            'exxdiv_treatment': None,
            'x_gamma_extrapolation': None,
            'fde_nspin': None,
            'fde_print_density': None,
            'fde_print_density_frag': None,
            'fde_print_embedpot': None,
            'fde_xc_funct': None,
            'fde_kin_funct': None
        }

        self.electrons = {
            'electron_maxstep': None,
            'diagonalization': None,
            'efield': None,
            # fat: freeze-and-thaw
            'fde_frag_charge': None,
            'fde_init_rho': None,
            'fde_fat': None,
            'fde_fat_thr': None,
            'fde_fat_maxstep': None,
            'fde_fat_mixing': None
        }

        self.ions = {
            'ion_dynamics': None
            }

        self.cell = {
            'cell_dynamics': None
            }

        self.kpoints = {
            'gamma': None,
            'type': 'automatic',
            'nk': [1, 1, 1],
            'sk': [0, 0, 0],
            'xk': None
        }

    def set_control(self, options):
        self.control.update(options)

        if self.runtype == 'pw' or self.runtype == 'fdepw':
            if not (self.control['calculation'] in ['scf', 'relax', 'nscf']):
                raise PyAdfError('Unknown calculation type for PW')

    def set_system(self, options):
        self.system.update(options)

        for k in ['cell_dim', 'cell_abc']:
            if k in list(self.system.keys()):
                dimension = 0
                for e in self.system[k]:
                    if e != 0:
                        dimension += 1
                if dimension == 0:
                    del self.system[k]

        if 'cell_dim' in list(self.system.keys()) and 'cell_abc' in list(self.system.keys()):
            raise PyAdfError('Both cell_dim and cell_abc specified')
        if 'cell_dim' not in list(self.system.keys()) and 'cell_abc' not in list(self.system.keys()):
            raise PyAdfError('Neither cell_dim nor cell_abc specified')

    def set_electrons(self, options):
        self.electrons.update(options)

    def set_ions(self, options):
        self.ions.update(options)

    def set_cell(self, options):
        self.cell.update(options)

    def set_kpoints(self, options):
        self.kpoints.update(options)

    def set_functional(self, functional):
        """
        Select the exchange-correlation functional for DFT.

        @param functional:
            A string identifying the functional.
            if None is specified, the functional is chosen according to the
            pseudopotentials defined for each atom
        @type functional: str
        """
        if self.functional.lower() == 'lda':
            self.functional = 'SLA VWN'
        else:
            self.functional = functional

    @staticmethod
    def iter_not_none(d):
        return ((k, v) for k, v in list(d.items()) if v is not None)

    def get_control_block(self):
        block = " &control\n"
        for opt, val in self.iter_not_none(self.control):
            if opt == 'calculation' and val not in ['scf', 'relax', 'nscf', 'band', 'cp']:
                raise PyAdfError('Unknown method in Quantum Espresso job')
            if isinstance(val, str):
                block += f"    {opt}='{val}'\n"
            else:
                block += f"    {opt}={val}\n"
        block += " /\n"
        return block

    def get_system_block(self, mol_opts_dict=None):
        block = " &system\n"

        system_plus_mol_dict = self.system.copy()
        if mol_opts_dict is not None:
            system_plus_mol_dict.update(mol_opts_dict)

        for opt, val in self.iter_not_none(system_plus_mol_dict):
            if opt == 'cell_dim':
                dimension = 1
                for e in val:
                    if e != 0 and e is not None:
                        block += '    celldm(' + str(dimension) + ')=' + str(e) + '\n'
                    dimension += 1
            elif opt == 'cell_abc':
                labels = ['a', 'b', 'c', 'cosab', 'cosac', 'cosbc']
                dimension = 0
                for e in val:
                    dimension += 1
                    if e != 0 and e is not None:
                        block += "    " + labels[dimension - 1] + '=' + str(e) + '\n'
                block += f"    {opt}={val}\n"
            else:
                block += f"    {opt}={val}\n"
        block += " /\n"
        return block

    def get_electrons_block(self):
        block = " &electrons\n"
        for opt, val in self.iter_not_none(self.electrons):
            block += f"    {opt}={val}\n"
        block += " /\n"
        return block

    def get_ions_block(self):
        block = ""
        for opt, val in self.iter_not_none(self.ions):
            block += f"    {opt}={val}\n"
        if len(block) > 0:
            block = " &ions\n" + block + " /\n"
        return block

    def get_cell_block(self):
        block = ""
        for opt, val in self.iter_not_none(self.cell):
            block += f"    {opt}={val}\n"
        if len(block) > 0:
            block = " &cell\n" + block + " /\n"
        return block

    def get_kpoints_block(self):
        block = "\nK_POINTS " + self.kpoints['type'] + "\n"
        if self.kpoints['type'] == 'automatic':
            for k in self.kpoints['nk']:
                block += str(k) + "  "
            for k in self.kpoints['sk']:
                block += str(k) + "  "
            block += "\n"
        else:
            raise PyAdfError('this KPOINTS option is not implemented yet')
        return block

    def __str__(self):
        """
        Returns a human-readable description of the settings.
        """
        ss = '   QE settings: summary of options\n\n'
        ss += '    runtype   : ' + str(self.runtype) + '\n'
        ss += '    control   : ' + str(self.control) + '\n'
        ss += '    system    : ' + str(self.system) + '\n'
        ss += '    electrons : ' + str(self.electrons) + '\n'
        ss += '    ions      : ' + str(self.ions) + '\n'
        ss += '    cell      : ' + str(self.cell) + '\n'
        return ss


class QEJob(job):
    """
    An abstract base class for Quantum Espresso jobs.

    Corresponding results class: L{QEResults}

    @group Initialization:
        __init__
    @group Running Internals:
        get_qefile
    """

    def __init__(self, runtype='pw'):
        """
        Constructor for Quantum Espresso jobs.
        """
        super().__init__()
        self.runtype = runtype

    def create_results_instance(self):
        return QEResults(self)

    def print_jobtype(self):
        pass

    def get_qefile(self):
        """
        Abstract method. Should be overwritten to return the Quantum Espresso input file.
        """
        return ""

    @property
    def checksum(self):
        import hashlib
        m = hashlib.md5()
        m.update(self.get_qefile().encode('utf-8'))
        return m.hexdigest()

    def get_runscript(self, nproc=1):
        runscript = "#!/bin/bash \n\n"
        runscript += "cat <<eor >qe.in\n"
        runscript += "echo\n"
        runscript += self.get_qefile()
        runscript += "eor\n"
        runscript += "cat qe.in \n"
        runscript += "echo \n"
        runscript += "touch qe.out \n"
        if self.runtype == "pw":
            runscript += 'if [ -f "$QEBINDIR/fdepw.x" ]; then\n'
            runscript += f"    mpirun -np {nproc:d} $QEBINDIR/pw.x -in qe \n"
            runscript += "else\n"
            runscript += f"    mpirun -np {nproc:d} $QEBINDIR/pw.x -in qe.in \n"
            runscript += "fi\n"
        elif self.runtype == "pp":
            runscript += f"mpirun -np {nproc:d} $QEBINDIR/pp.x -in qe.in \n"
        elif self.runtype == "fdepw":
            runscript += f"mpirun -np {nproc:d} $QEBINDIR/fdepw.x -in qe \n"
        else:
            raise PyAdfError('Unsupported calculation detected when creating runscript')
        runscript += "retcode=$?\n"

        runscript += "cat qe.out \n"
        if self.runtype in ("pw", "fdepw"):
            runscript += "cp pwscf.save/data-file.xml data-file.xml\n"
            runscript += "tar cvf pwscf.save.tar pwscf.save >/dev/null\n"

        runscript += "rm qe.in \n"
        runscript += "exit $retcode \n"

        return runscript

    def result_filenames(self):
        filelist = ['pwscf.save.tar', 'data-file.xml']
        return filelist

    def check_success(self, outfile, errfile):
        # check that Quantum Espresso terminated normally
        if self.runtype in ("pw", "pwscf"):
            with open(outfile, encoding='utf-8') as f:
                line = f.readlines()[-2]
            return "JOB DONE." in line
        else:
            return True


class QESinglePointJob(QEJob):
    """
    A class for Quantum Espresso single point runs.

    See the documentation of L{__init__} and L{QESettings} for details
    on the available options.

    Corresponding results class: L{QESinglePointResults}

    @Note: Right now, PWSCF jobs are supported.
    """

    def __init__(self, mol, pseudo=None, settings=None, options=None):
        """
        Constructor for Quantum Espresso single point jobs.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}

        @param pseudo:
            A dictionary specifying the pseudopotentials to use
            (e.g. C{pseudo={'O':'blyp-mt.UPF', 'H' : 'blyp-vbc.UPF'}}).
        @type pseudo: str or dict

        @param settings: The settings for the QE job. Currently not used.
        @type  settings: L{QESettings}

        @param options:
            Additional options.
            These will each be included directly in the Quantum Espresso input file.
            this will be handy to add the following optional input blocks

            CELL_PARAMETERS
            OCCUPATIONS
            CONSTRAINTS
            ATOMIC_FORCES
        @type options: list of str
        """
        super().__init__(runtype=settings.runtype)

        self.mol = mol

        self.pseudo = None
        self.atomic_pseudos = {}
        self.set_pseudos(pseudo)

        if settings is None:
            self.settings = QESettings()
        else:
            self.settings = settings

        if options is None:
            self._options = []
        else:
            self._options = options

    def create_results_instance(self):
        return QESinglePointResults(self)

    def get_molecule(self):
        return self.mol

    def set_pseudos(self, pseudo):
        self.pseudo = pseudo
        self.atomic_pseudos = {}

        atomtypes = set(self.mol.get_atom_symbols())

        if isinstance(self.pseudo, dict):
            for atom in atomtypes:
                self.atomic_pseudos[atom] = atom + "." + self.pseudo[atom]
        elif isinstance(self.pseudo, str):
            for atom in atomtypes:
                self.atomic_pseudos[atom] = atom + "." + self.pseudo
        else:
            raise PyAdfError('Error selecting pseudopotentials in QE job')

    def get_system_block(self):
        natoms = self.mol.get_number_of_atoms()
        ntypes = len(set(self.mol.get_atom_symbols()))
        charge = self.mol.get_charge()

        mol_opts_dict = {'nat': natoms, 'ntyp': ntypes, 'tot_charge': charge}

        block = self.settings.get_system_block(mol_opts_dict)
        return block

    def get_atomic_species_block(self):
        """
        Example:

        ATOMIC_SPECIES
         O 16.0d0 O.blyp-mt.UPF
         H 1.00d0 H.blyp-vbc.UPF
        """
        from .Utils import PeriodicTable as PT

        block = "\nATOMIC_SPECIES\n"
        for atom in set(self.mol.get_atom_symbols()):
            block += f" {atom} {PT.get_mass(atom):10.5f} {self.atomic_pseudos[atom]}\n"

        block += "\n"
        return block

    def get_atomic_positions_block(self):
        """
        Example:

        ATOMIC_POSITIONS (angstrom)
           O     0.0099    0.0099    0.0000
           H     1.8325   -0.2243   -0.0001
           H    -0.2243    1.8325    0.0002
        """

        block = "\nATOMIC_POSITIONS (angstrom)\n"
        block += self.mol.print_coordinates(index=False)
        block += "\n"
        return block

    def get_qefile(self):
        qefile = self.settings.get_control_block()
        qefile += self.get_system_block()
        qefile += self.settings.get_electrons_block()
        qefile += self.settings.get_ions_block()
        qefile += self.settings.get_cell_block()
        qefile += self.get_atomic_species_block()
        qefile += self.get_atomic_positions_block()
        qefile += self.settings.get_kpoints_block()
        qefile += self.get_options_block()
        qefile += self.get_other_blocks()
        return qefile

    def get_options_block(self):
        return '\n'.join(self._options)

    @abstractmethod
    def get_other_blocks(self):
        """
        Abstract method. Allows extending the QE input file in subclasses.
        """
        return ""

    def print_jobtype(self):
        return "Quantum Espresso single point job"

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
        print("   Pseudopotentials: ", str(self.atomic_pseudos))
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
