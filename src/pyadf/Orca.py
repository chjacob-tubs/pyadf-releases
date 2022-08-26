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


from .Errors import PyAdfError
from .BaseJob import job
from .DensityEvaluator import GTODensityEvaluatorMixin


class OrcaResults(GTODensityEvaluatorMixin):
    """

    Class for results of a Orca calculation.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_singlepointenergy, get_optimized_molecule,
        get_dipole_vector, get_dipole_magnitude,
        get_hess_value, get_frequencies, get_intensities

    """

    def __init__(self, j=None):
        """
        Constructor
        """
        super().__init__(j)
        self.resultstype = "Orca results"

    def get_gwb_filename(self):
        return self.files.get_results_filename(self.fileid, tape=21)

    def get_prop_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=67)
        except PyAdfError:
            raise PyAdfError("ORCA Properties file not found")
        return fn

    def get_engrad_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=68)
        except PyAdfError:
            raise PyAdfError("ORCA engrad gradients file not found")
        return fn

    def get_traj_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=13)
        except PyAdfError:
            raise PyAdfError("ORCA trajectory file not found")
        return fn

    def get_xyz_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=69)
        except PyAdfError:
            raise PyAdfError("ORCA geometry optimization xyz file not found")
        return fn

    def get_hess_filename(self):
        try:
            fn = self.files.get_results_filename(self.fileid, tape=42)
        except PyAdfError:
            raise PyAdfError("ORCA frequencies hessian file not found")
        return fn

    @staticmethod
    def _fix_orca_molden_file(file_contents):
        """
        Orca writes non-standard molden files, which need to be fixed
        in order to agree with the common convention.

        This method builds on the detective work that the developers
        of the multiwfn project (see http://sobereva.com/multiwfn/)
        have done, in particular the comments in multiwfn's fileIO.f90.
        """
        import math
        import scipy.special

        lval_dict = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'j': 7}

        lines = file_contents.splitlines()
        lines_basis_fixed = []

        section = None
        lval = None
        nbas = 0

        aos_to_invert = []

        for line in lines:
            if line.startswith('['):
                section = line[1:line.index(']')].upper()
                lines_basis_fixed.append(line)
            elif line and section == 'GTO':
                dat = line.split()
                if dat[0].lower() in lval_dict:
                    lval = lval_dict[dat[0].lower()]
                    nbas = nbas + (2 * lval + 1)

                    if lval == 3:  # f functions; f(+3) and f(-3) need to be changed
                        aos_to_invert.append(nbas - 1)
                        aos_to_invert.append(nbas)
                    elif lval == 4:  # g functions; (+/- 3) and (+/- 4) need to be changed
                        aos_to_invert.append(nbas - 3)
                        aos_to_invert.append(nbas - 2)
                        aos_to_invert.append(nbas - 1)
                        aos_to_invert.append(nbas)
                    elif lval == 5:  # h functions; (+/- 3) and (+/- 4) need to be changed
                        aos_to_invert.append(nbas - 5)
                        aos_to_invert.append(nbas - 4)
                        aos_to_invert.append(nbas - 3)
                        aos_to_invert.append(nbas - 2)
                    # there might be more problems with higher l, but usually
                    # molden files cannot handle those anyways

                    lines_basis_fixed.append(line)
                elif not dat[0].isdigit():
                    exponent = float(dat[0])
                    coeff = float(dat[1])

                    # correct the normalization of the basis functions
                    # Orca includes the normalization constants in the
                    # contraction coefficients, which needs to be corrected
                    renorm = math.sqrt(2.0 * (2.0 * exponent)**(lval + 1.5) / scipy.special.gamma(lval + 1.5))
                    renorm = renorm * math.sqrt(1.0 / (4.0 * math.pi))

                    # with this correction, the (contracted) Orca basis functions
                    # might not be normalized to one, but this is corrected by
                    # PySCF by default, and the MO coefficients correctly refer
                    # to the normalized basis functions

                    fixed_line = f'{exponent:21.10f} {coeff / renorm:21.10f}'
                    lines_basis_fixed.append(fixed_line)
                else:
                    lines_basis_fixed.append(line)

            else:
                lines_basis_fixed.append(line)

        # now we do a second pass in which we correct the MO coefficients
        # Orca molden files use the wrong sign for F(+/-3), G(+/-3), G(+/-4),
        # H(+/-3) and H(+/-4)
        # Above, the relevant AO numbers have been identified and collected
        # in the list aos_to_invert

        lines_mos_fixed = []

        for line in lines_basis_fixed:
            if line.startswith('['):
                section = line[1:line.index(']')].upper()
                lines_mos_fixed.append(line)
            elif line and section == 'MO':
                dat = line.split()
                if dat[0].isdigit():
                    nao = int(dat[0])
                    coeff = float(dat[1])
                    if nao in aos_to_invert:
                        coeff = -coeff
                    fixed_line = f"{nao:5d} {coeff:20.12f}"
                    lines_mos_fixed.append(fixed_line)
                else:
                    lines_mos_fixed.append(line)
            else:
                lines_mos_fixed.append(line)

        file_contents_fixed = '\n'.join(lines_mos_fixed)

        return file_contents_fixed

    def read_molden_file(self):
        """
        Returns Molden results file as a string.

        Orca writes non-standard molden files, which are fixed
        here on the fly (see L{_fix_orca_molden_file}).
        """

        try:
            molden_filename = self.files.get_results_filename(self.fileid, tape=41)
        except PyAdfError:
            raise PyAdfError("ORCA Molden file not found")

        with open(molden_filename, encoding='utf-8') as f:
            content = f.read()

        content = self._fix_orca_molden_file(content)

        return content

    def get_energy(self):
        import re

        output = self.get_output()
        energy = None
        en_re = re.compile(r"(FINAL SINGLE POINT ENERGY) *(?P<energy>-?\d+\.\d+)")
        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group('energy'))
        return energy

    def get_molecule(self):
        from .Molecule import molecule

        try:
            xyz_fn = self.get_xyz_filename()
        except PyAdfError:
            return self.job.mol
        return molecule(xyz_fn)

    def get_dipole_vector(self):
        prop_fn = self.get_prop_filename()

        with open(prop_fn) as f:
            lines = f.readlines()
        found = None
        for i, ll in enumerate(lines):
            if 'Total Dipole moment:' in ll:
                found = i
        if found is None:
            raise PyAdfError('Dipole moment not found in ORCA properties file')
        vector = [float(ii.split()[1]) for ii in lines[found+2:found+5]]
        return vector

    def get_hess_values(self, values):
        if max(values) > 5 :
            raise PyAdfError('IR spectrum entry only contains 6 colums!')

        hess_fn = self.get_hess_filename()

        with open(hess_fn) as f:
            lines = f.readlines()
        found = None
        for i, ll in enumerate(lines):
            if ll.startswith('$ir_spectrum'):
                found = i
        if found is None:
            raise PyAdfError('IR spectrum not found in ORCA hess file')

        n_values = int(lines[found+1].strip())

        hess_values = []

        for i in range(found+8,found+n_values+2):
            sequence = lines[i].strip().split()
            hess_values.append([float(sequence[n]) for n in values])

        return hess_values

    def get_frequencies(self):
        freqs = self.get_hess_values([0])
        return freqs

    def get_intensities(self):
        ints = self.get_hess_values([2])
        return ints

class OrcaSettings:
    """
    Class that holds the settings for a orca calculation
    """

    def __init__(self, method='DFT', basis='def2-SVP', functional='LDA', ri=None, disp=False, maxiter=None):
        """
        Constructor for OrcaSettings.

        All arguments are optional, leaving out an argument will choose default settings.

        @param method: the computational method
        @type method: str
        @param functional:
            exchange-correlation functional for DFT calculations, see L{set_functional}
        @type  functional: str
        @param basis: the basis set for the calculation
        @type  basis: str
        @param ri:  C{True} to switch on RI approximation, C{False} to switch off; C{None} to use method's default.
        @type  ri:  L{bool}
        @param maxiter: the maximum number of SCF iterations
        @type  maxiter: int
        """
        # declare all
        self.method = None
        self.basis = None
        self._functional = None
        self.ri = None
        self.disp = None
        self.maxiter = None
        self.extra_keywords = None
        self.extra_blocks = None

        # initialize the setter
        self.set_method(method)
        self.set_basis(basis)
        if self.method == 'DFT':
            self.set_functional(functional)
        self.set_ri(ri)
        self.set_disp(disp)
        self.set_maxiter(maxiter)

    def set_method(self, method):
        """
        Select the computational method.

        @param method: string identifying the selected method.
        @type  method: str
        """
        self.method = method.upper()

    def set_basis(self, basisset):
        """
        Select a basis set to be used for all atoms.

        @param basisset: Name of a basis set (e.g. C{def2-TZVP}).
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
        if self.method == 'DFT':
            self._functional = functional
        else:
            raise PyAdfError('Functional can only be set for DFT calculations')

    def set_ri(self, value):
        """
        Switch RI approximation on or off.

        @param value:  C{True} to switch on RI approximation, C{False} to switch off.
        @type  value:  L{bool}
        """
        if value is None:
            self.ri = (self.method == 'DFT')
        else:
            self.ri = value

    def set_disp(self, dispersion):
        """
        Set dispersion correction

        @param dispersion:  A string identifying the dispersion correction.
        @type  dispersion:  bool or str
        """
        if dispersion not in ['D3', 'D3BJ', 'D3Zero', 'D4', True, False]:
            raise PyAdfError('Unknown dispersion correction in OrcaJob')
        self.disp = dispersion

    def set_maxiter(self, maxiter):
        """
        Set the maximum number of iteration cycles to perform before giving up
        the optimization.

        @param maxiter: Max. number of cycles
        @type  maxiter: int
        """
        self.maxiter = maxiter

    def set_extra_keywords(self, keywords, append=False):
        """
        Set additional keywords for the calculation

        @param keywords: Avaible keywords. See Orca manual for available keywords.
        @type  keywords: str or list or None
        @param append: If True, append the extra keywords to those already set.
        @type  append: bool
        """
        if keywords is None:
            if not append:
                self.extra_keywords = None
        else:
            kws = keywords
            if not isinstance(keywords, list):
                kws = [keywords]
            if self.extra_keywords is None:
                self.extra_keywords = kws
            elif append:
                self.extra_keywords = self.extra_keywords + kws
            else:
                self.extra_keywords = kws

    def set_extra_block(self, block, append=False):
        """
        Set additional input blocks for the calculation

        @param block: Avaible keywords. See Orca manual for available keywords.
        @type  block: str or None
        @param append: If True, append the extra keywords to those already set.
        @type  append: bool
        """
        if block is None:
            if not append:
                self.extra_blocks = None
        else:
            if self.extra_blocks is None:
                self.extra_blocks = [block]
            elif append:
                self.extra_blocks = self.extra_blocks.append(block)
            else:
                self.extra_blocks = [block]

    def __str__(self):
        """
        Get a nicely formatted text block summarizing the settings.

        @returns: Text block
        @rtype:   L{str}
        """
        s = f'   Method: {self.method} '
        if self.ri:
            s += '(RI: ON) \n'
        else:
            s += '(RI: OFF) \n'
        s += f'   Basis Set: {self.basis} \n'
        if self.method == 'DFT':
            s += f'   Exchange-correlation functional: {self.functional} \n'
        if self.maxiter is not None:
            s += f'   Maximum number of SCF iterations {self.maxiter:d} \n'
        if self.extra_keywords is not None:
            s += '   Extra keywords: ' + str(self.extra_keywords)
        if self.extra_blocks is not None:
            s += '   Extra input blocks: \n'
            s += '\n'.join(self.extra_blocks)
        return s

    def get_keywords(self):
        kws = [self.basis, self.method]
        if self.method == 'DFT':
            kws.append(self.functional)
        if self.disp:
            kws.append(self.disp)
        if self.ri:
            kws.append('RI')
        else:
            kws.append('NoRI')
        if self.extra_keywords is not None:
            kws = kws + self.extra_keywords
        return kws

    def get_scf_block(self):
        block = ''
        if self.maxiter is not None:
            block += "%scf\n"
            block += f"MaxIter {self.maxiter:d} \n"
            block += "end\n"
        return block

    def get_input_blocks(self):
        blocks = ''
        blocks += self.get_scf_block()

        if self.extra_blocks is not None:
            blocks += '\n'.join(self.extra_blocks) + '\n'

        return blocks


class OrcaJob(job):
    """
    A class for Orca jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None):
        """
        Constructor
        """
        self.settings = None
        self.mol = mol

        if settings is None:
            self.settings = OrcaSettings()
        else:
            self.settings = settings

        super().__init__()

    def print_jobtype(self):
        return "Orca job"

    def get_molecule(self):
        return self.mol

    def get_runscript(self, nproc=1):

        runscript = ""
        runscript += "cat <<eor >INPUT.inp\n"
        runscript += self.get_orcafile(nproc)
        runscript += "eor\n"
        runscript += "cat INPUT.inp\n"

        runscript += 'if [ -z "$OPAL_PREFIX" ]; then\n'
        runscript += "  $ORCA_PATH/orca INPUT.inp\n"
        runscript += "else\n"
        runscript += '  $ORCA_PATH/orca INPUT.inp "--prefix $OPAL_PREFIX"\n'
        runscript += "fi\n"
        runscript += "retcode=$?\n"

        runscript += "$ORCA_PATH/orca_2mkl INPUT -molden\n"

        runscript += "rm INPUT.inp\n"
        runscript += "exit $retcode\n"

        return runscript

    def result_filenames(self):
        return ['INPUT.gbw', 'INPUT_property.txt', 'INPUT.engrad', 'INPUT.xyz',
                'INPUT_trj.xyz', 'INPUT.molden.input', 'INPUT.hess']

    def check_success(self, outfile, errfile):
        f = open(outfile, encoding="utf-8")
        success = False
        for ll in f.readlines()[-10:]:
            if '**ORCA TERMINATED NORMALLY**' in ll:
                success = True
            if 'ORCA finished by error termination' in ll:
                raise PyAdfError("Error termination in ORCA")
        return success

    def get_keywords(self):
        return self.settings.get_keywords()

    def get_input_blocks(self, nproc=1):
        blocks = ''
        blocks += self.get_parallel_block(nproc)
        blocks += self.settings.get_input_blocks()
        return blocks

    @staticmethod
    def get_parallel_block(nproc):
        block = ''
        if nproc > 1:
            block += "%pal\n"
            block += f"nprocs {nproc:d} \n"
            block += "end\n"
        return block

    def get_orcafile(self, nproc=1):

        orcafile = "! " + ' '.join(self.get_keywords()) + "\n"
        orcafile += self.get_input_blocks(nproc) + "\n"

        orcafile += f"*xyz {self.mol.get_charge():d} {self.mol.get_spin() + 1:d} \n"
        xyz_file = self.mol.get_xyz_file()
        orcafile += ''.join(xyz_file.splitlines(True)[2:])
        orcafile += "*\n"

        return orcafile

    def create_results_instance(self):
        return OrcaResults(self)

    @property
    def checksum(self):
        import hashlib

        m = hashlib.md5()
        m.update(self.get_orcafile().encode('utf-8'))
        return m.hexdigest()

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


class OrcaSinglePointJob(OrcaJob):
    """
    A class for Orca single point jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None):
        super().__init__(mol, settings)

    def print_jobtype(self):
        return "Orca single point job"

    def get_keywords(self):
        return ['SP'] + self.settings.get_keywords()


class OrcaGeometryOptimizationJob(OrcaJob):
    """
    A class for Orca geometry optimization jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None):
        super().__init__(mol, settings)

    def print_jobtype(self):
        return "Orca geometry optimization job"

    def get_keywords(self):
        return ['OPT'] + self.settings.get_keywords()


class OrcaFrequenciesJob(OrcaJob):
    """
    A class for Orca frequencies jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None):
        super().__init__(mol, settings)

    def print_jobtype(self):
        return "Orca frequencies job"

    def get_keywords(self):
        return ['FREQ'] + self.settings.get_keywords()

class OrcaOptFrequenciesJob(OrcaGeometryOptimizationJob):
    """
    A class for Orca optimization and frequencies jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None):
        super().__init__(mol, settings)

    def print_jobtype(self):
        return "Orca optimization and frequencies job"

    def get_keywords(self):
        return ['OPT FREQ'] + self.settings.get_keywords()
