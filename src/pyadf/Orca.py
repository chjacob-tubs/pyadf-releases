
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


from .Errors import PyAdfError
from .BaseJob import job
from pyadf.PyEmbed.DensityEvaluator import GTODensityEvaluatorMixin


class OrcaResults(GTODensityEvaluatorMixin):
    """
    Class for results of a Orca calculation.

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_singlepointenergy, get_optimized_molecule,
        get_dipole_vector, get_dipole_magnitude,
        get_hess_value, get_frequencies, get_ir_intensities

    """

    def __init__(self, j=None):
        """
        Constructor
        """
        super().__init__(j)
        self.resultstype = "Orca results"

    def get_gbw_filename(self):
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

    def get_total_energy(self):
        """
        Always returns Orca's "FINAL SINGLE POINT ENERGY".
        @return: float
        """
        import re

        output = self.get_output()
        energy = None
        en_re = re.compile(r"(FINAL SINGLE POINT ENERGY) *(?P<energy>-?\d+\.\d+)")
        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group('energy'))
        return energy

    def get_energy(self, level='SCF'):
        if level in ['SCF', 'HF']:
            energy = self.get_scf_energy('Total Energy')
        elif level == 'MP2':
            energy = self.get_scf_energy('Total Energy') \
                     + self.get_correlation_energy(r'E\(MP2\)')
        elif level == 'CCSD':
            energy = self.get_correlation_energy(r'E\(CCSD\)')
        elif level == 'CCSD(T)':
            energy = self.get_correlation_energy(r'E\(CCSD\(T\)\)')
        else:
            raise PyAdfError("Energy for level " + level + " not found in Orca output.")

        return energy

    def get_kinetic_energy(self):
        """
        Return the kinetic energy (requires TOTALENERGY option).

        @returns: the kinetic energy in atomic units
        @rtype: float
        """
        return self.get_scf_energy('Kinetic Energy')

    def get_scf_energy(self, what='Total Energy'):
        """
        Takes a string of what you want

        example from orca output
        ----------------
        TOTAL SCF ENERGY
        ----------------

        Total Energy       :         -455.84332339 Eh          -12404.12744 eV

        Components:
        Nuclear Repulsion  :          284.30889947 Eh            7736.43847 eV
        Electronic Energy  :         -740.15222286 Eh          -20140.56591 eV
        One Electron Energy:        -1196.67492193 Eh          -32563.18011 eV
        Two Electron Energy:          456.52269908 Eh           12422.61420 eV

        Virial components:
        Potential Energy   :         -910.32729499 Eh          -24771.26504 eV
        Kinetic Energy     :          454.48397159 Eh           12367.13760 eV
        Virial Ratio       :            2.00299098

        """
        import re

        output = self.get_output()
        found_section = None
        for i, ll in enumerate(output):
            if 'TOTAL SCF ENERGY' in ll:
                found_section = i
        if not found_section:
            raise PyAdfError('TOTAL SCF ENERGY not found in Orca output')
        else:
            output = output[found_section:found_section+20]

        energy = None
        en_re = re.compile(r"(" + what + r") *(:) *(?P<energy>-?\d+\.\d+)")
        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group('energy'))
        return energy

    def get_correlation_energy(self, what):
        """
        Takes a string of what you want

        example from orca output
        Triples Correction (T)                     ...     -0.006679230
        Scaling of triples based on CCSD energies (Peterson et al. Molecular Physics 113, 1551 (2015))
        E(T*) = f*E(T) where f = E(F12-CCSD)/E(CCSD)
        f = CCSD (with F12)/ CCSD (without F12)    ...      1.000000000
        Scaled triples correction (T)              ...     -0.006679230

        Final correlation energy                   ...     -0.433259968
        E(CCD)                                     ...   -152.352433731
        E(CCD(T))                                  ...   -152.359112961
        Initial guess performed in     0.781 sec
        E(0)                                       ...   -456.296109838
        E(MP2)                                     ...     -1.345881148
        Initial E(tot)                             ...   -457.641990986
        <T|T>                                      ...      0.717812866
        Number of pairs included                   ... 300
        Total number of pairs                      ... 300
        """
        import re
        output = self.get_output()
        energy = None
        en_re = re.compile(r"(" + what + r") *(\.\.\.) *(?P<energy>-?\d+\.\d+)")
        for line in output:
            m = en_re.match(line)
            if m:
                energy = float(m.group('energy'))
        return energy

    def get_nuclear_repulsion_energy(self):
        """
        Return the nuclear repulsion energy (as read from TAPE).

        @returns: the nuclear repulsion energy in atomic units
        @rtype: float
        """
        return self.get_scf_energy('Nuclear Repulsion')

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
        vector = [float(ii.split()[1]) for ii in lines[found + 2:found + 5]]
        return vector

    def get_hess_values(self, values):
        if max(values) > 5:
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

        n_values = int(lines[found + 1].strip())

        hess_values = []

        for i in range(found + 8, found + n_values + 2):
            sequence = lines[i].strip().split()
            hess_values.append([float(sequence[n]) for n in values])

        return hess_values

    def get_frequencies(self):
        freqs = [f[0] for f in self.get_hess_values([0])]
        return freqs

    def get_ir_intensities(self):
        ints = [i[0] for i in self.get_hess_values([2])]
        return ints


class OrcaExcitationResults(OrcaResults):

    def __init__(self, j=None):
        super().__init__(j)
        self.resultstype = "Orca excitation (TD-DFT) results"

    def read_cis_results(self):
        import numpy as np

        prop_fn = self.get_prop_filename()

        with open(prop_fn) as f:
            lines = f.readlines()
        found = None
        for i, ll in enumerate(lines):
            if ll.startswith('$ CIS_ABS'):
                found = i
        if found is None:
            raise PyAdfError('Excitation results not found in ORCA properties file')

        nroots = int(lines[found + 4].split()[-1])

        cis_results = []
        for i in range(found + 7, found + 7 + nroots):
            sequence = lines[i].strip().split()
            sequence = [float(s) for s in sequence[1:]]
            cis_results.append(sequence)

        cis_results = np.array(cis_results)
        return cis_results

    def get_excitation_energies(self):
        """
        Return excitation energies in eV.

        @return: excitation energies
        @rtype: np.array
        """
        from .Utils import au_in_eV

        exens = self.read_cis_results()[:, 0]
        exens = exens * au_in_eV

        return exens

    def get_oscillator_strengths(self):
        """
        Return oscillator strength (length representation).

        @return: oscillator strengths (dimensionless)
        @rtype: np.array
        """
        return self.read_cis_results()[:, 1]

    def get_transition_dipole_vector(self):
        """
        Electronic transition dipole moments (length representation).

        @returns: Numpy array containing transition dipole moments.
        @rtype: np.array(nroots, 3)
        """
        return self.read_cis_results()[:, 3:]


class OrcaSettings:
    """
    Class that holds the settings for a orca calculation
    """

    def __init__(self, method='DFT', basis='def2-SVP', functional='LDA', ri=None, disp=False, cpcm=None,
                 memory=None, converge=None, maxiter=None, printmos=False):
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
        @param ri:  C{True} to switch on RI approximation, C{False} to switch off,
                    string to set other method; C{None} to use method's default.
        @type  ri:  L{bool} or str
        @type  ri:  None, bool, or str
        @param cpcm: the selected solvent for the calculation
        @type cpcm: str
        @param memory: Changes the default memory per core (MB)
        @type memory: str or int
        @param converge: the chosen convergence tolerance of the SCF;
                         Default: NormalSCF (<1.0e06 au) for single point calculation and
                                  TightSCF (<1.0e-08 au) for geometry optimization
        @type converge: str
        @param maxiter: the maximum number of SCF iterations
        @type  maxiter: int
        @param printmos : Selects if MO coefficients and basis set should be printed into
                          output file
        @type printmos : bool
        """
        # declare all
        self.method = None
        self.basis = None
        self._functional = None
        self.ri = None
        self.disp = None
        self.cpcm = None
        self.memory = None
        self.converge = None
        self.maxiter = None
        self.printmos = None
        self.extra_keywords = None
        self.extra_blocks = None
        self.oocc_density = None
        self.brueckner_density = None
        self.simplified_z = None
        self.pointcharges = None
        self.ignoreconv = None

        # initialize the setter
        self.set_method(method)
        self.set_basis(basis)
        if self.method == 'DFT':
            self.set_functional(functional)
        self.set_ri(ri)
        self.set_disp(disp)
        self.set_cpcm(cpcm)
        self.set_memory(memory)
        self.set_converge(converge)
        self.set_maxiter(maxiter)
        self.set_printmos(printmos)

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

        @param value:  C{True} to switch on RI approximation, C{False} to switch off, string to set other
        approximation (RIJK, RIJCOSX, etc.)
        @type  value:  L{bool} or Lstr
        """
        if value is None:
            self.ri = None
        elif isinstance(value, str):
            self.ri = value
        elif value:
            self.ri = "RI"
        elif not value:
            self.ri = "NoRI"
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

    def set_cpcm(self, cpcm):
        """
        Sets the CPCM solvent model, e.g. "WATER"

        @param cpcm: Chosen solvent model
        @type cpcm: str
        """
        self.cpcm = cpcm

    def set_memory(self, memory):
        """
        Sets the memory per core for the calculation.

        @param memory: Choosen memory (MB)
        @type memory: str or int
        """

        self.memory = memory
        if self.memory is not None:
            self.set_extra_block("%maxcore " + str(self.memory) + "\n", append=False)

    def set_converge(self, converge):
        """
        Sets the SCF convergence threshold, e.g. "TightSCF"

        @param converge: Choosen convergence level
        @type converge: str
        """
        self.converge = converge

    def set_maxiter(self, maxiter):
        """
        Set the maximum number of iteration cycles to perform before giving up
        the optimization.

        @param maxiter: Max. number of cycles
        @type  maxiter: int
        """
        self.maxiter = maxiter

    def set_printmos(self, printmos):
        """
        Sets the option, wether MO coefficients and basis set should be printed into
        the output file.

        @param printmos : Choosen option (yes/no)
        @type printmos : bool
        """
        self.printmos = printmos

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
                self.extra_blocks.append(block)
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
        if self._functional not in ['PBEh-3C', 'HF-3C']:
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
        kws = [self.method]
        if self._functional not in ['PBEh-3C', 'HF-3C']:
            kws.append(self.basis)
        if self.method == 'DFT':
            kws.append(self.functional)
        if self.disp:
            kws.append(self.disp)
        if self.ri is not None:
            kws.append(self.ri)
        if self.converge is not None:
            kws.append(self.converge)
        if self.cpcm:
            kws.append('CPCM' + '(' + self.cpcm + ')')
        if self.printmos:
            kws.append("PrintMOs")
            kws.append("PrintBasis")
        if self.extra_keywords is not None:
            kws = kws + self.extra_keywords
        return kws

    def get_scf_block(self):
        block = ''
        block += "%scf\n"
        if self.maxiter is not None:
            block += f" MaxIter {self.maxiter:d}\n"
        if self.ignoreconv:
            block += " IgnoreConv True\n"
        block += "end\n"
        return block

    def get_mdci_block(self):
        block = ''
        block += "%mdci\n"
        if self.maxiter is not None:
            block += f" MaxIter {self.maxiter:d}\n"
        if self.oocc_density:
            block += " Denmat orbopt\n"
            block += " density orbopt\n"
        if self.brueckner_density:
            block += " Brueckner true\n"
        if self.simplified_z:
            block += " ZSimple true\n"
        block += "end\n"
        return block

    def get_input_blocks(self):
        blocks = ''
        blocks += self.get_scf_block()
        if 'CC' in self.method or 'CI' in self.method:
            blocks += self.get_mdci_block()

        if self.extra_blocks is not None:
            blocks += '\n'.join(self.extra_blocks) + '\n'

        return blocks


class OrcaTDDFTSettings:

    def __init__(self, nroots=None, maxdim=None, triplets=False, tda=False, maxiter=None):
        """
        Settings for ORCA TDDFT calculations.

        @param nroots Number of excited states
        @type nroots: int or str
        @param maxdim: size of Davidson expansion space (Davidson expansion space = MaxDim * NRoots,
                       see Orca manual sec. 8.4)
        @type maxdim: int or str
        @param triplets: Triple excitations allowed or not (no by default)
        @type triplets: bool
        @param tda: Is the Tamm Dancoff approximation requested (no by default)
        @type tda: bool
        @param maxiter: Number of max. steps for TDDFT calculation (default 100)
        @type maxiter: int
        """
        self.nroots = nroots
        self.maxdim = maxdim
        self.iroot = None
        self.triplets = triplets
        self.tda = tda
        self.maxiter = maxiter

    def get_tddft_block(self):
        block = "%tddft\n"

        if self.nroots is not None:
            block += f"NRoots {self.nroots}\n"
        if self.maxdim is not None:
            block += f"MaxDim {self.maxdim}\n"
        if self.iroot is not None:
            block += f"IRoot {self.iroot}\n"
        if self.triplets:
            block += "Triplets true\n"
        if not self.tda:
            block += "TDA false\n"
        if self.maxiter:
            block += "MaxIter " + str(self.maxiter) + "\n"
        block += "end\n"

        return block

    def __str__(self):
        """
        Get a nicely formatted text block summarizing the settings.

        @returns: Text block
        @rtype:   L{str}
        """
        if self.tda:
            s = f'   Excitation energies: TDDFT (TDA) \n'
        else:
            s = f'   Excitation energies: TDDFT \n'
        if self.triplets:
            s += f'   Triplet excitations: ON \n'
        else:
            s += f'   Triplet excitations: OFF \n'
        s += f'   Number of roots: {self.nroots} \n'
        if self.iroot is not None:
            s += f'   Root to follow: {self.iroot} \n'
        s += f'   Maxdim: {self.maxdim} \n'
        return s


class OrcaJob(job):
    """
    A class for Orca jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None, deuterium=None):
        """
        Constructor
        """
        self.settings = None
        self.deuterium = deuterium
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

        if self.settings.pointcharges:
            pointcharges = self.settings.pointcharges
            runscript += "cat <<EOF >pointcharges.pc\n"
            runscript += str(len(pointcharges))
            runscript += '\n'
            for i in range(len(pointcharges)):
                runscript += str(pointcharges[i][3])
                runscript += ' '
                runscript += str(pointcharges[i][0])
                runscript += ' '
                runscript += str(pointcharges[i][1])
                runscript += ' '
                runscript += str(pointcharges[i][2])
                runscript += '\n'
            runscript += 'EOF\n'
            runscript += 'cat pointcharges.pc\n'

        runscript += 'if [ -z "$OPAL_PREFIX" ]; then\n'
        runscript += "  $ORCA_PATH/orca INPUT.inp\n"
        runscript += "else\n"
        runscript += '  $ORCA_PATH/orca INPUT.inp "--prefix $OPAL_PREFIX"\n'
        runscript += "fi\n"
        runscript += "retcode=$?\n"

        if self.settings.oocc_density:
            runscript += "cp INPUT.mdci.optorb INPUT.mdci.optorb.gbw\n"
            runscript += "$ORCA_PATH/orca_2mkl INPUT.mdci.optorb -emolden\n"
            runscript += "mv INPUT.molden.input molden.old\n"
            runscript += "mv INPUT.mdci.optorb.molden.input INPUT.molden.input\n"
        elif self.settings.brueckner_density:
            runscript += "cp INPUT.mdci.brueck INPUT.mdci.brueck.gbw\n"
            runscript += "$ORCA_PATH/orca_2mkl INPUT.mdci.brueck -emolden\n"
            runscript += "mv INPUT.molden.input molden.old\n"
            runscript += "mv INPUT.mdci.brueck.molden.input INPUT.molden.input\n"
        else:
            runscript += "$ORCA_PATH/orca_2mkl INPUT -emolden\n"

        runscript += "exit $retcode\n"

        return runscript

    def result_filenames(self):
        return ['INPUT.gbw', 'INPUT_property.txt', 'INPUT.engrad', 'INPUT.xyz',
                'INPUT_trj.xyz', 'INPUT.molden.input', 'INPUT.mdci.optorb',
                'INPUT.scfp', 'INPUT.hess', 'INPUT.inp']

    def check_success(self, outfile, errfile):
        f = open(outfile, encoding="utf-8")
        success = False
        for ll in f.readlines()[-10:]:
            if '**ORCA TERMINATED NORMALLY**' in ll:
                success = True
            if 'ORCA finished by error termination' in ll:
                raise PyAdfError("Error termination in ORCA")
            if 'Error: Cannot open GBW file: INPUT.mdci.optorb.gbw' in ll:
                raise PyAdfError("Error termination in ORCA")
        return success

    def get_keywords(self):
        return self.settings.get_keywords()

    def get_input_blocks(self, nproc=1):
        blocks = ''
        blocks += self.get_parallel_block(nproc)
        blocks += self.settings.get_input_blocks()
        return blocks

    def get_parallel_block(self, nproc):
        block = ''
        if nproc > 1:
            # cc calculations cannot handle the case where there are
            # more processor cores in use than there are electrons
            # in the system
            if 'CC' in self.settings.method:
                num_elec = self.mol.get_number_of_electrons()
                if num_elec < nproc:
                    nproc = num_elec
            block += "%pal\n"
            block += f"nprocs {nproc:d} \n"
            block += "end\n"
        return block

    def get_orcafile(self, nproc=1):

        orcafile = "! " + ' '.join(self.get_keywords()) + "\n"
        orcafile += self.get_input_blocks(nproc) + "\n"

        if self.settings.pointcharges:
            orcafile += '%pointcharges "pointcharges.pc"\n'

        orcafile += f"*xyz {self.mol.get_charge():d} {self.mol.get_spin() + 1:d} \n"
        xyz_file = self.mol.get_xyz_file()

        if self.deuterium is not None:
            # sets atom mass to 2 for a list of molecules
            xyz_file_split = xyz_file.splitlines(True)[2:]
            for i in self.deuterium:
                xyz_file_split[i-1] = xyz_file_split[i-1].rstrip()+' M 2 \n'
            orcafile += ''.join(xyz_file_split)
        else:
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

    def __init__(self, mol, settings=None, deuterium=None):
        super().__init__(mol, settings, deuterium)

    def print_jobtype(self):
        return "Orca single point job"

    def get_keywords(self):
        return ['SP'] + self.settings.get_keywords()


class OrcaGeometryOptimizationJob(OrcaJob):
    """
    A class for Orca geometry optimization jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None, deuterium=None):
        super().__init__(mol, settings, deuterium)

    def print_jobtype(self):
        return "Orca geometry optimization job"

    def get_keywords(self):
        return ['OPT'] + self.settings.get_keywords()


class OrcaFrequenciesJob(OrcaJob):
    """
    A class for Orca frequencies jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None, deuterium=None):
        super().__init__(mol, settings, deuterium)

    def print_jobtype(self):
        return "Orca frequencies job"

    def get_keywords(self):
        return ['FREQ'] + self.settings.get_keywords()


class OrcaOptFrequenciesJob(OrcaGeometryOptimizationJob):
    """
    A class for Orca optimization and frequencies jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None, deuterium=None):
        super().__init__(mol, settings, deuterium)

    def print_jobtype(self):
        return "Orca optimization and frequencies job"

    def get_keywords(self):
        return ['OPT FREQ'] + self.settings.get_keywords()


class OrcaExcitationsJob(OrcaSinglePointJob):
    """
    A class for Orca excitation energy jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None, tddft=None):
        """
        Constructor for OrcaExcitationsJob.

        @param mol: the molecule
        @param settings: Orca settings
        @type settings: L{OrcaSettings}
        @param tddft: Orca TDDFT settings
        @type tddft: L{OrcaTDDFTSettings}
        """
        super().__init__(mol, settings)

        if tddft is None:
            self.tddft_settings = OrcaTDDFTSettings()
        else:
            self.tddft_settings = tddft

        if not self.settings.method == 'DFT':
            raise PyAdfError("OrcaExcitationsJob currently only supports TDDFT")

    def print_jobtype(self):
        return "Orca excitation energy job"

    def create_results_instance(self):
        return OrcaExcitationResults(self)

    def get_input_blocks(self, nproc=1):
        blocks = super().get_input_blocks(nproc=nproc)
        blocks += "\n"
        blocks += self.tddft_settings.get_tddft_block()
        return blocks

    def print_jobinfo(self):
        super().print_jobinfo()
        print()
        print(self.tddft_settings)


class OrcaExStateGeoOptJob(OrcaExcitationsJob):
    """
    A class for Orca excited state geometry optimization jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None, tddft=None, iroot=1):
        """
        Constructor for OrcaExStateGeoOptJob.

        @param mol: the molecule
        @param settings: Orca settings
        @type settings: L{OrcaSettings}
        @param tddft: Orca TDDFT settings
        @type tddft: L{OrcaTDDFTSettings}
        @param iroot: Solve geometry for state iroot
        @type iroot: int
        """
        super().__init__(mol, settings, tddft=tddft)
        self.tddft_settings.iroot = iroot

    def print_jobtype(self):
        return "Orca excited state geometry optimization job"

    def get_keywords(self):
        return ['OPT'] + self.settings.get_keywords()


class OrcaExStateFrequenciesJob(OrcaExcitationsJob):
    """
    A class for Orca excited state frequencies jobs.

    Corresponding results class: L{OrcaResults}
    """

    def __init__(self, mol, settings=None, tddft=None, iroot=1):
        """
        Constructor for OrcaExStateFrequenciesJob.

        @param mol: the molecule
        @param settings: Orca settings
        @type settings: L{OrcaSettings}
        @param tddft: Orca TDDFT settings
        @type tddft: L{OrcaTDDFTSettings}
        @param iroot: Solve geometry for state iroot
        @type iroot: int
        """
        super().__init__(mol, settings, tddft=tddft)
        self.tddft_settings.iroot = iroot

    def print_jobtype(self):
        return "Orca excited state frequencies job"

    def get_keywords(self):
        return ['FREQ'] + self.settings.get_keywords()
