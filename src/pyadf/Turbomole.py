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
Support for various flavours of I{Turbomole} computations.

@author:  Moritz Klammler
@contact: U{moritz.klammler@gmail.com<mailto:moritz.klammler@gmail.com>}

@group Jobs:      TurbomoleSinglePointJob,
                  TurbomoleGeometryOptimizationJob,
                  TurbomoleGradientJob,
                  TurbomoleForceFieldJob
@group Results:   TurbomoleSinglePointResults,
                  TurbomoleGeometryOptimizationResults,
                  TurbomoleGradientResults,
                  TurbomoleForceFieldResults
@group Settings:  TurbomoleSinglePointSettings,
                  TurbomoleGeometryOptimizationSettings,
                  TurbomoleGradientSettings,
                  TurbomoleForceFieldSettings
@group Internals: TurbomoleJob,
                  TurbomoleResults,
                  TurbomoleSettings
                  _TurbomoleAbInitoSettings_,
                  _TurbomoleDensityResults,
                  _TurbomoleEnergyResults,
                  _TurbomoleForceResults,
                  _TurbomolePlainResults

"""

from .BaseJob import results, job
from .Utils import f2f
from pyadf.PyEmbed.DensityEvaluator import GTODensityEvaluatorMixin

from .TurboDefinition import *

_nan = float('NaN')


class TurbomoleResults(results):
    """
    Results of a I{Turbomole} computation.

    @group Retrieval of specific results: get_scf_energies,
                                          get_energy,
                                          get_scf_energy,
                                          get_mp2_energy
    @group Access to result files:        get_molecule,
                                          get_result_file_list
    @group Obsolete:                      get_dipole_vector,
                                          get_dipole_magnitude

    """

    def __init__(self, j=None):
        """
        Initialize a new results object.

        @param j: L{job} object of the corresponding job.
        @type  j: L{job}

        """

        super().__init__(j=j)
        self.resultstype = "Turbomole results"

        if self.job is not None:
            if isinstance(self.job, TurbomoleForceFieldJob):
                self.method = 'uff'
            else:
                self.method = self.job.settings.method

    def get_molecule(self):
        """
        Get a L{molecule} object from the C{coord} file written after the
        computaton.

        If the C{coord} file can't be read, L{None} will be returned. This
        method will not raise any exceptions.

        @return: molecule
        @rtype:  L{molecule}

        """

        import os
        from .Molecule import molecule

        temp_coordfilename = self.get_temp_result_filename('coord')
        mol = molecule(filename=temp_coordfilename, inputformat='tmol')
        os.remove(temp_coordfilename)

        return mol

    def read_molden_file(self):
        """
        Returns Molden results file as a string.
        """
        moldenfile = self.files.read_file_from_archive(self.fileid, 'orbitals.molden')
        if moldenfile is None:
            raise PyAdfError('Turbomole Molden file not found.')
        return moldenfile

    # Since Turbomole produces many files and we keep them as one `tar' archive
    # in   the   file   manager,    we   define   us   an   additional   method
    # `get_temp_result_filename(FILENAME)'  that  will  extract  us  FILENAME's
    # content to a temporary file.
    def get_temp_result_filename(self, filename):
        """
        Access a result file from the archived Turbomole results.

        Extracts a result file from the archived data and writes its contents
        to a temporary file. The file name of this temporary file is returned
        and may be used to open / read as if it were the original file. You're
        self responsible to delete the temporary file afterwards, once you
        don't need it any longer.

        If the C{filename} can't be found in the archive, L{None} will be
        returned and no exception will raise.

        @param filename: Name of the file to extract.
                         (E.g. C{energy}.)
        @type  filename: L{str}
        @returns:        Absolute path to the temporary file.
        @rtype:          L{str}

        """
        return self.files.get_tempfile_from_archive(self.fileid, filename)

    def get_result_file_list(self):
        """
        Get a list of the output files written.

        Not included are the output logs I{PyADF} writes itself, that is
        I{stdout} and I{stderr} from the runscript. If the files can't be
        retrieved, an empty list is returned. No exceptions will be raised.

        @return: List with absolute filenames.
        @rtype:  L{str}C{[]}
        """

        import tarfile

        filenames = []
        try:
            tarfilename = self.files.get_results_filename(self.fileid)
            tar = tarfile.open(name=tarfilename)
            filenames = tar.getnames()
        except tarfile.TarError:
            pass
        return filenames

    # Unfortunately,  our   mother  implements  a   `get_dipole_vector'  and  a
    # `get_dipole_magnitude'  method.  This  is  bad  because not  all  of  our
    # children even YIELD  a dipole moment! The radical way  is to "delete" the
    # method here and re-introduce it for those result types that actually know
    # a dipole moment. However, things could be a lot more straight forward.
    def get_dipole_vector(self):
        """
        Not to be used.
        """
        return None

    def get_dipole_magnitude(self):
        """
        Not to be used.
        """
        return None


# CONCEPT OF INTERMEDIATE RESULTS CLASSES
#
# There  are  many kinds  of  Turbomole  computations  and all  yield  somewhat
# different results.  Hence,  we need many result classes.  But theire features
# will overlap heavily. In order to minimize redundancy, we define "intermdiate
# result  classes"  that  only  feature  one  specific  result.   Via  multiple
# inheritance, result  classes for any combination of  these results (featuring
# different kinds  of jobs)  can be  constructed with just  two lines  of code.
# Inorder to keep  Python's inheritance mechanism working, the  hirachy must be
# monotonous.  Therefore,  ALL  intermediate   classes  must  be  derived  form
# `TurbomoleResults' and  all user level  results classes must be  derived from
# classes  derived therefrom.  The intermediate  classes are  made  "privat" by
# prefixing  their names  with  an underscore.  A  user should  NEVER use  them
# directly.
class _TurbomolePlainResults(TurbomoleResults):
    """
    Clone of parent class.

    This class is identiacl to its mother but it should be used to derive user
    level result classes in order to keep the class hirachy monotone. See
    implementation comment above.

    """
    pass


class _TurbomoleDensityResults(TurbomoleResults, GTODensityEvaluatorMixin):
    """
    Intermediate class to let results classes for computations that yield a
    density inherit from.

    @group Retrieval of specific results: get_dipole_magnitude,
                                          get_dipole_vector

    """

    def __init__(self, j=None):
        super().__init__(j=j)
        self.dipole_vector = [None, None, None]

    def get_dipole_vector(self):
        """
        Get a 3-dimensional vector for the molecular dipole moment.

        The dipole moment is given in atomic units (M{e*a}). If a component
        can't be retireved, the entry will be a C{NaN}. This method will not
        raise any exceptions.

        @returns: Dipole moment vector, in atomic units.
        @rtype:   L{float}C{[3]}
        @bug:     More testing needed.

        """

        import re

        if not any(self.dipole_vector):
            tmoutput = self.get_output()
            dipole_vector = [_nan, _nan, _nan]
            attention = 0
            for line in tmoutput:
                line = line.lower()
                if attention >= 5:
                    break
                elif attention == 0:
                    if re.search(r'electrostatic\s+moments', line):
                        attention = 1
                        continue
                elif attention == 1:
                    if re.search(r'dipole\s+moment', line):
                        attention = 2
                        continue
                else:
                    words = line.split()
                    if len(words) == 4:
                        if words[0] == 'x':
                            dipole_vector[0] = words[3]
                            attention += 1
                        elif words[0] == 'y':
                            dipole_vector[1] = words[3]
                            attention += 1
                        elif words[0] == 'z':
                            dipole_vector[2] = words[3]
                            attention += 1
            for i in range(0, len(dipole_vector)):
                dipole_vector[i] = f2f(dipole_vector[i])
            self.dipole_vector = dipole_vector

        return self.dipole_vector

    def get_dipole_magnitude(self):
        """
        Get the magnitude of the molecular dipole moment.

        The dipole moment is given in atomic units (M{e*a}). If there is a
        problem computing the value, a C{NaN} will be returned. This method
        will not raise any exceptions.

        @returns: Absolute dipole moment in atomic units
        @rtype:   L{float}
        @bug:     More testing needed.
        @raises Hefeweizen: If it is given a glass of beer.

        """

        import math

        sqared = 0.0
        for component in self.get_dipole_vector():
            sqared += component**2
        return math.sqrt(sqared)


class _TurbomoleEnergyResults(TurbomoleResults):
    """
    Intermediate class to let results classes for computations that
    yield an energy inherit from.
    """

    def __init__(self, j=None):
        super().__init__(j=j)
        self.scf_energies = []
        self.scf_energy = None
        self.mp2energy = None

    def get_scf_energies(self, readagain=False):
        """
        Get a list of all SCF-converged energies encountered during the
        optimization.

        The energies are given in atomic units (Hartree).

        If there is only one energy (as for a single point job), this will be a
        list with one element. If there are many energies (as for a geometry
        optimization), the list will have many entries. If there is no energy
        (as for a broken job) the list will be empty. No exceptions
        will be raised.

        @param readagain: If C{True}, the energies are read again from the file
                          even if they are already known.
        @type  readagain: L{bool}
        @return:          List with energies.
        @rtype:           L{float}C{[]}

        """

        import os

        if len(self.scf_energies) == 0 or readagain:
            temp_energy_filename = self.get_temp_result_filename('energy')
            try:
                self.scf_energies = []
                transfer_mode = False
                with open(temp_energy_filename) as infile:
                    for line in infile:
                        words = line.split()
                        if len(words) == 0:
                            continue  # don't get confused by blank lines
                        if str(words[0]) == '$energy':
                            transfer_mode = True
                            continue
                        elif str(words[0]) == '$end':
                            break
                        if transfer_mode:
                            self.scf_energies.append(f2f(words[1]))
            except OSError:
                self.scf_energies = []
            finally:
                try:
                    os.remove(temp_energy_filename)
                except OSError:
                    pass
        return self.scf_energies

    def get_energy(self):
        """
        Get the final total energy for the considered method.
        """

        if self.method in ['hf', 'dft']:
            return self.get_scf_energy()
        elif self.method in ['mp2']:
            return self.get_scf_energy() + self.get_mp2_energy()
        else:
            raise PyAdfError('Invalid method in TurbomoleResults.get_energy')

    get_total_energy = get_energy

    def get_scf_energy(self):
        """
        Get the final (or, if it is a single point computation, the only)
        SCF-converged energy of the molecule in atomic units (Hartree).

        If there is no energy, C{None} will be returned. This method will not
        raise any exceptions.

        @return: SCF-converged energy.
        @rtype:  L{float}

        """

        self.scf_energy = None
        try:
            self.scf_energies = self.get_scf_energies()
            self.scf_energy = self.scf_energies[-1]
        except AttributeError:
            pass
        return self.scf_energy

    def get_mp2_energy(self, readagain=False):
        """
        Get the final MP2 energy.

        Final MP2 energy in atomic units (Hartree) If the MP2 energy has never
        been computed, C{None} will be returned. If it seems it has been
        computed but can't be read, a C{NaN} will be returned. This method will
        not raise any exceptions.

        @param readagain: Try to read the output again even if the energy is
                          allready known.
        @returns:         Final MP2 energy
        @rtype:           L{float}
        @bug:             More testing needed.

        """

        # import re
        #
        # if not self.mp2energy or readagain:
        #
        #    self.mp2energy = None
        #
        #    linepattern = re.compile('final\s+mp2\s+energy')
        #    energypattern = re.compile('[+-]?\d*\.\d*[ed]?[+-]?\d*')
        #
        #    for line in self.get_output():
        #        line = line.lower()
        #        if re.search(linepattern, line):
        #            m = re.search(energypattern, line)
        #            self.mp2energy = f2f(str(m.group(0)))
        #
        # return self.mp2energy

        import os

        if not self.mp2energy or readagain:
            temp_energy_filename = self.get_temp_result_filename('energy')

            energies = []
            transfer_mode = False
            with open(temp_energy_filename) as infile:
                for line in infile:
                    words = line.split()
                    if len(words) == 0:
                        continue  # don't get confused by blank lines
                    if str(words[0]) == '$energy':
                        transfer_mode = True
                        continue
                    elif str(words[0]) == '$end':
                        break
                    if transfer_mode:
                        energies.append(f2f(words[4]))
            self.mp2energy = energies[-1]

            try:
                os.remove(temp_energy_filename)
            except OSError:
                pass

        return self.mp2energy


class _TurbomoleForceResults(TurbomoleResults):
    """
    Intermediate class to let results classes for computations that yield a
    "force" (physically correct but unusual replacement for the word "gradient"
    to avoid name clashes) inherit from.

    @group Retrieval of specific results: get_gradients,
                                          get_gradient,
                                          get_gradient_vector

    """

    def __init__(self, j=None):
        super().__init__(j=j)
        self.gradients = []
        self.gradient = None

    def get_gradients(self, readagain=False):
        """
        Get a table with all gradients for all steps.

        The gradients are representated as a 3-dimensional list containing the
        gradient vectors for every atom for every iteration cycle. So to get
        the y-component of the gradient for atom #24 in iteration cycle 7,
        you'd say::

            g = res.get_gradients()</br>
            print g[6][23][1] # 0-based indexing!

        If there was just one cycle, the outermost list will ony have one
        element.

        The method will always succeed and not raise any exceptions. If the
        gradients can't be read at all, an empty list will be returned. If
        single items can't be read, they will show up as C{NaN}s in the list.

        The list is a normal I{Python} list and not a I{NumPy} array.

        @param readagain: Try to re-read the C{gradient} file even if
                          this is not needed.
        @returns:         Gradients table
        @rtype:           L{float}C{[M{C}][M{N}][3]}

        """

        import os

        if len(self.gradients) == 0 or readagain:

            temp_gradient_filename = self.get_temp_result_filename('gradient')
            atoms = self.get_molecule().get_number_of_atoms()

            # The `gradient' file is written like this:
            #
            # ----------------------------------------
            #  $grad ...
            #  cycle = 1 ...
            #  x1      y1      z1      A1
            #  x2      y2      z2      A2
            #  ...     ...     ...     ...
            #  xn      yn      zn      An
            #  dE/dx1  dE/dy1  dE/dz1
            #  dE/dx1  dE/dy1  dE/dz1
            #  ...     ...     ...
            #  dE/dxn  dE/dyn  dE/dzn
            #  cycle = 2 ...
            #  ...
            #  ...
            #  $end
            # ----------------------------------------
            #
            # Where  `x1' means  the x-coordinate  of the  first atom  and `A1'
            # stands for the atom type of atom no. 1. Above the first and below
            # the last cycle there is a line with some keyword.
            #
            # We read it line by line, looking for a line starting with `cycle'
            # (prefixed with some space) and then  go down for N lines (where N
            # is  the number  of  atoms in  our  molecule).  Then  we read  the
            # following N lines  as atomic gradients to get  a complete set for
            # this  iteration  step  (cycle).  We  append this  matrix  to  our
            # `self.gradients' and go on. The next line should be again `cycle'
            # or `$end'.
            #
            # The way  we obtain the data makes  it difficult to read  it as an
            # array directly. Of course we  could convert it to an array before
            # return  but there  would be  no point  in always  constructing an
            # array and a user maybe converting  it back to a list if she needs
            # it. After all, we don't get  the conversion for free and even for
            # a big molecule with many cycles the list isn't really big. (About
            # two MB for a molecule with 500 atoms and 200 cycles.)

            coord_block = False
            gradient_block = False
            corrupted = False

            self.gradients = []  # for each iteration step
            gradient = []  # for each atom
            atom = 0

            try:
                with open(temp_gradient_filename) as infile:
                    for line in infile:
                        if re.match(r'\s*cycle', line):
                            coord_block = True
                            atom = 0
                        elif coord_block:
                            atom += 1
                            if atom == atoms:
                                atom = 0
                                coord_block = False
                                gradient_block = True
                                corrupted = False
                                gradient = []
                        elif gradient_block:
                            atom += 1
                            words = line.split()
                            atom_gradient = []
                            if len(words) == 3 and not corrupted:
                                for word in words:
                                    g = f2f(word)
                                    atom_gradient.append(g)
                            else:
                                # If we enconter a corrupted line, we can't say
                                # what  the  meaning  of  the  following  lines
                                # is. All  we can  do is to  wait for  the next
                                # cycle and  put `NaN's for all of  the rest of
                                # this block.
                                corrupted = True
                                atom_gradient = [_nan, _nan, _nan]
                            gradient.append(atom_gradient)
                            if atom == atoms:
                                gradient_block = False
                                self.gradients.append(gradient)
            except OSError:
                self.gradients = []

            try:
                os.remove(temp_gradient_filename)
            except OSError:
                pass

        return self.gradients

    def get_gradient(self):
        """
        Get the final (or only) gradient.

        Equivalent to the last entry in L{get_gradients}. Please refere to the
        documentation there for the indexing scheme.

        @returns: Final gradient
        @rtype:   L{float}C{[M{N}][3]}
        """

        self.gradient = None
        try:
            self.gradients = self.get_gradients()
            self.gradient = self.gradients[-1]
        except Exception as e:
            print(e)
            pass
        return self.gradient

    def get_gradient_vector(self):
        """
        Get a one dimensional I{NumPy} array with the final I{gradient
        vector}.

        If arabic number count atoms and M{x}, M{y} and M{z} carthesian
        directions, then the vector is arranged like this::

            np.array([ 1.x, 1.y, 1.z, 2.x, 2.y, 2.z, ... N.x, N.y, N.z ])

        @returns: Final gradient vector
        @rtype:   C{np.array(float[3 M{N}])}
        @bug:     Not implemented yet!
        """

        import numpy as np
        gradient_vector = []
        for atom in self.get_gradient():
            for coordinate in atom:
                gradient_vector.append(coordinate)
        return np.array(gradient_vector)


class TurbomoleSinglePointResults(_TurbomoleDensityResults, _TurbomoleEnergyResults):
    """
    Results of a L{TurbomoleSinglePointJob}.
    """

    def __init__(self, j=None):
        super().__init__(j=j)
        self.resultstype = "Turbomole single point results"


class TurbomoleGeometryOptimizationResults(_TurbomoleDensityResults, _TurbomoleEnergyResults,
                                           _TurbomoleForceResults):
    """
    Results of a L{TurbomoleGeometryOptimizationJob}.
    """

    def __init__(self, j=None):
        super().__init__(j=j)
        self.resultstype = "Turbomole geometry optimization results"


class TurbomoleGradientResults(_TurbomoleDensityResults, _TurbomoleEnergyResults, _TurbomoleForceResults):
    """
    Results of a L{TurbomoleGradientJob}.
    """

    def __init__(self, j=None):
        super().__init__(j=j)
        self.resultstype = "Turbomole gradient results"


class TurbomoleForceFieldResults(_TurbomolePlainResults):
    """
    Results of a L{TurbomoleForceFieldJob} (using I{uff}).
    """
    pass


class TurbomoleSettings:
    """
    Container class for settings.
    """

    def __init__(self, verbose_level=1):
        """
        Initialize a new settings instance.

        @param verbose_level: The higher, the more debugging information will
                              be printed out by the L{TurboDefinition} part.
        @type  verbose_level: L{int}
        """

        self.verbose_level = verbose_level
        self.coordfilename = 'coord'
        self.charge = None
        self.spin = 0        # NOTE: assuming closed-shell facilitates many things and is reasonable
        self.unrestricted = False
        self.summary = []

    def __str__(self):
        """
        Get a nicely formatted text block summarizing the settings.

        @returns: Text block
        @rtype:   L{str}
        """

        # Before we do anything, let the respective method assemble the list of
        # interesting items.
        self.generate_summary()

        # First we dtermine the maximum length of a description.
        maxlength = 0
        for item in self.summary:
            if len(item[0]) > maxlength:
                maxlength = len(item[0])
        maxlength += 1  # for colon
        # Then we assemble the sequence.
        pattern = '  {i}) {desc:' + str(maxlength) + '} {val}' + '\n'
        text = ''
        for item in enumerate(self.summary):
            col1 = chr(item[0] + ord('a'))
            col2 = item[1][0] + ':'
            col3 = item[1][1]
            text += pattern.format(i=col1, desc=col2, val=col3)
        return text

    def generate_summary(self):
        """
        Generate the summary for making a string out of us.

        This method has to make sure that C{self.summary} is a list of the
        interesting settings.  C{__str__} calls this method before it makes a
        string of the list.  Child classes can call their mother's
        C{generate_summary} and then add their own stuff.
        """
        self.summary = []

    def _set_charge(self, charge):
        """
        Set the charge of the system to compute.

        This is I{not} a user level function! Users should I{always} use the
        C{L{molecule.set_charge}(M{Q})} method to specify the charge of their
        molecule. However, since the L{TurboDefinition} class doesn't know
        about molecules, it has to be toled the charge. But this property has
        to be set by the job class without need for the user to know.

        @param charge: Charge in atomic units
        @type  charge: L{int}
        """
        self.charge = charge

    def _set_spin(self, spin):
        """
        Set the spin of the system to compute.

        This is I{not} a user level function! Users should I{always} use the
        C{L{molecule.set_spin}(M{Q})} method to specify the spin of their
        molecule. However, since the L{TurboDefinition} class doesn't know
        about molecules, it has to be told the spin. But this property has
        to be set by the job class without need for the user to know. As
        restricted open-shell is not supported, yet, the unrestricted
        variable is set automatically.

        @param spin: Number of unpaired electrons
        @type  spin: L{int}
        """
        self.spin = spin
        if self.spin != 0:
            self.unrestricted = True

    def set_unrestricted(self, unrestricted):
        """
        Set wether the system is computed unrestricted or not.
        Essentially, this routine is only needed to calculate open-shell
        singulets.

        @param unrestricted: Boolean for unrestricted calculations
        @type  unrestricted: L{bool}
        """
        if unrestricted:
            self.unrestricted = True
        elif self.spin == 0:
            self.unrestricted = False
        else:
            # NOTE: restricted open-shell calculations are not supported, yet!!!
            #       --> non-singulet occupation not possible without unrestricted
            pass


class _TurbomoleAbInitioSettings(TurbomoleSettings):
    """
    Intermediate class to let settings for I{ab initio} jobs inherit from.
    """

    def __init__(self, verbose_level=1):

        super().__init__(verbose_level=verbose_level)

        self.method = None
        self.scf = None
        self.dft = None
        self.mp2 = None
        self.cc_memory = None
        self.ri = None
        self.ri_memory = None
        self.basis_set_all = None
        self.dft_grid = None
        self.dft_functional = None
        self.disp = None
        self.guess_initial_occupation_by = None
        self.ired = None
        self.idef = None
        self.idef_list = None
        self.scfconv = None
        self.scfiterlimit = None
        self.pointcharges = None
        self.cosmo = None
        self.cosmo_epsilon = None
        self.cosmo_rsolv = None
        self.cosmo_radii = None

    @property
    def num_pointcharges(self):
        if self.pointcharges is None:
            return 0
        else:
            return self.pointcharges.shape[0]

    def generate_summary(self):
        # See mother's docstring.
        super().generate_summary()
        self.summary.append(["Method", self.method])
        if self.dft:
            self.summary.append(["DFT Functional", self.dft_functional])
            self.summary.append(["Dispersion correction", self.disp])
            self.summary.append(["DFT integration grid", self.dft_grid])
        self.summary.append(["Use RI approximation", self.ri])
        if self.ri:
            self.summary.append(["Memory for RI", f"{self.ri_memory} MB"])
        if self.mp2:
            self.summary.append(["Memory for CC2", f"{self.cc_memory} MB"])
        self.summary.append(["Basis set", self.basis_set_all])
        self.summary.append(["Guess initial occupation by", self.guess_initial_occupation_by])
        self.summary.append(["Use red. int. coordinates", self.ired])
        self.summary.append(["Define add. int. coordinates", self.idef])
        if self.idef:
            self.summary.append(["List of add. int. coordinates", self.idef_list])
        if self.scfconv is None:
            self.summary.append(["SCF convergence criterion scfconv", "default"])
        else:
            self.summary.append(["SCF convergence criterion scfconv", self.scfconv])
        self.summary.append(["Limit of the number of the SCF iterations", self.scfiterlimit])
        if self.pointcharges is not None:
            self.summary.append(["Number of added point charges", self.num_pointcharges])
        if self.cosmo:
            self.summary.append(["Uses COSMO", self.cosmo])
        if self.cosmo_epsilon:
            self.summary.append(["COSMO epsilon parameter", self.cosmo_epsilon])
        if self.cosmo_rsolv:
            self.summary.append(["COSMO rsolv parameter", self.cosmo_rsolv])
        if self.cosmo_radii:
            self.summary.append(["COSMO radius definition menu parameter", self.cosmo_radii])

    def _set_method(self, method):
        """
        Select the computational method to be used.

        This is I{not} a user level function. Users should I{always} use the
        job class' initializatior's optional arguments to select the method.

        Known methods are:

            - C{hf}  Hartree-Fock with SCF
            - C{dft} SCF with DFT
            - C{mp2} HF-SCF followed by MP2

        @param method:      Name of a computational method
        @type  method:      L{str}
        @raises PyAdfError: If the method is none of the above three
        """

        if method not in ['hf', 'dft', 'mp2']:
            raise PyAdfError(f"""Sorry, I don't know about `{method}' as an ab inito method.""")

        self.method = method
        self.scf = (method == 'hf') or (method == 'dft')
        self.dft = method == 'dft'
        self.mp2 = method == 'mp2'

        if self.mp2:
            if self.cc_memory is None:
                self.set_cc_memory() # sets to default

    def set_basis_set(self, basis_set):
        """
        Select a basis set to be used for all atoms.

        There is no check done on the selection. If I{define} knows your basis
        set, fine, if it doesn't, your comoutation will crash.

        @param basis_set: Name of a basis set (e.g. C{def2-TZVP}).
        @type  basis_set: L{str}
        """

        self.basis_set_all = basis_set

    def set_dft_functional(self, dft_functional):
        """
        Select a DFT functional.

        There is no check done on the selection. If I{define} knows your
        functional, fine, if it doesn't, your comoutation will crash.

        @param dft_functional: Name of a DFT functional (e.g. C{b-p}).
        @type  dft_functional: L{str}
        @raises PyAdfError:    If DFT was not selected.
        """

        self.dft_functional = dft_functional

    def set_dft_grid(self, dft_grid):
        """
        Select the DFT integration grid.

        Available options m3-m5  or  1-7
        """

        self.dft_grid = dft_grid

    def set_ri(self, value, memory=2000):
        """
        Switch RI approximation on or off and assign a certain amount of memory
        to it.

        If you switch RI off, the memory is automatically set to C{None}
        whatever the parameter is.

        @param value:  C{True} to switch on, C{False} to switch off.
        @type  value:  L{bool}
        @param memory: Memory to be used for RI (in MB)
        @type  memory: L{int}
        """

        self.ri = value
        if value:
            self.ri_memory = memory
        else:
            self.ri_memory = None

    def set_cc_memory(self, memory=2000):
        """
        Assign an amount of memory to the CC2 program.

        @param memory:      Memory in MB
        @type  memory:      L{int}
        @raises PyAdfError: If the method is not MP2
        """
        self.cc_memory = memory

    def set_dispersion_correction(self, correction):
        """
        Select a dispersion correction.

        Possible corrections are:

            - C{None}   No correction
            - C{dft-d1} DFT-D1 correction
            - C{dft-d2} DFT-D2 correction
            - C{dft-d3} DFT-D3 correction (recommended)

        @param correction:  Name of a dispersion correction
        @type  correction:  L{str}
        @raises PyAdfError: For invalid choices

        """

        if correction not in ['dft-d1', 'dft-d2', 'dft-d3']:
            raise PyAdfError("""Sorry, I don't know the dispersion correction
            `{}'""".format(correction))

        self.disp = correction

    def set_scfconv(self, conv):
        """
        Set SCF convergence criterion.

        @param conv: SCF convergence criterion
        @type conv: L{int}
        """
        self.scfconv = int(conv)

    def set_scfiterlimit(self, maxit):
        """
        Set a maximum number of SCF iterations.

        @param maxit: Maximal number of SCF iterations
        @type maxit: L{int}
        @raises PyAdfError: For invalid values
        """

        if int(maxit) < 0:
            raise PyAdfError("""Sorry, the maximum number of iterations cannot be negative""")

        self.scfiterlimit = int(maxit)

    def set_initial_occupation_guess_method(self, method):
        """
        Select a method to guess the initial occupation.

        The only acceptable choice at the moment is C{eht} (extended Hueckel
        Theory). If you restart a job, the initial occupation is guessed
        nevertheless but the results of this estimation are immediately
        forgotten afterwards.

        @param method:      Method to be used.
        @type  method:      C{str}
        @raises PyAdfError: If the method is not known
        """

        if method != 'eht':
            raise PyAdfError("""Sorry, no support for `{}' as a mrthod to
            guess initial occupations.""".format(method))

        self.guess_initial_occupation_by = method

    def set_redundant_internal_coordinates(self, value):
        """
        Switch usage of redundant internal coordinates on or off.

        @param value:  C{True} to switch on, C{False} to switch off.
        @type  value:  L{bool}
        """
        self.ired = value

    def set_cosmo(self, cosmo):
        """
        Switch useage COSMO on or off.

        @param cosmo:  C{True} to switch on, C{False} to switch off.
        @type  cosmo:  L{bool}
        """

        self.cosmo = cosmo

    def set_cosmo_epsilon(self, cosmo_epsilon):
        """
        Set the epsilon value in COSMO.
        Default = infinity
        """
        self.cosmo_epsilon = cosmo_epsilon

    def set_cosmo_rsolv(self, cosmo_rsolv):
        """
        Set the rsolv value in COSMO.
        Default = min(rad(h))
        """

        self.cosmo_rsolv = cosmo_rsolv

    def set_cosmo_radii(self, cosmo_radii):
        """
        Set the radius definition menu parameter in COSMO. Defines the atomic radii.
        Default = 'r all o' (sets all atomic radii to optimized values, if such are available;
        if there is no optimized radius for one or more atoms (e.g. phosphorus), cosmoprep ends
        abnormally and no cosmoprep sequence is added to the control file (hence, no COSMO is
        used in the calculation).

        Other options (from cosmoprep user interface comments):
        enter your definitions in the following way:
        r <atoms> <radius>
        <atoms>  : all
                 : a list of atomic indices like 1-5,7,10
                 : an element name like "c"
        <radius> : o  - optimized radii only (see above)
                 : b  - optimized radii are used if defined, or not optimized radii (mostly bondii*1.17) else
                 : own proposal like 1.256 (in angstrom)

        @param cosmo_radii: string specifying what atomic radii should be used for which atoms (e.g. 'r all o')
        @type cosmo_radii: string
        @return: input string for cosmoprep radius definition menu
        @rtype: string
        """
        self.cosmo_radii = cosmo_radii

    def define_internal_coordinates(self, value):
        """
        Switch additional definition of internal coordinates on or off.

        @param value:  C{True} to switch on, C{False} to switch off.
        @type  value:  L{bool}
        """

        self.idef = value

    def add_internal_coordinates(self, icoord):
        """
        Add a list of additional internal coordinates.

        @param icoord:  List of internal coordinates to be defined. ['f bend 1 2 3','f tors 1 2 3 4']
        @type  icoord:  List of strings
        """
        self.define_internal_coordinates(True)
        self.idef_list = icoord

    def add_point_charges(self, pointcharges, bohr=False):
        """
        Add external point charges.

        @param pointcharges:
            Coordinates (x, y, z in Angstrom) and charges of point charges.
            If no charges are given, zero charges are used.
        @type pointcharges: float[4][n]

        @param bohr: Whether the point charge coordinates are given in bohr
            (default: Angstrom)
        @type bohr: bool
        """
        import numpy as np
        from .Utils import Bohr_in_Angstrom

        self.pointcharges = np.asarray(pointcharges)

        # point charge coordinates are stored in bohr; if necessary, convert units
        if not bohr:
            for i in range(self.num_pointcharges):
                self.pointcharges[i, 0] = self.pointcharges[i, 0] / Bohr_in_Angstrom
                self.pointcharges[i, 1] = self.pointcharges[i, 1] / Bohr_in_Angstrom
                self.pointcharges[i, 2] = self.pointcharges[i, 2] / Bohr_in_Angstrom


class TurbomoleSinglePointSettings(_TurbomoleAbInitioSettings):
    """
    Settings for a L{TurbomoleSinglePointJob}.
    """
    pass


class TurbomoleGeometryOptimizationSettings(_TurbomoleAbInitioSettings):
    """
    Intermediate class to let user level I{Turbomole} setting classes inherit from.
    """

    def __init__(self, verbose_level=1):

        super().__init__(verbose_level=verbose_level)

        self.gcart = None
        self.max_iterations = None
        self.frozen_atoms = None
        self.job_verbosity = None

    def generate_summary(self):
        # See mother's docstring.
        super().generate_summary()
        self.summary.append(["Max. cart. grad. norm", f"10^-{self.gcart} a.u."])
        self.summary.append(["Max. iteration cycles", self.max_iterations])

    def set_convergence_criterion(self, gcart):
        """
        Set the convergence cretirion for an iteration job.

        Here is the I{Turbomole} description of that value::

            -gcart integer:
            converge maximum norm of carthesian gradient up tp 10^(- integer)
            atomic units

        @param  gcart: Convergence criterion (recommended: C{gcart=4})
        @type   gcart: L{int}
        @raises PyAdfError: For invalid choices
        """

        if int(gcart) != gcart:
            raise PyAdfError(f"""So you think {gcart} is an integer?""")

        if gcart < 1:
            raise PyAdfError(f"""Rejecting 10^-({gcart}) a.u. as a convergence criterion.""")

        self.gcart = gcart

    def set_max_iterations(self, number):
        """
        Set the maximum number of iteration cycles to perform before giving up
        the optimization.

        @param number:      Max. number of cycles
        @type  number:      L{int}
        @raises PyAdfError: For invalid choices
        """

        number = int(number)
        if number < 1:
            raise PyAdfError("""Rejecting {} as the maximum number of
            iteration cycles.""".format(number))

        self.max_iterations = number

    def set_frozen_atoms(self, frozen_list):
        """
        Sets the atoms with atomnumbers in frozen_list to frozen in coord file.

        @param frozen_list:      List of frozen atoms
        @type  frozen_list:      list of int

        """

        self.frozen_atoms = frozen_list

    def set_job_verbosity(self, verbosity):
        """
        Sets the job verbosity, so that the optimization can be followed.

        @param verbosity:        whether or not jobex should be verbose
        @type  verbosity:        boolean

        """

        self.job_verbosity = verbosity


class TurbomoleGradientSettings(_TurbomoleAbInitioSettings):
    """
    Settings for a L{TurbomoleGradientJob}.
    """
    pass


class TurbomoleForceFieldSettings(TurbomoleSettings):
    """
    Settings for a L{TurbomoleForceFieldJob}.
    """

    def __init__(self, verbose_level=1):
        super().__init__(verbose_level=verbose_level)

        self.max_iterations = None

    def generate_summary(self):
        # See mother's docstring.
        super().generate_summary()
        self.summary.append(["Max. iteration cycles", self.max_iterations])

    def set_max_iterations(self, number):
        """
        Set the maximum number of iteration cycles to perform before giving up
        the optimization.

        @param number:      Max. number of cycles
        @type  number:      L{int}
        @raises PyAdfError: For invalid choices

        """

        number = int(number)
        if number < 1:
            raise PyAdfError("""Rejecting {} as the maximum number of
            iteration cycles.""".format(number))

        self.max_iterations = number


class TurbomoleJob(job):
    """
    Base class for all I{Turbomole} jobs and friends.

    This is I{not} a user level class. It is still "public" to allow
    other modules to test easily if an object is a I{Turbomole} job.

    @group Initialization:  set_restart
    @group Obsolete:        result_filenames
    @group Other Internals: create_results_instance,
                            get_molecule,
                            print_molecule,
                            print_settings,
                            print_extras
    """

    def __init__(self, mol):
        """
        Initialize a new job.

        @param mol: Molecule to work with
        @type mol:  L{molecule}
        """

        super().__init__()

        self.mol = mol
        self.settings = None

        self.jobtype = None
        self._checksum = None

        self.execute = []
        self.file_on_success = None
        self.file_on_fail = None

        self.restart = False
        self.restart_id = None
        self.restart_mos = None

    def set_restart(self, restart):
        """
        Use MOs derived from a previous job as starting point.

        This may speed up the converging of the SCF computation. Only the
        C{mos} file is copied but I{not} the molecular geometry or whatever
        else!

        @param restart:     Results object to use as starting point.
        @type  restart:     subclass of L{TurbomoleJob}
        @raises PyAdfError: If the results to restart from obviously make no
                            sense or the C{mos} file can't be extracted.
        """

        # We  first check  if using  the  old results  as a  starting point  is
        # reasonable.  The explicit  conversion to  strings helps  to  get more
        # informative error  messages even if a  completely  weird object might
        # be passed.

        if not issubclass(type(restart), type(TurbomoleResults())):
            raise PyAdfError("Cowardly refusing to restart a " + str(self.jobtype)
                             + " using the results of a non-Turbomole job.")

        # We also make sure that the molecule has the same number of electrons.

        restart_mol = restart.get_molecule()
        restart_electrons = (restart_mol.get_number_of_atoms() -
                             restart_mol.get_charge())
        own_electrons = self.mol.get_number_of_atoms() - self.mol.get_charge()

        if restart_electrons != own_electrons:
            raise PyAdfError("""I was given a molecule with {own} electrons but
            now you give me a `mos' file that was computed for a system with
            {res} electrons. I won't do that.""".format(own=own_electrons,
                                                        res=restart_electrons))

        # If these two exceptions don't apply, we trust the `mos'.

        self.restart = True
        self.restart_id = restart.fileid

        # We then  grab the `mos' file  and the molecular  coordinates from the
        # old job. The  methods setting up the job must be  aware of this. They
        # may  for  example check  the  status  of  the `restart'  switch.

        self.restart_mos = restart.get_temp_result_filename('mos')

        # The  `restart.get_temp_result_filename'   method  gracefully  returns
        # `None' in case the file can't be found.

        if self.restart_mos is None:
            raise PyAdfError("Can't grab the `mos' file from the result "
                             + "object you've asked me to restart from.")

    def before_run(self):
        """
        Runs I{define} with the current settings.

        @raise PyAdfError: In pathologic cases.
        """

        import os
        import shutil

        self.mol.write('coord', outputformat='tmol')
        td = TurboDefinition(self.settings)

        td.run()

        if self.restart:
            os.remove('mos')  # Some  nervous file  systems may  require this to
            #                  be done previously.
            try:
                shutil.move(self.restart_mos, 'mos')
            except OSError:  # Yes, `IOError', I've checked that.
                # This means  that we couldn't  copy the old `mos'  file. Crap,
                # the old one is already deleted!
                raise PyAdfError("""I deleted the `mos' file written by
                `define' and now it turned out that I can't copy the old `mos'
                file from the previous job you told me to restart from... Sorry
                for that.""")

    def get_runscript(self, nproc=1):
        """
        Get a shell script that runs the job.

        Output will not be redirected but written to I{stdout} and I{stderr} as
        managed by the application to be executed.

        @returns: Shell script
        @rtype:   L{str}
        """

        runscript = ''
        if nproc > 1:
            runscript += f"export PARNODES={nproc:d} \n\n"
            # Trying to get the results to stdout while it is running.
            # The results for Turbmole in parallel mode only appear in
            # the slave1.output file. For serial mode, this already works.
            runscript += "tail -f --retry slave1.output &\n"
            runscript += "TAILPID=$!\n"

        for ex in self.execute:
            runscript += ex + '\n'

        if nproc > 1:
            runscript += "kill $TAILPID\n"

        return runscript

    def after_run(self):
        """
        Some postprocessing functions.
        """
        self.molden_postprocessing()
        self.get_results_instance()

    def molden_postprocessing(self):
        """
        Try to create an orbitals.molden file.
        Additionally, there might be compatibility problems with d orbitals
        in tm-version < 7.2.
        On the other hand, there might be compatibility problems with the
        updated tm2molden program as well. Only tested with turbomole 7.1.
        """

        try:
            # Start the molden file generation
            returncode = self._tm2molden()
            tm2molden_status = (returncode == 0)
            if not tm2molden_status:
                print("ERROR: 'tm2molden' quit on error. no molden output")
        except Exception as e:
            print('an exception occured while trying to generate the molden file')
            print(str(e), e.args)
            print('this should not be a problem for the main calculation')
            raise
        else:
            try:
                try:
                    os.remove('orbitals.molden')
                    print('old molden file removed, this is unusual')
                except OSError:
                    pass
                os.rename('molden_std.input', 'orbitals.molden')
                self.settings.molden_file = True
            except OSError:
                print('could not rename molden file')

    def _tm2molden(self):
        """
        Generate a molden.input - file
        """
        from .JobRunner import DefaultJobRunner

        # Generate the input sequence to be passed.
        self.tm2molden_stdin = self._assembleInput()

        env = DefaultJobRunner().get_environ_for_local_command(TurbomoleJob)

        try:
            # Create the subprocess
            D = Popen(['tm2molden', 'norm'], stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env)

            # Pass it the input and wait for it to finish.
            self.tm2molden_stdout, self.tm2molden_stderr \
                = D.communicate(input=self.tm2molden_stdin.encode('utf-8'))
        except OSError:
            print("Couldn't start a `tm2molden' subprocess. "
                  + "Have you even installed it?")
            return None
        return D.returncode

    def _assembleInput(self):
        """
        Generates the input string to be passed to I{tm2molden}.

        @returns: Input sequence
        @rtype:   L{string} with \n

        """

        # skip entering a name
        sequence = ['']
        # does the file already exist? if so, overwrite it
        if os.path.isfile('molden_std.input'):
            sequence.append('')
        # yes, please write my data
        sequence.append('')
        # no, we don't want the optimization data
        if self.jobtype == "Turbomole geometry optimization job":
            sequence.append('n')

        inputstring = ''
        for word in sequence:
            inputstring += word + '\n'
        return inputstring

    @staticmethod
    def get_results_instance():
        """
        Get the results of this job.
        Make  a `archive.tar'  from `jobtempdir'.  The files  will be  in the
        arcive directly  with no containing directory. They  can therefore be
        extracted  via,  say,   `tar.extractfile'energy')'  (if  `tar'  is  a
        `TarFile'  object  opened in  reading  mode  with properly  specified
        gz compression.

        @return: Results
        """
        import tarfile

        tar = tarfile.open(name='archive.tar', mode='w:gz')
        for filename in os.listdir(os.getcwd()):
            tar.add(filename, arcname=filename)
        tar.close()

    def create_results_instance(self):
        """
        Create an instance of the matching results object for this job.

        This method should be overwritten in derived classes.

        @returns:    Results
        @rtype:      Respective subclass of L{TurbomoleResults}
        """
        return NotImplementedError("abstract method")

    def result_filenames(self):
        return ['archive.tar']

    def get_molecule(self):
        """
        Get the molecule of the curent job.

        @returns: Molecule
        @rtype:   L{molecule}
        """
        return self.mol

    def print_jobinfo(self):
        """
        Prints some information about the current job to I{stdout}.
        """

        print(' ' + '-' * 50)
        print(" Running " + self.jobtype)
        if self.restart:
            print(" (Restarted using `mos' from job "
                  + str(self.restart_id) + ".)")
        print()
        print("  MOLECULE:")
        print()
        self.print_molecule()
        print()
        print("  SETTINGS:")
        print()
        self.print_settings()
        print()
        print("  EXTRAS:")
        print()
        self.print_extras()
        print()
        print(' ' + '-' * 50)
        print()

    def print_molecule(self):
        """
        Prints the molecule to I{stdout}.
        """
        print(self.get_molecule())
        print(f"  charge: {self.mol.get_charge()}")

    def print_settings(self):
        """
        Prints the settings for the current job to I{stdout}.
        """
        print(self.settings)

    # noinspection PyMethodMayBeStatic
    def print_extras(self):
        """
        Prints the "extras" for this job. Whatever that is. There are no
        extras.
        """
        print("  There are no extras.")

    def print_jobtype(self):
        """
        Returns the name of this type of job.

        @return:     Type of the job
        @rtype:      L{str}
        @deprecated: This method was retained for consistency with the rest of
                     I{PyADF}. It should be avoided since its name is highly
                     confusing. It should be named C{get_jobtype} or should be
                     redefined to actually C{print} the name rather than
                     returning a string. Please use the class attribute
                     C{jobtype} instead.
        """
        return self.jobtype

    def check_success(self, outfile, errfile):
        """
        Run a brief check on the results.

        If a job obviously failed, C{False} is returned; otherwise C{True}.

        A job did not fail obviously if [C{file_on_success} is undefined or
        exists) and (C{file_on_fail} is undefined or does not exist)]. They
        have to be set by the derived job class.

        @param outfile: Filename of the file the I{stdout} by this job was
                        written to. Currently ignored.
        @type  outfile: L{str}
        @param errfile: Filename of the file the I{stderr} by this job was
                        written to. Currently ignored.
        @type  errfile: L{str}
        @return:        Success
        @rtype:         L{bool}

        """

        import os

        if self.file_on_success is None:
            success = True
        else:
            success = os.path.isfile(self.file_on_success)
        if self.file_on_fail is None:
            fail = False
        else:
            fail = os.path.isfile(self.file_on_fail)

        return success and not fail

    @property
    def checksum(self):
        """
        Generate a checksum quasi-unique to this kind of job with this molecule
        and this settings.

        This method composes a string from the type of and settings for this
        job (using I{Python}'s C{pickle} module) and generates a 128-bit md5
        checksum from it.  The concatenation of this checksum's hexadecimal
        representation and the checksum computed for the molecule (via the
        L{molecule} class' method) is returned as a string.

        This approach will not account for changes in the job class itself nor
        for changes in other methods employed. Hence, after updating I{PyADF}
        you should rerun any job regardless of the checksum matching or not.

        Please note that the checksums of identical jobs might differ due to
        internal floating point representation issues.  Calling this method
        mutiple times on the same object is guaranteed to yield the same
        hash. Calling L{set_restart} will not change the checksum as it will
        not change the results (only the speed they are gained).

        For information on the md5 algorithm, please see the RFC 1321
        U{https://tools.ietf.org/html/rfc1321.html}.

        @returns: Hexadicimal hash
        @rtype:   L{str}

        """
        import pickle
        import hashlib

        if self._checksum is None:

            m = hashlib.md5()
            m.update(str(type(self)).encode())

            # In case  pickeling should  fail, we might  re-read an empty  file and
            # generate  a  checksum  that  isn't  representative.  We  detect  this
            # potential  source of  error by  comparing the  hash before  and after
            # re-reading the file and insisting in them being different.
            failed_hash = m.hexdigest()

            m.update(pickle.dumps(self.settings, protocol=pickle.HIGHEST_PROTOCOL))
            settings_hash = m.hexdigest()

            if settings_hash == failed_hash:
                raise PyAdfError("""Error while trying to compute the md5 hash
                of the job. Hash equals empty hash.""")

            mol_hash = self.mol.checksum
            self._checksum = mol_hash + settings_hash

        return self._checksum


class TurbomoleSinglePointJob(TurbomoleJob):
    """
    Compute densities and energies for a fixed geometry with I{Turbomole} using
    DFT or MP2 methods.

    """

    def __init__(self, mol, method='dft', settings=None):
        """
        Initializes a new job.

        @param mol:      Molecule to work with
        @type mol:       L{molecule}
        @param method:   Computational method to apply. Available are:
                             - C{hf}  Hartree-Fock with SCF
                             - C{dft} SCF with DFT
                             - C{mp2} HF-SCF followed by MP2
        @type method:    L{str}
        @param settings: More specific settings
        @type  settings: L{TurbomoleSinglePointSettings}

        """

        super().__init__(mol)

        self.jobtype = "Turbomole single point job"
        self.file_on_success = 'energy'

        if settings is not None:
            if not isinstance(settings, TurbomoleSinglePointSettings):
                raise PyAdfError("""I'm a `TurbomoleSinglePointJob' so I need
            `TurbomoleSinglePointSettings' and nothing else, sorry.""")
            self.settings = settings
        else:
            self.settings = TurbomoleSinglePointSettings()

        # Chose reasonable default values for what the user hasn't specified.

        # noinspection PyProtectedMember
        self.settings._set_charge(self.mol.get_charge())
        # noinspection PyProtectedMember
        self.settings._set_spin(self.mol.get_spin())
        # noinspection PyProtectedMember
        self.settings._set_method(method)
        if self.settings.ri is None:
            if self.settings.method == 'dft':
                self.settings.set_ri(True)
            else:
                self.settings.set_ri(False)
        if self.settings.basis_set_all is None:
            self.settings.set_basis_set('def2-TZVP')
        if self.settings.method == 'dft':
            if self.settings.dft_functional is None:
                self.settings.set_dft_functional('b-p')
            if self.settings.dft_grid is None:
                self.settings.set_dft_grid('m3')
        if self.settings.guess_initial_occupation_by is None:
            self.settings.set_initial_occupation_guess_method('eht')
        if self.settings.ired is None:
            self.settings.set_redundant_internal_coordinates(False)
        if self.settings.idef is None:
            self.settings.define_internal_coordinates(False)

        # Chose the executing applications.

        if self.settings.ri and self.settings.method == 'dft':
            self.execute.append('ridft')
        else:
            self.execute.append('dscf')
        if self.settings.mp2:
            self.execute.append('mp2prep -e')
            self.execute.append('ricc2')

    def create_results_instance(self):
        # See docstring for mother method in `TurbomoleJob'.
        """
        @rtype:  L{TurbomoleSinglePointResults}

        """
        return TurbomoleSinglePointResults(self)


class TurbomoleGeometryOptimizationJob(TurbomoleJob):
    """
    Optimize molecular geometries with I{Turbomole} using DFT or MP2 methods.

    """

    def __init__(self, mol, method='dft', settings=None):
        """
        Initializes a new job.

        @param mol:      Molecule to work with
        @type mol:       L{molecule}
        @param method:   Computational method to apply. Available are:
                             - C{hf}  Hartree-Fock with SCF
                             - C{dft} SCF with DFT
                             - C{mp2} HF-SCF followed by MP2
        @type method:    L{str}
        @param settings: More specific settings
        @type  settings: L{TurbomoleGeometryOptimizationSettings}

        """

        super().__init__(mol)

        self.jobtype = "Turbomole geometry optimization job"

        if settings is not None:
            if not isinstance(settings, TurbomoleGeometryOptimizationSettings):
                raise PyAdfError("""I'm a `TurbomoleGeometryOptimizationJob' so I need
            `TurbomoleGeometryOptimizationSettings' and nothing else, sorry.""")
            self.settings = settings
        else:
            self.settings = TurbomoleGeometryOptimizationSettings()

        # Chose reasonable default values for what the user hasn't specified.

        # noinspection PyProtectedMember
        self.settings._set_charge(self.mol.get_charge())
        # noinspection PyProtectedMember
        self.settings._set_spin(self.mol.get_spin())
        # noinspection PyProtectedMember
        self.settings._set_method(method)
        if self.settings.ri is None:
            if settings.method == 'dft':
                self.settings.set_ri(True)
            else:
                self.settings.set_ri(False)
        if self.settings.basis_set_all is None:
            self.settings.set_basis_set('def2-TZVP')
        if self.settings.method == 'dft':
            if self.settings.dft_functional is None:
                self.settings.set_dft_functional('b-p')
            if self.settings.dft_grid is None:
                self.settings.set_dft_grid('m3')
        if self.settings.guess_initial_occupation_by is None:
            self.settings.set_initial_occupation_guess_method('eht')
        if self.settings.ired is None:
            # Turbomole is likely to mess up if `ired' is used for a linear
            # molecule. I don't know any common*) linear molecule with more
            # than 3 atoms. For such small atoms, ired doesn't help much
            # anyway.
            #
            # *) O=C=C=O would be one and you can insert an arbitrary number of
            #    carbons but let's hope nobody wants to copute THAT or if she
            #    does, takes care of the `ired' settings herself.

            self.settings.set_redundant_internal_coordinates(self.mol.get_number_of_atoms() > 3)
        if self.settings.idef is None:
            self.settings.define_internal_coordinates(False)
        if self.settings.gcart is None:
            self.settings.set_convergence_criterion(4)
        if self.settings.max_iterations is None:
            self.settings.set_max_iterations(500)
        if self.settings.job_verbosity is None:
            self.settings.set_job_verbosity(True)

        self.file_on_success = 'GEO_OPT_CONVERGED'
        self.file_on_fail = 'GEO_OPT_FAILED'

        if self.settings.mp2:
            self.execute.append('dscf')
            self.execute.append('mp2prep -g')

        shellstring = 'jobex'
        if self.settings.mp2:
            shellstring += ' -level=cc2'
        elif self.settings.ri:
            shellstring += ' -ri'
        if self.settings.job_verbosity:
            shellstring += ' -outfile /dev/stdout'
        shellstring += ' -gcart ' + str(self.settings.gcart)
        shellstring += ' -c ' + str(self.settings.max_iterations)
        self.execute.append(shellstring)

    def before_run(self):

        super().before_run()

        if self.settings.frozen_atoms:
            print("Found frozen atoms list:", self.settings.frozen_atoms)
            with open('coord_frozen', 'w') as out_file:
                with open('coord') as in_file:
                    for i, line in enumerate(in_file):
                        if i in self.settings.frozen_atoms:
                            out_file.write(line.rstrip('\n') + '    f' + '\n')
                        else:
                            out_file.write(line)
            os.rename('coord_frozen', 'coord')

    def create_results_instance(self):
        # See docstring for mother method in `TurbomoleJob'.
        """
        @rtype:  L{TurbomoleGeometryOptimizationResults}
        """
        return TurbomoleGeometryOptimizationResults(self)


class TurbomoleGradientJob(TurbomoleJob):
    """
    Compute gradients for fixed geometries with I{Turbomole} using DFT or MP2
    methods.

    """

    def __init__(self, mol, method='dft', settings=None):
        """
        Initializes a new job.

        @param mol:      Molecule to work with
        @type mol:       L{molecule}
        @param method:   Computational method to apply. Available are:
                             - C{hf}  Hartree-Fock with SCF
                             - C{dft} SCF with DFT
                             - C{mp2} HF-SCF followed by MP2
        @type method:    L{str}
        @param settings: More specific settings
        @type  settings: L{TurbomoleGradientSettings}
        """

        super().__init__(mol)

        self.jobtype = "Turbomole gradient job"
        self.file_on_success = None  # FIX THIS!
        self.file_on_fail = None  # FIX THIS!

        if settings is not None:
            if not isinstance(settings, TurbomoleGradientSettings):
                raise PyAdfError("""I'm a `TurbomoleGradientJob' so I need
            `TurbomoleGradientSettings' and nothing else, sorry.""")
            self.settings = settings
        else:
            self.settings = TurbomoleGradientSettings()

        # Chose reasonable default values for what the user hasn't specified.

        # noinspection PyProtectedMember
        self.settings._set_charge(self.mol.get_charge())
        # noinspection PyProtectedMember
        self.settings._set_spin(self.mol.get_spin())
        # noinspection PyProtectedMember
        self.settings._set_method(method)
        if self.settings.ri is None:
            self.settings.set_ri(True)
        if self.settings.basis_set_all is None:
            self.settings.set_basis_set('def2-TZVP')
        if self.settings.method == 'dft':
            if self.settings.dft_functional is None:
                self.settings.set_dft_functional('b-p')
            if self.settings.dft_grid is None:
                self.settings.set_dft_grid('m3')
        if self.settings.guess_initial_occupation_by is None:
            self.settings.set_initial_occupation_guess_method('eht')
        if self.settings.ired is None:
            self.settings.set_redundant_internal_coordinates(False)
        if self.settings.idef is None:
            self.settings.define_internal_coordinates(False)

        if self.settings.ri:
            self.execute.append('ridft')
            self.execute.append('rdgrad')
        else:
            self.execute.append('dscf')
            self.execute.append('grad')

    def create_results_instance(self):
        # See docstring for mother method in `TurbomoleJob'.
        """
        @rtype:  L{TurbomoleGradientResults}

        """
        return TurbomoleGradientResults(self)


class TurbomoleForceFieldJob(TurbomoleJob):
    """
    Preoptimize molecular geometries using I{uff} - the I{Turbomole} force
    field client.

    @group Obsolete: set_restart
    """

    def __init__(self, mol, settings=None):
        """
        Initializes a new job.

        @param mol:      Molecule to work with
        @type mol:       L{molecule}
        @param settings: More specific settings
        @type  settings: L{TurbomoleForceFieldSettings}

        """

        super().__init__(mol)

        self.jobtype = "Turbomole force field preoptimization job"
        self.file_on_success = None  # FIX THIS!
        self.file_on_fail = None  # FIX THIS!

        if settings is not None:
            if not isinstance(settings, TurbomoleForceFieldSettings):
                raise PyAdfError("""I'm a `TurbomoleForceFieldJob' so I need
            `TurbomoleForceFieldSettings' and nothing else, sorry.""")
            self.settings = settings
        else:
            self.settings = TurbomoleForceFieldSettings()

        # Chose reasonable default values for what the user hasn't specified.

        if self.settings.max_iterations is None:
            self.settings.set_max_iterations(500)

        # We have  to mess around  in the `control'  file in the middle  of the
        # computation.  That means, that  the shell  script has  to do  this. I
        # don't  want  to use  Python  for  this task  since  it  might not  be
        # available on the cluster that runs the job. Let's hope at least `sed'
        # is. The regexp has to make the line
        #
        #          1         1          0 ! maxcycle,modus,nqeq
        #
        # become
        #
        #   MAXCYCLES        1          0 ! maxcycle,modus,nqeq
        #
        # where MAXCYCLES is a number > 1.

        self.execute.append('uff')
        self.execute.append((r"sed -i 's/"
                             + r"^[ ]*\([0-9]\+\)[ ]*\([0-9]\+\)[ ]*\([0-9]\+\)[ !]*\(maxcycle,modus,nqeq\)"  # old
                             + r"/"
                             + r"\t{cycles}\t\2\t\3\t! \4"  # new
                             + r"/g' control").format(cycles=self.settings.max_iterations))
        self.execute.append('uff')

    def set_restart(self, restart):
        """
        Not to be used.

        Since there are no MOs in a force field job, this method makes no sense
        here.

        @raises NotImplementedError: Always

        """
        raise NotImplementedError("Calling this method makes no sense here!")

    def before_run(self):
        """
        Writes the I{coord} file.

        """
        self.mol.write('coord', outputformat='tmol')

    def create_results_instance(self):
        # See docstring for mother method in `TurbomoleJob'.
        """
        @rtype:  L{TurbomoleForceFieldResults}

        """
        return TurbomoleForceFieldResults(self)
