# -*- coding: utf-8 -*-

# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2021 by Christoph R. Jacob, Tobias Bergmann,
# S. Maya Beyhan, Julia Brüggemann, Rosa E. Bulo, Thomas Dresselhaus,
# Andre S. P. Gomes, Andreas Goetz, Michal Handzlik, Karin Kiewisch,
# Moritz Klammler, Lars Ridder, Jetze Sikkema, Lucas Visscher, and
# Mario Wolter.
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
 Job and results for numerical differentiations with ADF

 At present it is designed to do only numerical gradients
 The end user should use/check the following classes

 @group Settings:
   numgradsettings
 @group Jobs:
   adfnumgradjob
   adfnumgradsjob
 @group Results:
    adfnumgradresults
    adfnumgradsresults

 @author:       Andreas W. Goetz
 @organization: Vrije Universiteit Amsterdam (2008)

"""

from Utils import Bohr_in_Angstrom
from BaseJob import results, metajob
from Errors import PyAdfError


class numdiffsettings(object):
    """
    Class for the settings of a numerical differences job

    """

    def __init__(self, stepsize=0.01, atomicunits=True, method='3point'):
        """
        Initialize the settings for a numerical differentiation

        @parameter stepsize:
            step size for perturbation (default is in atomic units)
        @type stepsize: float

        @param atomicunits:
            Whether the stepsize is given in atomic units.
        @type atomicunits: bool

        @param method:
            The method of numerical differentiation
            Supported: 3point or 5point central differences
        @type method: str

        """
        if not isinstance(stepsize, float):
            raise PyAdfError("wrong stepsize in numdiffsettings")
        self.stepsize = stepsize

        if not isinstance(atomicunits, bool):
            raise PyAdfError("wrong atomicunits in numdiffsettings")
        self.atomicunits = atomicunits

        self.method = method


class numgradsettings(numdiffsettings):
    """
    Class for the settings of a numerical gradient job

    """

    def __init__(self, stepsize=0.01, atomicunits=True, method='3point', partial=False, energyterm=None):
        """
        Initialize the settings for the numerical differentiation

        @parameter stepsize:
            step size for displacement of atoms (in Bohr or Angstrom, default is Bohr)
        @type stepsize: float

        @param atomicunits:
            Whether the stepsize is given in atomic units.
            By default, it is in Bohr (atomic units).
        @type atomicunits: bool

        @param method:
            The method of numerical differentiation
            Supported: 3point or 5point central differences
        @type method: str

        @param partial:
            whether full or partial derivatives should be computed
            (i.e. with or without relaxed density)
        @type partial: bool

        """
        numdiffsettings.__init__(self, stepsize, atomicunits, method)

        # irrespective of the input, we will continue in atomic units
        # we will also extract the bond energy in atomic units
        if not atomicunits:
            self.stepsize = self.stepsize / Bohr_in_Angstrom
        self.atomicunits = True

        self.partial = partial
        self.energyterm = energyterm

    def get_info(self):
        """
        Get information about the settings for the numerical differentiation

        @returns: information about the settings
        @rtype:   string

        """
        if self.atomicunits:
            ssb = self.stepsize
            ssa = ssb * Bohr_in_Angstrom
        else:
            ssa = self.stepsize
            ssb = ssa / Bohr_in_Angstrom
        string = " stepsize: %6.4f Angstrom (%6.4f Bohr)\n" % (ssa, ssb)

        if self.method == '3point':
            string += " using 3 point central differences formula\n"
        elif self.method == '5point':
            string += " using 5 point central differences formula\n"

        if self.partial:
            string += " ... performing *partial* derivative\n"

        return string


class numdiffresults(results):
    """
    Results of a numdiffjob

    """

    def __init__(self, job):
        results.__init__(self, job)
        self.settings = self.job.settings
        self.points = None

    def get_points(self):
        """
        Method to retrieve the perturbed values at the points

        This is an abstract method that has to be overridden
        by child classes.

        @returns: the perturbed values at the points
        @rtype:   list of float

        """
        pass

    def compute_derivative(self):
        """
        Compute the numerical derivative from the perturbed values at the points

        This method works for scalar properties.
        (For non-scalar quantities, an extension needs to be coded)

        It should be used by child classes which implement certain properties as derivatives

        @returns: the numerical derivative
        @rtype:   float

        """
        stepsize = self.settings.stepsize

        if self.points is None:
            raise PyAdfError('points have not been initialized in compute_derivative')

        if self.settings.method == '3point':
            derivative = (self.points[0] - self.points[1]) / (2 * stepsize)
        elif self.settings.method == '5point':
            derivative = (self.points[3] - 8 * self.points[2] + 8 * self.points[1] - self.points[0]) / (12 * stepsize)
        else:
            raise PyAdfError('unsupported method for numerical differentiation, choose 3point or 5point')

        return derivative


class numgradresults(numdiffresults):
    """
    Results of a numgradjob

    @group Retrival of specific results:
        get_gradient

    """

    def __init__(self, job):
        numdiffresults.__init__(self, job)

    def get_gradient(self, energyterm=None):
        """
        Compute the numerical gradient from the energies
        collected at the displaced gemoetries

        @param energyterm:
            The energyterm for which partial derivatives should be calculated
        @type energyterm: str

        """
        self.get_points(energyterm)
        return self.compute_derivative()

    def get_points(self, energyterm=None):
        """
        Get the energies at the displaced geometries (point)

        This is an abstract method that has to be overridden
        by child classes (e.g.f or different QM programs)

        @param energyterm:
            The energyterm for which partial derivatives should be calculated
        @type energyterm: str

        @returns: the energy at perturbed geometries
        @rtype:   list of float

        """
        pass


class adfnumgradresults(numgradresults):
    """
    Results of an adfnumgradjob

    """

    def __init__(self, job):
        numgradresults.__init__(self, job)

    def get_points(self, energyterm=None):
        """
        Get the energies at the displaced geometries (points)

        """
        self.points = []
        for res in self.job.results:
            if energyterm is None:
                energy = res.get_bond_energy()
            elif energyterm.startswith('Total'):
                energy = res.get_result_from_tape('Total Energy', ' '.join(energyterm.split()[1:]))
            elif energyterm.startswith('FDE'):
                energy = res.get_result_from_tape('FDE Energy', ' '.join(energyterm.split()[1:]))
            else:
                energy = res.get_result_from_tape('Energy', energyterm)
            self.points.append(energy)


class numdiffjob(metajob):
    """
    Abstract base class for a numerical differences job

    """

    def __init__(self, mol, settings):
        """
        Initialize a numerical differences job

        @param mol:
            The molecule on which to perform the numerical differences calculation
        @type mol: Pyadf.Molecule.molecule

        @param settings:
           Settings for the numerical differentiation
           Can be overriden by child classes
        @type settings: numdiffsettings
        """
        metajob.__init__(self)

        self.molecule = mol

        if settings is None:
            self.settings = numdiffsettings()
        elif isinstance(settings, numdiffsettings):
            self.settings = settings
        else:
            raise PyAdfError("wrong settings object in numdiffjob")

    def metarun(self):
        """
        Run the numerical differences job

        This is an abstract method that has to be overridden by child classes

        """
        return numdiffresults(self)


class numgradjob(numdiffjob):
    """
    Class for a numerical gradient job

    """

    def __init__(self, mol, atom, coordinate, settings):
        """
        Initialize a numerical gradient job

        @param mol:
            The molecule on which to perform the numerical gradient calculation
        @type mol: Pyadf.Molecule.molecule

        @parameter atom:
            Atom number for which to compute the gradient
        @type atom: int

        @parameter coordinate:
            coordinate which the gradient shall be computed (x, y or z)
        @type coordinate: str

        @param settings:
            settings for the numerical gradient run
        @type settings: numgradsettings

        """
        numdiffjob.__init__(self, mol, settings)

        if not isinstance(atom, int):
            raise PyAdfError("non-integer atom number provided in numgradjob")
        self.atom = atom

        check = ['x', 'y', 'z', 'X', 'Y', 'Z']
        if coordinate not in check:
            raise PyAdfError("wrong coordinate provided in numgradjob")
        self.coordinate = coordinate.lower()

        if settings is None:
            self.settings = numgradsettings()
        elif isinstance(settings, numgradsettings):
            self.settings = settings
        else:
            raise PyAdfError("numgradsettings wrong in numgradjob")

    def print_jobtype(self):
        """
        Print info about the settings for this job

        """
        atom_symbol = self.molecule.get_atom_symbols(atoms=[self.atom])[0]
        string = "\nNumerical gradient job\n\n"
        string += " >> Atom %3i (%s), %s-coordinate <<\n" % (self.atom, atom_symbol, self.coordinate)
        string += self.settings.get_info()
        print string

    def get_displacements(self):
        """
        Determine the displacements required for this run

        """
        stepsize = self.settings.stepsize

        if self.settings.method == '3point':
            steps = [1, -1]
        elif self.settings.method == '5point':
            steps = [2, 1, -1, -2]
        else:
            raise PyAdfError('unsupported method for numerical differentiation, choose 3point or 5point')

        displacements = []
        for s in steps:
            displacements.append(s * stepsize)

        return displacements

    def metarun(self):
        """
        Run the numerical gradient job
        (perform the single point jobs required for numerical differentiation)

        This is an abstract method that has to be overridden by child classes

        @returns: the results object for a numerical gradient job
        @rtype:   numgradresults

        """
        return numgradresults(self)


class adfnumgradjob(numgradjob):
    """
    Class for a numerical gradient job

    Example usage:
    >>> # ADF settings
    >>> s_adf = adfsettings()
    >>> basis = 'DZP'
    >>> core = None
    >>> # ADF option list
    >>> o_adf = []
    >>> # total SCF settings
    >>> s_scf = adfscfsettings(basis, core, s_adf, o_adf)
    >>> # default numerical gradient run, atom 1, coordinate 'x'
    >>> # (no numgradsettings object required)
    >>> job = adfnumgradjob( mol, 1, 'x', None, s_scf)
    >>> res = job.run()
    >>> grad = res.get_gradient()
    >>> # now do partial derivatives
    >>> # define numgradsettings object, run the job and ...
    >>> s_numgrad = numgradsettings(partial=True)
    >>> job = adfnumgradjob( mol, 1, 'x', s_numgrad, s_scf)
    >>> res = job.run()
    >>> # ... now obtain the 'Electrostatic Interaction' energy contribution to the gradient
    >>> # ('Electrostatic Interaction' has to be on TAPE21)
    >>> grad_elstat = res.get_gradient('Electrostatic Interaction')

    """

    def __init__(self, mol, atom, coordinate, settings, scfsettings):
        """
        Initialize a numerical gradient job

        @param scfsettings:
            settings for the ADF runs
        @type scfsettings: adfscfsettings

        for the other parameters see class numgradjob

        """
        import ADFSinglePoint
        import copy

        numgradjob.__init__(self, mol, atom, coordinate, settings)

        if not isinstance(scfsettings, ADFSinglePoint.adfscfsettings):
            raise PyAdfError("scfsettings missing in numgradjob")
        self.scfsettings = copy.deepcopy(scfsettings)

        self.results = []

    def create_job(self, mol):
        """
        """
        from ADFSinglePoint import adfsinglepointjob

        s_scf = self.scfsettings

        if s_scf.create_job is None:
            job = adfsinglepointjob(mol=mol, basis=s_scf.basis, core=s_scf.core,
                                    settings=s_scf.settings, pointcharges=s_scf.pointcharges,
                                    options=s_scf.options)
        else:
            job = s_scf.create_job(s_scf, mol)

        return job

    def metarun(self):
        """
        Run the numerical gradient job

        @returns: the results object for a numerical gradient job
        @rtype:   numgradresults

        """

        print "-" * 50
        self.print_jobtype()

        # determine displacements
        displacements = self.get_displacements()

        # for partial derivatives we need the MOs at the reference geometry
        # the only way to do this in ADF is to write the Fock matrix at the reference geometry
        # and read it back in later
        # this can, at present be done with the keywords READFOCK and WRITEFOCK
        if self.settings.partial:
            raise PyAdfError('Partial derivatives no longer available, ADF interface changed')
            # s_scf.options.append('WRITEFOCK')
            # job = self.create_job(self.molecule, s_scf)
            # job.run()
            # s_scf.options.remove('WRITEFOCK')
            # s_scf.options.append('READFOCK')
            # s_scf.settings.set_ncycles(2)

        # compute and collect results for displaced geometries
        for dp in displacements:
            m = self.molecule.displace_atom(atom=self.atom, coordinate=self.coordinate, displacement=dp,
                                            atomicunits=self.settings.atomicunits)
            job = self.create_job(m)
            self.results.append(job.run())

        return adfnumgradresults(self)


class adfnumgradsresults(results):
    """
    Results of an adfnumgradsjob
    (The gradients are actually already computed in adfnumgradsjob.run() )

    """

    def __init__(self, job):
        results.__init__(self, job)
        self.settings = self.job.settings

    def get_gradients(self):
        """
        Method to retrieve the computed gradients

        @returns: the gradients
        @rtype:   numpy array

        """
        return self.job.gradients


class adfnumgradsjob(metajob):
    """
    Class for a numerical gradients job

    """

    def __init__(self, mol, settings, scfsettings, atoms='all', coordinates=('x', 'y', 'z')):
        """
        Initialize a numerical gradient job

        @param mol:
            The molecule on which to perform the numerical gradient calculation
        @type mol: Pyadf.Molecule.molecule

        @param settings:
            settings for the numerical gradients run
        @type settings: numgradsettings

        @param scfsettings:
            settings for the ADF single point runs at the displaced geometries
        @type scfsettings: scfsettings

        @param atoms:
            Atoms for which the gradient shall be computed
        @type atoms: int or list of int

        @param coordinates:
            coordinates for which the gradient shall be computed (x, y or z)
        @type coordinates: str or list of str

        """
        import ADFSinglePoint

        metajob.__init__(self)

        self.molecule = mol

        if settings is None:
            self.settings = numgradsettings()
        elif isinstance(settings, numgradsettings):
            self.settings = settings
        else:
            raise PyAdfError("numgradsettings missing in adfnumgradsjob")

        if not isinstance(scfsettings, ADFSinglePoint.adfscfsettings):
            raise PyAdfError("scfsettings missing in adfnumgradsjob")
        self.scfsettings = scfsettings

        if isinstance(atoms, int):
            atoms = [atoms]
        elif isinstance(atoms, list):
            for a in atoms:
                if not isinstance(a, int):
                    raise PyAdfError("int or list of int required for atoms to compute gradients")
        self.atoms = atoms

        if isinstance(coordinates, str):
            coordinates = [coordinates]
        check = ['x', 'y', 'z', 'X', 'Y', 'Z']
        if isinstance(coordinates, list) or isinstance(coordinates, tuple):
            self.coordinates = []
            for c in coordinates:
                if c not in check:
                    raise PyAdfError("wrong coordinates provided in adfnumgradsjob")
                else:
                    self.coordinates.append(c.lower())

        self.gradients = None

    def print_jobtype(self):
        """
        Print info about the settings for this job

        """
        m = self.molecule
        string = "\nNumerical gradient job\n\n"
        string += " Computing gradient for\n >>"
        for c in self.coordinates:
            string += " %s," % c
        string = string[:-1] + " coordinate <<\n"
        if self.atoms == 'all':
            string += " >> all atoms <<\n"
        else:
            string += " following atoms:\n"
            for at in self.atoms:
                at_symbol = m.get_atom_symbols(atoms=[at])[0]
                string += " >> %3i (%s) <<\n" % (at, at_symbol)
        string += self.settings.get_info()

        print string

    def metarun(self):
        """
        Run the numerical gradients run

        """
        import numpy

        mapping = {'x': 0, 'y': 1, 'z': 2}

        print "-" * 50
        self.print_jobtype()

        natoms = self.molecule.get_number_of_atoms()

        self.gradients = numpy.zeros((natoms, 3))

        if self.atoms == 'all':
            atomlist = range(1, natoms + 1)
        else:
            atomlist = self.atoms

        for at in atomlist:
            for coord in self.coordinates:
                job = adfnumgradjob(mol=self.molecule, atom=at, coordinate=coord,
                                    settings=self.settings, scfsettings=self.scfsettings)
                res = job.run()
                self.gradients[at - 1, mapping[coord]] = res.get_gradient(self.settings.energyterm)

        return adfnumgradsresults(self)
