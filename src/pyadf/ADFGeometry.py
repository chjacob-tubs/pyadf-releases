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
 Job and results for ADF geometry optimizations.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Jobs:
    adfgeometryjob, adfgradientsjob
 @group Settings:
     adfgeometrysettings
 @group Results:
    adfgradientsresults
"""

from .Errors import PyAdfError
from .ADFSinglePoint import adfsettings, \
    adfsinglepointjob, \
    adfsinglepointresults


class adfgradientsresults(adfsinglepointresults):
    """
    Results of an ADF gradients calculation

    @group Initialization:
        __init__
    @group Retrieval of specific results:
        get_gradients
    """

    def __init__(self, j=None):
        super().__init__(j)

    def get_gradients(self, energytype=''):
        nnuc = self.get_result_from_tape('Geometry', 'nnuc')
        if energytype == '':
            grad = self.get_result_from_tape('GeoOpt', 'Gradients_CART')
        else:
            grad = self.get_result_from_tape('Gradient', energytype)

        inpatm = self.get_result_from_tape('Geometry', 'atom order index')
        inpatm = inpatm.reshape(2, nnuc)[0]

        grad = grad.reshape(nnuc, 3)
        grad = grad[inpatm - 1]

        return grad


class adfgeometrysettings:
    """
    Class for the settings of an ADF geometry optimization job (adfgeometryjob)

    @group Constructor:
       __init__

    @group Manipulation:
       set_optim, set_iterations, set_converge

    """

    def __init__(self, optim=None, iterations=None, converge=None):
        """
        Create settings object for a geometry optimization job

        @param optim:
           optimization: 'Cartesian' or 'Delocalized' (default: 'Auto')
        @type optim: str

        @param iterations:
           maximum number of geometry optimization steps
        @type iterations: int

        @param converge:
           convergence settings (see method set_converge)
        @type converge: dict

        """
        self.optim = None
        self.iterations = None
        self.converge = None

        self.set_optim(optim)
        self.set_iterations(iterations)
        self.set_converge(converge)

    def __str__(self):
        s = '   Coordinates: '
        if self.optim is None:
            s += 'ADF default\n'
        else:
            s += self.optim + '\n'
        s += '   Iterations : '
        if self.iterations is None:
            s += 'ADF default\n'
        else:
            s += str(self.iterations) + '\n'
        s += '   Convergence criteria: '
        if self.converge == {}:
            s += 'ADF default\n'
        else:
            s += '\n'
            for k, v in self.converge.items():
                s += 16 * ' ' + k + ': ' + str(v) + '\n'
        return s

    def set_optim(self, optim):
        self.optim = optim

    def set_iterations(self, iterations):
        self.iterations = iterations

    def set_converge(self, converge):
        self.converge = {}
        if converge is not None:
            for k, v in converge.items():
                if k not in ('Energy', 'Gradients', 'Step'):
                    raise PyAdfError('Wrong key for converge in adgeometrysettings.set_converge()')
                self.converge[k] = v


class adfgeometryjob(adfsinglepointjob):
    """
    A job class for ADF geometry optimizations.

    See the documentation of L{__init__} and L{adfgeometrysettings} for details
    on the available options.

    Corresponding results class: L{adfsinglepointresults}

    @group Initialization:
        __init__
    @group Other Internals:
        print_geometrysettings
    """

    def __init__(self, mol, basis, settings=None, geometrysettings=None, frozen_atoms=None,
                 pointcharges=None, electricfield=None, core=None, options=None):
        """
        Create a new ADF geometry optimization job.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}

        @param basis:
            A string specifying the basis set to use (e.g. C{basis='TZ2P'}).
            Alternatively, a dictionary can be given listing different basis sets
            for different atom types. Such a dictionary must contain an entry "default"
            giving the basis set to use for all other atom types
            (e.g. C{basis={default:'DZP', 'C':'TZ2P'}}).
        @type basis: str or dict

        @param core:
            A string specifying which frozen cores to use (C{None}, C{Small}, C{Medium}, or C{Large}).
            Alternatively, a dictionary can be given to specify explicitly which core to
            use for each atom type (e.g. C{core={O:'1s', H:'None'}}).
            Such a dictionary can contain an entry 'default' giving the frozen core to
            use with all other atoms (possible values are C{None}, C{Small}, C{Medium}, or C{Large}).
        @type core: None or str or dict

        @param options:
            Additional options. These will each be included directly in the ADF input.
        @type options: list of str

        @param settings: The settings for this calculation, see L{adfsettings}
        @type settings: L{adfsettings}

        @param geometrysettings:
            Settings for the geometry optimization
        @type geometrysettings: L{adfgeometrysettings}

        @param frozen_atoms:
            Optionally, a list of atoms for which the coordinates should be frozen.
            (The atom numbering starts with 1). By default, all atoms are optimized.
        @type frozen_atoms:
            list of int or list of str

        @param pointcharges:
            Coordinates (x, y, z in Angstrom) and charges of point charges.
            If no charges are given, zero charges are used.
        @type pointcharges: float[3][n] or float[4][n]

        @param electricfield:
            Electric field (x, y, z in atomic units).
        @type pointcharges: float[3]

        """
        if settings is None:
            settings = adfsettings(accint=6.0)

        if geometrysettings is None:
            self.geometrysettings = adfgeometrysettings()
        else:
            self.geometrysettings = geometrysettings

        if frozen_atoms:
            if type(frozen_atoms[0]) == int:
                self._frozen_atoms = frozen_atoms
            else:
                symbs = mol.get_atom_symbols()
                self._frozen_atoms = []
                for i, s in enumerate(symbs):
                    if s in frozen_atoms:
                        self._frozen_atoms.append(i + 1)
        else:
            self._frozen_atoms = None

        super().__init__(mol, basis, core=core, settings=settings,
                         pointcharges=pointcharges, electricfield=electricfield,
                         options=options)
        self.task = 'GeometryOptimization'

    def get_properties_block(self):
        return self.get_geometry_block()

    def get_geometry_block(self):
        gs = self.geometrysettings
        block = "GeometryOptimization \n"
        block += "  Method Auto \n"
        if gs.optim is not None:
            block += "  CoordinateType " + gs.optim + "\n"
        else:
            block += "  CoordinateType Auto\n"

        if gs.iterations is not None:
            block += "  MaxIterations " + str(gs.iterations) + "\n"
        if gs.converge != {}:
            conv = "  Convergence \n"
            for k, v in gs.converge.items():
                conv += "   " + k + " " + str(v)
            block += conv + "\n"
            block += "  END\n"
        block += "END\n\n"
        if self._frozen_atoms is not None:
            block += self.get_constraints_block()
        return block

    def get_constraints_block(self):
        block = " Constraints\n"
        for i in self._frozen_atoms:
            block += f"  Atom {i:d} \n"
        block += " End\n\n"
        return block

    def print_jobtype(self):
        if self._frozen_atoms:
            return "ADF geometry optimization job (selected atoms only)"
        else:
            return "ADF geometry optimization job"

    def print_settings(self):
        super().print_settings()
        self.print_geometrysettings()

    def print_geometrysettings(self):
        print("   Geometry optimization settings")
        print("   ==============================")
        print()
        print(self.geometrysettings)
        print()


class adfgradientsjob(adfgeometryjob):
    """
    A job class for ADF gradient calculations.

    See the documentation of L{__init__} and L{adfgeometrysettings} for details
    on the available options.

    Corresponding results class: L{adfgradientsresults}

    @group Initialization:
        __init__
    """

    def __init__(self, mol, basis, settings=None, core=None, options=None):
        """
        Constructor for ADF gradients jobs.

        @param mol:
            The molecular coordinates.
        @type mol: L{molecule}

        @param basis:
            A string specifying the basis set to use (e.g. C{basis='TZ2P'}).
            Alternatively, a dictionary can be given listing different basis sets
            for different atom types. Such a dictionary must contain an entry "default"
            giving the basis set to use for all other atom types
            (e.g. C{basis={default:'DZP', 'C':'TZ2P'}}).
        @type basis: str or dict

        @param settings: The settings for this calculation, see L{adfsettings}
        @type settings: L{adfsettings}

        @param core:
            A string specifying which frozen cores to use (C{None}, C{Small}, C{Medium}, or C{Large}).
            Alternatively, a dictionary can be given to specify explicitly which core to
            use for each atom type (e.g. C{core={O:'1s', H:'None'}}).
            Such a dictionary can contain an entry 'default' giving the frozen core to
            use with all other atoms (possible values are C{None}, C{Small}, C{Medium}, or C{Large}).
        @type core: None or str or dict

        @param options:
            Additional options. These will each be included directly in the ADF input.
        @type options: list of str
        """

        import copy
        if settings is None:
            settings = adfsettings(accint=6.0)
        if options is None:
            opts = []
        else:
            opts = copy.copy(options)

        gs = adfgeometrysettings(iterations=0, converge={'Gradients': '1e3'})

        super().__init__(mol, basis, core=core, geometrysettings=gs, settings=settings, options=opts)

    def create_results_instance(self):
        return adfgradientsresults(self)

    def print_jobtype(self):
        return "ADF gradients job"
