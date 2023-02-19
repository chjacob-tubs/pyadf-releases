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
 PyADF Overview
 ==============

 PyADF is a scripting framework for quantum chemistry, with a special
 focus on multiscale simulations. For an in-depth introduction, see
 the overview article in U{J. Comput. Chem. B{32} (2011),
 2328-2338<https://dx.doi.org/10.1002/jcc.21810>}.

 PyADF input files are Python scripts, in which the classes defined by
 the PyADF scripting framework are available by default.

 This is an example of a minimal PyADF input file::

     mol = molecule('h2o.xyz')
     job = adfsinglepointjob( mol, basis='TZ2P')
     results = job.run()
     print results.get_dipole_magnitude()

 B{Running PyADF calculations}

 To run a PyADF script, use

 C{pyadf input.pyadf}

 Type C{pyadf --help} for command line options.

 B{Basic steps of a PyADF calculation}

 The minimal PyADF script given above shows the typical steps for
 setting up a calculation using PyADF.

     1. B{Setting up the molecule}

     Molecules are represented using the L{molecule} class.
     In the simplest case, you just have to give the filename
     of an xyz file containing the coordinates.

     >>> mol = molecule('h2o.xyz')

     More complicated things are also possible, e.g., obtaining
     individual residues from a pdb file.


     2. B{Creating the job}

     Next, you have to create a job. Different kinds of jobs are
     available, among others L{adfsinglepointjob}, L{adffragmentsjob}
     and L{adfgeometryjob}.

     A simple ADF single point calculation, can be set up like this
     (C{myfiles} is the global file manager provided by PyADF):

     >>> job = adfsinglepointjob( mol, basis='TZ2P')

     More options for this job could be provided using the L{adfsettings} class.
     Other kinds of jobs might require more information for initialization,
     please look at the documentation.


     3. B{Running the job}

     Now you are ready to actually run your job:

     >>> results = job.run()

     This will now run the actual ADF calculation.


     4. B{Doing something with the results}

     The C{job.run()} method returns a results object, which is an
     instance of some subclass of L{adfresults} (depending on the
     type of job you ran).
     In the case of an ADF single point calculation, an instance of
     L{adfsinglepointjob} is returned. This object can now be used
     to obtain things that have been calculated, for instance the
     dipole moment:

     >>> print results.get_dipole_magnitude()
     0.6629161489

     It is up to you what you do with these results, you can just print them
     or do more complicated manipulations with them.

 Using these basic building blocks, more complicated workflows can be built.
 For more details, please have a look at the documentation of the
 individual classes and at the examples in test/testinputs.

 B{Further information}

 Detailed documentation of the available classes is available.
 The following classes are most commonly used in PyADF input files:

  - B{Molecular Coordinates}
    - For more on the manipulation of the molecular geometry, see L{molecule}
  - B{ADF Calculations}
    - The most important job and result classes for simple ADF calculations are:
      - ADF single point calculations: L{adfsinglepointjob} and L{adfsinglepointresults}
      - ADF geometry optimizations: L{adfgeometryjob}
      - ADF frequency calculations: L{adffreqjob}
      - ADF chemical shifts: L{adfnmrjob} and L{adfnmrresults}
      - ADF spin-spin couplings : L{adfcpljob} and L{adfcplresults}
    - For all ADF calculations, settings are defined with L{adfsettings}
    - For more on plotting and related functionality, see L{Plot}
  - B{Dalton, Dirac, and NWChem Calculations}
    - Dalton single point calculations: L{daltonsinglepointjob} and L{daltonsinglepointresults}
    - Dalton CC2 excitation energies: L{daltonCC2job} and L{daltonCC2results}
    - Dirac single point calculations: L{diracsinglepointjob} and L{diracsinglepointresults}
    - NWChem single point calculations: L{nwchemsinglepointjob} and L{nwchemsinglepointresults}
  - B{Multiscale Quantum Chemistry}
    - FDE calculations with ADF: L{adffragmentsjob} and L{adffragmentsresults}
    - MFCC calculations with ADF: L{adfmfccjob} and L{mfccresults}
    - 3-FDE calculations with ADF: L{adf3fdejob} and L{adf3fdejob}
    - WFT-in-DFT embedding calculations: L{wftindftjob}

 B{Tests and Examples}

 The directory C{tests/testinputs} contains a number of examples.
 To run one of these examples, either use the C{test_pyadf}
 testing program, or copy the C{.pyadf} input file together
 with all required coordinates files to a new directory.
 In this directory, run C{pyadf input.pyadf}

 More examples (including those from the JCC paper) will be
 made available shortly.

 B{Finally...}

 Any suggestions for improvements are welcome!

 If you are doing cool things with PyADF, please let us know!


 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Molecule:
    Molecule
 @group ADF Calculations:
    ADFSinglePoint,
    ADFGeometry,
    ADFFrequencies,
    ADFFragments,
    ADF_NMR,
    ADF_CPL
 @group ADF Frozen Density Embedding:
    ADF_FDE,
    ADF_3FDE,
    ADFPotential,
    ADF_FDE_AccurateEmbedding,
    WFTinDFT
 @group ADF Utilities:
    ADFNumDiff,
    ADF_FDE_Analysis
 @group Dalton Calculations:
    DaltonSinglePoint
    DaltonCC2
 @group Dirac Calculations:
    Dirac
 @group NWChem Calculations:
    NWChem
 @group Turbomole and Friends:
    Turbomole
    SNF
    TurboDefinition
 @group Plotting:
    Plot.Grids
    ADF_Densf
 @group PyADF infrastructure:
    Files, JobRunner, BaseJob, ADFBase
 @group Utilities:
    Utils, Errors, PatternsLib

 @undocumented: __package__

 @newfield example: Example, Examples
 @newfield exampleuse: Example of Usage, Examples of Usage

"""

__version__ = '1.1'
from .Utils import VersionInfo

# noinspection PyUnresolvedReferences
import kf

from .Utils import pse, Bohr_in_Angstrom, au_in_eV, au_in_Debye, conversion
from .Errors import PyAdfError
from .Molecule import molecule, MoleculeFactory

from .Files import adf_filemanager
from .JobRunnerConfiguration import JobRunnerConfiguration
from .JobRunner import DefaultJobRunner, SerialJobRunner
from .BaseJob import job, results
from .ADFBase import scmjob, amsjob, adfjob, adfresults

from .ADFSinglePoint import adfsettings, adfscfsettings, adfsinglepointjob, \
    adfsinglepointresults, adfspjobdecorator
from .ADFFragments import fragment, FrozenDensFragment, fragmentlist, adffragmentsjob
from .ADFGeometry import adfgeometrysettings, adfgeometryjob, adfgradientsjob, adfgradientsresults
from .ADFFrequencies import adffreqjob, adfsinglepointfreqjob
from .ADF_NMR import adfnmrjob, adfnmrresults
from .ADF_CPL import cplsettings, adfcpljob, adfcplresults
from .ADF_DFTB import dftbsettings, dftbsinglepointjob, dftbgeometryjob, dftbfreqjob
from .ADFNumDiff import numgradsettings, adfnumgradjob, adfnumgradsjob

from .DaltonSinglePoint import daltonsettings, daltonjob, daltonsinglepointjob
from .DaltonCC2 import daltonCC2settings, daltonCC2job

from .Orca import OrcaSettings, OrcaTDDFTSettings, OrcaJob, OrcaSinglePointJob, \
    OrcaGeometryOptimizationJob, OrcaFrequenciesJob, OrcaOptFrequenciesJob, \
    OrcaExcitationsJob, OrcaExStateGeoOptJob, OrcaExStateFrequenciesJob, \
    OrcaResults

from .Dirac import diracsettings, diracjob, diracsinglepointjob
from .QuantumEspresso import QEJob, QESinglePointJob, QEResults, QESettings
from .QuantumEspresso_PostProc import QEPostProcJob, QEPostProcResults, QEPostProcSettings


from .Molcas import MolcasJob, MolcasSinglePointJob, MolcasResults, MolcasSettings

from .NWChem import nwchemsettings, nwchemjob, nwchemsinglepointjob
from .NWChemCC2 import nwchemCC2job

from .Turbomole import TurbomoleSinglePointSettings, TurbomoleGeometryOptimizationSettings, \
    TurbomoleGradientSettings, TurbomoleForceFieldSettings, \
    TurbomoleJob, TurbomoleSinglePointJob, \
    TurbomoleGeometryOptimizationJob, TurbomoleGradientJob, \
    TurbomoleForceFieldJob

from .SNF import SNFJob, SNFResults

from .ADF_FDE import adffdejob, adffderesults, adffdesettings

try:
    from .ADF_3FDE import cappedfragment, cappedfragmentlist, capmolecule, \
        adfmfccjob, mfccresults, adf3fdejob
except ImportError:
    pass

try:
    from .MFCCMBE2 import mfccmbe2results, mfccmbe2job, mfccmbe2interactionresults, mfccmbe2interactionjob,\
        generalmfccresults, generalmfccjob, mfccinteractionresults, mfccinteractionjob
except ImportError:
    pass

from .ADF_FDE_AccurateEmbedding import adfaccurateembeddingjob

from .ADF_Densf import densfjob
from .Plot.Grids import cubegrid, adfgrid, customgrid

from .WFTinDFT import diracfragment, wftindftjob

from .ADF_FDE_Analysis import adffdeanalysisjob, adffdeanalysissettings, datasetjob

from .ADFPotential import adfimportgridjob, adfpotentialjob, adfimportembpotjob

try:
    import xcfun
    from .PyEmbed import *
    from .ManyBodyExpansion import MBEJob, MBEResults, DensityBasedMBEJob, DensityBasedMBEResults
except ImportError:
    pass
