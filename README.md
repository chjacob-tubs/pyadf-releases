PyADF - A Scripting Framework for Multiscale Quantum Chemistry

## Installation Instructions

The recommended way to install PyADF is to install it within a `conda`
environment. 
You should also be able to manually add the src-folder to your `$PYTHONPATH`
and the src/scripts-folder to your `$PATH`.
In that case you will have to install the (optional and mandatory) requirements manually.

### Installation with conda

To install PyADF within a new `conda` environment with full functionality,
follow these steps:

1. Create and activate a new `conda` environment:

```bash
conda create -n your_env_name
conda activate your_env_name
```

2. Install `numpy` and other dependencies via `conda`:

```bash
conda install -c conda-forge numpy scipy xcfun pyscf openbabel rdkit
```

3. Install PyADF from the top level folder (where the `setup.py` resides):

```bash
pip install .
```

4. Now you should be able to run the `pyadf` and `test_pyadf` scripts
from the `conda` environment where you installed PyADF. There are some
command line options which you can examine using the following commands:

```bash
pyadf --help
test_pyadf --help
```

### Requirements and Dependencies

There are two types of requirements for PyADF:

1. PyADF is written in Python and depends on some Python-packages and libraries.

2. PyADF is a scripting framework that can be used with a wide range of quantum chemistry
software packages.

#### Python Requirements

`scipy` and `numpy` are necessary to start the `pyadf`-script.
`xcfun` and `pyscf` are needed for some calculations that PyADF can perform directly,
especially embedding calculations.
`openbabel` and `rdkit` are needed for extended functionality while handling molecules,
but a molecule class that relies on neither exists as well.
The `src/pyadf/test/test_env.yml`-file tells you which versions of the Python dependencies
the provided version of PyADF was most recently tested against.

#### QC Dependencies

Similarly, the `src/pyadf/test/test_pyadf.conf`-file tells you which versions of
the QC dependencies the provided version of PyADF was most recently tested against.
The usage of such files is optional and depends on the Environment Modules package.
A default configuration file can be provided in your home folder as `~/.pyadfconfig`.

The file also includes some settings regarding the parallelization with an MPI im-
plementation. Please correspond the manuals of the QC packages to set up your own
MPI implementation. Often, the `$LD_LIBRARY_PATH`-variable has to be set for MPI
to work.

Irrespective of whether you use the Environment Modules package and the `jobrunnerconf`
functionality for this, you have to set some environment variables for each of the
programs to function properly with PyADF. PyADF often uses these variables to find
the scripts via explicit paths rather than relying on the `$PATH`-variable:

##### AMS

You should set the variables AMS needs (cf. the AMS documentation) like `$AMSHOME`,
`$AMSBIN`, `$AMSRESOURCES` etc. PyADF directly depends on `$AMSBIN` to find the
executables. Some tests rely on `$AMSRESORCES`.

##### Dalton

You should set the variables DALTON needs (cf. the DALTON documentation) like
`$DALTONHOME`, `$DALTONBIN` etc. PyADF directly depends on `$DALTONBIN` to find
the executables.

##### DIRAC

You should set the variables DIRAC needs (cf. the DIRAC documentation) like
`$DIRACHOME`, `$DIRACBIN` etc. PyADF directly depends on `$DIRACBIN` to find
the executables.

##### MOLCAS

For MOLCAS, the main folder has to be set as `$MOLCAS`. The `bin` folder within the
main folder has to be added to the `$PATH`-varibale as the current implementation
depends on the `$PATH`-variable.

##### NWChem

NWChem depends on the `$NWCHEMBIN`-variable explicitly. Check the NWChem documentation
whether variables like `$NWCHEM_NWPW_LIBRARY`, `$NWCHEM` and `$NWCHEM_BASIS_LIBRARY`
need to be set as well as whether `$NWCHEMBIN` has to be added to the `$PATH`-variable.

##### ORCA

The main folder of ORCA is usually added to the `$PATH`-variable. However,
PyADF utilizes the `$ORCA_PATH`-variable. The MPI implementation is set via
the `$OPAL_PREFIX`.

##### PySCF

Has already been covered under Python Requirements and is currently not supported
in the same way the other QC packages are.

##### Quantum ESPRESSO

Quantum ESPRESSO in PyADF depends on the `$QEBINDIR`-variable which has to be set
on the `bin` directory in the main directory of Quantum ESPRESSO. Quantum ESPRESSO
generally uses the variables `$BIN_DIR`, `$PSEUDO_DIR` and `$TMP_DIR`, with the
first two depending on the path of the installed files.

##### TURBOMOLE (+ SNF)

PyADF currently utilizes the `$PATH`-variable to start the different scripts for
pure TURBOMOLE calculations. So the `$TURBOBIN`-variable, which is used for SNF
calculations, has to be added to the `$PATH`-variable. The `$TURBODIR`-variable
is always needed for the installation to work.

## Using PyADF

The general workflow for PyADF relies on input scripts which are themselves
written in Python and are run through `pyadf`:

```bash
pyadf --save example_script.pyadf
```

You can try running the full (some tests depend on OpenBabel and only run with OpenBabel
installed) set of tests provided with PyADF by using:

```bash
test_pyadf --keep --timings --save --molclass=openbabel --tests=all --jobrunnerconf=YOURPATHTOPYADF/pyadf/test/test_pyadf.conf
```

Whether you can/need to use the `jobrunnerconf`-option depends on how you install
and manage your QC software (see section QC dependencies).

A single test can be run by using:

```bash
test_pyadf --keep --save --singletest=Orca_SinglePoint_Dipole
```

see also:

```bash
pyadf --help
test_pyadf --help
```

The test-inputs which you can find in (`src/pyadf/test/testinputs/SOME_TEST/SOME_TEST.pyadf`)
can give you further hints regarding the features of PyADF.
