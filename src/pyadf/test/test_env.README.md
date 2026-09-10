How to create the conda environment for testing
-----------------------------------------------
```
conda create -c conda-forge -n SOMENAME  numpy "libopenblas=*=*openmp*" scipy pyscf h5py openbabel rdkit xcfun fortranformat
conda activate SOMENAME
conda env export > $YOUR_PYADF_HOME/src/pyadf/test/test_env.yml
```
remove last line (prefix=...) from the file `test_env.yml`
