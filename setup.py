#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(
    name='PyADF',
    version='1.4',
    license='GPLv3',
    author='Christoph Jacob and others',
    author_email='c.jacob@tu-braunschweig.de',
    url='https://github.com/chjacob-tubs/pyadf-releases',
    description='PyADF - A Scripting Framework for Multiscale Quantum Chemistry',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    scripts=[
        'src/scripts/pyadf',
        'src/scripts/test_pyadf'
    ],
    install_requires=[
        'numpy',
        'scipy',
        # 'pyscf', via pip or conda, see README.md
        # 'xcfun', via conda, see README.md
        # 'openbabel', via conda, see README.md
        # 'rdkit', via conda, alternative to openbabel
    ],
    python_requires='>=3.6'
    # other setup parameters
)
