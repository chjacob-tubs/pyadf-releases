#!/usr/bin/env python3

from distutils.core import setup

setup(name='PyADF',
      version='0.99',
      description='PyADF - A Scripting Framework for Multiscale Quantum Chemistry',
      author='Christoph Jacob and others',
      author_email='c.jacob@tu-braunschweig.de',
      url='https://www.pyadf.org',
      license='GPLv3',
      package_dir={'': 'src'},
      packages=['pyadf', 'pyadf.Molecule', 'pyadf.Plot', 'pyadf.PyEmbed'],
      py_modules=['pyadfenv', 'kf', 'xml2kf'],
      scripts=['src/scripts/pyadf', 'src/scripts/create_pwd_on_nodes.py']
      )
