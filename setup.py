#!/usr/bin/env python

from distutils.core import setup

setup(name         = 'PyADF', 
      version      = '0.9', 
      description  = 'PyADF - A Scripting Framework for Multiscale Quantum Chemistry', 
      author       = 'Christoph Jacob and others', 
      author_email = 'christoph.jacob@kit.edu', 
      url          = 'http://www.pyadf.org', 
      license      = 'GPLv3',
      package_dir  = {'': 'src'}, 
      packages     = ['pyadf'], 
      py_modules   = ['pyadfenv', 'kf', 'xml2kf'], 
      scripts      = ['src/scripts/pyadf', 'src/scripts/create_pwd_on_nodes.py']
     )

