#!/usr/bin/env python3

"""
kf.py - Implementation of a Python interface to read and write KF files.
        This is an implementation that uses the KF utilities to do the actual work,
        so you need a working ADF installation.

Copyright (C) 2006-2008 by Scientific Computing and Modelling NV.
For support, contact SCM support (support at scm . com)

This file is part of the ADF software
For more information, see <https://www.scm.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

SCM owns the intellectual property right for this file and reserves the
right to distribute it under a license other than LGPL
"""

import numpy as np

from . import _kftools


# --------------
# Exceptions
# --------------
class PyADFException(Exception):
    def __init__(self, aMessage='A PyADFException occurred.'):
        super().__init__(aMessage)

    # --------------


# KF data types
# --------------
IntegerType = 1
RealType = 2
CharacterType = 3
LogicalType = 4


class KFType:

    def __init__(self):
        pass

    @staticmethod
    def typeEnum(self):
        pass

    @staticmethod
    def regExForDataElement(self):
        pass

    @staticmethod
    def stringToDataConversionFunc(self):
        pass

    def stringForData(self, data):
        pass

    @staticmethod
    def formatData(data, nperline, fmt):
        fmt_data = [fmt.format(r) for r in data]
        fmt_lines = [''.join(fmt_data[i:i+nperline]) for i in range(0, len(fmt_data), nperline)]
        return '\n'.join(fmt_lines)

    def len(self, d):
        return len(d)


class KFIntegerType(KFType):
    @staticmethod
    def typeEnum(self):
        return IntegerType

    @staticmethod
    def regExForDataElement(self):
        return r'\s*[+-]?\d+\s*'

    @staticmethod
    def stringToDataConversionFunc(self):
        return int

    def stringForData(self, data):
        return self.formatData(data, 8, "{:10d}")


class KFRealType(KFType):
    @staticmethod
    def typeEnum(self):
        return RealType

    @staticmethod
    def regExForDataElement(self):
        return r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?|[+-]?Infinity|[+-]?NaN'

    @staticmethod
    def stringToDataConversionFunc(self):
        return float

    def stringForData(self, data):
        return self.formatData(data, 3, "{:28.16e}")


class KFCharacterType(KFType):
    @staticmethod
    def typeEnum(self):
        return CharacterType

    @staticmethod
    def regExForDataElement(self):
        return r'\n?.'

    @staticmethod
    def stringToDataConversionFunc(self):
        return None

    def len(self, d):
        return 160 * len(d)

    def stringForData(self, data):
        s = ""
        for sstr in data:
            longstr = sstr.ljust(160)
            s1 = longstr[0:79]
            s2 = longstr[80:159]
            s = s + s1 + "\n" + s2 + "\n"
        return s[:-1]


class KFLogicalType(KFType):

    def stringForData(self, data):
        count = 0
        s = ""
        for ll in data:
            if count == 80:
                s = s + "\n"
                count = 0
            count = count + 1
            if ll:
                s = s + "T"
            else:
                s = s + "F"
        return s

    @staticmethod
    def typeEnum(self):
        return KFLogicalType

    @staticmethod
    def regExForDataElement(self):
        return r'[TF]'

    @staticmethod
    def stringToDataConversionFunc(self):
        return self.stringToLogical

    @staticmethod
    def stringToLogical(sstr):
        return sstr == "T"


def KFTypeForEnum(enum):
    """Factory for creating KF Type instances"""
    if enum == 1:
        t = KFIntegerType()
    elif enum == 2:
        t = KFRealType()
    elif enum == 3:
        t = KFCharacterType()
    elif enum == 4:
        t = KFLogicalType()
    else:
        raise PyADFException('Invalid type in KFTypeForEnum')
    return t


# ----------------------
# KF file wrapper class
# ----------------------
class kffile:
    """
    A class wrapper for an ADF KF file. Allows reading from and writing
    to binary KF files from python. Makes use of the ADF utilities
    dmpkf, udmpkf, and cpkf.
    """

    env = None

    def __init__(self, fileName, buffered=True):
        import os
        self._fileName = fileName
        if self.env is not None:
            self._kfpath = self.env.setdefault('AMSBIN', '')
        else:
            self._kfpath = os.environ.setdefault('AMSBIN', '')
        self._contentsdict = None

        self._changed = False

        self._buffered = buffered
        self._write_buffer = None
        self._write_buffer_vars = []

        self._kftools_file = _kftools.KFFile(fileName)

    def close(self):
        if self._buffered:
            self._write_from_buffer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def _readcontents(self):
        """Read the table of contents using dmpkf."""
        import os
        import subprocess
        curdir = os.getcwd()
        try:
            newdir = os.path.dirname(self._fileName)  # to shorten the filename. dmpkf does not like names too long
            if newdir != '':
                os.chdir(newdir)
            dumpCmd = os.path.join(self._kfpath, 'dmpkf')
            dumpCmd = [dumpCmd, os.path.basename(self._fileName), '--xmltoc']

            DEVNULL = open(os.devnull, 'wb')
            s = subprocess.Popen(dumpCmd, stdout=subprocess.PIPE, stderr=DEVNULL,
                                 env=self.env).communicate()[0]
        except Exception:
            os.chdir(curdir)
            raise
        os.chdir(curdir)

        from xml.dom.minidom import parseString
        dom = parseString(s)
        contentdict = {}
        for section in dom.getElementsByTagName('section'):
            secdict = {}
            contentdict[section.getAttribute('id')] = secdict
            for variable in section.getElementsByTagName('variable'):
                varname = variable.getAttribute('id')
                vartype = int(variable.getAttribute('type'))
                varlength = int(variable.getAttribute('length'))
                secdict[varname] = {'type': vartype, 'length': varlength}

        return contentdict

    def contents(self):
        """Returns data structure describing sections and variables in file."""
        if not self._contentsdict:
            self._contentsdict = self._readcontents()
        return self._contentsdict

    def sections(self):
        """Returns array of strings containing all section names."""
        if not self._contentsdict:
            self._contentsdict = self._readcontents()
        sortedKeys = list(self._contentsdict.keys())
        sortedKeys.sort()
        return sortedKeys

    def sectionvars(self, section):
        """
        Returns a dictionary containing dictionaries for each variable
        in a given section. The keys of the enclosing dictionary are the variable names.
        The dictionaries contained therein each correspond to a variable and have the
        keys 'type' and 'length'.
        """
        if not self._contentsdict:
            self._contentsdict = self._readcontents()
        return self._contentsdict[section]

    def _write(self, section, variable, data, dataEnum):
        """
        Sets the data for a particular variable in the kf file.
        """

        d = data
        if isinstance(d, str):
            d = [d]
        if isinstance(d, np.ndarray):
            d = d.flat

        # test it d is iterable; if not, make list out of it
        try:
            _ = iter(d)
        except TypeError:
            d = [d]

        typ = KFTypeForEnum(dataEnum)
        ll = typ.len(d)

        varString = f'{section}\n{variable}\n{ll:10d}{ll:10d}{dataEnum:10d}\n{typ.stringForData(d)}\n'

        if self._buffered:
            self._write_buffer_vars.append((section, variable))
            if self._write_buffer is None:
                self._write_buffer = varString
            else:
                self._write_buffer = self._write_buffer + varString
        else:
            self._storeString(varString, [(section, variable)])

        self._changed = True

    def _write_from_buffer(self):
        if self._write_buffer is not None:
            self._storeString(self._write_buffer, self._write_buffer_vars)

            self._write_buffer = None
            self._write_buffer_vars = []

            self._kftools_file.reinit_reader()

    def writereals(self, sec, var, data):
        self._write(sec, var, data, RealType)

    def writeints(self, sec, var, data):
        self._write(sec, var, data, IntegerType)

    def writelogicals(self, sec, var, data):
        self._write(sec, var, data, LogicalType)

    def writechars(self, sec, var, data):
        self._write(sec, var, data, CharacterType)

    def read(self, section, variable):
        if self._buffered and self._changed:
            self._write_from_buffer()
        if self._changed:
            self._kftools_file.reinit_reader()
            self._changed = False
        return self._kftools_file.read(section, variable, return_as_list=True)

    def _storeString(self, sstr, sec_vars):
        """
        Copies the string passed, into the binary kf file.
        Assumes udmpkf can parse the string.
        """
        import os
        import subprocess
        import tempfile

        # Undump string data with udmpkf
        with tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False) as tf:
            path = tf.name
            tf.file.close()
            os.remove(path)

        udumpCmd = os.path.join(self._kfpath, 'udmpkf')

        # The following code is much nicer, but adds an approximately
        # 20-fold overhead due to the communication of the string to
        # the subprocess.
        #
        # DEVNULL = open(os.devnull, 'wb')
        # subprocess.Popen([udumpCmd, path], stdin=subprocess.PIPE, stderr=DEVNULL,
        #                  env=self.env).communicate(input=sstr)

        # This is an ugly, but much faster workaround:
        # We write the string to a temporary file, and then use a shell pipe

        with tempfile.NamedTemporaryFile() as tf_sstr:
            path_sstr = tf_sstr.name
            tf_sstr.write(sstr.encode())
            tf_sstr.file.close()

            subprocess.call(f'{udumpCmd} {path} < {path_sstr}', shell=True, env=self.env)

        # Work around start script bug: __0 files only renamed in current directory
        if os.path.isfile(path + '__0'):
            os.rename(path + '__0', path)

        # Use cpkf to merge the two binary files
        copyCmd = os.path.join(self._kfpath, 'cpkf')
        copyCmd = [copyCmd, path, self._fileName]

        for sec, var in sec_vars:
            copyCmd.append(sec + '%' + var)

        DEVNULL = open(os.devnull, 'wb')
        subprocess.Popen(copyCmd, stderr=DEVNULL, env=self.env).wait()

        # Close temporary file
        os.remove(path)


def setup_kf_environment(env):
    kffile.env = env
