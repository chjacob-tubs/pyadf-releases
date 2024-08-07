"""
This file is based on kftools.py from the PLAMS package, see https://github.com/SCM-NV/PLAMS,
with modifications by Christoph Jacob
Copyright (C) Scientific Computing and Modelling NV.

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


import os
import shutil
import struct

from bisect import bisect
from collections import OrderedDict

import numpy as np
import subprocess

# from ..core.functions import log

__all__ = ['KFFile', 'KFReader']


class KFFileError(Exception):
    """File or filesystem related error."""
    pass


class KFReader:
    """A class for efficient Python-native reader of binary files in KF format.

    This class offers read-only access to any fragment of data from a KF file. Unlike other
    Python KF readers, this one does not use the Fortran binary ``dmpkf`` to process KF files,
    but instead reads and interprets raw binary data straight from the file, on Python level.
    That approach results in significant speedup (by a factor of few hundreds for large files
    extracted variable by variable).

    The constructor argument *path* should be a string with a path (relative or absolute) to
    an existing KF file.

    *blocksize* indicates the length of basic KF file block. So far, all KF files produced by
    any of Amsterdam Modeling Suite programs have the same block size of 4096 bytes. Unless
    you're doing something *very* special, you should not touch this value.

    Organization of data inside KF file can depend on a machine on which this file was produced.
    Two parameters can vary: the length of integer (32 or 64 bit) and endian (little or big).
    These parameters have to be determined before any reading can take place, otherwise the
    results will have no sense. If the constructor argument *autodetect* is ``True``, the constructor
    attempts to automatically detect the format of a given KF file, allowing to read files created
    on a machine with different endian or integer length. This automatic detection is enabled by
    default and it is advised to leave it that way. If you wish to disable it, you should set
    ``endian`` and ``word`` attributes manually before reading anything (see the code for details).

    .. note ::

        This class consists of quite technical, low level code. If you don't need to modify or
        extend |KFReader|, you can safely ignore all private methods, all you need
        is :meth:`~KFReader.read` and occasionally :meth:`~KFReader.__iter__`

    """

    _sizes = {'s': 1, 'i': 4, 'd': 8, 'q': 8}

    def __init__(self, path, blocksize=4096, autodetect=True):
        if os.path.isfile(path):
            self.path = os.path.abspath(path)
        else:
            raise KFFileError(f'File {path} not present')

        self._blocksize = blocksize
        self.endian = '<'  # endian: '<' = little, '>' = big
        self.word = 'i'  # length of int: 'i' = 4 bytes, 'q' = 8 bytes

        self._data = {}
        self.sections = None

        if autodetect:
            self._autodetect()

    def read(self, section, variable):
        """Extract and return data for a *variable* located in a *section*.

        For single-value numerical or boolean variables returned value is a single number or bool.
        For longer variables this method returns a list of values.
        """

        if self.sections is None:
            self.create_index()

        try:
            tmp = self.sections[section]
        except KeyError:
            raise KeyError(f'Section {section} not present in {self.path}')
        try:
            vtype, vlb, vstart, vlen = tmp[variable]
        except KeyError:
            raise KeyError(f'Variable {variable} not present in section {section} of {self.path}')

        if vtype == 3:
            # string data
            ret = []
            first = True
            with open(self.path, 'rb') as f:
                for i in KFReader._datablocks(self._data[section], vlb):
                    if first:
                        ret = self._get_data(self._read_block(f, i), vtype)[vstart - 1:]
                        first = False
                    else:
                        ret += self._get_data(self._read_block(f, i), vtype)

            ret = ret[:vlen]
            try:
                ret = ret.decode()
            except UnicodeDecodeError:
                ret = ret.decode("Latin-1")

            s = []
            mystr = ret
            for n in range(len(ret) // 160):
                s.append(mystr[0:159].rstrip())
                mystr = mystr[160:]
            if len(s) == 1:
                s = s[0]
            return s

        else:
            if (vtype == 1) and (self.word == 'i'):
                ret = np.empty(vlen, dtype=np.int32)
            elif (vtype == 1) and (self.word == 'q'):
                ret = np.empty(vlen, dtype=np.int64)
            elif vtype == 2:
                ret = np.empty(vlen, dtype=np.float64)
            elif vtype == 4:
                ret = np.empty(vlen, dtype=bool)
            else:
                raise KeyError('Unknown vtype')

            first = True
            ii = 0
            with open(self.path, 'rb') as f:
                for i in KFReader._datablocks(self._data[section], vlb):
                    if first:
                        data = self._get_data(self._read_block(f, i), vtype)[vstart - 1:vstart - 1 + vlen - ii]
                        first = False
                    else:
                        data = self._get_data(self._read_block(f, i), vtype)
                        data = data[:vlen - ii]
                    ret[ii:ii + len(data)] = data
                    ii = ii + len(data)

        return ret

    def __iter__(self):
        """Iteration yields pairs of section name and variable name."""
        if self.sections is None:
            self.create_index()
        for section in self.sections:
            for variable in self.sections[section]:
                yield section, variable

    def _autodetect(self):
        """Try to automatically detect the format (int size and endian) of this KF file."""
        with open(self.path, 'rb') as f:
            b = f.read(128)

        blocksize = struct.unpack(b'i', b[28:32])[0]
        self._blocksize = 4096 if blocksize == 538976288 else blocksize
        # log('Block size of {} detected as {}'.format(self.path, self._blocksize), 7)

        one = b[80:84]

        if struct.unpack(b'32s', b[48:80])[0] == b'SUPERINDEX                      ':
            self.word = 'i'
        elif struct.unpack(b'32s', b[64:96])[0] == b'SUPERINDEX                      ':
            self.word = 'q'
            one = b[96:104]
        else:
            # log('WARNING: Unable to autodetect integer size and endian of {}. Using'
            #     'defaults (4 bytes and little endian)'.format(self.path), 3)
            return

        for e in ['<', '>']:
            if struct.unpack(str(e + self.word), one)[0] == 1:
                self.endian = e
                # d = {'q': '8 bytes', 'i': '4 bytes', '<': 'little endian', '>': 'big endian', }
                # log(('Format of {0} detected to {'+self.word+'} and {'+self.endian+'}').format(self.path, **d), 7)

    def _read_block(self, f, pos):
        """Read a single block of binary data from posistion *pos* in file *f*."""
        f.seek((pos - 1) * self._blocksize)
        return f.read(self._blocksize)

    # noinspection PyShadowingBuiltins
    def _parse(self, block, format):  # format = [(32,'s'),(4,'i'),(2,'d')]
        """Translate a *block* of binary data into list of values in specified *format*.

        *format* should be a list of pairs *(a,t)* where *t* is one of the following
        characters: ``'s'`` for string (bytes), ``'i'`` for 32-bit integer, ``'q'`` for
        64-bit integer and *a* is the number of occurrences (or length of a string).

        For example, if *format* is equal to ``[(32,'s'),(4,'i'),(2,'d'),(2,'i')]``, the
        contents of *block* are divided into 72 bytes (32*1 + 4*4 + 2*8 + 2*4 = 72) chunks
        (possibly droping the last one, if it's shorter than 72 bytes). Then each chunk is
        translated to a 9-tuple of bytes, 4 ints, 2 floats and 2 ints. List of such tuples
        is the returned value.
        """
        step = 0
        formatstring = self.endian
        for a, t in format:
            step += a * self._sizes[t]
            formatstring += str(a) + t

        if step > 0:
            # end = (len(block) // step) * step
            # return list(struct.iter_unpack(formatstring, block[:end]))

            res = []
            for i in range(len(block) // step):
                res.append(struct.unpack_from(formatstring, block, offset=i * step))
            return res
        else:
            return []

    # noinspection PyShadowingBuiltins
    def _parse_one_chunk(self, block, format):  # format = [(32,'s'),(4,'i'),(2,'d')]
        step = 0
        formatstring = self.endian
        for a, t in format:
            step += a * self._sizes[t]
            formatstring += str(a) + t

        # end = (len(block) // step) * step
        # return list(struct.iter_unpack(formatstring, block[:end]))

        res = struct.unpack_from(formatstring, block)
        return res

    def _get_data(self, datablock, vtype):
        """
        Extract all data of a given type from a single data block.

        Returned value is a list of values (int, float, or bool) or a single "bytes" object.
        """
        hlen = 4 * self._sizes[self.word]
        i, d, s, b = self._parse_one_chunk(datablock[:hlen], [(4, self.word)])
        contents = self._parse_one_chunk(datablock[hlen:], list(zip((i, d, s, b), (self.word, 'd', 's', self.word))))
        if vtype == 1:
            return contents[:i]
        elif vtype == 2:
            return contents[i:i + d]
        elif vtype == 3:
            return contents[i + d]
        elif vtype == 4:
            return list(map(bool, contents[i + d + 1:]))
        else:
            raise KeyError('Unknown vtype')

    def create_index(self):
        """
        Find and parse relevant index blocks of KFFile to extract the information
        about location of all sections and variables.

        Two dictionaries are populated during this process. ``_data`` contains, for each section,
        a list of triples describing how logical blocks of data are mapped into physical ones. For
        example, ``_data['General'] = [(3,6,12), (9,40,45)]`` means that logical blocks 3-8 of section
        ``General`` are located in physical blocks 6-11 and logical blocks 9-13 in physical blocks 40-44.
        This list is always sorted via first tuple elements allowing efficient access to arbitrary
        logical block of each section.

        The second dictionary, ``_sections``, is used to locate each variable within its section.
        For each section, it contains another dictionary of each variable of this section. So
        ``_section[sec][var]`` contains all information needed to extract variable ``var`` from
        section ``sec``. This is a 4-tuple containing the following information: variable type,
        logic block in which the variable first occurs, position within this block where its data
        start and the length of the variable. Combining this information with mapping stored in
        ``_data`` allows to extract each single variable.
        """

        hlen = 32 + 7 * self._sizes[self.word]  # length of index block header

        with open(self.path, 'rb') as f:
            superlist = self._parse(self._read_block(f, 1), [(32, 's'), (4, self.word)])
            nextsuper = superlist[0][4]
            while nextsuper != 1:
                nsl = self._parse(self._read_block(f, nextsuper), [(32, 's'), (4, self.word)])
                nextsuper = nsl[0][4]
                superlist += nsl

            self._data = {}  # list of triples to convert logical to physical block numbers
            self.sections = {}
            # pb=physical block, lb=logical block, le=length, ty=type (3 for index, 4 for data)
            for key, pb, lb, le, ty in superlist:
                try:
                    key = key.decode()
                except UnicodeDecodeError:
                    key = key.decode("Latin-1")
                key = key.rstrip(' ')
                if key in ['SUPERINDEX', 'EMPTY']:
                    continue
                if ty == 4:  # data block
                    if key not in self._data:
                        self._data[key] = []
                    # noinspection PyTypeChecker
                    self._data[key].append((lb, pb, pb + le))
                elif ty == 3:  # index block
                    if key not in self.sections:
                        self.sections[key] = {}
                    for i in range(le):
                        indexblock = self._read_block(f, pb + i)
                        body = self._parse(indexblock[hlen:], [(32, 's'), (6, self.word)])
                        for var, vlb, vstart, vlen, _xx1, vused, vtype in body:
                            try:
                                var = var.decode()
                            except UnicodeDecodeError:
                                var = var.decode("Latin-1")
                            var = var.rstrip(' ')
                            if var == 'EMPTY':
                                continue
                            self.sections[key][var] = (vtype, vlb, vstart, vused)

            for k, v in list(self._data.items()):
                lbs = []
                pbs = []
                for lb, first, last in sorted(v):
                    lbs.append(lb)
                    pbs.append((first, last))
                self._data[k] = (lbs, pbs)

    @staticmethod
    def _datablocks(lst, n=1):
        """Transform a tuple of lists ``([x1,x2,...], [(a1,b1),(a2,b2),...])`` into an iterator
        over ``range(a1,b1)+range(a2,b2)+...`` Iteration starts from nth element of this list."""
        i = bisect(lst[0], n) - 1
        lb = lst[0][i]
        first, last = lst[1][i]
        ret = first + n - lb
        while i < len(lst[1]):
            while ret < last:
                yield ret
                ret += 1
            i += 1
            if i < len(lst[1]):
                ret, last = lst[1][i]


class KFFile:
    """A class for reading and writing binary files in KF format.

    This class acts as a wrapper around |KFReader| collecting all the data written by
    user in some "temporary zone" and using Fortran binaries ``udmpkf`` and ``cpkf`` to
    write this data to the physical file when needed.

    The constructor argument *path* should be a string with a path to an existing KF file
    or a new KF file that you wish to create. If a path to existing file is passed, new
    |KFReader| instance is created allowing to read all the data from this file.

    When :meth:`~KFFile.write` method is used, the new data is not immediately written to
    a disk. Instead of that, it is temporarily stored in ``tmpdata`` dictionary. When
    method :meth:`~KFFile.save` is invoked, contents of that dictionary are written to
    a physical file and ``tmpdata`` is emptied.

    Other methods like :meth:`~KFFile.read` or :meth:`~KFFile.delete_section` are aware
    of ``tmpdata`` and work flawlessly, regardless if :meth:`~KFFile.save` was called or
    not.

    By default, :meth:`~KFFile.save` is automatically invoked after each :meth:`~KFFile.write`,
    so physical file on a disk is always "actual". This behavior can be adjusted with *autosave*
    constructor parameter. Having autosave enabled is usually a good idea, however, if you need
    to write a lot of small pieces of data to your file, the overhead of calling ``udmpkf`` and
    ``cpkf`` after *every* :meth:`~KFFile.write` can lead to significant delays. In such a case
    it is advised to disable autosave and call :meth:`~KFFile.save` manually, when needed.

    Dictionary-like bracket notation can be used as a shortcut to read and write variables::

        mykf = KFFile('someexistingkffile.kf')
        #all three below are equivalent
        x = mykf['General%Termination Status']
        x = mykf[('General','Termination Status')]
        x = mykf.read('General','Termination Status')

        #all three below are equivalent
        mykf['Geometry%xyz'] = somevariable
        mykf[('Geometry','xyz')] = somevariable
        mykf.write('Geometry','xyz', somevariable)

    """
    _types = {int: (1, 8, lambda x: '{:10d}'.format(x)),
              float: (2, 3, lambda x: '{:26.16e}'.format(x)),
              str: (3, 80, lambda x: x),
              bool: (4, 80, lambda x: 'T' if x else 'F')}

    def __init__(self, path, autosave=True):
        self.autosave = autosave
        self.path = os.path.abspath(path)
        self.tmpdata = OrderedDict()
        self.reader = KFReader(self.path) if os.path.isfile(self.path) else None

    def reinit_reader(self):
        self.reader = KFReader(self.path) if os.path.isfile(self.path) else None

    def read(self, section, variable, return_as_list=False):
        """
        Extract and return data for a *variable* located in a *section*.

        By default, for single-value numerical or boolean variables returned value is
        a single number or bool. For longer variables this method returns a list of values.
        For string variables a single string is returned. This behavior can be changed by
        setting *return_as_list* parameter to ``True``. In that case the returned value is
        always a list of numbers (possibly of length 1) or a single string.
        """
        if section in self.tmpdata and variable in self.tmpdata[section]:
            ret = self.tmpdata[section][variable]
        else:
            ret = self.reader.read(section, variable)

        if return_as_list and not isinstance(ret, (list, np.ndarray)):
            ret = [ret]

        return ret

    def write(self, section, variable, value):
        """
        Write a *variable* with a *value* in a *section* . If such a variable already
        exists in this section, the old value is overwritten.
        """

        if not isinstance(value, (int, bool, float, str, list)):
            raise ValueError('Trying to store improper value in KFFile')
        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError('Cannot store empty lists in KFFile')
            if any(not isinstance(i, type(value[0])) for i in value):
                raise ValueError('Lists stored in KFFile must have all elements of the same type')
            if not isinstance(value[0], (int, bool, float, str)):
                raise ValueError('Only lists of int, float, str or bool can be stored in KFFile')

        if section not in self.tmpdata:
            self.tmpdata[section] = OrderedDict()
        self.tmpdata[section][variable] = value

        if self.autosave:
            self.save()

    def save(self):
        """Save all changes stored in ``tmpdata`` to physical file on a disk."""
        if len(self.tmpdata) > 0 and any(len(i) > 0 for i in list(self.tmpdata.values())):
            txt = ''
            newvars = []
            for section in self.tmpdata:
                for variable in self.tmpdata[section]:
                    val = self.tmpdata[section][variable]
                    txt += f'{section}\n{variable}\n{KFFile._str(val)}\n'
                    newvars.append(section + '%' + variable)
            self.tmpdata = OrderedDict()

            tmpfile = self.path + '.tmp' if self.reader else self.path
            FNULL = open(os.devnull, 'wb')
            subprocess.run(['udmpkf', tmpfile], input=txt.encode(), stdout=FNULL, stderr=FNULL)
            if self.reader:
                subprocess.run(['cpkf', tmpfile, self.path] + newvars, stdout=FNULL, stderr=FNULL)
                os.remove(tmpfile)
            self.reader = KFReader(self.path)

    def delete_section(self, section):
        """Delete the entire *section* from this KF file."""
        if section in self.tmpdata:
            del self.tmpdata[section]
        if self.reader:
            if not self.reader.sections:
                self.reader.create_index()
            if section in self.reader.sections:
                tmpfile = self.path + '.tmp'
                FNULL = open(os.devnull, 'wb')
                subprocess.run(['cpkf', self.path, tmpfile, '-rm', section], stdout=FNULL, stderr=FNULL)
                shutil.move(tmpfile, self.path)
                self.reader = KFReader(self.path)

    def sections(self):
        """Return a list with all section names, ordered alphabetically."""
        ret = set(self.tmpdata)
        if self.reader:
            if self.reader.sections is None:
                self.reader.create_index()
            ret |= set(self.reader.sections)
        ret = list(ret)
        ret.sort()
        return ret

    def read_section(self, section):
        """
        Return a dictionary with all variables from a given *section*.

        .. note::

            Some sections can contain very large amount of data. Turning them into dictionaries
            can cause memory shortage or performance issues. Use this method carefully.

        """
        ret = {}
        for sec, var in self:
            if sec == section:
                ret[var] = self.read(sec, var)
        # if len(ret) == 0:
        #    log("WARNING: Section '{}' not present in {} or present, but empty. "+
        #    "Returning empty dictionary".format(section, self.path), 1)
        return ret

    def get_skeleton(self):
        """
        Return a dictionary reflecting the structure of this KF file. Each key in that dictionary
        corresponds to a section name of the KF file with the value being a set of variable names.
        """
        ret = {}
        for sec, var in self:
            if sec not in ret:
                ret[sec] = set()
            ret[sec].add(var)
        return ret

    def __getitem__(self, name):
        """
        Allow to use ``x = mykf['section%variable']`` or ``x = mykf[('section','variable')]``
        instead of ``x = kf.read('section', 'variable')``.
        """
        section, variable = KFFile._split(name)
        return self.read(section, variable)

    def __setitem__(self, name, value):
        """
        Allow to use ``mykf['section%variable'] = value`` or ``mykf[('section','variable')] = value``
         instead of ``kf.write('section', 'variable', value)``.
        """
        section, variable = KFFile._split(name)
        self.write(section, variable, value)

    def __iter__(self):
        """Iteration yields pairs of section name and variable name."""
        ret = set()
        if self.reader:
            for sec, var in self.reader:
                ret.add((sec, var))
        for sec in self.tmpdata:
            for var in self.tmpdata[sec]:
                ret.add((sec, var))
        ret = list(ret)
        ret.sort(key=lambda x: x[0] + x[1])
        yield from ret

    def __contains__(self, arg):
        """
        Implements Python ``in`` operator for KFFiles. *arg* can be a single string with a section
        name or a pair of strings (section, variable).
        """
        if isinstance(arg, str):
            return arg in self.sections()
        if isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], str) \
                and isinstance(arg[1], str):
            try:
                self.read(*arg)
                return True
            except KeyError:
                return False
        raise TypeError("'in <KFFile>' requires string of a pair of strings as left operand")

    @staticmethod
    def _split(name):
        """
        Ensure that a key used in bracket notation is of the form ``'section%variable'``
        or ``('section','variable')``. If so, return a tuple ``('section','variable')``.
        """
        if isinstance(name, tuple) and len(name) == 2:
            return name[0], name[1]
        if isinstance(name, str):
            s = name.split('%')
            if len(s) == 2:
                return s[0], s[1]
        raise ValueError('Improper key used in KFFile dictionary-like notation')

    @staticmethod
    def _str(val):
        """Return a string representation of *val* in the form that can be understood by ``udmpkf``."""
        if isinstance(val, (int, bool, float)):
            val = [val]
        valtype = type(val[0])
        t, step, f = KFFile._types[valtype]
        ll = len(val)
        if (valtype == str or valtype == str) and isinstance(val, list):
            # udmpkf reads 160 characters per variable, split over max. 80 per line, to make a string array
            ll = ll * 160
            step = 1
            splitstrings = [[s[0:80], s[80:160]] for s in val]
            val = [item for sublist in splitstrings for item in sublist]

        ret = f'{ll:10d}{ll:10d}{t:10d}'
        for i, el in enumerate(val):
            if i % step == 0:
                ret += '\n'
            ret += f(el)
        return ret
