# This file is part of 
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2014 by Christoph R. Jacob, S. Maya Beyhan,
# Rosa E. Bulo, Andre S. P. Gomes, Andreas Goetz, Michal Handzlik,
# Karin Kiewisch, Moritz Klammler, Jetze Sikkema, and Lucas Visscher 
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
 Defines plotting related classes.

 The central class for plotting is L{densfresults}. Instances of
 L{densfresults} represent densities (or related quantities like
 the Laplacian of the density on a grid. The easiest way to
 obtain such a density is to use the L{adfsinglepointresults.get_density}
 method.
 
 >>> res = adfsinglepointjob(mol, basis='DZP').run()
 >>> dens = res.get_density()
 >>> dens.get_tape41('density.t41')    # save the density to a TAPE41 file
 >>> dens.get_cubfile('density.cub')  # save the density to a Gaussian-type cube file

 The C{get_density} method takes an optional argument C{grid}, which
 specifies which type of grid is used. This can be instances of either
 L{cubegrid}, for an evenly spaced cubic grid, and L{adfgrid}, for the
 integration grid used by ADF (imported from a TAPE10 file). See the
 documentation for L{cubegrid} and L{adfgrid}.

 Densities can be added and substracted. 
 
     - Obtain the sum of the densities of fragments 1 and 2:
     
     >>> grid = cubegrid(mol1 + mol2) # create a common grid for both fragments
     >>>
     >>> # density of fragment 1
     >>> res_frag1 = frag1_job.run()
     >>> dens1 = res_frag1.get_density(grid=grid)
     >>>
     >>> # density of fragment 2
     >>> res_frag2 = frag2_job.run()
     >>> dens2 = res_frag2.get_density(grid=grid)
     >>>
     >>> # total density
     >>> tot_dens = dens1 + dens2
     >>> tot_dens.get_cubfile ('totdens.cub')

     - Obtain the difference density (e.g., between supermolecule and FDE):
     
     >>> res_supermol = supermol_job.run()
     >>> res_fde = fde_job.run()
     >>>
     >>> grid = adfgrid(res_supermol)
     >>> 
     >>> dens_supermol = res_supermol.get_density(grid=grid)     # supermolecular density
     >>> dens_fde      = res_fde.get_density(grid=grid)          # total FDE density
     >>>
     >>> diffdens = dens_supermol - dens_fde
     >>> diffdens.get_cubfile('diffdens.cub')

 Finally, it is also possible to integrate densities and functions
 of the density. Not that this should be done using L{adfgrid} in
 order to obtain accurate results.
 
     - Number of electrons:
     
     >>> print dens.integral()
 
     - Integrated absolute and RMS error in the density:
 
     >>> print "integrated absolute error: ", diffdens.integral (lambda x: abs(x))
     >>> print "integrated RMS error: ", math.sqrt(diffdens.integal (lambda x: x*x))

 More examples on the use of the routines in the L{Plot} module can be
 found in the tests C{ADFPlot} and C{ADF3FDE_Dialanine}.

 @author:       Christoph Jacob and others
 @organization: Karlsruhe Institute of Technology (KIT)
 @contact:      christoph.jacob@kit.edu

 @group Density Results:
     densfresults
 @group Density Generation:
     densfjob
"""

from ADFBase import adfjob, adfresults
from PlotGrids import grid, cubegrid, interpolation
from Errors  import PyAdfError
from Utils   import Bohr_in_Angstrom
from itertools import izip


class densfresults (adfresults):
    """
    Class representing densities (and related quantities) as produced 
    by the ADF utility program C{densf} (or its replacement C{cjdensf}).

    Instances of C{densfresults} are returned by the run method of
    L{densfjob}. Alternatively, the methods L{adfsinglepointresults.get_density}
    and L{adfsinglepointresults.get_laplacian} can be used to obtain the
    electron density or the Laplacian of the electron density directly from
    an L{adfsinglepointresults} object.
    The documentation of the respective methods provides more detailed
    information.

    Even though an instance of C{densfresults} can contain different 
    quantities given on a grid, in the following it will mostly be
    refered to as "density".
    
    The most common tasks that can be done with C{densfresults} are:
      - obtain the numerical values using L{get_values}
      - get a TAPE41 file of the density with L{get_tape41}
      - get a Gaussian-type cube file of the density with L{get_cubfile}
      - adding and substracting of different densities (see L{__add__} and L{__sub__})
      - integration over (a function of) the density using L{integral}
      
    @group Access to density:
        get_values, valueiter, get_tape41, get_cubfile
    @group Manipulation of densities:
        __add__, __sub__
    @group Integration over density:
        integral
    @group Retrieval of properties derived from density/potential:
        get_efield_in_point_grid, get_electronic_dipole_moment_grid
    @group Analysis of densities per Voronoi cell:
        integral_voronoi, get_electronic_dipole_voronoi
    @undocumented:
        _add_with_factor, _delete_values, _read_values_from_tape41, _write_tape41
    """

    def __init__ (self, j=None, grid=None) :
        """
        Constructor for densfresults.
        """
        adfresults.__init__(self, j)
        self.nspin = 1

        if j is not None :
            if grid is not None:
                raise PyAdfError('grid must not be passed if adfres is present')
            self.grid = self.job.grid
            self.prop = self.job.prop
            self._values = None
        elif grid is not None:
            import numpy
            self.grid = grid
            self.prop = None
            self._values = numpy.zeros((self.grid.get_number_of_points(),))
        else :
            self.job   = None
            self.files = None
            self.grid  = None
            self.prop  = None
            self._values = None

        self._can_delete_values = False

    def get_values (self):
        """
        Returns the values of the density on the grid.
        
        Depending on the type of the grid, the values
        are given in a one-dimensional or tree-dimensional
        array.
        
        @rtype: array of float
        """
        if self._values == None:
            self._can_delete_values = True
            self._read_values_from_tape41()
        return self._values.view()

    def _delete_values (self) :
        if self._can_delete_values :
            del self._values
            self._values = None

            # run garbage collection in order to release memory
            import gc
            gc.collect()

    def _get_tape41_section_variable (self):

        prop = self.prop.lower()

        if prop.endswith('alpha') :
            prop = prop[:-5].strip()
        elif prop.endswith('beta') :
            prop = prop[:-4].strip()

        if prop == 'density scf' :
            section  = 'SCF'
            variable = 'Density'
        elif prop == 'density fit scf' :
            section  = 'SCF'
            variable = 'Fitdensity'
        elif prop == 'gradient' :
            section  = 'SCF'
            variable = 'SGradient'
        elif prop == 'laplacian' :
            section  = 'SCF'
            variable = 'DensityLap'
        elif prop == 'laplacian fit' :
            section  = 'SCF'
            variable = 'FitdensityLap'
        elif prop == 'potential total' :
            section  = 'Potential'
            variable = 'Total'
        elif prop == 'potential nuc' :
            section  = 'Potential'
            variable = 'Nuclear'
        elif prop == 'potential coul' :
            section  = 'Potential'
            variable = 'Coulomb'
        elif prop == 'potential xc' :
            section  = 'Potential'
            variable = 'XC'
        elif prop == 'potential reconstructed' :
            section  = 'Potential'
            variable = 'Reconstructed'
        elif prop.startswith('kinpot') :
            section  = 'Potential'
            variable = 'Kinetic'
        elif prop.startswith('orbital') :
            split = prop.split()
            if split[1] == "loc" :
                section = "LocOrb" 
            else:
                section = "SCF_%s" % split[1].upper()
            variable = split[2].upper()
        elif prop.startswith('orblapl') :
            split = prop.split()
            section = "OrbLapl SCF_%s" % split[1].upper().replace('LOC', 'A')
            variable = split[2].upper()
        elif prop.startswith('locorbkinpot') or prop.startswith('orbdenskinpot'):
            section  = 'Potential'
            variable = 'Kinetic'           
        elif prop.startswith('locorbxcpot') :
            section  = 'Potential'
            variable = 'XC'           
        elif prop.startswith('locorbcoulpot') :
            section  = 'Potential'
            variable = 'Coulomb'           
        elif prop.startswith('orbdenslapl') :
            section  = 'SCF'
            variable = 'DensityLap'
        elif prop.startswith('locorbgrad') :
            section  = 'SCF'
            variable = 'SGradient'
        elif prop.startswith('orbdensgrad'):
            if prop.startswith('orbdensgradx'):
                section  = 'SCF'
                variable = 'Gradient x'
            elif prop.startswith('orbdensgrady'):
                section  = 'SCF'
                variable = 'Gradient y'
            elif prop.startswith('orbdensgradz'):
                section  = 'SCF'
                variable = 'Gradient z'
            else:
                section  = 'SCF'
                variable = 'SGradient'
        elif prop.startswith('locorbdens') or prop.startswith('orbdens') :
            section  = 'SCF'
            variable = 'Density'
        elif prop.startswith('embpot') :
            section  = 'Potential'
            variable = 'EmbeddingPot'           
        elif prop.startswith('nadkin') :
            section  = 'Potential'
            variable = 'nadKinFrozen'           
        else :
            section  = None
            variable = None

        if self.prop.lower().endswith('alpha') :
            variable += "_A"       
        elif self.prop.lower().endswith('beta') :
            variable += "_B"       
 
        return section, variable    

    def _read_values_from_tape41 (self):

        section, variable = self._get_tape41_section_variable()
        if section is None :
            raise PyAdfError('Unknown property requested')

        self._values = self.get_result_from_tape(section, variable, tape=41)
        if len(self.grid.shape) == 1 :
            self._values = self._values[:self.grid.shape[0]]
        else :
            self._values = self._values.reshape(self.grid.shape, order='Fortran')

    def valueiter (self):
        """
        Returns an iterator over the values of the density in the grid points.
        
        The order of these values is the same as in the 
        grid point iterators of L{grid} and its subclasses,
        see L{grid.coorditer} and L{grid.weightiter}.

        @exampleuse:

            Print coordinates of all grid points where the density is larger than 1.0
            
            >>> # 'dens' is an instance of densfresults
            >>> 
            >>> for coord, val in zip(dens.grid.coorditer(), dens.valueiter()) :
            ...     if val > 1.0 :
            ...         print "Grid point: ", coord
        
        @rtype: iterator
        """
        return self.get_values().flat

    def get_tape41 (self, filename):
        """
        Obtain a copy of the TYPE41 file associated with the density.
        
        @param filename: The filename to which TAPE41 is copied.
        @type  filename: str
        """
        if self.fileid == None :
            self.fileid = self.files.get_id (self.get_checksum())
            if (self.fileid == None):
                self._write_tape41()
        self.copy_tape (tape=41, name=filename)

    def _write_tape41 (self):
        import os, kf

        cwd = os.getcwd()
        self.files.change_to_basedir()       
 
        f = kf.kffile('TAPE41')
        self.grid.write_grid_to_t41(f)

        section, variable = self._get_tape41_section_variable()
        f.writereals(section, variable, 
                     self._values.reshape((self._values.size,), order='Fortran'))
        f.close()
    
        self.files.add_results(self)
    
        os.chdir(cwd)

    def get_cubfile (self, filename, involume=None):
        """
        Obtain a Gaussian-type cube file of the associated density.
        
        @param filename: The filename of the cube file to be written.
        @type  filename: str

        @param involume: 
            a function of coordinates defining a box or volume, for which values are
            written to the cubfile. Returns a boolean. For example:
            >>> def withinradius(coord) :
            ...     geom_center = [2.0, 4.0, 3.5]
            ...     radius = coord-geom_center
            ...     rad_abs = math.sqrt(radius[0]**2 + radius[1]**2 + radius[2]**2)
            ...     coord_within_radius = rad_abs < 5.0
            ...     return coord_within_radius

        @type involume : function

        """
        if not isinstance(self.grid, cubegrid) :
            raise PyAdfError('cubfile can only be written if cubegrid has been used')
        
        values = self.get_values()
        
        f = open(filename, 'w')
        # first two lines are title
        f.write('Cube file generated by PyADF \n')
        f.write(str(self.prop)+'\n')
        # cube file grid specification
        f.write(self.grid.get_cube_header())

        if involume is not None :
            import numpy
            for ix in range(self.grid.shape[0]) :
                for iy in range(self.grid.shape[1]) :
                    for iz in range(self.grid.shape[2]) :
                        coord = numpy.array([self.grid._startpoint[0] + ix*self.grid._spacing,
                                             self.grid._startpoint[1] + iy*self.grid._spacing,
                                             self.grid._startpoint[2] + iz*self.grid._spacing])
                        if (involume(coord)) :
                            f.write('%14.6e ' % values[ix, iy, iz])
                        else :
                            f.write('%14.6e ' % 0.0)
                        if (iz % 6 == 5) :
                            f.write('\n')
                    f.write('\n')
        else :
            for ix in range(self.grid.shape[0]) :
                for iy in range(self.grid.shape[1]) :
                    for iz in range(self.grid.shape[2]) :
                        f.write('%14.6e ' % values[ix, iy, iz])
                        if (iz % 6 == 5) :
                            f.write('\n')
                    f.write('\n')
        f.close()

    def get_xsffile (self, filename):
        """
        Obtain a XSF file of the associated density.

        @attention: Density values in the file are in atomic units!
        
        @param filename: The filename of the xsf file to be written.
        @type  filename: str
        """
        if not isinstance(self.grid, cubegrid) :
            raise PyAdfError('XSF file can only be written if cubegrid has been used')
        
        f = open(filename, 'w')
        f.write(self.grid.get_xsf_header())

        # now write the values (in the correct order)
        values = self.get_values()
        for iz in range(self.grid.shape[2]) :
            for iy in range(self.grid.shape[1]) :
                for ix in range(self.grid.shape[0]) :                
                    f.write('%26.18e \n' % values[ix, iy, iz])

        f.write(self.grid.get_xsf_footer())

        f.close()
    
    def get_xyzwvfile (self, filename, bohr=True, endmarker=False, add_comment=True, empty=False) :
        """
        Obtain a xyzwv file of the associated density.
        
        File will be in the following format:
        
        On the first line is the number of grid points, 
        on the second line is a comment (title of the file).
        
        On each of the following lines, there are five reals numbers: X, Y, Z, W, V,
        where X,Y,Z are the coordinates of the grid point (in Angstrom), 
        W are the integration weights for the grid, and V is the value 
        (i.e., density) in this grid point.

        if endmarker is set to true, a negative integer serving as end-of-file 
        marker, is written in the last line
        
        @param filename: The filename of the file to be written.
        @type  filename: str
        """
        f = open(filename, 'w')
        f.write('%i \n' % self.grid.get_number_of_points() )

        if add_comment :
            if bohr :
                f.write('# XYZWV file generated by PyADF (Coordinates in Bohr)\n')
            else:
                f.write('# XYZWV file generated by PyADF (Coordinates in Angstrom)\n')

        if not empty :
            for c, w, v in zip(self.grid.coorditer(bohr=bohr), self.grid.weightiter(), self.valueiter()) :
                f.write("%26.18e  %26.18e  %26.18e  %26.18e     %26.18e \n" % (c[0], c[1], c[2], w, v))
        else :
            v = 0.0
            for c, w in zip(self.grid.coorditer(bohr=bohr), self.grid.weightiter()) :
                f.write("%26.18e  %26.18e  %26.18e  %26.18e     %26.18e \n" % (c[0], c[1], c[2], w, v))
 
        if endmarker :
            f.write('  -42\n') 

        f.close()
    
    def get_xyzvfile (self, filename, bohr=False):
        """
        Obtain a xyzv file of the associated density.
        
        File will be in the following format:
        
        On the first line is the number of grid points, 
        on the second line is a comment (title of the file).
        
        On each of the following lines, there are four reals numbers: X, Y, Z, V,
        where X,Y,Z are the coordinates of the grid point (in Angstrom) 
        and V is the value (i.e., density) in this grid point.
        
        @param filename: The filename of the file to be written.
        @type  filename: str
        """
        f = open(filename, 'w')
        f.write('%i \n' % self.grid.get_number_of_points() )

        if bohr :
            f.write('XYZV file generated by PyADF (Coordinates in Bohr)\n')
        else:
            f.write('XYZV file generated by PyADF (Coordinates in Angstrom)\n')

        for c, v in zip(self.grid.coorditer(bohr=bohr), self.valueiter()) :
            f.write("%16.8e  %16.8e  %16.8e     %16.8e \n" % (c[0], c[1], c[2], v))
        f.close()
        
    def _add_with_factor (self, other, fact=1.0) :

        # first check that the grid is the same
        if not (self.grid is other.grid) :
            raise PyAdfError('grids have to be the same for addition of densities')
        
        # copy self
        import copy
        summ = copy.copy(self)
        summ.job    = None
        summ.fileid = None
        summ._can_delete_values = False

        # calculate checksum for added density
        import hashlib
        m = hashlib.md5()
        m.update("Density obtained by adding :\n" )
        m.update(self.get_checksum())
        m.update("with factor %18.10f times \n"%fact)
        m.update(other.get_checksum())
        summ._checksum = m.digest()
        
        # and finally add
        #pylint: disable-msg=W0212
        summ._values = self.get_values() + fact*other.get_values()

        self._delete_values()
        other._delete_values()
    
        return summ

    def _add_constant (self, const) :

        # copy self
        import copy
        summ = copy.copy(self)
        summ.job    = None
        summ.fileid = None
        summ._can_delete_values = False

        # calculate checksum for added density
        import hashlib
        m = hashlib.md5()
        m.update("Density obtained by adding :\n" )
        m.update(self.get_checksum())
        m.update("Constant shift %18.10f" % const)
        summ._checksum = m.digest()
        
        # and finally add
        #pylint: disable-msg=W0212
        summ._values = self.get_values() + const
        self._delete_values()

        return summ
    
    def __add__ (self, other):
        """
        Addition of two densities.

        For two instances of C{densfresults}, an addition 
        is defined. The result of such an addition is another
        instance, which contains the sum of the two densities.
    
        The two instances must use the same grid.
        
        @exampleuse:
        
            Addition of densities.
            
            >>> # 'dens1' is a densfresult associated with the density of fragment 1
            >>> # 'dens2' is a densfresult associated with the density of fragment 2
            >>>
            >>> dens_tot = dens1 + dens2
            >>> dens_tot.get_cubfile('total_density.cub')

        """
        if isinstance(other, float) :
            return self._add_constant(other)
        elif isinstance(other, densfresults):
            return self._add_with_factor(other)
        else: 
            return other + self

    def __sub__ (self, other):
        """
        Subtraction of two densities.

        For two instances of C{densfresults}, a subtraction
        is defined. The result of such a subtraction is another
        instance, which contains the difference of the two densities.

        The two instances must use the same grid.
        
        @exampleuse:
        
            Calculation of a difference density.
            
            >>> # 'dens1' is a densfresult associated with the exact density
            >>> # 'dens2' is a densfresult associated with an approximate density (e.g. FDE)
            >>>
            >>> diffdens = dens1 - dens2
            >>> diffdens.get_cubfile('difference_density.cub')
            >>> print "RMS density deviation: ", math.sqrt(diffdens.integral(lambda x: x*x))

        """
        if isinstance(other, float) :
            return self._add_constant(-other)
        elif isinstance(other, densfresults):
            return self._add_with_factor(other, fact=-1.0)
        else:
            return other - self
  
    def filter_negative (self, thresh=0.0) :
                                                            
        # copy self
        import copy
        
        negative = copy.copy(self)
        negative.job    = None
        negative.fileid = None
        negative._can_delete_values = False
                                
        import numpy 
        negative._values = numpy.minimum(self.get_values(), [thresh])
        
        # calculate checksum for negative density
        import md5
        m = md5.new()
        m.update("Density obtained by keeping only negative values, thresh %18.8f :\n" % thresh )
        m.update(self.get_checksum())
        negative._checksum = m.digest()
         
        self._delete_values()
        
        return negative

    def filter_positive (self, thresh=0.0) :
                                                            
        # copy self
        import copy
        
        negative = copy.copy(self)
        negative.job    = None
        negative.fileid = None
        negative._can_delete_values = False
                                
        import numpy 
        negative._values = numpy.maximum(self.get_values(), [thresh])
        
        # calculate checksum for negative density
        import md5
        m = md5.new()
        m.update("Density obtained by keeping only positive values, thresh %18.8f :\n" % thresh )
        m.update(self.get_checksum())
        negative._checksum = m.digest()
         
        self._delete_values()
        
        return negative

    def filter_zeros (self, thresh=1e-4) :
                                                            
        # copy self
        import copy
        
        negative = copy.copy(self)
        negative.job    = None
        negative.fileid = None
        negative._can_delete_values = False
                                
        import numpy 
        values = self.get_values()
        new_values = numpy.zeros_like(values)
        new_values.shape = (values.size,)
       
        for i in range(new_values.size) :
            if abs(values.flat[i]) < thresh :
                if values.flat[i] > 0.0 :
                    new_values[i] = thresh
                else :
                    new_values[i] = -thresh
            else :
                new_values[i] = values.flat[i]

        new_values.shape = values.shape
        negative._values = new_values
 
        # calculate checksum for negative density
        import md5
        m = md5.new()
        m.update("Density obtained from filter_zeros, thresh %18.8f :\n" % thresh )
        m.update(self.get_checksum())
        negative._checksum = m.digest()
         
        self._delete_values()
        
        return negative

 
    def __mul__ (self, other):
        """
        Pointwise multiplication of two densities.

        For two instances of C{densfresults}, a multiplication
        is defined. The result of such a multiplication is another
        instance, which contains the pointwise productof the two densities.
    
        The two instances must use the same grid.
        
        @exampleuse:
        
            Integral over density times potential
            
            >>> # 'dens' is a densfresult associated with the density 
            >>> # 'pot' is a densfresult associated with the potential
            >>>
            >>> temp = dens * pot
            >>> print temp.integral()

        """
        if isinstance(other, float) :
            return self._mul_with_constant(other)
        elif isinstance(other, densfresults):
            return self._mul(other)
        else:
            return other * self
    
    def _mul (self, other):
        # first check that the grid is the same
        if not (self.grid is other.grid) :
            raise PyAdfError('grids have to be the same for addition of densities')
        
        # copy self
        import copy
        prod = copy.copy(self)
        prod.job    = None
        prod.fileid = None

        # calculate checksum for added density
        import hashlib
        m = hashlib.md5()
        m.update("Density obtained by multiplying :\n" )
        m.update(self.get_checksum())
        m.update(other.get_checksum())
        prod._checksum = m.digest()
        
        # and finally add
        #pylint: disable-msg=W0212
        prod._values = self.get_values() * other.get_values()

        self._delete_values()
        other._delete_values()
    
        return prod
    
    def _mul_with_constant (self, fact):
        # copy self
        import copy
        prod = copy.copy(self)
        prod.job    = None
        prod.fileid = None

        # calculate checksum for added density
        import hashlib
        m = hashlib.md5()
        m.update("Density obtained by multiplying with factor :\n" )
        m.update(self.get_checksum())
        m.update(str(fact))
        
        prod._values = self.get_values() * fact

        self._delete_values()
        return prod

    def __pow__ (self, exp):
        # copy self
        import copy
        power = copy.copy(self)
        power.job    = None
        power.fileid = None

        # calculate checksum for added density
        import hashlib
        m = hashlib.md5()
        m.update("Density obtained by taking\n") 
        m.update(self.get_checksum())
        m.update("to the power of \n" )
        m.update(str(exp))
        power._checksum = m.digest()
        
        # and finally add
        #pylint: disable-msg=W0212
        power._values = self.get_values()**exp

        self._delete_values()
    
        return power

    def __div__ (self, other):
        """
        Pointwise division of two densities.

        For two instances of C{densfresults}, a multiplication
        is defined. The result of such a multiplication is another
        instance, which contains the pointwise productof the two densities.
    
        The two instances must use the same grid.
        """
        if isinstance(other, float) :
            return self._mul_with_constant(1.0/other)
        elif isinstance(other, densfresults):
            return self._mul(other**(-1.0))
        else:
            return other**(-1.0) * self

    def apply_function (self, func):
        # copy self
        import copy
        prod = copy.copy(self)
        prod.job    = None
        prod.fileid = None

        # calculate checksum for added density
        import md5
        m = md5.new()
        m.update("Density obtained by applying function :\n" )
        m.update(self.get_checksum())
        m.update(str(func))
        
        prod._values = func(self.get_values())

        self._delete_values()
        return prod
     
    def integral (self, func=None, ignore=None, involume=None):
        """
        Returns the integral of (a function of) the density.
        
        This calculates the integral S{integral} f(S{rho}(r)) dr, 
        where f is the function given as argument.
        
        @exampleuse:
        
            Calculate integral of the square of the density
            
            >>> # 'dens' is an instance of densfresults
            >>> 
            >>> int = dens.integral(f=lambda x: x*x, ignore=(dens.get_values() < 0))
            >>> print "Integral of the squared density: ", int
        
        @param func: 
            A function of one variable that is applied to the density
            before the integration. If None, the density is integrated
            directly, i.e., the identity function is used.
        @type  func: function

        @param ignore: 
            An array of booleans to exclude certain points
        @type ignore:
            numpy.ndarray

        @param involume:
            a function of coordinates defining a box or volume over which to integrate, 
            returns a boolean, e.g.
            >>> def withinradius(coord) :
            ...     geom_center = [2.0, 4.0, 3.5]
            ...     radius = coord-geom_center
            ...     rad_abs = math.sqrt(radius[0]**2 + radius[1]**2 + radius[2]**2)
            ...     coord_within_radius = rad_abs < 5.0
            ...     return coord_within_radius
        @type involume : function
           
        @returns: integral of (a function of) the density
        @rtype: float
        """
        if func == None :
            func = lambda x: x
        ii = 0.0

        if involume is None :  
            if ignore is None :
                for w, val in izip(self.grid.weightiter(), self.valueiter()) :
                    ii += w * func(val)
            else:
                for w, val, ig in izip(self.grid.weightiter(), self.valueiter(), ignore.flat) :
                    if not ig:
                        ii += w * func(val)
        else :
            if ignore is None :
                for w, val, c in izip(self.grid.weightiter(), self.valueiter(), self.grid.coorditer() ) :
                    if (involume(c)) :
                        ii += w * func(val)
            else:
                for w, val, c, ig in izip(self.grid.weightiter(), self.valueiter(), 
                                          self.grid.coorditer(), ignore.flat) :
                    if involume(c) and not ig:
                        ii += w * func(val)

        return ii

    def get_electronic_dipole_moment_grid(self, involume=None):
        """
        Gets a (local) electronic dipole moment in atomic units by calculating the integral
        S{integral} S{rho}(r)*(x,y,z) dr 
        for a given volume. Nuclear contribution in class molecule.
        Caution: inaccuracies introduced by numerical integration introduce an origin
        dependence of the total dipole moment (and a mismatch with the ADF results). 
        One possible solution is to use the integral of the
        density to renormalize the dipole moments.
        
        @param involume:
            a function of coordinates defining a box or volume over which to integrate, returns a boolean
        @type involume : function
           
        @returns: electronic dipole moment (complete molecule or part of it)
        @rtype: numpy.array of floats
        """
        import numpy
        import math

        ii_x = 0.0
        ii_y = 0.0
        ii_z = 0.0

        if involume is None :
            for w, val, c in izip(self.grid.weightiter(), self.valueiter(), self.grid.coorditer()) :

                ii_x += -w * val * c[0] 
                ii_y += -w * val * c[1] 
                ii_z += -w * val * c[2] 

        else :
            for w, val, c in izip(self.grid.weightiter(), self.valueiter(), self.grid.coorditer()) :
                if (involume(c)) :
                    ii_x += -w * val * c[0] 
                    ii_y += -w * val * c[1] 
                    ii_z += -w * val * c[2] 

        dipole = numpy.array([ii_x,ii_y,ii_z])

        return dipole/Bohr_in_Angstrom

    def integral_voronoi (self, atoms, func=None):
        """
        Returns the integral of (a function of) the density over the Voronoi cells 
        of a given list of atoms. Intended for analysis of the (difference) density 
        per Voronoi cell.
        
        This calculates the integral S{integral} f(S{rho}(r)) dr, 
        where f is the function given as argument.
        
        @param func: 
            A function of one variable that is applied to the density
            before the integration. If None, the density is integrated
            directly, i.e., the identity function is used.
        @type  func: function

        @param atoms: for which atoms (more precisely their Voronoi cells) to perform the 
                      analysis (counting starts at 1)
        @type atoms: list of ints
           
        @returns: integral over density per Voronoi cell
        @rtype: list of floats
        """
        if func == None :
            func = lambda x: x

        vor_int = [0.0 for i in range(len(atoms))]

        for w, val, c, v in izip(self.grid.weightiter(), self.valueiter(), self.grid.coorditer(),
                              self.grid.voronoiiter() ) :
          
            for iatom in range(len(atoms)) :
                 if (atoms[iatom] == v) :
                     vor_int[iatom] += w * func(val) 
            
        return vor_int

    def get_electronic_dipole_voronoi(self, atoms):
        """
        Gets a (local) electronic dipole moment in atomic units by calculating the integral
        S{integral} S{rho}(r)*(x,y,z) dr 
        for the Voronoi cell around a given atom (list). Nuclear contribution in class molecule.
        Caution: inaccuracies introduced by numerical integration introduce an origin
        dependence of the total dipole moment (and a mismatch with the ADF results). 
        One possible solution is to use the integral of the total
        density to renormalize the dipole moments.
        
        @param atoms:
            atom numbers
        @type atoms:
            list of integers
        @returns: electronic dipole moment (per Voronoi cell)
        @rtype: list of numpy.array of floats
        """
        #func = lambda x: x*r_x, r_y, r_z
        import numpy
        import math

        ii_x = 0.0
        ii_y = 0.0
        ii_z = 0.0
        
        voronoidip = []
        for iatom in range(len(atoms)) :
            voronoidip.append( [ii_x, ii_y, ii_z] )

        for w, val, c, v in izip(self.grid.weightiter(), self.valueiter(), self.grid.coorditer(),
                              self.grid.voronoiiter() ) :
          
            for iatom in range(len(atoms)) :
                 if (atoms[iatom] == v) :
                     voronoidip[iatom][0] += -w * val * c[0]
                     voronoidip[iatom][1] += -w * val * c[1]
                     voronoidip[iatom][2] += -w * val * c[2]

        lvoronoidip = []
        sum = numpy.array([0.0, 0.0, 0.0])
        for iatom in range(len(atoms)) :
            lvoronoidip.append(numpy.array(voronoidip[iatom])/Bohr_in_Angstrom)
         
        return lvoronoidip

    def get_efield_in_point_grid(self, pointcoord):
        """
        Calculates the electronic contribution to the electric field in a point in 
        atomic units. Nuclear contribution in class molecule.
        This subroutine employs the density on the grid, preferably the potential
        should be used to calculate the electronic contribution.
        
        @param pointcoord:
        @type pointcoord: array of float, in Angstrom coordinates

        @returns: the electronic contribution to the electric field in atomic units
        @rtype: numpy.array of float
           
        """
        import numpy

        E_x = 0.0
        E_y = 0.0
        E_z = 0.0
        for w, val, c in izip(self.grid.weightiter(), self.valueiter(), self.grid.coorditer()) :
            dist = numpy.sqrt( (c[0]-pointcoord[0])**2 + (c[1]-pointcoord[1])**2 + (c[2]-pointcoord[2])**2 )
            E_x += - w * val * (c[0]-pointcoord[0]) / dist**3
            E_y += - w * val * (c[1]-pointcoord[1]) / dist**3
            E_z += - w * val * (c[2]-pointcoord[2]) / dist**3

        return numpy.array([E_x, E_y, E_z])*(Bohr_in_Angstrom*Bohr_in_Angstrom)

    def interpolate(self, int_grid):
        '''
        Convert the density/potential to another grid using interpolation.
        
        This method returns the same density, but on another grid.
        For the interpolation, the IMLS algorithm is used (see
        L{interpolation} for details). 
        
        The main purpose of this routine is to obtain a density/potential on 
        a L{cubegrid} that is suitable for visualization from one that is
        available only on an L{adfgrid}.
        
        @param int_grid: the grid to use for the interpolated density
        @type  int_grid: subclass of L{grid}
        
        @return: the interpolated density/potential
        @rtype:  L{densfresults}
        '''
        int_dens = densfresults(grid=int_grid)

        interp = interpolation(self)
        for i, point in enumerate(int_dens.grid.coorditer()) :
            if i % 500 == 0 :
                print "Interpolating point %i of %i " % (i, int_dens.grid.get_number_of_points())
            #pylint: disable-msg=W0212
            int_dens._values[i] = interp.get_value_at_point(point)

        #pylint: disable-msg=W0212
        int_dens._values.shape = int_dens.grid.shape
        
        import hashlib
        m = hashlib.md5()
        m.update("Interpolated from :\n" )
        m.update(self.get_checksum())
        m.update("on grid :\n")
        m.update(int_grid.get_grid_block(True))
        int_dens._checksum = m.digest()

        return int_dens
    
    def get_value_at_point (self, point):
        '''
        Get value at one point by interpolation.
        
        @param point: the point for which the interpolated density/potential is needed.
        '''
        interp = interpolation(self)
        return interp.get_value_at_point(point)    


class densfresults_unrestricted (object):
    """
    A wrapper class for unrestricted L{densfresults}.
    """
    
    def __init__ (self, alpha, beta):
        """
        Constructor for densfresults_unrestricted.
        
        @param alpha: the alpha results
        @type  alpha: L{densfresults}
        @param beta: the alpha results
        @type  beta: L{densfresults}
        """
        if not (alpha.grid is beta.grid) :
            raise PyAdfError('alpha and beta grids have to be the same in densfresults_unrestricted')
        self.nspin = 2
        self._checksum = None
        
        self._alpha = alpha
        self._beta  = beta
        
    def _get_grid (self) :
        return self._alpha.grid

    grid = property(_get_grid, None, None)
    """
    Returns the grid.
    """
    
    def get_checksum (self):
        if self._checksum is None:
            import hashlib

            m = hashlib.md5()
            m.update("Unrestricted density with :\n" )
            m.update("alpha: " +self._alpha.get_checksum())
            m.update("beta:  " +self._beta.get_checksum())
            self._checksum = m.digest()
            
        return self._checksum
  
    def __getitem__ (self, key):
        if key == 'alpha' :
            return self._alpha
        elif key == 'beta' :
            return self._beta
        elif key == 'tot' :
            return self._alpha + self._beta
        elif key == 'spin' :
            return self._alpha - self._beta
        else:
            raise KeyError
        
    def __add__(self, other):
        if isinstance(other, densfresults_unrestricted) :
            alpha = self._alpha + other['alpha']
            beta = self._beta + other['beta']
        else :
            alpha = self._alpha + other
            beta = self._beta + other
        return densfresults_unrestricted(alpha, beta)
    
    def __sub__(self, other):
        if isinstance(other, densfresults_unrestricted) :
            alpha = self._alpha - other['alpha']
            beta = self._beta - other['beta']
        else :
            alpha = self._alpha - other
            beta = self._beta - other
        return densfresults_unrestricted(alpha, beta)
    
    def __mul__(self, other):
        if isinstance(other, densfresults_unrestricted) :
            alpha = self._alpha * other['alpha']
            beta = self._beta * other['beta']
        else :
            alpha = self._alpha * other
            beta = self._beta * other
        return densfresults_unrestricted(alpha, beta)

    def __div__(self, other):
        if isinstance(other, densfresults_unrestricted) :
            alpha = self._alpha / other['alpha']
            beta = self._beta / other['beta']
        else :
            alpha = self._alpha / other
            beta = self._beta / other
        return densfresults_unrestricted(alpha, beta)

    def __pow__(self, exp):
        alpha = self._alpha**exp 
        beta = self._beta**exp 
        return densfresults_unrestricted(alpha, beta)

    def filter_negative (self, thresh=0.0) :
        alpha = self._alpha.filter_negative(thresh) 
        beta = self._beta.filter_negative(thresh)
        return densfresults_unrestricted(alpha, beta)

    def filter_positive (self, thresh=0.0) :
        alpha = self._alpha.filter_positive(thresh) 
        beta = self._beta.filter_positive(thresh)
        return densfresults_unrestricted(alpha, beta)

    def filter_zeros (self, thresh=1e-4) :
        alpha = self._alpha.filter_zeros(thresh)
        beta = self._beta.filter_zeros(thresh)
        return densfresults_unrestricted(alpha, beta)

    def apply_function (self, func):
        alpha = self._alpha.apply_function(func)
        beta = self._beta.apply_function(func)
        return densfresults_unrestricted(alpha, beta)


class densfjob (adfjob):
    """
    A class for densf jobs.
    
    This can be used to obtain the density (and related quantities)
    on a grid, e.g. for plotting or integration.
    
    @undocumented: _get_prop, _get_grid
    """
    
    def __init__ (self, adfres, prop, grid=None, frag=None) :
        """
        Constructor for densfjob.
        
        @param adfres: 
            The results of the ADF job for which the density 
            (or related quantity) should be calculated.
        @type  adfres: L{adfsinglepointresults} or subclass.
        @param prop: 
            The property to calculate. Possible values are
            'Density SCF', 'Density Fit SCF', 'Laplacian', and 'Laplacian Fit'.
        @type  prop: str
        @param grid: 
            The grid to use. If None, a default L{cubegrid} is used.
        @type  grid: subclass of L{grid}
        @param frag:
            Which fragment to use. Default is 'Active'.
            This can be used to get the densities of specific frozen
            fragments.
        @type frag: str
        """
        #pylint: disable-msg=W0621

        adfjob.__init__ (self)
        
        self._adfresults = adfres
        self._prop = prop
        if grid == None :
            self._grid = cubegrid(adfres.get_molecule())
        else :
            self._grid = grid
        if frag == None:
            self._frag = 'Active'
        else :
            self._frag = frag
        
        if self._prop.lower().startswith('orbital loc') :    
            self._olddensf = True
        else:
            self._olddensf = False

    def create_results_instance (self):
        return densfresults(self)
        
    def get_runscript (self):
        """
        Return a runscript for CJDENSF.
        """
        #pylint: disable-msg=W0221
        if self._olddensf :
            runscript = adfjob.get_runscript (self, program='densf', serial=True)
        else:
            runscript = adfjob.get_runscript (self, program='cjdensf', serial=True)
        return runscript

    def get_input (self):
        """
        Return an input file for CJDENSF.
        """
        inp = ""
        inp += self._grid.get_grid_block(self._checksum_only)
        if not self._olddensf :
            if self._frag == "ALL" :
                inp += 'ALLFRAGMENTS \n'
            else:
                inp += 'FRAGMENT ' + self._frag + '\n'

        prop = self._prop.upper().strip()

        if prop.endswith('ALPHA') :
            prop = prop[:-5].strip()
        elif prop.endswith('BETA') :
            prop = prop[:-4].strip()

        if prop.startswith('ORBITAL') :
            split = prop.split()
            if split[1].upper() == "LOC" :
                inp += "ORBITALS LOC \n"
                inp += "  %s \n" % split[2]
                inp += "END\n"
            else :
                inp += "ORBITALS \n" 
        elif prop.startswith('ORBLAPL') :
            split = prop.split()
            if split[1].upper() == "LOC" :
                inp += "LOCORBITALS \n"
            else :
                inp += "ORBITALS \n" 
            inp += "ORBITALS \n" 
        elif prop.startswith('NADKIN') :
            inp += 'EMBPOT\n'
            inp += self.prop + '\n'
        elif prop.startswith('LOCORBDENS') :
            orbs = eval(prop.strip().split(' ', 1)[1])
            inp += "LOCORBDENS \n"
            for i in orbs :
                inp += "%i \n" % i
            inp += "END\n"
        elif prop.startswith('ORBDENSLAPL') :
            orbs = eval(prop.strip().split(' ', 1)[1])
            inp += "ORBDENS \n"
            for i in orbs :
                inp += "%i \n" % i
            inp += "END\n"
            inp += "LAPLACIAN \n" 
        elif prop.startswith('ORBDENSGRAD') :
            orbs = eval(prop.strip().split(' ', 1)[1])
            inp += "ORBDENS \n"
            for i in orbs :
                inp += "%i \n" % i
            inp += "END\n"
            inp += "GRADIENT \n" 
        elif prop.startswith('ORBDENSKINPOT') :
            split = prop.strip().split(' ', 2)
            orbs = eval(split[2])
            inp += "ORBDENS \n"
            for i in orbs :
                inp += "%i \n" % i
            inp += "END\n"
            inp += "KINPOT %s \n" % split[1]
        elif prop.startswith('ORBDENS') :
            orbs = eval(prop.strip().split(' ', 1)[1])
            inp += "ORBDENS \n"
            for i in orbs :
                inp += "%i \n" % i
            inp += "END\n"
        elif prop.startswith('LOCORBGRAD') :
            orbs = eval(prop.strip().split(' ', 1)[1])
            inp += "LOCORBDENS \n"
            for i in orbs :
                inp += "%i \n" % i
            inp += "END\n"
            inp += "GRADIENT \n" 
        elif prop.startswith('LOCORBKINPOT') :
            split = prop.strip().split(' ', 2)
            orbs = eval(split[2])
            inp += "LOCORBDENS \n"
            for i in orbs :
                inp += "%i \n" % i
            inp += "END\n"
            inp += "KINPOT %s \n" % split[1]
        elif prop.startswith('LOCORBXCPOT') :
            split = prop.strip().split(' ', 1)
            orbs = eval(split[1])
            inp += "LOCORBDENS \n"
            for i in orbs :
                inp += "%i \n" % i
            inp += "END\n"
            inp += "POTENTIAL XC \n" 
        elif prop.upper().startswith('LOCORBCOULPOT') :
            split = prop.strip().split(' ', 1)
            orbs = eval(split[1])
            inp += "LOCORBDENS \n"
            for i in orbs :
                inp += "%i \n" % i
            inp += "END\n"
            inp += "POTENTIAL COUL \n" 
        else:
            inp += prop + "\n"
        inp += "END INPUT\n"
        
        if self._checksum_only :
            inp += self._adfresults.get_checksum()

        return inp

    def print_jobtype (self):
        return "DENSF job"

    def print_jobinfo (self):
        print " "+50*"-"
        print " Running " + self.print_jobtype()
        print
        print "   SCF taken from ADF job ", self._adfresults.fileid," (results id)"
        print
        print "   Fragment used: ", self._frag
        print
        print "   Calculated property : ", self._prop
        print
        
    def before_run (self):
        self._adfresults.link_tape (21)
        self.grid.before_densf_run()

    def after_run (self):
        import os
        self.grid.after_densf_run()
        os.remove('TAPE21')

    def _get_prop (self):
        return self._prop
    
    prop = property (_get_prop, None, None)
    """
    The property calculated (e.g., SCF Density).
    """

    def _get_grid (self):
        return self._grid
    
    grid = property (_get_grid, None, None)
    """
    The grid used.
    """
    
