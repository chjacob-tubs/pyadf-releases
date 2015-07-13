#!/usr/bin/env python
#
# Simple Python module for numerically solving a one-electron 
# Schrodinger equation for a given potential.
#
# Copyright (C) 2010-2012 by Christoph R. Jacob
# 
# when using, please cite:
# Ch. R. Jacob, J. Chem. Phys. 135, 244102 (2011).
#
# for details on the logarthmic grid used here, see
# D. Andrae, J. Hinze, Int. J. Quantum Chem. 63, 65 (1997).
# G. Eickerling, M. Reiher, J. Chem. Theory Comput. 4, 286 (2008).
#
# This file is part of
# PyADF - A Scripting Framework for Multiscale Quantum Chemistry.
# Copyright (C) 2006-2012 by Christoph R. Jacob, S. Maya Beyhan,
# Rosa E. Bulo, Andre S. P. Gomes, Andreas Goetz, Karin Kiewisch,
# Jetze Sikkema, and Lucas Visscher
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

import math
import numpy

Bohr_in_Angstrom = 0.5291772108

class Grid (object) :

    def __init__ (self, N):
        self.N = N             # number of inner grid points
        self.h = 1.0/(N+1)     # step size
        self.s = numpy.arange(N+1)*self.h
        self.s = self.s[1:]

        # these have to be defined for each grid
        self.r = None
        self.w2 = None
        self.fac_wp = 0.0

    def get_discrete_laplacian (self):
        A = numpy.zeros((self.N, self.N))
    
        # initialize A to T (Eq. 45 in Andrae/Hinze paper)

        A[0,0] =  -75.0
        A[0,1] =  -20.0
        A[0,2] =   70.0
        A[0,3] =  -30.0
        A[0,4] =    5.0
    
        A[1,0] =   80.0
        A[1,1] = -150.0
        A[1,2] =   80.0
        A[1,3] =   -5.0
        for i in range(2,self.N-2) :
            A[i,i-2] =   -5.0
            A[i,i-1] =   80.0
            A[i,i]   = -150.0
            A[i,i+1] =   80.0
            A[i,i+2] =   -5.0
        A[-2,-4] =   -5.0
        A[-2,-3] =   80.0
        A[-2,-2] = -150.0
        A[-2,-1] =   80.0
    
        A[-1,-5] =    5.0
        A[-1,-4] =  -30.0
        A[-1,-3] =   70.0
        A[-1,-2] =  -20.0
        A[-1,-1] =  -75.0
    
        # V = 60 * h^2 * F mit F = -V
        A = numpy.dot(numpy.diag(self.w2*self.w2), A)
        A = (1.0/(60.0*self.h*self.h)) * A - numpy.diag(self.fac_wp*self.w2*self.w2)
        return A

    def calc_integral(self, func) :
        return self.h * numpy.dot(1.0/self.w2,func)
    
class RationalGrid (Grid) :

    def __init__ (self, N, b) :
        Grid.__init__(self, N)

        self.r = b*self.s/(1-self.s)

        self.w2 = (1-self.s)*(1-self.s) / b
        self.fac_wp = 0.0

class LogGrid (Grid) :
    
    def __init__ (self, N, b=0.1, rmax=1000.0) :
        Grid.__init__(self, N)
        self.b = b
        self.rmax = rmax
        
        T = 1.0/(math.log(rmax+b) - math.log(b))
    
        self.r = b*numpy.exp(self.s/T) - b

        # w^2 = ds/dr
        self.w2 = T/(self.r + b)
        # w''/w = (1/w) * d2w/ds^2
        self.fac_wp = 1.0/(4*T*T) 
        
def calc_orbitals(grid, pot, l, norbs) :

    A = grid.get_discrete_laplacian()
    A = -0.5*A + numpy.diag(pot + 0.5*l*(l+1)/(grid.r*grid.r))

    evals, evecs  = numpy.linalg.eig(A)
    order = evals.argsort()

    evals = evals[order[:norbs]].real
    evecs = evecs[:,order[:norbs]].real
 
    for i in range(norbs) :
        # convert from wP back to P
        evecs[:,i] = 1.0/numpy.sqrt(grid.w2) * evecs[:,i]
        # normalize P
        N = grid.calc_integral(evecs[:,i]*evecs[:,i])
        evecs[:,i] = 1.0/math.sqrt(N) * evecs[:,i]

        # multiply by radial part
        if l == 0 :  # s orbital
            evecs[:,i] = 1.0/math.sqrt(4.0*math.pi) * evecs[:,i]
        if l == 1 :  # p_x orbital along x-axis
            evecs[:,i] = math.sqrt(3.0/(4.0*math.pi)) * evecs[:,i]
        if l == 2 :  # d_z2 orbital along z-axis
            evecs[:,i] = 0.5 * math.sqrt(5.0/math.pi) * evecs[:,i]
            
    return evals, evecs

def calc_density(grid, pot, occs, output=True) :

    dens = numpy.zeros_like(grid.r)

    for l, norbs in occs.iteritems() :
        evals, evecs = calc_orbitals(grid, pot, l, norbs)
        if output:
            print "Eigenvalues for l=%2i : " % l, evals

        for i in range(norbs) :
            dens = dens + 2.0*evecs[:,i]**2
    
    return dens

def reconstruct_potential(grid, refdens, startpot, occs, denserr=1e-4) :
    recpot = numpy.zeros_like(startpot)

    err = 1e10
    it = 0
    while err > denserr :
        it = it + 1
        dens = calc_density(grid, startpot+recpot, occs, output=False)
        err = 4.0*math.pi*grid.calc_integral(abs(dens-refdens))
        print "Iteration %4i: error=%14.6e " % (it, err)

        recpot = recpot - (refdens-dens)/grid.r

    print refdens-dens
    return recpot

#def main() :
#    
#    grid = LogGrid(400)
#    dens = calc_density(grid, 36.0, {0:4, 1:3, 2:1})
#    
#    import pylab
#    pylab.plot(grid.r, dens)
#
#    pylab.xlim(0, 2.0)
#    
#    pylab.show()    
#    
#main()
