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
import scipy.sparse
import scipy.sparse.linalg

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
        self.fac_wp = None
        self.cs1 = None

        self.lapl_symm = None
        self.lapl = None

    def get_discrete_nabla(self):

        w2 = scipy.sparse.diags([self.w2], [0], format='csc')

        # Bickley five-point (error h^5) with m=4, j=2

        A2 = scipy.sparse.diags([2.0, -16.0, 16.0, -2.0], [-2, -1, 1, 2], 
                               shape=(self.N,self.N), format='lil')  

        # Bickley five-point (error h^4) with m=5, j=0 - to avoid any assumption for func[-1] 
        A2[0,0] = -50.0
        A2[0,1] =  96.0
        A2[0,2] = -72.0
        A2[0,3] =  32.0
        A2[0,4] =  -6.0

        # Bickley five-point (error h^4) with m=5, j=1 - to avoid any assumption for func[-1] 
        A2[1,0] =  -6.0
        A2[1,1] = -20.0
        A2[1,2] =  36.0
        A2[1,3] = -12.0
        A2[1,4] =   2.0

        # Bickley five-point (error h^4) with m=5, j=3,4 - to avoid any assumption for func[-1] 
        A2[-2,-5] =  -2.0
        A2[-2,-4] =  12.0
        A2[-2,-3] = -36.0
        A2[-2,-2] =  20.0
        A2[-2,-1] =   6.0

        A2[-1,-5] =    6.0
        A2[-1,-4] =  -32.0
        A2[-1,-3] =   72.0
        A2[-1,-2] =  -96.0
        A2[-1,-1] =   50.0

        A2 = (1.0/(24.0*self.h)) * A2.tocsc()
        A2 = w2.dot(A2)

        return A2

    def get_discrete_laplacian (self):
    
        w2 = scipy.sparse.diags([self.w2], [0], format='csc')
        w4 = scipy.sparse.diags([self.w2*self.w2], [0], format='csc')
        w  = scipy.sparse.diags([numpy.sqrt(self.w2)], [0], format='csc')

        # symmetric Laplacian, requires l-dependent correction for the first grid-point

        A1 = scipy.sparse.diags([-5.0, 80.0, -150.0, 80.0, -5.0], [-2, -1, 0, 1, 2], 
                               shape=(self.N,self.N), format='csc')  

        A1 = (1.0/(60.0*self.h*self.h)) * A1

        A1 = w2.dot(A1.dot(w2))
        A1 = A1 - scipy.sparse.diags([self.fac_wp*self.w2*self.w2], [0], format='csc')

        # non-symmetric Laplacian using forward-differentiation for the first grid-point

        A2 = scipy.sparse.diags([-5.0, 80.0, -150.0, 80.0, -5.0], [-2, -1, 0, 1, 2], 
                               shape=(self.N,self.N), format='lil')  

        # Bickley six-point (error h^6) with m=5, j=1 - assuming/enforcing func[-1] = 0 
        A2[0,0] = -75.0
        A2[0,1] = -20.0
        A2[0,2] =  70.0
        A2[0,3] = -30.0
        A2[0,4] =   5.0

        # Bickley six-point (error h^6) with m=5, j=4,5 - to avoid any assumption for func[-1] 
        A2[-2,-6] =   5.0
        A2[-2,-5] = -30.0
        A2[-2,-4] =  70.0
        A2[-2,-3] = -20.0
        A2[-2,-2] = -75.0
        A2[-2,-1] =  50.0

        A2[-1,-6] =  -50.0
        A2[-1,-5] =  305.0
        A2[-1,-4] = -780.0
        A2[-1,-3] = 1070.0
        A2[-1,-2] = -770.0
        A2[-1,-1] =  225.0

        A2 = (1.0/(60.0*self.h*self.h)) * A2.tocsc()

        A2 = w.dot(A2.dot(w))
        A2 = A2 - scipy.sparse.diags([self.fac_wp*self.w2], [0], shape=(self.N,self.N), format='csc')

        return A1, A2

    def calc_integral(self, func) :
        ii = numpy.dot(1.0/self.w2,func)
        ii = ii + (1.0/6.0)*func[0]/self.w2[0] - (1.0/24.0)*func[1]/self.w2[1] \
                + (1.0/24.0)*func[-2]/self.w2[-2] - (1.0/6.0)*func[-1]/self.w2[-1]
        return self.h * ii
   
    def extrapolate_to_zero(self, func) :

        # Bickley six-point (error h^4) with m=5, j=0 - to avoid any assumption for func[-1] 

        f1 = (1.0/(120.0*self.h))   * (-274.0*func[0] + 600.0*func[1] - 600.0*func[2] + 400.0*func[3] - 150.0*func[4] + 24.0*func[5])
        f2 = (1.0/(60.0*self.h**2)) * ( 225.0*func[0] - 770.0*func[1] +1070.0*func[2] - 780.0*func[3] + 305.0*func[4] - 50.0*func[5])
        f3 = (1.0/(20.0*self.h**3)) * ( -85.0*func[0] + 355.0*func[1] - 590.0*func[2] + 490.0*func[3] - 205.0*func[4] + 35.0*func[5])

        return func[0] - f1*self.h + f2*self.h**2 - f3*self.h**3 
 
class RationalGrid (Grid) :

    def __init__ (self, N, b) :
        Grid.__init__(self, N)

        self.b = b
        self.r = b*self.s/(1-self.s)

        self.w2 = (1-self.s)*(1-self.s) / b
        self.fac_wp = 0.0
        self.cs1 = self.b

        self.nabla = self.get_discrete_nabla()
        self.lapl_symm, self.lapl = self.get_discrete_laplacian()

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
        self.cs1 = self.b/T

        self.nabla = self.get_discrete_nabla()
        self.lapl_symm, self.lapl = self.get_discrete_laplacian()

def eigh_davidson (A, k, startvecs, convmax=1e-7, convnorm=1e-12) :

    bvecs = startvecs
    n = A.shape[0]

    maxres = 1e10
    resnorm = 1e10
    it = 0

    while (maxres > convmax) and (resnorm > convnorm) and (it < n) :
        it = it+1

        # orthogonalize bvecs
        bvecs, dum = numpy.linalg.qr(bvecs) 

        # construct subspace Hamiltonian
        sigma_sub = A.dot(bvecs)
        A_sub = numpy.dot(bvecs.transpose(), sigma_sub)

        # diagonalize subspace Hamiltonian
        evals, evecs_sub = numpy.linalg.eigh(A_sub)

        order = evals.argsort()
        evals = evals[order]
        evecs_sub = evecs_sub[:,order]

        # update basis vectors
        bvecs = numpy.dot(bvecs, evecs_sub)

        # calculate residue vectors
        resid = numpy.zeros((n,k))
        for i in range(k) :
            resid[:,i] = numpy.dot(sigma_sub, evecs_sub[:,i]) - evals[i]*bvecs[:,i]
        maxres = numpy.max(numpy.abs(resid))
        resnorm = numpy.linalg.norm(resid)

        #print "Davidson It %3i, Res %14.6e " % (it, maxres), evals

        # construct new basis vectors (diagonal preconditioner)
        b_new = numpy.zeros((n,k))
        for i in range(k) :
            aa = scipy.sparse.diags([evals[i]], [0], shape=(n,n), format='csc')
            b_new[:,i] = scipy.sparse.linalg.splu(A - aa).solve(resid[:,i])

        bvecs = numpy.concatenate((bvecs, b_new), axis=1) 

    if (it >= n-2) :
        raise Exception('Davidson did not converge')

    return evals[:k], bvecs[:,:k]


class OneDimSolver (object) :

    def __init__(self, grid) :
        self.grid = grid
        self.old_evecs = {} 

        self.lapl_symmetrized = 0.5 * (grid.lapl + grid.lapl.transpose())
        
    def calc_orbitals(self, pot, l, norbs, kmat=None) :

        A = self.grid.lapl_symm.copy()

        # apply correction for first grid point
        if (l == 0) :
            Z = -pot[0]*self.grid.r[0]
            d0 = 5.0 * (1.0 + self.grid.h*Z*self.grid.cs1)/(1.0 - self.grid.h*Z*self.grid.cs1)
        elif (l == 1) :
            d0 = -5.0 
        elif (l == 2) :
            d0 = 5.0 
        else :
            d0 = 0.0

        A[0,0] = A[0,0] + d0* (1.0/(60.0*self.grid.h*self.grid.h)) * self.grid.w2[0]*self.grid.w2[0]

        # add potential
        A = -0.5 * A + scipy.sparse.diags([pot + 0.5*l*(l+1)/(self.grid.r*self.grid.r)], [0], format='csc')

        A = A.toarray()
        if kmat is not None :
            A = A + kmat

        if True :
        #if not self.old_evecs.has_key(l) :

            #evals, evecs  = numpy.linalg.eigh(A.toarray())
            evals, evecs  = numpy.linalg.eigh(A)

            order = evals.argsort()
            evals = evals[order[:norbs]]
            evecs = evecs[:,order[:norbs]]

        else :

            evals, evecs = eigh_davidson(A, norbs, self.old_evecs[l])

        self.old_evecs[l] = evecs.copy(order='K')

        for i in range(norbs) :
            # convert from (P/w) back to P
            evecs[:,i] = numpy.sqrt(self.grid.w2) * evecs[:,i]
            # normalize P
            evecs[:,i] = 1.0/math.sqrt(self.grid.h) * evecs[:,i]

            # multiply by angular part
            if l == 0 :  # s orbital
                evecs[:,i] = 1.0/math.sqrt(4.0*math.pi) * evecs[:,i]
            if l == 1 :  # p_x orbital along x-axis
                evecs[:,i] = math.sqrt(3.0/(4.0*math.pi)) * evecs[:,i]
            if l == 2 :  # d_z2 orbital along z-axis
                evecs[:,i] = 0.5 * math.sqrt(5.0/math.pi) * evecs[:,i]
                
        return evals, evecs

    def calc_density(self, pot, occs, ons=None, output=True) :

        dens = numpy.zeros_like(self.grid.r)

        for l, norbs in occs.iteritems() :
            evals, evecs = self.calc_orbitals(pot, l, norbs)
            if output:
                print "Eigenvalues for l=%2i : " % l, evals

            for i in range(norbs) :
                if ons is None :
                    dens = dens + 2.0*evecs[:,i]**2
                else :
                    dens = dens + ons[l][i]*evecs[:,i]**2
        
        return dens

    def calc_density_and_energy(self, pot, occs, ons=None, output=True) :

        dens = numpy.zeros_like(self.grid.r)
        e_tot = 0.0

        for l, norbs in occs.iteritems() :
            evals, evecs = self.calc_orbitals(pot, l, norbs)
            if output:
                print "Eigenvalues for l=%2i : " % l, evals

            for i in range(norbs) :
                if ons is None :
                    dens = dens + 2.0*evecs[:,i]**2
                    e_tot = e_tot + 2.0 * (2*l+1) * evals[i]    
                else :
                    dens = dens + ons[l][i]*evecs[:,i]**2
                    e_tot = e_tot + ons[l][i] * (2*l+1) * evals[i]    

        e_pot = 4.0*math.pi*self.grid.calc_integral(pot*dens)

        return dens, e_tot-e_pot, e_pot

def calc_orbitals (grid, pot, l, norbs, kmat=None) :
    solver = OneDimSolver(grid)
    return solver.calc_orbitals(pot, l, norbs, kmat)

def calc_density(grid, pot, occs, ons=None, output=True) :
    solver = OneDimSolver(grid)
    return solver.calc_density(pot, occs, ons=ons, output=output) 

def calc_density_and_energy(grid, pot, occs, ons=None, output=True) :
    solver = OneDimSolver(grid)
    return solver.calc_density_and_energy(pot, occs, ons=ons, output=output) 

def reconstruct_potential(grid, refdens, startpot, occs, denserr=1e-4) :
    recpot = numpy.zeros_like(startpot)

    err = 1e10
    it = 0
    while err > denserr :
        it = it + 1
        dens = calc_density(grid, startpot+recpot, occs, output=False)
        err = 4.0*math.pi*grid.calc_integral(abs(dens-refdens))
        print "Iteration %4i: error=%14.6e " % (it, err)

        recpot = recpot - (refdens-dens)/ grid.r

    #print refdens-dens
    return recpot

class CalcFuncGrad(object):
    def __init__(self, grid, occs, refdens, startpot, ons=None, lambd=0.0):
        self.solver = OneDimSolver(grid)

        self.value, self.grad = None, None
        self.pot = None
        
        self.grid = grid
        self.occs = occs

        maxl = max(occs.keys()) 
        if ons is None :
            self.ons = [ [2.0]*occs[l] for l in range(maxl+1) ]
        else :
            self.ons = ons
            for l in range(maxl+1) :
                if not (len(ons[l]) == occs[l]) :
                    raise Exception('inconsistent occs and ons')

        self.refdens = refdens
        self.startpot = startpot
        self.lambd = lambd

        self.ncomp = 0
        self.nit   = 0

    def gradient (self, pot, lambda_smooth=None):
        self.dens = self.solver.calc_density(pot/self.grid.r, self.occs, ons=self.ons, output=False)

        w = -4.0*math.pi * (self.grid.h/self.grid.w2) * (self.dens - self.refdens) / self.grid.r

        if lambda_smooth is None :
             self.grad = w + self.lambd * self.grad_smooth(pot)
        else :
             self.grad = w + lambda_smooth * self.grad_smooth(pot)

        return self.grad

    def grad_smooth (self, pot) :
        return -8.0*math.pi * self.grid.h * self.solver.lapl_symmetrized.dot(pot-self.startpot) 

    def hess (self, pot, lambda_smooth=None) :
        H = numpy.zeros((self.grid.N, self.grid.N))
        
        maxl = max(self.occs.keys()) 

        for l in range(maxl+1) :
            print "Constructing Hessian for l = ", l

            ens, orbs = OneDimSolver(self.grid).calc_orbitals(pot/self.grid.r, l, self.grid.N)

            Hl = numpy.zeros((self.grid.N, self.grid.N))

            for iocc in range(self.occs[l]) :
                #for ivirt in range(self.occs[l], self.grid.N) :
                for ivirt in range(self.grid.N) :
                    if not (iocc == ivirt) :
                        prodorbs = orbs[:,iocc]*orbs[:,ivirt]
                        Hl = Hl + self.ons[l][iocc] * numpy.outer(prodorbs, prodorbs) / (ens[iocc] - ens[ivirt])

            if l == 0 :  # s shell
                H = H + 4.0*math.pi * Hl
            if l == 1 :  # p shell
                H = H + (4.0*math.pi / 3.0) * Hl
            if l == 2 :  # d shell
                H = H + (4.0*math.pi / 5.0) * Hl
       
        H = numpy.outer(1.0/(self.grid.r*self.grid.w2), 1.0/(self.grid.r*self.grid.w2)) * H
        H = - (8.0 * math.pi * self.grid.h*self.grid.h) * H 

        if lambda_smooth is None :
             H = H + self.lambd * self.hess_smooth()
        else :
             H = H + lambda_smooth * self.hess_smooth()

        return H

    def invhess (self, pot) :
        H = self.hess(pot)

        evals, evecs = numpy.linalg.eigh(H) 

        evals = numpy.abs(1.0/(evals + 1e-8))    # cutoff small eigenvalues here
        Hinv = numpy.dot(evecs, numpy.dot(numpy.diag(evals), evecs.transpose()))

        return Hinv

    def hess_smooth (self):
        return -8.0*math.pi * self.grid.h * self.solver.lapl_symmetrized.toarray()

    def error (self) :
        abserr = 4.0*math.pi * self.grid.calc_integral(numpy.abs(self.dens-self.refdens))
        maxdens = numpy.max(numpy.abs(self.dens-self.refdens)/(self.grid.r*self.grid.r)) 

        return abserr, maxdens

    def info (self, pot) :
        self.nit = self.nit + 1

        abserr, maxdens = self.error()
        gradnorm = numpy.linalg.norm(self.grad)

        print "BFGS Iteration %4i: ncomp=%4i     abserr=%14.6e maxdens=%14.6e norm=%14.6e" % (self.nit, self.ncomp, abserr, maxdens, gradnorm)

class CalcFuncGradSpin (object):

    def __init__(self, grid, occs_alpha, occs_beta, refdens_alpha, refdens_beta, startpot, lambd_tot=0.0, lambd_spin=0.0):
        self.nit  = 0
        self.grad = None 

        self.grid = grid
        self.lambd_tot = lambd_tot
        self.lambd_spin = lambd_spin

        self.func_alpha = CalcFuncGrad(grid, occs_alpha, refdens_alpha, startpot, lambd=0.0)
        self.func_beta  = CalcFuncGrad(grid, occs_beta, refdens_beta, startpot, lambd=0.0)

    def lagrangian(self, pot):
        return self.func_alpha(pot[:self.grid.N]) + self.func_beta[self.grid.N:]

    def gradient (self, pot):
        grad_alpha = self.func_alpha.gradient(pot[:self.grid.N], lambda_smooth=0.0)
        grad_beta  = self.func_beta.gradient(pot[self.grid.N:], lambda_smooth=0.0)

        grad_alpha = grad_alpha + (self.lambd_tot + self.lambd_spin) * self.func_alpha.grad_smooth(pot[:self.grid.N]) \
                                + (self.lambd_tot - self.lambd_spin) * self.func_beta.grad_smooth(pot[self.grid.N:])
        grad_beta  = grad_beta + (self.lambd_tot + self.lambd_spin) * self.func_beta.grad_smooth(pot[self.grid.N:]) \
                               + (self.lambd_tot - self.lambd_spin) * self.func_alpha.grad_smooth(pot[:self.grid.N])

        self.grad = numpy.concatenate([grad_alpha, grad_beta])
        return self.grad

    def hess (self, pot) :
        Haa = self.func_alpha.hess(pot[:self.grid.N], lambda_smooth=0.0)
        Haa = Haa + (self.lambd_tot + self.lambd_spin) * self.func_alpha.hess_smooth()

        Hbb = self.func_beta.hess(pot[self.grid.N:], lambda_smooth=0.0)
        Hbb = Hbb + (self.lambd_tot + self.lambd_spin) * self.func_beta.hess_smooth()

        Hab = (self.lambd_tot - self.lambd_spin) * self.func_alpha.hess_smooth()
        Hba = (self.lambd_tot - self.lambd_spin) * self.func_beta.hess_smooth()

        H = numpy.vstack( (numpy.hstack((Haa, Hab)), numpy.hstack((Hba, Hbb)) ) )
        return H

    def invhess (self, pot) :
        H = self.hess(pot)

        evals, evecs = numpy.linalg.eigh(H) 

        evals = numpy.abs(1.0/(evals + 1e-8))    # cutoff small eigenvalues here
        Hinv = numpy.dot(evecs, numpy.dot(numpy.diag(evals), evecs.transpose()))

        return Hinv

    def info (self, pot) :
        self.nit = self.nit + 1

        abserr_alpha, maxdens_alpha = self.func_alpha.error()
        abserr_beta, maxdens_beta = self.func_beta.error()
        gradnorm = numpy.linalg.norm(self.grad)

        print "BFGS Iteration %4i:  ALPHA: abserr=%14.6e maxdens=%14.6e     BETA: abserr=%14.6e maxdens=%14.6e     GRADNORM=%14.6e" % \
                                    (self.nit, abserr_alpha, maxdens_alpha, abserr_beta, maxdens_beta, gradnorm)


def reconstruct_potential_sd(grid_in, refdens_in, startpot_in, occs_in, denserr=1e-4, method=None) :
    from bfgs_only_fprime import fmin_bfgs_onlygrad, fmin_newton_onlygrad
    from scipy.optimize import fmin_bfgs

    funcs = CalcFuncGrad(grid_in, occs_in, refdens_in, startpot_in*grid_in.r, lambd=1e-8)
    #funcs = CalcFuncGrad(grid_in, occs_in, refdens_in, startpot_in*grid_in.r, lambd=0.0)

    if (method is None) or (method == 'BFGS') :
        recpot = fmin_bfgs(funcs.lagrangian, startpot_in*grid_in.r, fprime=funcs.gradient, callback=funcs.info, norm=2, disp=True, gtol=1e-8) / grid_in.r
    elif (method == 'BFGS_OnlyGrad') :
        recpot = fmin_bfgs_onlygrad(funcs.gradient, startpot_in*grid_in.r, invhess=funcs.invhess, callback=funcs.info, gtol=1e-10) / grid_in.r
    else :
        raise Exception('Unknown optimization method')

    return recpot - startpot_in


def reconstruct_spinpotential_sd (grid_in, refdens_alpha_in, refdens_beta_in, startpot_in, occs_alpha_in, occs_beta_in, 
                                 lamb_tot=0.0, lamb_spin=0.0, gradnorm=1e-8) :

    from bfgs_only_fprime import fmin_bfgs_onlygrad, fmin_newton_onlygrad
    from scipy.optimize import fmin_bfgs

    funcs = CalcFuncGradSpin(grid_in, occs_alpha_in, occs_beta_in, refdens_alpha_in, refdens_beta_in, startpot_in*grid_in.r, lamb_tot, lamb_spin)

    startpot = numpy.concatenate([startpot_in*grid_in.r, startpot_in*grid_in.r]) 
    recpot = fmin_bfgs_onlygrad(funcs.gradient, startpot, invhess=funcs.invhess, callback=funcs.info, gtol=gradnorm)

    return recpot[:grid_in.N] / grid_in.r - startpot_in, recpot[grid_in.N:] / grid_in.r - startpot_in, 


def reconstruct_restrpotential_sd (grid_in, refdens_tot_in, startpot_in, occs_in, ons, lamb, gradnorm=1e-8) :

    from bfgs_only_fprime import fmin_bfgs_onlygrad, fmin_newton_onlygrad
    from scipy.optimize import fmin_bfgs

    funcs = CalcFuncGrad(grid_in, occs_in, refdens_tot_in, startpot_in*grid_in.r, ons, lamb)

    recpot = fmin_bfgs_onlygrad(funcs.gradient, startpot_in*grid_in.r, invhess=funcs.invhess, callback=funcs.info, gtol=gradnorm)

    return recpot / grid_in.r - startpot_in 

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
