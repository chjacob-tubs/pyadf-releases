#
# This is a modified version of the file downloaded from 
# https://gist.github.com/rmcgibbo/4735287 
#

# Copyright (c) 2013, SciPy Developers, Robert McGibbon
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 

import numpy as np
from collections import namedtuple
Result = namedtuple('Result', ['xopt', 'gopt', 'Hopt', 'n_grad_calls'])

def fmin_bfgs_onlygrad(fprime, x0, args=(), invhess=None, gtol=1e-5, callback=None, maxiter=None):
    """Minimize a function, via only information about its gradient, using BFGS
    
    The difference between this and the "standard" BFGS algorithm is that the
    line search component uses a weaker criterion, because it can't check
    for sure that the function value actually decreased.
    
    Parameters
    ----------
    fprime : callable f(x, *args)
        gradient of the objective function to be minimized
    x0 : ndarray
        Initial guess
    args : tuple, optional
        Extra arguments to be passed to `fprime`
    gtol : float, optional
        gradient norm must be less than `gtol` before succesful termination
    callback : callable, optional
        An optional user-supplied function to call after each iteration.
        Called as `callback(xk)`, where `xk` is the current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    
    Returns
    -------
    xopt : ndarray
        Parameters which minimize `f`, and which are a root of the gradient,
        `fprime`
    gopt : ndrarray
        value of the gradient at `xopt`, which should be near zero
    Hopt : ndarray
        final estimate of the hessian matrix at `xopt`
    n_grad_calls : int
        number of gradient calls made
    """
    
    x0 = np.asarray(x0).flatten()
    if maxiter is None:
            maxiter = len(x0)*200
    
    gfk = fprime(x0, *args)  # initial gradient
    n_grad_calls = 1  # number of calls to fprime()
    
    k = 0  # iteration counter
    N = len(x0)  # degreees of freedom
    I = np.eye(N, dtype=int)
   
    if invhess is None : 
        Hk = I  # initial guess of the Hessian
    else :
        Hk = invhess(x0, *args)
    xk = x0
    sk = [2*gtol]
    
    gnorm = np.linalg.norm(gfk)
    while (gnorm > gtol) and (k < maxiter):
        # search direction
        pk = -np.dot(Hk, gfk)
        
        alpha_k, gfkp1, ls_grad_calls = _line_search(fprime, xk, gfk, pk, args, maxiters=20)
        print "CJCJ line search ", ls_grad_calls, alpha_k
        
        n_grad_calls += ls_grad_calls
        
        # advance in the direction of the step
        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1

        if gfkp1 is None:
            gfkp1 = fprime(xkp1, *args)
            n_grad_calls += 1
        
        yk = gfkp1 - gfk
        gfk = gfkp1
        
        if callback is not None:
            callback(xk)

        k += 1
        gnorm = np.linalg.norm(gfk)
        if gnorm < gtol:
            break
            
        try:  #this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (np.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            print "Divide-by-zero encountered: rhok assumed large"
        if np.isinf(rhok):  #this is patch for numpy
            rhok = 1000.0
            print "Divide-by-zero encountered: rhok assumed large"

        # main bfgs update here. this is copied straight from
        # scipy.optimize._minimize_bfgs
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok

        if True :
        #if False :
        #if rhok < 0.0 :
        #if (k == 20) :
            Hk = invhess(xk, *args)
        else :
            Hk = np.dot(A1, np.dot(Hk, A2)) + rhok * sk[:, np.newaxis] * sk[np.newaxis, :]

        evals, evecs = np.linalg.eigh(Hk)
        print "CJCJ bfgs 1 ", np.linalg.norm(sk), np.linalg.norm(yk), rhok
        print "CJCJ bfgs 2 ", np.min(evals), np.max(evals), 1.0/np.max(evals), 1.0/np.min(evals)

    
    if k >= maxiter:
        print "Warning: %d iterations exceeded" % maxiter
        print "         Current gnorm: %e" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k
        
    
    elif gnorm < gtol:
        print "Optimization terminated successfully."
        print "         Current gnorm: %e" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k
        
    #return Result(xopt=xk, gopt=gfk, Hopt=Hk, n_grad_calls=n_grad_calls)
    return xk

def fmin_newton_onlygrad(fprime, fhess, x0, args=(), gtol=1e-5, callback=None, maxiter=None):
    """Minimize a function, via only information about its gradient, using second-order Newton
    
    The difference between this and the "standard" algorithm is that the
    line search component uses a weaker criterion, because it can't check
    for sure that the function value actually decreased.
    
    Parameters
    ----------
    fprime : callable f(x, *args)
        gradient of the objective function to be minimized
    x0 : ndarray
        Initial guess
    args : tuple, optional
        Extra arguments to be passed to `fprime`
    gtol : float, optional
        gradient norm must be less than `gtol` before succesful termination
    callback : callable, optional
        An optional user-supplied function to call after each iteration.
        Called as `callback(xk)`, where `xk` is the current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    
    Returns
    -------
    xopt : ndarray
        Parameters which minimize `f`, and which are a root of the gradient,
        `fprime`
    gopt : ndrarray
        value of the gradient at `xopt`, which should be near zero
    Hopt : ndarray
        final estimate of the hessian matrix at `xopt`
    n_grad_calls : int
        number of gradient calls made
    """
    
    x0 = np.asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0)*200
    
    gfk = fprime(x0, *args)  # initial gradient
    n_grad_calls = 1  # number of calls to fprime()
    
    k = 0  # iteration counter
    N = len(x0)  # degreees of freedom
    I = np.eye(N, dtype=int)
   
    xk = x0
    sk = [2*gtol]
    
    gnorm = np.linalg.norm(gfk)
    while (gnorm > gtol) and (k < maxiter):
        H = fhess(xk, *args)

        # search direction
        pk = np.linalg.lstsq(H, -gfk, rcond=1e-12)[0]

        alpha_k, gfkp1, ls_grad_calls = _line_search(fprime, xk, gfk, pk, args)
        
        n_grad_calls += ls_grad_calls
        
        # advance in the direction of the step
        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1

        if gfkp1 is None:
            gfkp1 = fprime(xkp1, *args)
            n_grad_calls += 1
        
        yk = gfkp1 - gfk
        gfk = gfkp1
        
        if callback is not None:
            callback(xk)
        
        k += 1
        gnorm = np.linalg.norm(gfk)
        if gnorm < gtol:
            break
            
    if k >= maxiter:
        print "Warning: %d iterations exceeded" % maxiter
        print "         Current gnorm: %e" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k
        
    
    elif gnorm < gtol:
        print "Optimization terminated successfully."
        print "         Current gnorm: %e" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k
        
    #return Result(xopt=xk, gopt=gfk, Hopt=Hk, n_grad_calls=n_grad_calls)
    return xk


def _line_search(fprime, xk, gk, pk, args=(), alpha_guess=1.0, curvature_condition=0.9,
    update_rate=0.5, maxiters=5):
    """Inexact line search with only the function gradient
    
    The step size is found only to satisfy the strong curvature condition, (i.e
    the second of the strong Wolfe conditions)
    
    Parameters
    ----------
    fprime : callable f(x, *args)
        gradient of the objective function to be minimized
    xk : ndarray
        current value of x
    gk : ndarray
        current value of the gradient
    pk : ndarray
        search direction
    args : tuple, optional
        Extra arguments to be passed to `fprime`
        
    Returns
    -------
    alpha : float
        The step length
    n_evaluations : int
        The number of times fprime() was evaluated
    gk : ndarray
        The gradient value at the alpha, `fprime(xk + alpha*pk)`
        
    Other Parameters
    -----------------
    alpha_guess : float, default = 1.0
        initial guess for the step size
    curvature_condition : float, default = 0.9
        strength of the curvature condition. this is the c_2 on the wikipedia
        http://en.wikipedia.org/wiki/Wolfe_conditions, and is recommended to be
        0.9 according to that page. It should be between 0 and 1.
    update_rate : float, default=0.5
        Basically, we keep decreasing the alpha by multiplying by this constant
        every iteration. It should be between 0 and 1.
    maxiters : int, default=5
        Maximum number of iterations. The goal is that the line search step is
        really quick, so we don't want to spend too much time fiddling with alpha
    """
    alpha = alpha_guess
    initial_slope = np.dot(gk, pk)

    for j in xrange(maxiters):
        gk_new = fprime(xk + alpha * pk, *args)
        if np.abs(np.dot(gk_new, pk)) < np.abs(curvature_condition * initial_slope):
            break
        else:
            alpha *= update_rate
            
    # j+1 is the final number of calls to fprime()
    return alpha, gk_new, j+1
