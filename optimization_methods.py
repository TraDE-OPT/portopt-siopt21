# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021 Kristian Bredies (kristian.bredies@uni-graz.at)
#                       Enis Chenchene (enis.chenchene@uni-graz.at)
#                       Dirk A. Lorenz (d.lorenz@tu-braunschweig.de)
#                       Emanuele Naldi (e.naldi@tu-braunschweig.de)
#
#    This file is part of the example code repository for the paper:
#
#      K. Bredies, E. Chenchene, D. Lorenz, E. Naldi.
#      Degenerate Preconditioned Proximal Point Algorithms,
#      SIAM Journal on Optimization, 2021. In press.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains all the methods used for comparison.

For details and references, see Section 3.2 in:

K. Bredies, E. Chenchene, D. Lorenz, E. Naldi.
Degenerate Preconditioned Proximal Point Algorithms.
SIAM J. Optim. 2021 (in press).
"""

import numpy as np
from proximity_operators import func, shrink, shrink_power, proj_simplex


def find_opt(r, S, W0, t1, t2, delta, L, maxit=350):
    K = len(r)  # number of assets

    I = np.eye(K)
    B = 0.5*(S+delta*I)  # compute it only once
    r = 0.5*r

    def C(w):
        return B@w-r

    # Parameter tuning
    gamma = 2/(L+delta)

    # Initialize variables
    w1 = np.ones(K)/K
    w2 = np.ones(K)/K

    for k in range(maxit):
        x1 = proj_simplex(w1)
        x2 = shrink_power(t2*gamma/2, x1-gamma/2*C(x1)+(w2-w1)/2-W0)+W0
        x3 = shrink(t1*gamma, 2*x2-gamma*C(x2)-w2-W0)+W0

        w1 = w1+1.4*(x2-x1)
        w2 = w2+1.4*(x3-x2)

    return x1


def sequential_FDR_1(r, S, W0, x_true, t1, t2, delta, L, tol=10**-10, maxit=350):
    '''
    Implements the Seqeuntial Forward DRS method introduced in Section 3.1.2
    splitting the forward term according to point 1. in Section  3.2.
    '''

    K = len(r)  # number of assets
    dists = []  # Initialize distances to minimum

    def C1(w):
        return S@w-r

    def C2(w):
        return delta*w

    # Parameter tuning
    gamma = 1/(L+delta)

    # Initialize variables
    w1 = np.ones(K)/K
    w2 = np.ones(K)/K

    k = 0
    err = 1
    while (k < maxit) and (err > tol):
        x1 = shrink_power(t2*gamma, w1-W0)+W0
        x2 = shrink(t1*gamma/2, x1-gamma/2*C1(x1)+(w2-w1)/2-W0)+W0
        x3 = proj_simplex(2*x2-gamma*C2(x2)-w2)

        w1 = w1+1.4*(x2-x1)
        w2 = w2+1.4*(x3-x2)

        err = np.linalg.norm(x3-x_true)
        dists.append(err)
        k = k+1

    return (x1+x2+x3)/3, dists


def sequential_FDR_2(r, S, W0, x_true, t1, t2, delta, L, tol=10**-10, maxit=350):
    '''
    Implements the Sequential Forward DRS method introduced in Section 3.1.2
    computing the forward term only once according to point 2. in Section  3.2.
    '''

    K = len(r)  # number of assets
    dists = []  # Initialize distances to minimum

    # Forward operator
    I = np.eye(K)
    B = (S+delta*I)  # compute it only once

    def C(w):
        return B@w-r

    # Parameter tuning
    gamma = 1/(L+delta)

    # Initialize variables
    w1 = np.ones(K)/K
    w2 = np.ones(K)/K

    k = 0
    err = 1
    while (k < maxit) and (err > tol):
        x1 = shrink_power(t2*gamma, w1-W0)+W0
        x2 = shrink(t1*gamma/2, x1+(w2-w1)/2-W0)+W0
        x3 = proj_simplex(2*x2-gamma*C(x2)-w2)

        w1 = w1+1.4*(x2-x1)
        w2 = w2+1.4*(x3-x2)

        err = np.linalg.norm(x3-x_true)
        dists.append(err)
        k = k+1

    return x3, dists


def sequential_FDR_3(r, S, W0, x_true, t1, t2, delta, L, tol=10**-10, maxit=350):
    '''
    Implements the Seqeuntial Forward DRS method introduced in Section 3.1.2
    computing the forward term twice according to point 2. in Section  3.2.

    '''

    K = len(r)  # number of assets
    dists = []  # Initialize distances to minimum

    # Forward operator
    I = np.eye(K)
    B = 0.5*(S+delta*I)  # compute it only once
    r = 0.5*r

    def C(w):
        return B@w-r

    # Parameter tuning
    gamma = 2/(L+delta)

    # Initialize variables
    w1 = np.ones(K)/K
    w2 = np.ones(K)/K

    k = 0
    err = 1
    while (k < maxit) and (err > tol):
        x1 = shrink_power(t2*gamma, w1-W0)+W0
        x2 = shrink(t1*gamma/2, x1-gamma/2*C(x1)+(w2-w1)/2-W0)+W0
        x3 = proj_simplex(2*x2-gamma*C(x2)-w2)

        w1 = w1+1.4*(x2-x1)
        w2 = w2+1.4*(x3-x2)

        err = np.linalg.norm(x3-x_true)
        dists.append(err)
        k = k+1

    return (x1+x2+x3)/3, dists


def generalized_FB(r, S, W0, x_true, t1, t2, delta, L, tol=10**-10, maxit=350):
    '''
    Implements the so-called Generalized Forward-Backward introduced in:

    H. Raguet, J. Fadili, and G. Peyré, A generalized forward-backward splitting,
    SIAM J. Imaging Sci., 6 (2013), pp. 1199–1226.
    '''

    K = len(r)  # number of assets
    dists = []  # Initialize distances to minimum

    # Forward operator
    I = np.eye(K)
    B = (S+delta*I)  # compute it only once

    def C(w):
        return B@w-r

    # Parameter tuning
    gamma = 1/(L+delta)

    # Initialize variables
    z1 = np.ones(K)/K
    z2 = np.ones(K)/K
    z3 = np.ones(K)/K

    x = (z1+z2+z3)/3
    k = 0
    err = 1
    while (k < maxit) and (err > tol):
        z1 = z1+1.4*(proj_simplex(2*x-z1-gamma*C(x))-x)
        z2 = z2+1.4*(shrink(t1*3*gamma, 2*x-z2-gamma*C(x)-W0)+W0-x)
        z3 = z3+1.4*(shrink_power(t2*3*gamma, 2*x-z3-gamma*C(x)-W0)+W0-x)

        x = (z1+z2+z3)/3

        err = np.linalg.norm(x-x_true)
        dists.append(err)
        k = k+1

    return x, dists


def parallel_FDR_1(r, S, W0, x_true, t1, t2, delta, L, tol=10**-10, maxit=350):
    '''
    Implements the Parallel Forward DRS method version 1, cf. Section 3.1.1
    '''

    K = len(r)  # number of assets
    dists = []  # Initialize distances to minimum

    # Forward operator
    I = np.eye(K)
    B = 0.5*(S+delta*I)  # compute it only once
    r = 0.5*r

    def C(w):
        return B@w-r

    # Parameter tuning
    gamma = 2/(L+delta)

    # Initialize variables
    z1 = np.ones(K)/K
    z2 = np.ones(K)/K

    x = (z1+z2)/2
    k = 0
    err = 1
    while (k < maxit) and (err > tol):
        x = proj_simplex((z1+z2)/2)
        z1 = z1+1.4*(shrink(t1*gamma, 2*x-gamma*C(x)-z1-W0)+W0-x)
        z2 = z2+1.4*(shrink_power(t2*gamma, 2*x-gamma*C(x)-z2-W0)+W0-x)

        err = np.linalg.norm(x-x_true)
        dists.append(err)
        k = k+1

    return x, dists


def parallel_FDR_2(r, S, W0, x_true, t1, t2, delta, L, tol=10**-10, maxit=350):
    '''
    Implements the Parallel Forward DRS method version 2, cf. Section 3.1.1
    '''

    K = len(r)  # number of assets
    dists = []  # Initialize distances to minimum

    # Forward operators

    def C1(w):
        return S@w-r

    def C2(w):
        return delta*w

    # Parameter tuning
    gamma = 1/(L+delta)

    # Initialize variables
    z1 = np.ones(K)/K
    z2 = np.ones(K)/K

    x = (z1+z2)/2
    k = 0
    err = 1
    while (k < maxit) and (err > tol):
        x = proj_simplex((z1+z2)/2)

        z1 = z1+1.4*(shrink(t1*gamma, 2*x-gamma*C1(x)-z1-W0)+W0-x)
        z2 = z2+1.4*(shrink_power(t2*gamma, 2*x-gamma*C2(x)-z2-W0)+W0-x)

        err = np.linalg.norm(x-x_true)
        dists.append(err)
        k = k+1

    return x, dists


def parallel_DR(r, S, W0, x_true, t1, t2, delta, L, tol=10**-10, maxit=350):
    '''
    Implements the Parallel DRS method.
    '''

    K = len(r)  # number of assets
    dists = []  # Initialize distances to minimum

    # Parameter tuning
    gamma = 2/L

    # Compute Prox F
    tau = gamma/3
    # In = np.linalg.inv(np.eye(K)+tau/(1+delta*tau)*S)
    # prox_F = lambda x: In @ (1/(1+delta*tau)*x+tau/(1+delta*tau)*r)

    In = np.linalg.inv((1+delta*tau)*np.eye(K)+tau*S)

    def prox_F(x):
        return In @ (x + tau*r)

    # Initialize variables
    z1 = np.ones(K)/K
    z2 = np.ones(K)/K
    z3 = np.ones(K)/K

    k = 0
    err = 1
    while (k < maxit) and (err > tol):
        x = prox_F((z1+z2+z3)/3)

        z1 = z1+1.8*(shrink(t1*gamma, 2*x-z1-W0)+W0-x)
        z2 = z2+1.8*(shrink_power(t2*gamma, 2*x-z2-W0)+W0-x)
        z3 = z3+1.8*(proj_simplex(2*x-z3)-x)

        err = np.linalg.norm(x-x_true)
        dists.append(err)
        k = k+1

    return x, dists


def PPXA(r, S, W0, x_true, t1, t2, delta, L, tol=10**-10, maxit=350):
    '''
    Implements the so-called PPXA method introduced in:

    N. Pustelnik, C. Chaux, and J.-C. Pesquet, Parallel proximal algorithm for image restoration using hybrid regularization,
    IEEE Trans. Image Process., 20 (2011), pp. 2450–2462.
    '''

    K = len(r)  # number of assets
    dists = []  # Initialize distances to minimum

    # Parameter tuning
    gamma = 2/L

    # Compute Prox F
    tau = gamma
    In = np.linalg.inv((1+delta*tau)*np.eye(K)+tau*S)

    def prox_F(x):
        return In @ (x + tau*r)

    # Initialize variables
    z1 = np.ones(K)/K
    z2 = np.ones(K)/K
    z3 = np.ones(K)/K
    z4 = np.ones(K)/K
    x = np.ones(K)/K

    k = 0
    err = 1
    while (k < maxit) and (err > tol):
        p1 = prox_F(z1)
        p2 = shrink(t1*gamma, z2-W0)+W0
        p3 = shrink_power(t2*gamma, z3-W0)+W0
        p4 = proj_simplex(z4)
        p = (p1+p2+p3+p4)/4

        z1 = z1+1.8*(2*p-x-p1)
        z2 = z2+1.8*(2*p-x-p2)
        z3 = z3+1.8*(2*p-x-p3)
        z4 = z4+1.8*(2*p-x-p4)

        x = x+1.8*(p-x)

        err = np.linalg.norm(x-x_true)
        dists.append(err)
        k = k+1

    return x, dists
