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
This file contains all proximity operators.

For details and references, see Section 3.2 in:

K. Bredies, E. Chenchene, D. Lorenz, E. Naldi.
Degenerate Preconditioned Proximal Point Algorithms.
SIAM J. Control Optim. 2021 (in press).
"""

import numpy as np


def func(w, S, r, delta, n, W0, t1, t2):
    return np.dot(w, (S+delta*np.eye(n)) @ w)/2-np.dot(r, w)+t1*np.linalg.norm((w - W0), ord=1)+t2*np.linalg.norm(np.absolute(w - W0)**(3/2), ord=1)


def shrink(tau, w):
    return np.sign(w)*np.maximum(0, np.abs(w)-tau)


def shrink_power(tau, w):
    return w + 9/8*tau**2*np.sign(w)*(1-np.sqrt(1+16/(9*tau**2)*np.abs(w)))


def proj_simplex(v, z=1, random_state=None):
    rs = np.random.RandomState(random_state)
    n_features = len(v)
    U = np.arange(n_features)
    s = 0
    rho = 0
    while len(U) > 0:
        G = []
        L = []
        k = U[rs.randint(0, len(U))]
        ds = v[k]
        for j in U:
            if v[j] >= v[k]:
                if j != k:
                    ds += v[j]
                    G.append(j)
            elif v[j] < v[k]:
                L.append(j)
        drho = len(G) + 1
        if s + ds - (rho + drho) * v[k] < z:
            s += ds
            rho += drho
            U = L
        else:
            U = G
    theta = (s - z) / float(rho)

    return np.maximum(v - theta, 0)
