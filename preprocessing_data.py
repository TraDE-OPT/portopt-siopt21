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
Data preprocessing.

For details and references, see Section 3.2 in:

K. Bredies, E. Chenchene, D. Lorenz, E. Naldi.
Degenerate Preconditioned Proximal Point Algorithms.
SIAM J. Optim. 2021 (in press).
"""

import numpy as np


def generate_data(returns, n, kappa, step=1, days=100):
    returns_l = list(returns)
    returns_l = returns_l[1:n+1]
    A = np.random.rand(days, len(returns_l))
    k = 0
    for i in returns_l:
        for ind in range(days):
            A[ind][k] = returns[i][ind+step*kappa]
        k = k+1

    A = A-A.mean(axis=0, keepdims=True)

    S = A.T @ A

    r = np.sum(A, axis=0)/np.shape(A)[0]

    return S, r
