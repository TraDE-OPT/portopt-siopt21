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
Run this script to reproduce the numerical experiment in Section 3.2 in:

K. Bredies, E. Chenchene, D. Lorenz, E. Naldi.
Degenerate Preconditioned Proximal Point Algorithms.
SIAM J. Optim. 2021 (in press).
"""

import numpy as np
import optimization_methods as opt
import time
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing_data import generate_data
from proximity_operators import proj_simplex


def compare_algorithms(r, S, W0, x_true, t1, t2, delta, L, tol, maxit=3000):
    tic = time.time()
    x_SeqFDRv1, Dist_SeqFDRv1 = opt.sequential_FDR_1(r, S, W0, x_true, t1, t2, delta, L, tol, maxit)
    print("Sequential FDR 1 ended in {} seconds".format(time.time()-tic))

    tic = time.time()
    x_SeqFDRv2, Dist_SeqFDRv2 = opt.sequential_FDR_2(r, S, W0, x_true, t1, t2, delta, L, tol, maxit)
    print("Sequential FDR 2 ended in {} seconds".format(time.time()-tic))

    tic = time.time()
    x_SeqFDRv3, Dist_SeqFDRv3 = opt.sequential_FDR_3(r, S, W0, x_true, t1, t2, delta, L, tol, maxit)
    print("Sequential FDR 3 ended in {} seconds".format(time.time()-tic))

    tic = time.time()
    x_GenBF, Dist_GenFB = opt.generalized_FB(r, S, W0, x_true, t1, t2, delta, L, tol, maxit)
    print("Generalized FB ended in {} seconds".format(time.time()-tic))

    tic = time.time()
    x_GenParFDR, Dist_ParFDR = opt.parallel_FDR_1(r, S, W0, x_true, t1, t2, delta, L, tol, maxit)
    print("Parallel FDR 1 ended in {} seconds".format(time.time()-tic))

    tic = time.time()
    x_GenParFDRv2, Dist_ParFDRv2 = opt.parallel_FDR_2(r, S, W0, x_true, t1, t2, delta, L, tol, maxit)
    print("Parallel FDR 2 ended in {} seconds".format(time.time()-tic))

    tic = time.time()
    x_GenParDR, Dist_ParDR = opt.parallel_DR(r, S, W0, x_true, t1, t2, delta, L, tol, maxit)
    print("Parallel DR ended in {} seconds".format(time.time()-tic))

    tic = time.time()
    x_PPXA, Dist_PPXA = opt.PPXA(r, S, W0, x_true, t1, t2, delta, L, tol, maxit)
    print("PPXA ended in {} seconds".format(time.time()-tic))
    return Dist_SeqFDRv1, Dist_SeqFDRv2, Dist_SeqFDRv3, Dist_GenFB, Dist_ParFDR, Dist_ParFDRv2, Dist_ParDR, Dist_PPXA


if __name__ == '__main__':
    n = 53  # testing using a real-world data-set with n=53
    W0 = proj_simplex(np.random.rand(n))  # portfolio at time 0

    # other possible choices:
    # W0 = np.array([1/n for _ in range(n)])
    # W0 = np.array([1/n for _ in range(n)])

    tol = 1e-9
    iters = 500
    x_true_maxit = 10000

    returns = pd.read_csv("data/returns.txt")

    x_true = np.copy(W0)
    step = 20
    days = 200

    kmax = 2
    plt.figure(figsize=(30, 10*kmax))

    for k in range(kmax):
        if k == 0:
            print('\nStarting case 0. Initial portfolio position: random')
        else:
            print('\nStarting case {}. Initial portfolio position: output of case {} considered on the same instance but 20 days later'.format(k, k-1))

        W0 = np.copy(x_true)
        S, r = generate_data(returns, n, k, step, days)

        L = np.linalg.norm(S)

        t1 = 0.001
        t2 = 0.001
        delta = 1

        x_true = opt.find_opt(r, S, W0, t1, t2, delta, L, x_true_maxit)
        D_SeqFDRv1, D_SeqFDRv2, D_SeqFDRv3, D_GenFB, D_ParFDR, D_ParFDRv2, D_ParDR, D_PPXA = compare_algorithms(r, S, W0, x_true, t1, t2, delta, L, tol, maxit=iters)

        plt.subplot(kmax, 2, 2*k+1)
        plt.semilogy(D_SeqFDRv1, 'b', ls=('dashed'), linewidth=3, label='SeqFDRv1')
        plt.semilogy(D_SeqFDRv2, 'r', ls=('dotted'), linewidth=3, label='SeqFDRv2')
        plt.semilogy(D_SeqFDRv3, 'g', ls=('solid'), linewidth=3, label='SeqFDRv3')
        plt.legend(fontsize=25)
        plt.grid()
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)

        plt.subplot(kmax, 2, 2*k+2)
        plt.semilogy(D_SeqFDRv3, 'g', ls=('solid'), linewidth=3, label='SeqFDRv3')
        plt.semilogy(D_GenFB, 'm', ls=('dashed'), linewidth=3, label='GenFB')
        plt.semilogy(D_ParFDR, 'k', ls=('dotted'), linewidth=3, label='ParFDR')
        plt.semilogy(D_ParDR, 'y', ls=('dashdot'), linewidth=3, label='ParDR')
        plt.semilogy(D_PPXA, 'c', ls=(0, (3, 1, 1, 1, 1, 1)), linewidth=3, label='PPXA')
        plt.legend(fontsize=25)
        plt.grid()
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        if k > 0:
            print('#### Total transaction from previous position (%):', sum(np.absolute(x_true-W0)))

    del D_SeqFDRv1, D_SeqFDRv2, D_SeqFDRv3, D_GenFB, D_ParFDR, D_ParFDRv2, D_ParDR, D_PPXA
    plt.savefig('test2.pdf')
