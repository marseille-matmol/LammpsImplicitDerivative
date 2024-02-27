#!/usr/bin/env python3

import os
import numpy as np

# local imports
from lammps_implicit_der.tools import mpi_print, \
                                      get_default_data_path, \
                                      initialize_mpi, \
                                      error_tools, \
                                      TimingGroup

from lammps_implicit_der.systems import Dislo


def run_minimization(method,
                     maxiter=100,
                     step=1e-2,
                     adaptive_step=True,
                     comm=None,
                     error_tol=1e-6,
                     der_alpha=0.5,
                     der_adaptive_alpha=True,
                     pickle_name='dislo.pickle',
                     minimize_at_iters=True,
                     apply_hard_constraints=True):

    trun = TimingGroup('Minimize Dislo')
    trun.add('total', level=2).start()

    data_path = get_default_data_path()
    datafile_path_easy = os.path.join(data_path, 'dislo_relaxed_easy_core.lammps-data')
    datafile_path_hard = os.path.join(data_path, 'dislo_hard_core.lammps-data')

    mpi_print('Dislo easy core initial relaxation...', comm=comm)
    with trun.add('easy core init'):
        dislo_easy = Dislo(snapcoeff_filename='W.snapcoeff',
                           datafile=datafile_path_easy,
                           logname='dislo.log',
                           comm=comm,
                           verbose=True)

    mpi_print('Dislo hard core initialization (no relaxation)...', comm=comm)
    with trun.add('hard core init'):
        dislo_hard = Dislo(snapcoeff_filename='W.snapcoeff',
                           datafile=datafile_path_hard,
                           minimize=False,
                           logname='dislo_hard_init.log',
                           comm=comm)

    X_hard = dislo_hard.X_coord

    X_easy = dislo_easy.X_coord

    X_hard = X_easy.copy() + dislo_easy.minimum_image(X_hard-X_easy)

    mpi_print('\nParameter optimization...\n', comm=comm)

    dislo_final, error_array = error_tools.minimize_loss(
        dislo_easy,
        X_hard,
        comm=comm,
        step=step,
        adaptive_step=adaptive_step,
        maxiter=maxiter,
        error_tol=error_tol,
        der_method=method,
        der_ftol=1e-2,
        der_adaptive_alpha=der_adaptive_alpha,
        der_alpha=der_alpha,
        der_maxiter=500,
        pickle_name=pickle_name,
        verbosity=3,
        minimize_at_iters=minimize_at_iters,
        apply_hard_constraints=apply_hard_constraints,
        )

    trun.timings['total'].stop()

    mpi_print(trun, comm=comm)
    mpi_print(dislo_easy.timings, comm=comm)


def main():

    comm, rank = initialize_mpi()

    #method = 'inverse'
    method = 'energy'
    #method = 'sparse'

    step = 1e-3
    #step = 1e-4
    #step = 0.1

    der_alpha = 0.5
    #der_alpha = 10.0
    der_adaptive_alpha = True
    minimize_at_iters = True
    apply_hard_constraints = True
    #apply_hard_constraints = False

    adaptive_step = True

    run_minimization(method,
                     maxiter=30,
                     comm=comm,
                     step=step,
                     adaptive_step=adaptive_step,
                     error_tol=1e-6,
                     der_alpha=der_alpha,
                     der_adaptive_alpha=der_adaptive_alpha,
                     minimize_at_iters=minimize_at_iters,
                     apply_hard_constraints=apply_hard_constraints)


if __name__ == '__main__':
    main()
