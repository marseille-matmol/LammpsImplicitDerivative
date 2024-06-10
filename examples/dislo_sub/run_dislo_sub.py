#!/usr/bin/env python3

import os
import numpy as np

# local imports
from lammps_implicit_der.tools import initialize_mpi, mpi_print, error_tools, TimingGroup, minimize_loss
from lammps_implicit_der.systems import DisloSub


def run_minimization(
                     # Implicit derivative parameters
                     der_method,
                     der_min_style,
                     der_adaptive_alpha,
                     der_alpha,
                     der_ftol,
                     der_maxiter,
                     # Minimization parameters
                     maxiter,
                     step,
                     adaptive_step,
                     error_tol,
                     minimize_at_iters,
                     apply_hard_constraints,
                     comm=None):

    trun = TimingGroup('Minimize Dislo')
    trun.add('total', level=2).start()

    datafile_path_start = 'dislo_sub_easy_core.lammps-data'
    datafile_path_target = 'dislo_sub_hard_core.lammps-data'

    mpi_print('Dislo start initial relaxation...', comm=comm)
    with trun.add('start init'):
        dislo_start = DisloSub(snapcoeff_filename='WX.snapcoeff',
                               datafile=datafile_path_start,
                               logname='dislo_start.log',
                               minimize=True,
                               comm=comm,
                               verbose=True)

    mpi_print('Dislo target initialization (no relaxation)...', comm=comm)
    with trun.add('target init'):
        dislo_target = DisloSub(snapcoeff_filename='WX.snapcoeff',
                                datafile=datafile_path_target,
                                logname='dislo_target.log',
                                minimize=False,
                                comm=comm,
                                verbose=True)

    X_target = dislo_target.X_coord

    X_start = dislo_start.X_coord

    X_target = X_start.copy() + dislo_start.minimum_image(X_target-X_start)

    mpi_print('\nParameter optimization...\n', comm=comm)

    dislo_final, minim_dict = error_tools.minimize_loss(
                                    dislo_start,
                                    X_target,
                                    comm=comm,
                                    step=step,
                                    adaptive_step=adaptive_step,
                                    maxiter=maxiter,
                                    error_tol=error_tol,
                                    der_method=der_method,
                                    der_min_style=der_min_style,
                                    der_ftol=der_ftol,
                                    der_adaptive_alpha=der_adaptive_alpha,
                                    der_alpha=der_alpha,
                                    der_maxiter=der_maxiter,
                                    verbosity=3,
                                    minimize_at_iters=minimize_at_iters,
                                    apply_hard_constraints=apply_hard_constraints,
                                    )

    trun.timings['total'].stop()

    mpi_print(trun, comm=comm)


def main():

    comm, rank = initialize_mpi()

    #
    # Minimization parameters
    #
    step = 1e-3  # 0.1 # 1e-4
    minimize_at_iters = True
    apply_hard_constraints = False  # True
    adaptive_step = True  # False
    maxiter = 100
    error_tol = 1e-6

    #
    # Implicit derivative parameters
    #
    der_method = 'energy'  # 'sparse' # 'inverse'
    der_adaptive_alpha = True

    #der_min_style = 'cg'
    #der_alpha = 1e-6

    der_min_style = 'fire'
    der_alpha = 0.5

    der_ftol = 1e-2
    der_maxiter = 100

    run_minimization(
                     # Implicit derivative parameters
                     der_method,
                     der_min_style=der_min_style,
                     der_adaptive_alpha=der_adaptive_alpha,
                     der_alpha=der_alpha,
                     der_ftol=der_ftol,
                     der_maxiter=der_maxiter,
                     # Minimization parameters
                     maxiter=maxiter,
                     step=step,
                     adaptive_step=adaptive_step,
                     error_tol=error_tol,
                     minimize_at_iters=minimize_at_iters,
                     apply_hard_constraints=apply_hard_constraints,
                     comm=comm)


if __name__ == '__main__':
    main()
