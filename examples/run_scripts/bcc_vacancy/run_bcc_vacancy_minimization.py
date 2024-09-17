#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

# local imports
from lammps_implicit_der.tools import initialize_mpi, mpi_print, error_tools, TimingGroup, minimize_loss
from lammps_implicit_der.systems import BCC_VACANCY


def run_minimization(method,
                     step,
                     maxiter,
                     error_tol=1e-5,
                     adaptive_step=True,
                     comm=None,
                     der_alpha0=0.5,
                     der_adaptive_alpha=True,
                     pickle_name=None,
                     minimize_at_iters=True,
                     apply_hard_constraints=True):

    trun = TimingGroup('Minimize Vacancy')
    trun.add('total', level=2).start()

    path = '.'
    non_perturb_data = os.path.join(path, "bcc_vacancy.data")

    # If datafile is None, the system is initialized from scratch
    datafile = non_perturb_data if os.path.exists(non_perturb_data) else None

    vac_non_perturb = BCC_VACANCY(datafile=datafile,
                                 snapcoeff_filename='W.snapcoeff',
                                 ncell_x=3,
                                 verbose=True, comm=comm, logname='vac.log')

    # Save the datafile if it was initialized from scratch
    if datafile is None:
        vac_non_perturb.write_data(non_perturb_data)

    vac_perturb = BCC_VACANCY(datafile=non_perturb_data,
                             snapcoeff_filename='W_perturbed.snapcoeff',
                             comm=comm, logname='vac_perturbed.log')

    X_target = vac_perturb.X_coord.copy()

    vac_final, error_array, \
    min_X, min_Theta = minimize_loss(
                                vac_non_perturb,
                                X_target,
                                comm=comm,
                                error_tol=error_tol,
                                step=step,
                                adaptive_step=adaptive_step,
                                maxiter=maxiter,
                                der_method=method,
                                der_adaptive_alpha=der_adaptive_alpha,
                                der_alpha0=der_alpha0,
                                der_maxiter=500,
                                pickle_name=pickle_name,
                                verbosity=3,
                                minimize_at_iters=minimize_at_iters,
                                apply_hard_constraints=apply_hard_constraints,
                                )

    vac_final.write_data('vac_final.data')

    trun.timings['total'].stop()
    mpi_print(trun, comm=comm)


def main():

    comm, rank = initialize_mpi()

    error_tol = 1e-3

    method = 'inverse'
    #method = 'energy'
    #method = 'sparse'

    step = 1e-2
    adaptive_step = True
    maxiter = 30

    der_alpha0 = 0.5
    der_adaptive_alpha = True

    #minimize_at_iters = False
    minimize_at_iters = True

    apply_hard_constraints = True

    if method == 'sparse':
        der_alpha0 = 1e-4
        der_adaptive_alpha = False
    elif method == 'energy':
        der_alpha0 = 0.5
        der_adaptive_alpha = True

    #one_run = False
    one_run = True

    if one_run:
        run_minimization(method,
                         comm=comm,
                         error_tol=error_tol,
                         step=step,
                         adaptive_step=adaptive_step,
                         maxiter=maxiter,
                         der_alpha0=der_alpha0,
                         der_adaptive_alpha=der_adaptive_alpha,
                         minimize_at_iters=minimize_at_iters,
                         apply_hard_constraints=apply_hard_constraints,
                         pickle_name=None)

    #loop_run = True
    loop_run = False
    if loop_run:

        #der_alpha_list = [0.001, 0.05]
        #der_alpha_list = list(np.arange(0.05, 0.1, 0.01))
        #der_alpha_list += list(np.arange(0.1, 1.1, 0.1))

        der_alpha_list = list(np.arange(1.5, 5.5, 0.5))

        step_list = list(np.geomspace(1e-4, 1e-1, 10))
        maxiter = 100

        der_alpha_fix = 1.0

        #for der_alpha_iter in der_alpha_list:
        for iloop, step_iter in enumerate(step_list):

            mpi_print('\n'*3)
            mpi_print('*'*80)
            #mpi_print(f'Running minimization with {der_alpha_iter=}')
            mpi_print(f'{iloop+1}/{len(step_list)}: running minimization with {step_iter=}')
            mpi_print('*'*80)
            mpi_print('\n')

            method = 'energy'
            #pickle_name = f'minim_dict_{method}_alpha_{der_alpha_iter:09.6f}.pkl'
            pickle_name = f'minim_dict_{method}_step_{step_iter:10.8f}.pkl'

            run_minimization(method,
                             step=step_iter,
                             maxiter=maxiter,
                             der_alpha0=der_alpha_fix,
                             der_adaptive_alpha=der_adaptive_alpha,
                             pickle_name=pickle_name)


if __name__ == '__main__':
    main()
