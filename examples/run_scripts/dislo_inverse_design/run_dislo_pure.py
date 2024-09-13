#!/usr/bin/env python3

import os
import numpy as np
import yaml

# local imports
from lammps_implicit_der.tools import initialize_mpi, mpi_print, error_tools, TimingGroup, minimize_loss
from lammps_implicit_der.systems import DisloSub, DISLO
from lammps_implicit_der.tools.io import setup_minimization_dict


def run_minimization(param_dict, comm=None):

    trun = TimingGroup('Minimize DISLO')
    trun.add('total', level=2).start()

    # Unpack the parameter dictionary
    # Implicit derivative parameters
    der_method = param_dict['implicit_derivative']['method']
    der_min_style = param_dict['implicit_derivative']['min_style']
    der_adaptive_alpha = param_dict['implicit_derivative']['adaptive_alpha']
    der_alpha = param_dict['implicit_derivative']['alpha']
    der_ftol = param_dict['implicit_derivative']['ftol']
    der_maxiter = param_dict['implicit_derivative']['maxiter']

    # Minimization parameters
    maxiter = param_dict['minimization']['maxiter']
    step = param_dict['minimization']['step']
    adaptive_step = param_dict['minimization']['adaptive_step']
    error_tol = param_dict['minimization']['error_tol']
    minimize_at_iters = param_dict['minimization']['minimize_at_iters']
    apply_hard_constraints = param_dict['minimization']['apply_hard_constraints']

    # System parameters
    datafile_path_start = param_dict['system']['lammps_data_start']
    datafile_path_target = param_dict['system']['lammps_data_target']
    snapcoeff_filename = param_dict['system']['snapcoeff_filename']
    sub_element = param_dict['system']['sub_element']

    mpi_print('DISLO start initial relaxation...', comm=comm)
    with trun.add('start init'):
        dislo_start = DISLO(snapcoeff_filename=snapcoeff_filename,
                               datafile=datafile_path_start,
                               logname='dislo_start.log',
                               minimize=True,
                               comm=comm,
                               verbose=True)

    mpi_print('DISLO target initialization (no relaxation)...', comm=comm)
    with trun.add('target init'):
        dislo_target = DISLO(snapcoeff_filename=snapcoeff_filename,
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
                                    sub_element,
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

    param_dict = setup_minimization_dict(input_name='minimize_param.yml', comm=comm)

    run_minimization(param_dict, comm=comm)


if __name__ == '__main__':
    main()
