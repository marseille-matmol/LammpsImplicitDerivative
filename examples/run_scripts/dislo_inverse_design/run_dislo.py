#!/usr/bin/env python3

import os
import numpy as np
import yaml

# local imports
from lammps_implicit_der.tools import initialize_mpi, mpi_print, error_tools, TimingGroup, minimize_loss
from lammps_implicit_der.systems import SCREW_DISLO
from lammps_implicit_der.tools.io import setup_default_minimization_dict, load_parameters, print_parameters
from lammps_implicit_der.tools.generate_masks import generate_mask_dX, generate_mask_radius, plot_mask

def run_minimization(param_dict, comm=None):

    trun = TimingGroup('Minimize DISLO')
    trun.add('total', level=2).start()

    # Unpack the parameter dictionary
    # Implicit derivative parameters
    der_method = param_dict['implicit_derivative']['method']
    der_min_style = param_dict['implicit_derivative']['min_style']
    der_adaptive_alpha = param_dict['implicit_derivative']['adaptive_alpha']
    der_alpha0 = param_dict['implicit_derivative']['alpha0']
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
    # fixed cylinder for dislocation
    fixed_cyl_axis = param_dict['system']['fixed_cyl_axis']
    fixed_cyl_x1 = param_dict['system']['fixed_cyl_x1']
    fixed_cyl_x2 = param_dict['system']['fixed_cyl_x2']
    fixed_cyl_r = param_dict['system']['fixed_cyl_r']
    fixed_cyl_lo = param_dict['system']['fixed_cyl_lo']
    fixed_cyl_hi = param_dict['system']['fixed_cyl_hi']

    mpi_print('DISLO start initial relaxation...', comm=comm)
    with trun.add('start init'):
        dislo_start = SCREW_DISLO(snapcoeff_filename=snapcoeff_filename,
                                 datafile=datafile_path_start,
                                 sub_element=sub_element,
                                 fixed_cyl_axis=fixed_cyl_axis,
                                 fixed_cyl_x1=fixed_cyl_x1,
                                 fixed_cyl_x2=fixed_cyl_x2,
                                 fixed_cyl_r=fixed_cyl_r,
                                 fixed_cyl_lo=fixed_cyl_lo,
                                 fixed_cyl_hi=fixed_cyl_hi,
                                 logname='dislo_start.log',
                                 minimize=True,
                                 comm=comm,
                                 verbose=True)

    mpi_print('DISLO target initialization (no relaxation)...', comm=comm)
    with trun.add('target init'):
        dislo_target = SCREW_DISLO(snapcoeff_filename=snapcoeff_filename,
                                  datafile=datafile_path_target,
                                  sub_element=sub_element,
                                  fixed_cyl_axis=fixed_cyl_axis,
                                  fixed_cyl_x1=fixed_cyl_x1,
                                  fixed_cyl_x2=fixed_cyl_x2,
                                  fixed_cyl_r=fixed_cyl_r,
                                  fixed_cyl_lo=fixed_cyl_lo,
                                  fixed_cyl_hi=fixed_cyl_hi,
                                  logname='dislo_target.log',
                                  minimize=False,
                                  comm=comm,
                                  verbose=True)

    X_target = dislo_target.X_coord

    X_start = dislo_start.X_coord

    X_target = X_start.copy() + dislo_start.minimum_image(X_target-X_start)

    # Hessian mask
    if param_dict['implicit_derivative']['apply_hess_mask']:

        hess_mask_type = param_dict['implicit_derivative']['hess_mask_type']
        if hess_mask_type == 'dX':
            dX = dislo_start.minimum_image(X_target - X_start)
            threshold = param_dict['implicit_derivative']['hess_mask_threshold']
            hess_mask, hess_mask_3D = generate_mask_dX(dX, threshold=threshold, comm=comm)

        elif hess_mask_type == 'radius':
            radius = param_dict['implicit_derivative']['hess_mask_radius']
            center_specie = 2
            species = dislo_start.species
            hess_mask, hess_mask_3D = generate_mask_radius(X_start, radius=radius, center_specie=center_specie, species=species, comm=comm)

        plot = True
        if plot and (comm is None or comm.Get_rank() == 0):
            plot_mask(X_start, hess_mask_3D)

    else:
        hess_mask = None
        hess_mask_3D = None

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
                                    der_alpha0=der_alpha0,
                                    der_maxiter=der_maxiter,
                                    der_hess_mask=hess_mask,
                                    verbosity=3,
                                    minimize_at_iters=minimize_at_iters,
                                    apply_hard_constraints=apply_hard_constraints,
                                    )

    trun.timings['total'].stop()

    mpi_print(trun, comm=comm)


def main():

    comm, rank = initialize_mpi()

    param_dict = setup_default_minimization_dict()
    param_dict = load_parameters(param_dict, 'minimize_param.yml')
    print_parameters(param_dict, comm=comm)

    run_minimization(param_dict, comm=comm)


if __name__ == '__main__':
    main()
