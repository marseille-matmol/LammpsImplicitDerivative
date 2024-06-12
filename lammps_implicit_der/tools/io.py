#!/usr/bin/env python3
"""
Input/output tools.
"""

import yaml
from .utils import mpi_print


def setup_minimization_dict(input_name=None, comm=None):
    """
    Setup a dictionary with the parameters for the minimization (inverse design).

    Example of input YAML file:

    system:
        lammps_data_start: relaxed_WBe_screw.lammps-data
        lammps_data_target: target_WBe_screw.lammps-data
        snapcoeff_filename: WBe-NEW.snapcoeff
        sub_element: Be

    minimization:
        step: 1e-3
        minimize_at_iters: True
        apply_hard_constraints: False
        adaptive_step: True
        maxiter: 50
        error_tol: 1e-6

    implicit_derivative:
        method: energy
        min_style: fire
        adaptive_alpha: True
        alpha: 0.5
        ftol: 1e-4
        maxiter: 500
    """

    param_dict = {}

    param_dict['system'] = {
        'lammps_data_start': 'relaxed_WBe_screw.lammps-data',
        'lammps_data_target': 'target_WBe_screw.lammps-data',
        'snapcoeff_filename': 'WBe.snapcoeff',
        'sub_element': 'Be',
    }

    param_dict['minimization'] = {
        'step': 1e-3,
        'minimize_at_iters': True,
        'apply_hard_constraints': False,
        'adaptive_step': True,
        'maxiter': 50,
        'error_tol': 1e-6,
        'lammps_data_start': 'relaxed_WBe_screw.lammps-data',
        'lammps_data_target': 'target_WBe_screw.lammps-data',
    }

    param_dict['implicit_derivative'] = {
        'method': 'energy',
        'min_style': 'fire',
        'adaptive_alpha': True,
        'alpha': 0.5,
        'ftol': 1e-4,
        'maxiter': 500,
    }

    # Load parameters from file
    if input_name is not None:

        with open(input_name, 'r') as f:
            param_dict_input = yaml.load(f, Loader=yaml.FullLoader)

        # check that param_dict_input has allowed keys only
        for key1 in param_dict_input.keys():
            if key1 not in param_dict.keys():
                raise ValueError(f'Key {key1} not allowed in input file')

            for key2 in param_dict_input[key1].keys():
                if key2 not in param_dict[key1].keys():
                    raise ValueError(f'Key {key2} not allowed in input file')

                # apply the same type as in the default dictionary
                default_type = type(param_dict[key1][key2])
                param_dict[key1][key2] = default_type(param_dict_input[key1][key2])

    # Print the parameters
    mpi_print('Running with the following parameters:', comm=comm)
    for key1 in param_dict.keys():
        mpi_print(f'  {key1}:', comm=comm)

        for key2 in param_dict[key1].keys():
            val = param_dict[key1][key2]
            val_type = str(type(val)).replace('<class \'', '').replace('\'>', '')
            mpi_print(f'      {key2}: {val} ({val_type})', comm=comm)

        mpi_print('', comm=comm)

