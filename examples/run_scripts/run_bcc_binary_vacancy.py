#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

# local imports
from lammps_implicit_der.tools import mpi_print, \
                                      get_default_data_path, \
                                      initialize_mpi, \
                                      error_tools, \
                                      TimingGroup

from lammps_implicit_der.systems import BccBinaryVacancy


def run_minimization(method,
                     step,
                     maxiter,
                     comm=None,
                     der_alpha=0.5,
                     error_tol=1e-5,
                     der_adaptive_alpha=True,
                     pickle_name=None,
                     minimize_at_iters=True,
                     apply_hard_constraints=False):

    trun = TimingGroup('Minimize Vacancy')
    trun.add('total', level=2).start()

    data_path = get_default_data_path()

    # Both non-perturbed and perturbed LAMMPS objects must be
    # initialized from the same datafile to match the atom indexing.
    # Important for MPI runs.

    non_perturb_data = os.path.join(data_path, "bcc_binary_vacancy.data")

    # If datafile is None, the system is initialized from scratch
    datafile = non_perturb_data if os.path.exists(non_perturb_data) else None

    binvac_non_perturb = BccBinaryVacancy(datafile=datafile,
                                          data_path=data_path,
                                          snapcoeff_filename='NiMo.snapcoeff',
                                          num_cells=3,
                                          verbose=True, comm=comm, logname='binvac.log')

    # Save the datafile if it was initialized from scratch
    if datafile is None:
        binvac_non_perturb.write_data(non_perturb_data)

    binvac_perturb = BccBinaryVacancy(datafile=non_perturb_data,
                                      data_path=data_path,
                                      snapcoeff_filename='NiMo_Mo_perturbed.snapcoeff',
                                      comm=comm, logname='binvac_perturbed.log')

    X_target = binvac_perturb.X_coord.copy()

    #exit()

    binvac_final, error_array, \
    min_X, min_Theta = error_tools.minimize_loss(
                                binvac_non_perturb,
                                X_target,
                                comm=comm,
                                step=step,
                                maxiter=maxiter,
                                error_tol=error_tol,
                                der_ftol=1e-4,
                                der_method=method,
                                der_adaptive_alpha=der_adaptive_alpha,
                                der_alpha=der_alpha,
                                der_maxiter=500,
                                pickle_name=pickle_name,
                                verbosity=3,
                                minimize_at_iters=minimize_at_iters,
                                apply_hard_constraints=apply_hard_constraints,
                                binary=True
                                )

    binvac_final.write_data('binvac_final.data')

    trun.timings['total'].stop()
    mpi_print(trun, comm=comm)
    mpi_print(binvac_non_perturb.timings, comm=comm)


def main():

    comm, rank = initialize_mpi()

    method = 'inverse'
    #method = 'energy'
    #method = 'sparse'

    step = 1e-4
    maxiter = 100

    error_tol = 1e-1

    der_alpha = 0.1
    der_adaptive_alpha = True

    minimize_at_iters = True

    # Hard constraints are not implemented for NiMo
    apply_hard_constraints = False

    if method == 'sparse':
        der_alpha = 1e-4
        der_adaptive_alpha = False
    elif method == 'energy':
        der_alpha = 0.5
        der_adaptive_alpha = True

    run_minimization(method,
                     comm=comm,
                     step=step,
                     maxiter=maxiter,
                     error_tol=error_tol,
                     der_alpha=der_alpha,
                     der_adaptive_alpha=der_adaptive_alpha,
                     minimize_at_iters=minimize_at_iters,
                     apply_hard_constraints=apply_hard_constraints,
                     pickle_name=None)


if __name__ == '__main__':
    main()
