#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

# local imports
from lammps_implicit_der.tools import initialize_mpi, mpi_print, error_tools, TimingGroup, minimize_loss
from lammps_implicit_der.systems import DisloWBe


def run_minimization(
                     # Implicit derivative parameters
                     der_method,
                     der_min_style,
                     der_adaptive_alpha,
                     der_alpha,
                     der_ftol,
                     der_maxiter,
                     apply_hess_mask,
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

    datafile_path_start = 'relaxed_WBe_screw.lammps-data'
    datafile_path_target = 'target_WBe_screw.lammps-data'

    mpi_print('Dislo start initial relaxation...', comm=comm)
    with trun.add('start init'):
        dislo_start = DisloWBe(snapcoeff_filename='WBe-NEW.snapcoeff',
                               datafile=datafile_path_start,
                               logname='dislo_start.log',
                               minimize=True,
                               comm=comm,
                               verbose=True)

    mpi_print('Dislo target initialization (no relaxation)...', comm=comm)
    with trun.add('target init'):
        dislo_target = DisloWBe(snapcoeff_filename='WBe-NEW.snapcoeff',
                                datafile=datafile_path_target,
                                logname='dislo_target.log',
                                minimize=False,
                                comm=comm,
                                verbose=True)

    X_target = dislo_target.X_coord

    X_start = dislo_start.X_coord

    dX = dislo_start.minimum_image(X_target - X_start)

    X_target = X_start.copy() + dX

    if apply_hess_mask:

        #mask_algo = 'dX_norm'
        mask_algo = 'radius'

        if mask_algo == 'dX_norm':

            dX_3D = dX.reshape(-1, 3)
            dX_norm = np.linalg.norm(dX_3D, axis=1)

            threshold = 0.12
            #threshold = 0.05
            der_hess_mask_3D = dX_norm > np.max(dX_norm) * threshold

        elif mask_algo == 'radius':

            X_target_3D = X_target.reshape(-1, 3)

            # Define the mask
            radius_threshold = 5.0

            center_specie = 2

            center = X_target_3D[dislo_target.species == center_specie]

            mpi_print(center, comm=comm)

            der_hess_mask_3D = np.linalg.norm(X_target_3D - center, axis=1) < radius_threshold

        Natom_mask = np.sum(der_hess_mask_3D)
        mpi_print(f'Number of atoms in the mask: {Natom_mask}', comm=comm)

        der_hess_mask = np.outer(der_hess_mask_3D, np.ones(3)).flatten().astype(bool)

        plot = True
        #plot = False
        if plot and (comm is None or comm.Get_rank() == 0):
            # Plot x and y of the entire system and in red, the atoms in the mask
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            X_3D = X_target.reshape(-1, 3)

            ax.plot(X_3D[:, 0], X_3D[:, 1], ls='', marker='o', color='tab:blue')
            ax.plot(X_3D[der_hess_mask_3D, 0], X_3D[der_hess_mask_3D, 1], ls='', marker='o', color='red', ms=10)

            fsize = 16
            ax.set_aspect('equal')
            ax.set_xlabel(r'$X$ ($\mathrm{\AA}$)', fontsize=fsize)
            ax.set_ylabel(r'$Y$ ($\mathrm{\AA}$)', fontsize=fsize)

            plt.show()

        #mpi_print(np.where(der_hess_mask)[0], comm=comm)

    else:
        der_hess_mask = None

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
                                    der_hess_mask=der_hess_mask,
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
    #der_method = 'dense'  # 'sparse' # 'inverse'
    der_method = 'energy'  # 'sparse' # 'inverse'
    der_adaptive_alpha = True

    der_ftol = 1e-6
    der_maxiter = 500
    #der_min_style = 'cg'
    der_min_style = 'fire'
    der_alpha = 1e-2

    #apply_hess_mask = True
    apply_hess_mask = False

    #der_min_style = 'fire'
    #der_alpha = 0.5

    run_minimization(
                     # Implicit derivative parameters
                     der_method,
                     der_min_style=der_min_style,
                     der_adaptive_alpha=der_adaptive_alpha,
                     der_alpha=der_alpha,
                     der_ftol=der_ftol,
                     der_maxiter=der_maxiter,
                     apply_hess_mask=apply_hess_mask,
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
