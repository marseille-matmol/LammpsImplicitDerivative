#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lammps_implicit_der.systems import VacW, BccVacancyConcentration, BCC_VACANCY
from lammps_implicit_der.tools import mpi_print, initialize_mpi, TimingGroup, plot_tools, error_tools, get_size


def main():

    trun = TimingGroup('vac_W_perturb')
    trun.add('total', level=2).start()

    comm, rank = initialize_mpi()

    alat = 3.1855

    fix_box_relax = False
    #fix_box_relax = True

    # Command line arguments
    ncell_x = int(sys.argv[1])
    method = sys.argv[2]
    #sample = int(sys.argv[3])

    kwargs = {}
    if method == 'sparse':
        kwargs = {
            'adaptive_alpha': True,
            'alpha': 0.01,
            'maxiter': 100,
        }
    elif method == 'energy':
        kwargs = {
            'adaptive_alpha': True,
            'alpha': 0.01,
        }

    # Non-perturbed system
    vac0 = BCC_VACANCY(snapcoeff_filename='W.snapcoeff',
                      snapparam_filename='W.snapparam',
                      data_path='.',
                      comm=comm,
                      minimize=True,
                      ncell_x=ncell_x,
                      alat=alat,
                      logname='vac0.log',
                      fix_box_relax=fix_box_relax,
                      datafile=None)

    Theta0 = vac0.pot.Theta_dict['W']['Theta'].copy()
    X0 = vac0.X_coord.copy()

    #print_coords = True
    print_coords = False
    if print_coords:
        if rank == 0:
            vac0.write_xyz_file(filename=f"vac0_{ncell_x}.xyz", verbose=True)

    energy0 = vac0.energy
    #stress0 = vac0.stress
    # Convert to MPa
    #stress0 *= 0.1 / (ncell_x * alat)**3

    # Implicit derivative
    dX_dTheta = vac0.implicit_derivative(method=method, **kwargs)

    #for sample in range(1, 101):
    for sample in range(1, 2):

        comm.Barrier()

        mpi_print('', comm=comm)
        mpi_print('*'*30+f'Running for sample {sample}'+'*'*30, comm=comm)

        # Perturbed system
        vac_perturb = BCC_VACANCY(snapcoeff_filename=f'W_perturb_{sample}.snapcoeff',
                                 snapparam_filename='W.snapparam',
                                 data_path='.',
                                 comm=comm,
                                 minimize=True,
                                 ncell_x=ncell_x,
                                 alat=alat,
                                 logname='vac_perturb.log',
                                 fix_box_relax=fix_box_relax,
                                 datafile=None)

        X_true = vac_perturb.X_coord
        dX_true = vac0.minimum_image(X_true - X0)
        Theta_perturb = vac_perturb.pot.Theta_dict['W']['Theta'].copy()
        dTheta = Theta_perturb - Theta0

        dX_pred = dTheta @ dX_dTheta
        X_pred = vac0.minimum_image(X0 + dX_pred)

        energy_true = vac_perturb.energy
        #stress_true = vac_perturb.stress
        # Convert to MPa
        #stress_true *= 0.1 / (ncell_x * alat)**3

        # Compute the predicted energy stress: send coordinates to vac_perturb
        # and compute the energy and stress
        #vac_perturb.lmp.scatter("x", 1, 3, np.ctypeslib.as_ctypes(vac_perturb._X_coord))

        #energy_pred = vac_perturb.energy(compute_id='PredictedEnergy')
        #stress_pred = vac_perturb.stress
        # Convert to MPa
        #stress_pred *= 0.1 / (ncell_x * alat)**3

        output_dict = {
            'Natom': vac0.Natom,
            'energy0': energy0,
            #'stress0': stress0,
            'energy_true': energy_true,
            #'stress_true': stress_true,
            #'energy_pred': energy_pred,
            #'stress_pred': stress_pred,
            'sample': sample,
            'dTheta': dTheta,
            'method': method,
            #'dX_dTheta': dX_dTheta,
            'Theta0': Theta0,
            'Theta_perturb': Theta_perturb,
            'dX_pred': dX_pred,
            'dX_true': dX_true,
            'X_pred': X_pred,
            'X_true': X_true,
            'X0': X0,
            'timings_system': trun.to_dict(),
            'timings_run': vac0.timings.to_dict(),
        }

        #plot = False
        plot = True

        mpi_print(f'{energy0=:.6f} {energy_true=:.6f}', comm=comm)

        if rank == 0 and plot:

            hist_true, bin_edges_true = np.histogram(dX_true, bins=50, density=True)
            hist_pred, bin_edges_pred = np.histogram(dX_pred, bins=50, density=True)

            fig, axes = plt.subplots(1, 3, figsize=(8, 5))
            plot_tools.plot_coords(axes, X0.reshape(-1, 3), c='gray', s=32, label='original')
            plot_tools.plot_coords(axes, X_true.reshape(-1, 3), c='tab:red', s=35, label='perturbed')
            plot_tools.plot_coords(axes, X_pred.reshape(-1, 3), s=45, label='predicted', facecolors='none', edgecolors='black', marker='o')
            plt.tight_layout()

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(bin_edges_true[:-1], hist_true, label='true', color='tab:red', lw=3)
            ax.plot(bin_edges_pred[:-1], hist_pred, label=f'predicted {method}', color='black', lw=3)
            ax.set_yscale('log')
            ax.set_xlabel('dX')

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(dX_true, dX_pred, ls='', marker='o')
            ax.plot([dX_true.min(), dX_true.max()], [dX_true.min(), dX_true.max()], ls='-', color='black')
            ax.set_xlabel('True Position Changes')
            ax.set_ylabel('Predicted Position Changes')
            plt.show()

        if rank == 0:
            output_filename = f'vac_W_{ncell_x:03d}_{method}_{sample:03d}.pkl'
            with open(output_filename, 'wb') as f:
                pickle.dump(output_dict, f)

    trun.timings['total'].stop()
    mpi_print(trun, comm=comm)


if __name__ == '__main__':
    main()
