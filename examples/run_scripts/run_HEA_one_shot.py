#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lammps_implicit_der.systems import BccBinary, get_bcc_alloy_A_delta_B
from lammps_implicit_der.tools import mpi_print, initialize_mpi, TimingGroup


def main():

    # Read method, delta, num_cells as command line arguments
    method = sys.argv[1]
    delta = float(sys.argv[2])
    num_cells = int(sys.argv[3])

    trun = TimingGroup(f'HEA {method} delta={delta} num_cells={num_cells}')
    trun.add('total', level=2).start()

    comm, rank = initialize_mpi()

    # Not-perturbed random alloy
    bcc_alloy_Ni_Mo = BccBinary(datafile=None,
                                comm=comm,
                                snapcoeff_filename='NiMo.snapcoeff',
                                num_cells=num_cells,
                                specie_B_concentration=0.5,
                                verbose=True,
                                minimize=True)

    # Perturbed random alloy
    bcc_alloy_Ni_Mo_pert = get_bcc_alloy_A_delta_B(delta=delta,
                                                   comm=comm,
                                                   num_cells=num_cells,
                                                   minimize=True,
                                                   datafile=None,
                                                   specie_B_concentration=0.5)

    # Compute the implicit derivative with respect to Mo parameters
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

    dX_dTheta = bcc_alloy_Ni_Mo.implicit_derivative(method=method, **kwargs)

    # Perturbed parameters
    Theta_Mo_pert = bcc_alloy_Ni_Mo_pert.pot.Theta_dict['Mo']['Theta'].copy()
    Theta_Mo = bcc_alloy_Ni_Mo.pot.Theta_dict['Mo']['Theta'].copy()
    dTheta = Theta_Mo_pert - Theta_Mo

    # Predicted coordinate perturbation
    dX_pred = dTheta @ dX_dTheta
    X_pred = bcc_alloy_Ni_Mo.minimum_image(bcc_alloy_Ni_Mo.X_coord + dX_pred)

    # True change
    X_true = bcc_alloy_Ni_Mo_pert.X_coord.copy()
    dX_true = bcc_alloy_Ni_Mo.minimum_image(X_true - bcc_alloy_Ni_Mo.X_coord)

    trun.timings['total'].stop()

    output_dict = {
        'num_cells': num_cells,
        'Natom': bcc_alloy_Ni_Mo.Natom,
        'delta': delta,
        'dTheta': dTheta,
        'method': method,
        'dX_dTheta': dX_dTheta,
        'Theta_Ni': bcc_alloy_Ni_Mo.pot.Theta_dict['Ni']['Theta'].copy(),
        'Theta_Mo': bcc_alloy_Ni_Mo.pot.Theta_dict['Mo']['Theta'].copy(),
        'Theta_Mo_pert': Theta_Mo_pert,
        'dX_dTheta': dX_dTheta,
        'dX_pred': dX_pred,
        'dX_true': dX_true,
        'X_pred': X_pred,
        'X_true': X_true,
        'X0': bcc_alloy_Ni_Mo.X_coord.copy(),
        'species': bcc_alloy_Ni_Mo.species.copy(),
        'timings_system': trun.to_dict(),
        'timings_run': bcc_alloy_Ni_Mo.timings.to_dict()
    }

    # Save to pickle
    if rank == 0:
        output_filename = f'HEA_NiMo_{method}_{delta:.4f}_{num_cells:03d}.pkl'
        with open(output_filename, 'wb') as f:
            pickle.dump(output_dict, f)

    mpi_print(trun, comm=comm)
    mpi_print(bcc_alloy_Ni_Mo.timings, comm=comm)

    plot = True
    #plot = False

    if plot and rank == 0:
        plot_results(dX_true, dX_pred, method, delta, num_cells)


def plot_results(dX_true, dX_pred, method, delta, num_cells):

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    plt.subplots_adjust(left=0.086, right=0.957, bottom=0.136)

    fsize = 20
    ax[0].plot(dX_true, dX_pred, 'o', label=method, color='tab:blue')
    ax[0].set_xlabel('True Position Changes', fontsize=fsize)
    ax[0].set_ylabel('Predicted Position Changes', fontsize=fsize)
    ax[0].set_title(f'method: {method}; number of cells: {num_cells}', fontsize=fsize)
    ax[0].plot([dX_true.min(), dX_true.max()], [dX_true.min(), dX_true.max()], ls="--", c=".3")

    xmin, xmax = -0.05, 0.05
    bins = 50
    hist, bin_edges = np.histogram(dX_true[ (dX_true > xmin) & (dX_true < xmax) ], bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax[1].plot(bin_centers, hist, label='true', color='black', lw=3)

    hist, bin_edges = np.histogram(dX_pred[ (dX_pred > xmin) & (dX_pred < xmax) ], bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax[1].plot(bin_centers, hist, label=method, color='tab:blue', lw=3)
    ax[1].set_xlabel('Position Change', fontsize=fsize)
    ax[1].set_ylabel('Frequency', fontsize=fsize)
    ax[1].set_xlim(xmin, xmax)

    ax[1].legend()

    fig.savefig(f'HEA_NiMo_{method}_{delta:.4f}_{num_cells:03d}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
