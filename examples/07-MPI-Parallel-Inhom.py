#!/usr/bin/env python3
"""
Example of a parallel MPI run for inhomogeneous implicit derivative calculation (requires mpi4py installed).

Run with MPI:
mpirun -n 8 ./07-MPI-Parallel-Inhom.py

OpenMP parallelization is also supported by LAMMPS. One can set the number of OpenMP threads as:
export OMP_NUM_THREADS=8

Parameters to change:
- ncell_x: Number of unit cells in each direction.
- method: Method to compute the implicit derivative. Options are 'sparse', 'energy', and 'dense'.
- sample: Sample number from the ensemble of perturbed potentials.
- delta: Perturbation strength.

"""

# Standard libraries
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Package imports
from lammps_implicit_der import SNAP
from lammps_implicit_der.systems import BCC_VACANCY
from lammps_implicit_der.tools import plot_tools, initialize_mpi, TimingGroup, mpi_print

plotparams = plot_tools.plotparams.copy()
plotparams['font.size'] = 16
plotparams['figure.subplot.wspace'] = 0.2
plotparams['axes.labelsize'] = 22
plt.rcParams.update(plotparams)


def main():

    # Initialize MPI
    comm, rank = initialize_mpi()

    # Initialize timing
    trun = TimingGroup('vac_W_perturb')
    trun.add('total', level=2).start()

    # Lattice parameter
    alat = 3.16316

    # Number of unit cells in each direction
    ncell_x = 3

    # Not-perturbed system
    vac0 = BCC_VACANCY(alat=alat, ncell_x=ncell_x, del_coord=[0.0, 0.0, 0.0],
                       snapcoeff_filename='W_REF.snapcoeff',
                       minimize=True, fix_box_relax=False,
                       logname='vac0.log', verbose=True, comm=comm)

    Theta0 = vac0.pot.Theta_dict['W']['Theta'].copy()
    X_coord0 = vac0.X_coord.copy()

    # Select the method to compute the implicit derivative. Options are 'sparse', 'energy', and 'dense'.
    method = 'dense'

    # Some implicit derivative options (can be done differently)
    method_kwargs = {
        'sparse':
            {
                'adaptive_alpha': True,
                'alpha0': 0.01,
                'maxiter': 100,
                'atol': 1e-4,
            },
        'energy':
            {
                'adaptive_alpha': True,
                'alpha0': 0.01,
                'ftol': 1e-6,
                'min_style': 'fire', # options: 'fire', 'cg', 'sd'
            },
        'dense': {},
    }

    # Compute the implicit derivative
    with trun.add(f'impl. der. {method}', level=2):
        dX_dTheta = vac0.implicit_derivative(method=method, **method_kwargs[method])

    # Read the potential ensemble
    with open('Theta_ens.pkl', 'rb') as f:
        Theta_ens = pickle.load(f)

    # Select an arbitrary sample from the ensemble and perturb the potential with delta=40.0
    delta = 10.0
    sample = 1
    Theta_perturb = Theta_ens['Theta_mean'] + delta * (Theta_ens['Theta_ens_list'][sample] - Theta_ens['Theta_mean'])

    # Potential perturbation
    dTheta = Theta_perturb - Theta0

    # Predict the position change
    dX_pred = dTheta @ dX_dTheta

    # Save the perturbed Theta to a file
    pot = SNAP.from_files(snapcoeff_filename='W_REF.snapcoeff', snapparam_filename='W_REF.snapparam', comm=comm)
    pot.Theta_dict['W']['Theta'] = Theta_perturb.copy()
    pot.to_files(snapcoeff_filename='W_perturb_new.snapcoeff', snapparam_filename='W_perturb_new.snapparam',
                 overwrite=True, verbose=False)

    # This step is required for the true positions change from the LAMMPS minimization
    # Since the potential is written to the current folder, we specify  data_path='.'
    vac_perturb = BCC_VACANCY(alat=alat, ncell_x=ncell_x, del_coord=[0.0, 0.0, 0.0],
                              data_path='.',
                              snapcoeff_filename='W_perturb_new.snapcoeff',
                              minimize=True, fix_box_relax=False,
                              logname='vac0.log', verbose=False, comm=comm)

    # True minimized positions
    X_true = vac_perturb.X_coord.copy()
    # True difference in positions with minimum image convention applied
    dX_true = vac0.minimum_image(X_true - X_coord0)

    # Stop timner and print timings
    trun.timings['total'].stop()
    mpi_print(trun, comm=comm)

    if comm is None or rank == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.11)
        ax.plot(dX_true, dX_pred, ls='', marker='o', label='Predicted')
        ax.plot([dX_true.min(), dX_true.max()], [dX_true.min(), dX_true.max()], ls='--', color='gray', label='Ideal')
        ax.set_xlabel(r'True Position Change ($\mathrm{\AA}$)')
        ax.set_ylabel(r'Predicted Position Change ($\mathrm{\AA}$)')
        ax.legend()
        plt.show()


if __name__ == '__main__':
    main()
