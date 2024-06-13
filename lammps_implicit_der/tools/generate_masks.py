#!/usr/bin/env python3
"""
Generate the Hessian masks for the implicit derivative calculations using the dense method.
"""

import numpy as np
import matplotlib.pyplot as plt

from .utils import mpi_print


def generate_mask_dX(dX, threshold=0.12, comm=None):

    if dX.ndim != 1:
        raise ValueError('dX must be a 1D flat array.')

    dX_3D = dX.reshape(-1, 3)
    dX_norm = np.linalg.norm(dX_3D, axis=1)

    hess_mask_3D = dX_norm > np.max(dX_norm) * threshold

    hess_mask = np.outer(hess_mask_3D, np.ones(3)).flatten().astype(bool)

    nmask = np.sum(hess_mask)
    ntotal = len(hess_mask)
    natom_mask = nmask // 3
    mpi_print(f'>>>dX mask. Number of elements in the mask: {nmask} out of {ntotal} ({nmask/ntotal:.1%})', comm=comm)
    mpi_print(f'>>>Number of atoms in the mask: {natom_mask}', comm=comm)

    return hess_mask, hess_mask_3D


def generate_mask_radius(X_coord, radius=5.0, center_coord=np.array([0.0, 0.0, 0.0]), center_specie=None, species=None, comm=None):

    if X_coord.ndim != 1:
        raise ValueError('X_coord must be a 1D flat array.')

    if center_specie is not None and species is None:
        raise ValueError('If center_species is not None, species must be provided.')

    X_3D = X_coord.reshape(-1, 3)

    # check shapes
    if (species is not None) and (len(X_3D) != len(species)):
        mpi_print(f'{len(X_coord)=} {len(X_3D)=}', comm=comm)
        raise ValueError('X_coord and species must have the same length.')

    if center_specie is not None:
        center_coord = X_3D[species == center_specie]

    hess_mask_3D = np.linalg.norm(X_3D - center_coord, axis=1) < radius
    hess_mask = np.outer(hess_mask_3D, np.ones(3)).flatten().astype(bool)

    nmask = np.sum(hess_mask)
    ntotal = len(hess_mask)
    natom_mask = nmask // 3

    mpi_print(f'>>>Radius mask. Center of the mask: {center_coord}, with radius of {radius} A', comm=comm)
    mpi_print(f'>>>Number of elements in the mask: {nmask} out of {ntotal} ({nmask/ntotal:.1%})', comm=comm)
    mpi_print(f'>>>Number of atoms in the mask: {natom_mask}', comm=comm)

    return hess_mask, hess_mask_3D


def plot_mask(X_coord, hess_mask_3D):

    if X_coord.ndim != 1:
        raise ValueError('X_coord must be a 1D flat array.')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    X_3D = X_coord.reshape(-1, 3)

    ax.plot(X_3D[:, 0], X_3D[:, 1], ls='', marker='o', color='tab:blue')
    ax.plot(X_3D[hess_mask_3D, 0], X_3D[hess_mask_3D, 1], ls='', marker='o', color='red', ms=10)

    fsize = 16
    ax.set_aspect('equal')
    ax.set_xlabel(r'$X$ ($\mathrm{\AA}$)', fontsize=fsize)
    ax.set_ylabel(r'$Y$ ($\mathrm{\AA}$)', fontsize=fsize)

    plt.show()