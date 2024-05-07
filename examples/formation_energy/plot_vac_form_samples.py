#!/usr/bin/env python
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.interpolate import CubicSpline

# Local imports
from lammps_implicit_der import LammpsImplicitDer, SNAP
from lammps_implicit_der.tools import mpi_print, initialize_mpi, TimingGroup, plot_tools, \
                                      compute_energy_volume, create_perturbed_system, run_npt_implicit_derivative
from lammps_implicit_der.tools.error_tools import coord_error
from lammps_implicit_der.systems import BccVacancy, Bcc
#from lammps_implicit_der.tools.error_tools import coord_error

from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

plotparams = plot_tools.plotparams.copy()
plotparams['figure.figsize'] = (9, 6)
plotparams['font.size'] = 16
plotparams['figure.subplot.wspace'] = 0.2
plt.rcParams.update(plotparams)


def main():

    # Read the pickle file: run_dict.pkl
    pickle_filename = 'run_dict.pkl'
    with open(pickle_filename, 'rb') as f:
        run_dict = pickle.load(f)

    # Get deltas for which the calculations were successful
    idelta_done_dict = {}
    success_matrix = np.zeros((len(run_dict['sample_list']), len(run_dict['delta_array'])), dtype=bool)
    print(f'{success_matrix.shape = }')
    for isample, sample in enumerate(run_dict['sample_list']):
        s_str = f'sample_{sample}'

        delta_list = []
        for i, delta in enumerate(run_dict['delta_array']):
            d_str = f'delta_{i}'
            if run_dict[s_str][d_str]['pure']['npt'] is not None and run_dict[s_str][d_str]['vac']['npt'] is not None:
                delta_list.append(delta)
                success_matrix[isample, i] = True

        idelta_done_dict[sample] = delta_list

    print(success_matrix)

    # Plot success_matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(success_matrix, cmap='coolwarm', aspect='auto')
    # xticklabels from delta_array
    ax.set_xticks(range(len(run_dict['delta_array'])))
    ax.set_xticklabels([f'{delta:.2f}' for delta in run_dict['delta_array']])
    ax.set_xlabel('Perturbation magnitude $\delta$')
    ax.set_ylabel('Sample index')
    # Integer yticks
    ax.yaxis.set_major_locator(MultipleLocator(1))
    # Black grid around the cells
    #ax.grid(color='black', linestyle='-', linewidth=1)
    plt.show()

    exit()

    sample = 1

    ncell_x = run_dict['ncell_x']
    delta_array = run_dict['delta_array']
    ndelta = len(delta_array)

    s_str = f'sample_{sample}'
    alat_pure_array = [run_dict[s_str][f'delta_{d}']['pure']['npt']['volume_true'] ** (1/3) / ncell_x for d in range(ndelta)]
    print(run_dict[s_str].keys())
    print(alat_pure_array)
    exit()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fsize = 22
    plt.subplots_adjust(wspace=0.2, hspace=0.01)
    # Lattice constant
    axes[0, 0].plot(delta_array,
                [bcc_pure_list[i]['volume_true']**(1/3) / ncell_x for i in range(ndelta)], lw=4.0, label='Bulk latt')

    #axes[0, 0].plot(delta_array,
    #            [bcc_vac_list[i]['volume_true']**(1/3) / ncell_x for i in range(ndelta)], lw=4.0, label='Vacancy')

    # Create an inset in axes[0, 0] with energy-volume curves
    ax_inset = axes[0, 0].inset_axes([0.14, 0.14, 0.4, 0.4])
    cmap = plt.get_cmap('coolwarm')
    color_array = cmap(np.linspace(0, 1, len(epsilon_array_en_vol)))
    epsilon_array_inset = en_vol_delta_pure_dict['epsilon_array']
    for i, delta in enumerate(delta_array_en_vol):
        energy_array_delta_pure = en_vol_delta_pure_dict[f'delta_{delta:06.2f}']['energy_array']
        energy_array_delta_vac = en_vol_delta_vac_dict[f'delta_{delta:06.2f}']['energy_array']

        if abs(delta) < 1e-6:
            color = 'black'
            lw = 4.0
        else:
            lw = 2.0
            color = color_array[i]

        ax_inset.plot(100.0 * epsilon_array_inset, energy_array_delta_pure, c=color, lw=lw)

    fsize_inset = 12
    # Colorbar for inset
    #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax_inset, orientation='vertical')
    divider = make_axes_locatable(ax_inset)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    norm = plt.Normalize(vmin=delta_array_en_vol[0], vmax=delta_array_en_vol[-1])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation='vertical',format='%.0f')
    cbar.set_label('$\delta$')
    # Fontsize for colorbar
    cbar.ax.tick_params(labelsize=fsize_inset)

    ax_inset.set_xlabel('Strain Magnitude (%)', fontsize=fsize_inset)
    ax_inset.set_ylabel('Energy (eV)', fontsize=fsize_inset)

    # Ticklabel size for inset
    ax_inset.tick_params(axis='both', which='major', labelsize=fsize_inset)

    #axes[0, 0].set_xlabel('Perturbation Magnitude $\delta$')
    # remove tick labels
    axes[0, 0].set_xticklabels([])
    axes[0, 0].set_ylabel('Lattice constant ($\mathrm{\AA}$)', fontsize=fsize)
    axes[0, 0].set_ylim(3.02, 3.2)
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(0.05))

    # Formation energies
    Natom_pure = bcc_pure.Natom
    Natom_vac = bcc_vac.Natom
    E_form_full_pred = compute_formation_property(Natom_vac, Natom_pure, bcc_vac_list, bcc_pure_list, 'energy_full_pred')
    E_form_hom_pred = compute_formation_property(Natom_vac, Natom_pure, bcc_vac_list, bcc_pure_list, 'energy_hom_pred')
    E_form_inhom_pred = compute_formation_property(Natom_vac, Natom_pure, bcc_vac_list, bcc_pure_list, 'energy_inhom_pred')
    E_form_true = compute_formation_property(Natom_vac, Natom_pure, bcc_vac_list, bcc_pure_list, 'energy_true')
    E_form_pred0 = compute_formation_property(Natom_vac, Natom_pure, bcc_vac_list, bcc_pure_list, 'energy_pred0')

    axes[1, 0].plot(delta_array, E_form_pred0, label='No position change', marker='o', c='purple')
    axes[1, 0].plot(delta_array, E_form_hom_pred, label='Homogeneous', marker='o', c='goldenrod')
    axes[1, 0].plot(delta_array, E_form_inhom_pred, label='Inhomogeneous', marker='o', c='blue')
    axes[1, 0].plot(delta_array, E_form_full_pred, label='Hom. + Inhom.', marker='o', c='red')
    axes[1, 0].plot(delta_array, E_form_true, label='True', marker='o', c='black')


    axes[1, 0].set_xlabel('Perturbation Magnitude $\delta$', fontsize=fsize)
    axes[1, 0].set_ylabel('Formation Energy (eV)', fontsize=fsize)

    # Absolute volume
    axes[0, 1].plot(delta_array, [bcc_pure_list[i]['volume_true'] for i in range(ndelta)], label='Pure True', marker='s', c='dimgray')
    axes[0, 1].plot(delta_array, [bcc_pure_list[i]['volume_pred'] for i in range(ndelta)], label='Pure Pred. from dP/dV', marker='s')
    axes[0, 1].plot(delta_array, [bcc_pure_list[i]['volume_pred_DT'] for i in range(ndelta)], label='Pure Pred. from dP/dV', marker='s', ls='--')

    axes[0, 1].plot(delta_array,[bcc_vac_list[i]['volume_true'] for i in range(ndelta)], label='Vac. True', marker='o', c='black')
    axes[0, 1].plot(delta_array, [bcc_vac_list[i]['volume_pred'] for i in range(ndelta)], label='Vac. Pred. from dP/dV', marker='o')
    axes[0, 1].plot(delta_array, [bcc_vac_list[i]['volume_pred_DT'] for i in range(ndelta)], label='Vac. Pred. from dP/dV', marker='o', ls='--')

    axes[0, 1].set_xticklabels([])
    axes[0, 1].set_ylabel('Volume ($\mathrm{\AA}^3$)', fontsize=fsize)

    # Formation volume
    vol_form_full_pred = compute_formation_property(Natom_vac, Natom_pure, bcc_vac_list, bcc_pure_list, 'volume_pred')
    vol_form_true = compute_formation_property(Natom_vac, Natom_pure, bcc_vac_list, bcc_pure_list, 'volume_true')

    axes[1, 1].plot(delta_array, vol_form_true, label='True', marker='o', c='black')
    axes[1, 1].plot(delta_array, vol_form_full_pred, label='Predicted', marker='o')

    axes[1, 1].set_xlabel('Perturbation Magnitude $\delta$', fontsize=fsize)
    axes[1, 1].set_ylabel('Formation Volume ($\mathrm{\AA}^3$)', fontsize=fsize)

    for ax in axes.flatten():
        ax.grid()
        ax.legend()
        #ax.set_xlim(delta_min, delta_max)

    axes[0,0].legend(loc='upper right')
    fig.savefig('formation_energy_2x2.pdf')

    plt.show()


if __name__ == '__main__':
    main()