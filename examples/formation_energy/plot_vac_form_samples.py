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

# font size for x and y axis labels
plotparams['axes.labelsize'] = 22

plt.rcParams.update(plotparams)


def compute_formation_property(run_dict, sample, property_name_pure, property_name_vac=None):

    if property_name_vac is None:
        property_name_vac = property_name_pure

    Natom_pure = run_dict['Natom pure']
    Natom_vac = run_dict['Natom vac']

    s_str = f'sample_{sample}'
    prop_pure = np.array([run_dict[s_str][f'delta_{i}']['pure']['npt'][property_name_pure] for i in run_dict[s_str]['conv_idelta_list']])
    prop_vac = np.array([run_dict[s_str][f'delta_{i}']['vac']['npt'][property_name_vac] for i in run_dict[s_str]['conv_idelta_list']])

    print(f'{prop_pure = } {prop_vac = } {Natom_pure = } {Natom_vac = }')
    prop_formation = prop_vac - prop_pure * Natom_vac / Natom_pure

    return prop_formation


def plot_success_matrix(ax, run_dict):

    delta_array = run_dict['delta_array']
    # Get deltas for which the calculations were successful
    success_matrix = np.zeros((len(run_dict['sample_list']), len(run_dict['delta_array'])), dtype=bool)
    success_matrix[:, :] = False
    for isample, sample in enumerate(run_dict['sample_list']):
        s_str = f'sample_{sample}'
        for idelta in run_dict[s_str]['conv_idelta_list']:
            success_matrix[isample, idelta] = True

    cmap = plt.get_cmap('coolwarm')
    ax.imshow(success_matrix, cmap='coolwarm_r', aspect='auto')
    ax.set_xticks(range(len(delta_array)))
    ax.set_xticklabels([f'{delta:.0f}' for delta in delta_array])
    ax.set_xlabel('Perturbation magnitude $\delta$')
    ax.set_ylabel('Sample index')
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Legend: red for successful, blue for unsuccessful
    c1 = plt.Line2D([0], [0], color=cmap(0.0), lw=10)
    c2 = plt.Line2D([0], [0], color=cmap(1.0), lw=10)
    # Short line for the legend
    ax.legend([c1, c2], ['Converged', 'Unconverged'], loc='upper right')


def plot_lattice_constant(ax, run_dict, sample):

    ncell_x = run_dict['ncell_x']
    delta_array = run_dict['delta_array']
    ndelta = len(delta_array)

    s_str = f'sample_{sample}'

    idelta_en_vol_list = [i for i in range(ndelta) if run_dict[s_str][f'delta_{i}']['pure'] is not None]
    alat_pure_list = [run_dict[s_str][f'delta_{i}']['pure']['npt']['volume_true'] ** (1/3) / ncell_x for i in idelta_en_vol_list]

    ax.plot(delta_array[idelta_en_vol_list], alat_pure_list, lw=4.0, label='Bulk latt')

    ax.set_xlabel('Perturbation Magnitude $\delta$')
    ax.set_ylabel('Lattice constant ($\mathrm{\AA}$)')
    ax.set_ylim(3.02, 3.2)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))


def plot_energy_volume_deltas(ax, run_dict, sample, fsize_bar=12):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']
    ndelta = len(delta_array)
    idelta_en_vol_list = [i for i in range(ndelta) if run_dict[s_str][f'delta_{i}']['pure'] is not None]

    cmap = plt.get_cmap('coolwarm')
    epsilon_array_en_vol = run_dict['epsilon_array_en_vol']
    color_array = cmap(np.linspace(0, 1, len(epsilon_array_en_vol)))
    for idelta in idelta_en_vol_list:
        delta = delta_array[idelta]
        d_str = f'delta_{idelta}'
        energy_array_delta_pure = run_dict[s_str][d_str]['pure']['en_vol']['energy_array']

        if abs(delta) < 1e-6:
            color = 'black'
            lw = 4.0
        else:
            lw = 2.0
            color = color_array[idelta]

        ax.plot(100.0 * epsilon_array_en_vol, energy_array_delta_pure, c=color, lw=lw)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    norm = plt.Normalize(vmin=np.min(delta_array), vmax=np.max(delta_array))
    cmap = plt.get_cmap('coolwarm')
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation='vertical',format='%.0f')
    cbar.set_label('$\delta$')
    cbar.ax.tick_params(labelsize=fsize_bar)
    ax.set_xlabel('Strain Magnitude (%)', fontsize=fsize_bar)
    ax.set_ylabel('Energy (eV)', fontsize=fsize_bar)
    ax.tick_params(axis='both', which='major', labelsize=fsize_bar)


def main():

    # Read the pickle file: run_dict.pkl
    #pickle_filename = 'run_dict.pkl'
    pickle_filename = 'run_dict_ncellx_3_subset.pkl'
    with open(pickle_filename, 'rb') as f:
        run_dict = pickle.load(f)

    # Plot success_matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_success_matrix(ax, run_dict)

    #
    # Plotting physical data
    #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plt.subplots_adjust(wspace=0.2, hspace=0.01)

    sample = 1

    ncell_x = run_dict['ncell_x']
    delta_array = run_dict['delta_array']
    ndelta = len(delta_array)

    s_str = f'sample_{sample}'

    # Lattice constant
    plot_lattice_constant(axes[0, 0], run_dict, sample)
    axes[0, 0].set_xticklabels([])
    # Energy-volume as inset
    ax_inset = axes[0, 0].inset_axes([0.14, 0.14, 0.4, 0.4])
    plot_energy_volume_deltas(ax_inset, run_dict, sample)

    # Formation energies
    E_form_full_pred = compute_formation_property(run_dict, sample, 'energy_full_pred')
    E_form_hom_pred = compute_formation_property(run_dict, sample, 'energy_hom_pred')
    E_form_inhom_pred = compute_formation_property(run_dict, sample, 'energy_inhom_pred')
    E_form_true = compute_formation_property(run_dict, sample, 'energy_true')
    E_form_pred0 = compute_formation_property(run_dict, sample, 'energy_pred0')

    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    axes[1, 0].plot(delta_array_sample, E_form_pred0, label='No position change', marker='o', c='purple')
    axes[1, 0].plot(delta_array_sample, E_form_hom_pred, label='Homogeneous', marker='o', c='goldenrod')
    axes[1, 0].plot(delta_array_sample, E_form_inhom_pred, label='Inhomogeneous', marker='o', c='blue')
    axes[1, 0].plot(delta_array_sample, E_form_full_pred, label='Hom. + Inhom.', marker='o', c='red')
    axes[1, 0].plot(delta_array_sample, E_form_true, label='True', marker='o', c='black')

    axes[1, 0].set_xlabel('Perturbation Magnitude $\delta$')
    axes[1, 0].set_ylabel('Formation Energy (eV)')
    axes[1, 0].legend()

    plt.show()
    exit()

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