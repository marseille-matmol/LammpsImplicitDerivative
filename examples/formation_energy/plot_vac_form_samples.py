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

    #cmap = plt.get_cmap('coolwarm')
    #cmap = plt.get_cmap('Dark2_r')
    cmap = plt.get_cmap('Accent_r')
    ax.imshow(success_matrix, cmap=cmap, aspect='auto')
    ax.set_xticks(2*np.array(range(len(delta_array[::2]))))
    ax.set_xticklabels([f'{delta:.0f}' for delta in delta_array[::2]])
    ax.set_xlabel('Perturbation magnitude $\delta$')
    ax.set_ylabel('Sample index')
    #ax.yaxis.set_major_locator(MultipleLocator(1))

    # Legend: red for successful, blue for unsuccessful
    c1 = plt.Line2D([0], [0], color=cmap(1.0), lw=10)
    c2 = plt.Line2D([0], [0], color=cmap(0.0), lw=10)
    # Short line for the legend
    ax.legend([c1, c2], ['Converged', 'Unconverged'], loc='upper right', handlelength=0.3)


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
    ax.set_ylim(3.05, 3.22)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))


def plot_energy_volume_deltas(ax, run_dict, sample, cmap_name='coolwarm', label_pad=-10, fsize_bar=None, second_xaxis=False):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']
    ndelta = len(delta_array)
    #idelta_en_vol_list = [i for i in range(ndelta) if run_dict[s_str][f'delta_{i}']['pure'] is not None]
    idelta_conv_list = run_dict[s_str]['conv_idelta_list']

    cmap = plt.get_cmap(cmap_name)
    epsilon_array_en_vol = run_dict['epsilon_array_en_vol']
    color_array = cmap(np.linspace(0, 1, len(delta_array)))

    #for i, idelta in enumerate(idelta_en_vol_list):
    for i, idelta in enumerate(idelta_conv_list):
        delta = delta_array[idelta]
        d_str = f'delta_{idelta}'
        energy_array_delta_pure = run_dict[s_str][d_str]['pure']['en_vol']['energy_array']

        if abs(delta) < 1e-6:
            color = 'black'
            lw = 4.0
        else:
            lw = 2.0
            #color = color_array[i]
            color = color_array[idelta]

        ax.plot(100.0 * epsilon_array_en_vol, energy_array_delta_pure, c=color, lw=lw)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    norm = plt.Normalize(vmin=np.min(delta_array), vmax=np.max(delta_array))
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation='vertical', format='%.0f', shrink=0.2)
    cbar.set_label('$\delta$')

    if fsize_bar is not None:
        cbar.ax.tick_params(labelsize=fsize_bar)
        cbar.set_label('$\delta$')
        ax.set_xlabel('Strain Magnitude (%)', fontsize=fsize_bar)
        ax.set_ylabel('Energy (eV)', fontsize=fsize_bar)
        ax.tick_params(axis='both', which='major', labelsize=fsize_bar)
    else:
        ax.set_xlabel('Strain Magnitude (%)')
        ax.set_ylabel('Energy (eV)')
        cbar.set_label('Perturbation Magnitude $\delta$', labelpad=label_pad)

    if second_xaxis:
        ax2 = ax.twiny()
        d_str = f'delta_{run_dict[s_str]["conv_idelta_list"][0]}'
        volume0 = run_dict[s_str][d_str]['pure']['npt']['volume0']
        volume_array = [volume0 * (1.0 + epsilon)**3 for epsilon in epsilon_array_en_vol]
        ax2.set_xlim(volume_array[0], volume_array[-1])
        ax2.set_xlabel('Volume ($\mathrm{\AA}^3$)')


def plot_formation_energy(ax, run_dict, method_plot_dict, sample):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']

    E_form_full_pred = compute_formation_property(run_dict, sample, 'energy_full_pred')
    E_form_hom_pred = compute_formation_property(run_dict, sample, 'energy_hom_pred')
    E_form_inhom_pred = compute_formation_property(run_dict, sample, 'energy_inhom_pred')
    E_form_true = compute_formation_property(run_dict, sample, 'energy_true')
    E_form_pred0 = compute_formation_property(run_dict, sample, 'energy_pred0')

    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    ax.plot(delta_array_sample, E_form_pred0, **method_plot_dict['formation']['energy_pred0'])
    ax.plot(delta_array_sample, E_form_hom_pred, **method_plot_dict['formation']['energy_hom_pred'])
    ax.plot(delta_array_sample, E_form_inhom_pred, **method_plot_dict['formation']['energy_inhom_pred'])
    ax.plot(delta_array_sample, E_form_full_pred, **method_plot_dict['formation']['energy_full_pred'])
    ax.plot(delta_array_sample, E_form_true, **method_plot_dict['formation']['energy_true'])

    ax.set_xlabel('Perturbation Magnitude $\delta$')
    ax.set_ylabel('Formation Energy (eV)')
    ax.legend()


def plot_formation_energy_error(ax, run_dict, method_plot_dict, sample, error_type='abs', legend=True):

    if error_type == 'abs':
        error_func = lambda x, y: np.abs(x - y)
    elif error_type == 'rel':
        error_func = lambda x, y: np.abs(x - y) / np.abs(y) * 100.0

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']

    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    E_form_true = compute_formation_property(run_dict, sample, 'energy_true')

    E_form_full_pred_error = error_func(compute_formation_property(run_dict, sample, 'energy_full_pred'), E_form_true)
    E_form_hom_pred_error = error_func(compute_formation_property(run_dict, sample, 'energy_hom_pred'), E_form_true)
    E_form_inhom_pred_error = error_func(compute_formation_property(run_dict, sample, 'energy_inhom_pred'), E_form_true)
    E_form_pred0_error = error_func(compute_formation_property(run_dict, sample, 'energy_pred0'), E_form_true)

    ax.plot(delta_array_sample, E_form_pred0_error, **method_plot_dict['formation']['energy_pred0'])
    ax.plot(delta_array_sample, E_form_hom_pred_error, **method_plot_dict['formation']['energy_hom_pred'])
    ax.plot(delta_array_sample, E_form_inhom_pred_error, **method_plot_dict['formation']['energy_inhom_pred'])
    ax.plot(delta_array_sample, E_form_full_pred_error, **method_plot_dict['formation']['energy_full_pred'])

    ax.set_xlabel('Perturbation Magnitude $\delta$')

    if error_type == 'abs':
        ax.set_ylabel('Formation Energy Error (eV)')
    elif error_type == 'rel':
        ax.set_ylabel('Formation Energy Error (%)')

    if legend:
        ax.legend()


def get_prop_array(run_dict, sample, system, prop_name):
    prop_array = \
        np.array([run_dict[f'sample_{sample}'][f'delta_{i}'][system]['npt'][prop_name] for i in run_dict[f'sample_{sample}']['conv_idelta_list']])

    return prop_array


def plot_absolute_volume(ax, run_dict, sample):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']

    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'volume_true'), label='Pure True', marker='s', c='black')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'volume_pred'), label='Pure Pred. from dP/dV', marker='s')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'volume_pred_DT'), label='Pure Pred. from D@T', marker='s', ls='--')

    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'volume_true'), label='Vac. True', marker='o', c='black')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'volume_pred'), label='Vac. Pred. from dP/dV', marker='o')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'volume_pred_DT'), label='Vac. Pred. from D@T', marker='o', ls='--')

    ax.set_xlabel('Perturbation Magnitude $\delta$')
    ax.set_ylabel('Volume ($\mathrm{\AA}^3$)')
    ax.legend()


def plot_absolute_energy(ax, run_dict, sample):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']

    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'energy_true'), label='Pure True', marker='s', c='black')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'energy_pred0'), label='Pure No pos. change', marker='s', c='purple')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'energy_hom_pred'), label='Pure Hom.', marker='s', c='goldenrod')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'energy_inhom_pred'), label='Pure Inhom.', marker='s', c='blue')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'energy_full_pred'), label='Pure Hom. + Inhom.', marker='s', c='red')

    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'energy_true'), label='Vac. True', marker='o', c='black')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'energy_pred0'), label='Vac. No pos. change', marker='o', c='purple')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'energy_hom_pred'), label='Vac. Hom.', marker='o', c='goldenrod')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'energy_inhom_pred'), label='Vac. Inhom.', marker='o', c='blue')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'energy_full_pred'), label='Vac. Hom. + Inhom.', marker='o', c='red')

    ax.set_xlabel('Perturbation Magnitude $\delta$')
    ax.set_ylabel('Energy (eV)')
    ax.legend()


def plot_coord_error(ax, run_dict, method_plot_dict, sample):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']
    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    # coord_error_full, coord_error_hom, coord_error_inhom, coord_error_hom_DT

    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'coord_error_hom'), **method_plot_dict['pure']['energy_hom_pred'])
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'coord_error_inhom'), **method_plot_dict['pure']['energy_inhom_pred'])
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'coord_error_full'), **method_plot_dict['pure']['energy_full_pred'])

    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'coord_error_hom'), **method_plot_dict['vac']['energy_hom_pred'])
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'coord_error_inhom'), **method_plot_dict['vac']['energy_inhom_pred'])
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'coord_error_full'), **method_plot_dict['vac']['energy_full_pred'])

    #ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'coord_error_full'), label='Pure Hom. + Inhom.', marker='s')

    ax.set_xlabel('Perturbation Magnitude $\delta$')
    ax.set_ylabel(r"Coordinate Error, $\mathrm{\AA}$")
    ax.legend()


def plot_energy_error(ax, run_dict, method_plot_dict, sample, error_type='abs'):

    if error_type == 'abs':
        error_func = lambda x, y: np.abs(x - y)
    elif error_type == 'rel':
        error_func = lambda x, y: np.abs(x - y) / np.abs(y) * 100.0

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']

    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    # Pure
    true_pure = get_prop_array(run_dict, sample, 'pure', 'energy_true')

    error_pred0 = error_func(get_prop_array(run_dict, sample, 'pure', 'energy_pred0'), true_pure)
    ax.plot(delta_array_sample, error_pred0, **method_plot_dict['pure']['energy_pred0'])

    error_hom = error_func(get_prop_array(run_dict, sample, 'pure', 'energy_hom_pred'), true_pure)
    ax.plot(delta_array_sample, error_hom, **method_plot_dict['pure']['energy_hom_pred'])

    error_inhom = error_func(get_prop_array(run_dict, sample, 'pure', 'energy_inhom_pred'), true_pure)
    ax.plot(delta_array_sample, error_inhom, **method_plot_dict['pure']['energy_inhom_pred'])

    error_full = error_func(get_prop_array(run_dict, sample, 'pure', 'energy_full_pred'), true_pure)
    ax.plot(delta_array_sample, error_full, **method_plot_dict['pure']['energy_full_pred'])

    # Vacancy
    vac_true = get_prop_array(run_dict, sample, 'vac', 'energy_true')

    error_pred0 = error_func(get_prop_array(run_dict, sample, 'vac', 'energy_pred0'), vac_true)
    ax.plot(delta_array_sample, error_pred0, **method_plot_dict['vac']['energy_pred0'])

    error_hom = error_func(get_prop_array(run_dict, sample, 'vac', 'energy_hom_pred'), vac_true)
    ax.plot(delta_array_sample, error_hom, **method_plot_dict['vac']['energy_hom_pred'])

    error_inhom = error_func(get_prop_array(run_dict, sample, 'vac', 'energy_inhom_pred'), vac_true)
    ax.plot(delta_array_sample, error_inhom, **method_plot_dict['vac']['energy_inhom_pred'])

    error_full = error_func(get_prop_array(run_dict, sample, 'vac', 'energy_full_pred'), vac_true)
    ax.plot(delta_array_sample, error_full, **method_plot_dict['vac']['energy_full_pred'])

    ax.set_xlabel('Perturbation Magnitude $\delta$')

    if error_type == 'abs':
        ax.set_ylabel('Energy Error (eV)')
    elif error_type == 'rel':
        ax.set_ylabel('Energy Error (%)')

    ax.legend()


def plot_formation_volume(ax, run_dict, sample):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']
    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    vol_form_full_pred = compute_formation_property(run_dict, sample, 'volume_pred')
    vol_form_pred_DT = compute_formation_property(run_dict, sample, 'volume_pred_DT', 'volume_pred')
    vol_form_true = compute_formation_property(run_dict, sample, 'volume_true')

    ax.plot(delta_array_sample, vol_form_true, label='True', marker='o', c='black')
    ax.plot(delta_array_sample, vol_form_full_pred, label='Pred. from dP/dV', marker='o')
    ax.plot(delta_array_sample, vol_form_pred_DT, label='Pred. from D@T', marker='o', ls='--')

    ax.set_xlabel('Perturbation Magnitude $\delta$')
    ax.set_ylabel('Formation Volume ($\mathrm{\AA}^3$)')
    ax.legend()


def plot_formation_energy_error_bins(ax, bin_dict, method_plot_dict, spline_fill=True):

    # Energies
    # en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']
    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_full_pred']

    dE_bin_centers = bin_dict['energy']['bin_centers']

    for i, en_key in enumerate(en_key_list):

        bin_average = bin_dict['energy'][en_key]['average']
        bin_perc_33 = bin_dict['energy'][en_key]['perc_33']
        bin_perc_66 = bin_dict['energy'][en_key]['perc_66']
        bin_std = bin_dict['energy'][en_key]['std']

        ax.plot(dE_bin_centers, bin_average, **method_plot_dict['formation'][en_key])

        if spline_fill:
            dE_grid, bin_33_interp = interpolate(dE_bin_centers, bin_perc_33)
            dE_grid, bin_66_interp = interpolate(dE_bin_centers, bin_perc_66)
            ax.fill_between(dE_grid, bin_33_interp, bin_66_interp, alpha=0.2,
                            color=method_plot_dict['formation'][en_key]['c'])
        else:
            ax.fill_between(dE_bin_centers, bin_perc_33, bin_perc_66, alpha=0.2,
                            color=method_plot_dict['formation'][en_key]['c'])

    ax.set_ylabel('Energy Error (eV)')

    ax.set_xlabel('Formation Energy Change (eV)')
    ax.set_ylabel('Formation Energy Error (eV)')


def plot_formation_volume_error_bins(ax, bin_dict, method_plot_dict, spline_fill=True):

    # vol_key_list = ['volume_pred', 'volume_pred_DT']
    vol_key_list = ['volume_pred']

    dV_bin_centers = bin_dict['volume']['bin_centers']

    for i, vol_key in enumerate(vol_key_list):

        bin_average = bin_dict['volume'][vol_key]['average']
        bin_perc_33 = bin_dict['volume'][vol_key]['perc_33']
        bin_perc_66 = bin_dict['volume'][vol_key]['perc_66']
        bin_std = bin_dict['volume'][vol_key]['std']

        ax.plot(dV_bin_centers, bin_average, **method_plot_dict['formation'][vol_key])

        if spline_fill:
            dV_grid, bin_33_interp = interpolate(dV_bin_centers, bin_perc_33)
            dV_grid, bin_66_interp = interpolate(dV_bin_centers, bin_perc_66)
            ax.fill_between(dV_grid, bin_33_interp, bin_66_interp, alpha=0.2, color=method_plot_dict['formation'][vol_key]['c'])
        else:
            ax.fill_between(dV_bin_centers, bin_perc_33, bin_perc_66, color=method_plot_dict['formation'][vol_key]['c'], alpha=0.2)

    ax.set_ylabel('Volume Error ($\mathrm{\AA}^3$)')

    ax.set_xlabel('Formation Volume Change ($\mathrm{\AA}^3$)')
    ax.set_ylabel('Formation Volume Error ($\mathrm{\AA}^3$)')


def plot_formation_energy_error_average(ax, average_dict, method_plot_dict):

    delta_array = average_dict['detla_array']
    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']

    for i, en_key in enumerate(en_key_list):

        form_error_array = np.array([average_dict['error']['formation'][en_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        form_error_std = np.array([average_dict['error']['formation'][en_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        ax.plot(delta_array, form_error_array, **method_plot_dict['formation'][en_key])
        ax.fill_between(delta_array, form_error_array - form_error_std, form_error_array + form_error_std,
                        color=method_plot_dict['formation'][en_key]['c'], alpha=0.2)

    ax.set_xlabel('Perturbation Magnitude $\delta$')
    ax.set_ylabel('Formation Energy Error (eV)')


def plot_formation_volume_error_average(ax, average_dict, method_plot_dict):

    delta_array = average_dict['detla_array']

    vol_key_list = ['volume_pred', 'volume_pred_DT']

    for i, vol_key in enumerate(vol_key_list):

        form_error_array = np.array([average_dict['error']['formation'][vol_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        form_error_std = np.array([average_dict['error']['formation'][vol_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        ax.plot(delta_array, form_error_array, **method_plot_dict['formation'][vol_key])
        ax.fill_between(delta_array, form_error_array - form_error_std, form_error_array + form_error_std,
                        color=method_plot_dict['formation'][vol_key]['c'], alpha=0.2)

    ax.set_xlabel('Perturbation Magnitude $\delta$')
    ax.set_ylabel('Formation Volume Error ($\mathrm{\AA}^3$)')


def plot_energy_error_average(ax, average_dict, method_plot_dict):
    delta_array = average_dict['detla_array']

    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']

    for i, en_key in enumerate(en_key_list):

        pure_error_array = np.array([average_dict['error']['pure'][en_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        pure_error_std = np.array([average_dict['error']['pure'][en_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        vac_error_array = np.array([average_dict['error']['vac'][en_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        vac_error_std = np.array([average_dict['error']['vac'][en_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        form_error_array = np.array([average_dict['error']['formation'][en_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        form_error_std = np.array([average_dict['error']['formation'][en_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        ax.plot(delta_array, pure_error_array, **method_plot_dict['pure'][en_key])
        #ax.fill_between(delta_array, pure_error_array - pure_error_std, pure_error_array + pure_error_std, color=color_list[i], alpha=0.2)

        ax.plot(delta_array, vac_error_array, **method_plot_dict['vac'][en_key])
        #ax.fill_between(delta_array, vac_error_array - vac_error_std, vac_error_array + vac_error_std, color=color_list[i], alpha=0.2)

    ax.set_ylabel('Energy Error (eV)')


def plot_volume_error_average(ax, average_dict, method_plot_dict):
    delta_array = average_dict['detla_array']

    vol_key_list = ['volume_pred', 'volume_pred_DT']
    color_list = ['tab:blue', 'tab:orange']
    legend_list = ['From dP/dV', 'From D@T']

    for i, vol_key in enumerate(vol_key_list):

        pure_error_array = np.array([average_dict['error']['pure'][vol_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        pure_error_std = np.array([average_dict['error']['pure'][vol_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        vac_error_array = np.array([average_dict['error']['vac'][vol_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        vac_error_std = np.array([average_dict['error']['vac'][vol_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        ax.plot(delta_array, pure_error_array, **method_plot_dict['pure'][vol_key])
        ax.fill_between(delta_array, pure_error_array - pure_error_std, pure_error_array + pure_error_std, color=color_list[i], alpha=0.2)

        ax.plot(delta_array, vac_error_array, **method_plot_dict['vac'][vol_key])
        ax.fill_between(delta_array, vac_error_array - vac_error_std, vac_error_array + vac_error_std, color=color_list[i], alpha=0.2)

    ax.set_ylabel('Volume Error ($\mathrm{\AA}^3$)')


def interpolate(x_data, y_data, npoints=1000):
    cs = CubicSpline(x_data, y_data)
    x_grid = np.linspace(x_data.min(), x_data.max(), npoints)
    return x_grid, cs(x_grid)


def filter_data(run_dict, abs_threshold=1e5, rel_threshold=10.0, rel_form_en_thr=100.0, verbose=False):

    error_func = lambda x, y: np.abs(x - y) / np.abs(y) * 100.0

    print(f'Filtering data with abs. energy threshold {abs_threshold:.1e} eV and relative threshold {rel_threshold:.1f}%...')
    delta_array = run_dict['delta_array']
    for sample in run_dict['sample_list']:
        s_str = f'sample_{sample}'
        delta_remove_list = []
        for d_str in run_dict[s_str].keys():
            if not d_str.startswith('delta_'):
                continue
            if run_dict[s_str][d_str]['pure'] is None or run_dict[s_str][d_str]['vac'] is None:
                continue
            if run_dict[s_str][d_str]['pure']['npt'] is None or run_dict[s_str][d_str]['vac']['npt'] is None:
                continue
            idelta = int(d_str.split('_')[1])
            for en_key in ['energy_full_pred', 'energy_hom_pred', 'energy_inhom_pred', 'energy_pred0', 'energy_true']:

                en_pure = run_dict[s_str][d_str]['pure']['npt'][en_key]
                en_vac = run_dict[s_str][d_str]['vac']['npt'][en_key]

                en_pure_true = run_dict[s_str][d_str]['pure']['npt']['energy_true']
                en_vac_true = run_dict[s_str][d_str]['vac']['npt']['energy_true']

                rel_error_pure = error_func(en_pure, en_pure_true)
                rel_error_vac = error_func(en_vac, en_vac_true)

                en_form = en_vac - en_pure * run_dict['Natom vac'] / run_dict['Natom pure']
                en_form_true = en_vac_true - en_pure_true * run_dict['Natom vac'] / run_dict['Natom pure']
                rel_error_form = error_func(en_form, en_form_true)

                if np.abs(en_pure) > abs_threshold or np.abs(en_vac) > abs_threshold or \
                    rel_error_pure > rel_threshold or rel_error_vac > rel_threshold or \
                        rel_error_form > rel_form_en_thr:

                    delta_remove_list.append(delta_array[idelta])
                    run_dict[s_str]['conv_idelta_list'].remove(idelta)
                    break

        if verbose and len(delta_remove_list) > 0:
            print(f'   Sample: {sample:3d} | Removed deltas: {delta_remove_list}')

    return run_dict


def average_data(run_dict):

    delta_array = run_dict['delta_array']
    average_dict = {}

    average_dict['detla_array'] = delta_array
    average_dict['error'] = {}
    average_dict['error']['pure'] = {}
    average_dict['error']['vac'] = {}
    average_dict['error']['formation'] = {}

    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']
    vol_key_list = ['volume_pred', 'volume_pred_DT']
    prop_key_list = en_key_list + vol_key_list

    # For all the deltas, create empty lists
    for prop_key in prop_key_list:
        average_dict['error']['pure'][prop_key] = {}
        average_dict['error']['vac'][prop_key] = {}
        average_dict['error']['formation'][prop_key] = {}

        for delta in range(len(delta_array)):
            d_str = f'delta_{delta}'
            average_dict['error']['pure'][prop_key][d_str] = []
            average_dict['error']['vac'][prop_key][d_str] = []
            average_dict['error']['formation'][prop_key][d_str] = []

    # Append the deltas
    for sample in run_dict['sample_list']:
        s_str = f'sample_{sample}'
        # Only the filtered deltas
        delta_sample_list = run_dict[s_str]['conv_idelta_list']
        for delta in delta_sample_list:
            d_str = f'delta_{delta}'

            # True values
            en_pure_true = run_dict[s_str][d_str]['pure']['npt']['energy_true']
            en_vac_true = run_dict[s_str][d_str]['vac']['npt']['energy_true']
            en_form_true = en_vac_true - en_pure_true * run_dict['Natom vac'] / run_dict['Natom pure']

            vol_pure_true = run_dict[s_str][d_str]['pure']['npt']['volume_true']
            vol_vac_true = run_dict[s_str][d_str]['vac']['npt']['volume_true']
            vol_form_true = vol_vac_true - vol_pure_true * run_dict['Natom vac'] / run_dict['Natom pure']

            for en_key in en_key_list:
                en_pure = run_dict[s_str][d_str]['pure']['npt'][en_key]
                en_vac = run_dict[s_str][d_str]['vac']['npt'][en_key]

                average_dict['error']['pure'][en_key][d_str].append(en_pure - en_pure_true)
                average_dict['error']['vac'][en_key][d_str].append(en_vac - en_vac_true)

                en_form = en_vac - en_pure * run_dict['Natom vac'] / run_dict['Natom pure']
                average_dict['error']['formation'][en_key][d_str].append(en_form - en_form_true)

            for vol_key in vol_key_list:
                vol_pure = run_dict[s_str][d_str]['pure']['npt'][vol_key]
                vol_vac = run_dict[s_str][d_str]['vac']['npt'][vol_key]

                average_dict['error']['pure'][vol_key][d_str].append(vol_pure - vol_pure_true)
                average_dict['error']['vac'][vol_key][d_str].append(vol_vac - vol_vac_true)

                vol_form = vol_vac - vol_pure * run_dict['Natom vac'] / run_dict['Natom pure']
                average_dict['error']['formation'][vol_key][d_str].append(vol_form - vol_form_true)

    # Compute averages
    for prop_key in prop_key_list:
        for delta in range(len(delta_array)):
            d_str = f'delta_{delta}'
            pure_array = np.array(average_dict['error']['pure'][prop_key][d_str])
            vac_array = np.array(average_dict['error']['vac'][prop_key][d_str])
            form_array = np.array(average_dict['error']['formation'][prop_key][d_str])

            # Replace the list of values with the average
            average_dict['error']['pure'][prop_key][d_str] = [np.mean(pure_array), np.std(pure_array)]
            # TODO: percentile -> 33, 66
            #average_dict['error']['pure'][prop_key][d_str] = [np.mean(pure_array), np.percentile(pure_array, 33, 66)]
            average_dict['error']['vac'][prop_key][d_str] = [np.mean(vac_array), np.std(vac_array)]
            average_dict['error']['formation'][prop_key][d_str] = [np.mean(form_array), np.std(form_array)]

    return average_dict


def compute_error_bins(run_dict, E_range=None, V_range=None):

    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']
    vol_key_list = ['volume_pred', 'volume_pred_DT']
    prop_key_list = en_key_list + vol_key_list

    bin_dict = {}
    bin_dict['energy'] = {}
    bin_dict['volume'] = {}

    # Find emin and emax
    delta_array = run_dict['delta_array']
    idelta0 = np.argmin(np.abs(delta_array))
    en_pure_true0 = run_dict['sample_0'][f'delta_{idelta0}']['pure']['npt']['energy_true']
    en_vac_true0 = run_dict['sample_0'][f'delta_{idelta0}']['vac']['npt']['energy_true']
    en_form_true0 = en_vac_true0 - en_pure_true0 * run_dict['Natom vac'] / run_dict['Natom pure']

    vol_pure_true0 = run_dict['sample_0'][f'delta_{idelta0}']['pure']['npt']['volume_true']
    vol_vac_true0 = run_dict['sample_0'][f'delta_{idelta0}']['vac']['npt']['volume_true']
    vol_form_true0 = vol_vac_true0 - vol_pure_true0 * run_dict['Natom vac'] / run_dict['Natom pure']

    if E_range is None or V_range is None:

        emin, emax = 0.0, 0.0
        vmin, vmax = 0.0, 0.0

        for sample in run_dict['sample_list']:
            s_str = f'sample_{sample}'
            # Only the filtered deltas
            delta_sample_list = run_dict[s_str]['conv_idelta_list']
            for delta in delta_sample_list:
                d_str = f'delta_{delta}'
                en_pure_true = run_dict[s_str][d_str]['pure']['npt']['energy_true']
                en_vac_true = run_dict[s_str][d_str]['vac']['npt']['energy_true']
                en_form_true = en_vac_true - en_pure_true * run_dict['Natom vac'] / run_dict['Natom pure']

                vol_pure_true = run_dict[s_str][d_str]['pure']['npt']['volume_true']
                vol_vac_true = run_dict[s_str][d_str]['vac']['npt']['volume_true']
                vol_form_true = vol_vac_true - vol_pure_true * run_dict['Natom vac'] / run_dict['Natom pure']

                dE_form_true = en_form_true - en_form_true0
                dV_form_true = vol_form_true - vol_form_true0

                if dE_form_true < emin:
                    emin = dE_form_true
                elif dE_form_true > emax:
                    emax = dE_form_true

                if dV_form_true < vmin:
                    vmin = dV_form_true
                elif dV_form_true > vmax:
                    vmax = dV_form_true

        if E_range is not None:
            emin, emax = E_range
        if V_range is not None:
            vmin, vmax = V_range
    else:
        emin, emax = E_range
        vmin, vmax = V_range

    print(f'Formation energy at delta=0: {en_form_true0:.3f} eV')
    print(f'Max. deviations -+: {emin:.3f} to {emax:.3f} eV')
    print(f'Formation volume at delta=0: {vol_form_true0:.3f} A^3')
    print(f'Max. deviations -+: {vmin:.3f} to {vmax:.3f} A^3')

    dE_num_bins = 20
    dE_bins = np.linspace(np.ceil(emin), np.floor(emax), dE_num_bins + 1)
    dE_bin_centers = 0.5 * (dE_bins[:-1] + dE_bins[1:])
    bin_dict['energy']['bins'] = dE_bins
    bin_dict['energy']['bin_centers'] = dE_bin_centers
    bin_dict['energy']['at delta 0'] = en_form_true0

    dV_num_bins = 20
    dV_bins = np.linspace(np.ceil(vmin), np.floor(vmax), dV_num_bins + 1)
    dV_bin_centers = 0.5 * (dV_bins[:-1] + dV_bins[1:])
    bin_dict['volume']['bins'] = dV_bins
    bin_dict['volume']['bin_centers'] = dV_bin_centers
    bin_dict['volume']['at delta 0'] = vol_form_true0

    for en_key in en_key_list:
        bin_dict['energy'][en_key] = {}
        bin_dict['energy'][en_key]['bin_errors'] = [[] for _ in range(dE_num_bins)]
    for vol_key in vol_key_list:
        bin_dict['volume'][vol_key] = {}
        bin_dict['volume'][vol_key]['bin_errors'] = [[] for _ in range(dV_num_bins)]

    for sample in run_dict['sample_list']:
        s_str = f'sample_{sample}'
        # Only the filtered deltas
        delta_sample_list = run_dict[s_str]['conv_idelta_list']
        for delta in delta_sample_list:
            d_str = f'delta_{delta}'
            en_pure_true = run_dict[s_str][d_str]['pure']['npt']['energy_true']
            en_vac_true = run_dict[s_str][d_str]['vac']['npt']['energy_true']
            en_form_true = en_vac_true - en_pure_true * run_dict['Natom vac'] / run_dict['Natom pure']

            vol_pure_true = run_dict[s_str][d_str]['pure']['npt']['volume_true']
            vol_vac_true = run_dict[s_str][d_str]['vac']['npt']['volume_true']
            vol_form_true = vol_vac_true - vol_pure_true * run_dict['Natom vac'] / run_dict['Natom pure']

            dE_form_true = en_form_true - en_form_true0
            dV_form_true = vol_form_true - vol_form_true0

            dE_bin_index = np.digitize(dE_form_true, dE_bins) - 1
            dV_bin_index = np.digitize(dV_form_true, dV_bins) - 1

            if 0 <= dE_bin_index < dE_num_bins:
                for en_key in en_key_list:
                    en_pure = run_dict[s_str][d_str]['pure']['npt'][en_key]
                    en_vac = run_dict[s_str][d_str]['vac']['npt'][en_key]
                    en_form = en_vac - en_pure * run_dict['Natom vac'] / run_dict['Natom pure']
                    en_form_error = en_form - en_form_true
                    bin_dict['energy'][en_key]['bin_errors'][dE_bin_index].append(en_form_error)

            if 0 <= dV_bin_index < dV_num_bins:
                for vol_key in vol_key_list:
                    vol_pure = run_dict[s_str][d_str]['pure']['npt'][vol_key]
                    vol_vac = run_dict[s_str][d_str]['vac']['npt'][vol_key]
                    vol_form = vol_vac - vol_pure * run_dict['Natom vac'] / run_dict['Natom pure']
                    vol_form_error = vol_form - vol_form_true
                    bin_dict['volume'][vol_key]['bin_errors'][dV_bin_index].append(vol_form_error)

    # Compute the average and percentiles
    for en_key in en_key_list:
        bin_errors = bin_dict['energy'][en_key]['bin_errors']
        bin_dict['energy'][en_key]['average'] = np.array([np.percentile(errors, 50) if errors else np.nan for errors in bin_errors])
        bin_dict['energy'][en_key]['perc_33'] = np.array([np.percentile(errors, 33) if errors else np.nan for errors in bin_errors])
        bin_dict['energy'][en_key]['perc_66'] = np.array([np.percentile(errors, 66) if errors else np.nan for errors in bin_errors])
        bin_dict['energy'][en_key]['std'] = np.array([np.std(errors) if errors else np.nan for errors in bin_errors])

    for vol_key in vol_key_list:
        bin_errors = bin_dict['volume'][vol_key]['bin_errors']
        bin_dict['volume'][vol_key]['average'] = np.array([np.percentile(errors, 50)  if errors else np.nan for errors in bin_errors])
        bin_dict['volume'][vol_key]['perc_33'] = np.array([np.percentile(errors, 33) if errors else np.nan for errors in bin_errors])
        bin_dict['volume'][vol_key]['perc_66'] = np.array([np.percentile(errors, 66) if errors else np.nan for errors in bin_errors])
        bin_dict['volume'][vol_key]['std'] = np.array([np.std(errors) if errors else np.nan for errors in bin_errors])

    return bin_dict


def cut_data(run_dict, delta_min=-50.0, delta_max=50.0, verbose=False):

    print(f'Cutting data with delta_min={delta_min} and delta_max={delta_max}...')
    delta_array = run_dict['delta_array']
    for sample in run_dict['sample_list']:
        s_str = f'sample_{sample}'
        delta_remove_list = []
        for idelta in range(len(delta_array)):
            delta = delta_array[idelta]
            if (delta < delta_min or delta > delta_max) and (idelta in run_dict[s_str]['conv_idelta_list']):
                delta_remove_list.append(delta_array[idelta])
                run_dict[s_str]['conv_idelta_list'].remove(idelta)

        if verbose and len(delta_remove_list) > 0:
            print(f'   Sample: {sample:3d} | Removed deltas: {delta_remove_list}')

    return run_dict


def filter_data_energy_volume(run_dict, atol=1e-7, rtol=50.0, verbose=False):

    delta_array = run_dict['delta_array']
    delta0_idx = np.argmin(np.abs(delta_array))
    print(f'Filtering based on E-V with abs. energy threshold {atol:.1e} eV and relative threshold {rtol:.1f}%...')
    for sample in run_dict['sample_list']:
        s_str = f'sample_{sample}'
        # Only the filtered deltas
        delta_sample_list = run_dict[s_str]['conv_idelta_list'].copy()
        energy_array_delta0_pure = run_dict[s_str][f'delta_{delta0_idx}']['pure']['en_vol']['energy_array']
        #idx_nonzero = np.where(np.abs(energy_array_delta0_pure) > atol)[0]
        for delta in delta_sample_list:
            d_str = f'delta_{delta}'
            energy_array_delta_pure = run_dict[s_str][d_str]['pure']['en_vol']['energy_array']

            #diff_abs = np.abs(energy_array_delta_pure[idx_nonzero] - energy_array_delta0_pure[idx_nonzero])
            #diff_rel = np.abs(diff_abs / energy_array_delta0_pure[idx_nonzero]) * 100.0

            # atol + rtol * abs(desired) for each element of energy_array_delta0_pure and energy_array_delta_pure

            diff_abs = np.abs(energy_array_delta_pure - energy_array_delta0_pure)

            if np.any(diff_abs > atol + (rtol / 100.0) * np.abs(energy_array_delta0_pure)):
                run_dict[s_str]['conv_idelta_list'].remove(delta)
                run_dict[s_str][d_str]['pure'] = None
                run_dict[s_str][d_str]['vac'] = None

    return run_dict


def main():

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Read the pickle file: run_dict.pkl
    pickle_filename = 'run_dict.pkl'
    #pickle_filename = './ncell_x_3_dense_npt2/run_dict.pkl'
    #pickle_filename = './NERSC/ncell_x_4_energy/run_dict.pkl'

    print(f'Reading {pickle_filename}...')
    with open(pickle_filename, 'rb') as f:
        run_dict = pickle.load(f)

    print(f'Number of atoms in pure system: {run_dict["Natom pure"]}')
    print(f'Number of atoms in vacancy system: {run_dict["Natom vac"]}')

    # filter data
    run_dict = filter_data(run_dict)
    run_dict = filter_data_energy_volume(run_dict, atol=1e-2, rtol=50.0)

    # Hard-remove deltas from -50.0 to 50.0
    #run_dict = cut_data(run_dict, delta_min=-50.0, delta_max=50.0)

    # Plot success_matrix
    plot_succ_matrix = False
    #plot_succ_matrix = True
    if plot_succ_matrix:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_success_matrix(ax, run_dict)
        fig.savefig(os.path.join('plots', 'success_matrix.pdf'))

    method_plot_dict = {
        'formation': {
            'energy_true': {'label': 'True', 'marker': 'o', 'c': 'black'},
            'energy_pred0': {'label': 'No pos. change', 'marker': 'o', 'c': 'purple'},
            'energy_hom_pred': {'label': 'Hom.', 'marker': 'o', 'c': 'goldenrod'},
            'energy_inhom_pred': {'label': 'Inhom.', 'marker': 'o', 'c': 'blue'},
            'energy_full_pred': {'label': 'Hom. + Inhom.', 'marker': 'o', 'c': 'red'},
            'volume_pred': {'label': 'From dP/dV', 'marker': 'o', 'c': 'tab:blue'},
            'volume_pred_DT': {'label': 'From D@T', 'marker': 'o', 'c': 'tab:orange'},
        },
        'pure': {
            'energy_true': {'label': 'Bulk True', 'marker': 's', 'c': 'black', 'ls': '--'},
            'energy_pred0': {'label': 'Bulk No pos. change', 'marker': 's', 'c': 'purple', 'ms': 12, 'ls': '--'},
            'energy_hom_pred': {'label': 'Bulk Hom.', 'marker': 's', 'c': 'goldenrod', 'ls': '--'},
            'energy_inhom_pred': {'label': 'Bulk Inhom.', 'marker': 's', 'c': 'blue', 'ms': 12, 'ls': '--'},
            'energy_full_pred': {'label': 'Bulk Hom. + Inhom.', 'marker': 's', 'c': 'red', 'ls': '--'},
            'volume_pred': {'label': 'From dP/dV', 'marker': 'o', 'c': 'tab:blue', 'ls': '--'},
            'volume_pred_DT': {'label': 'From D@T', 'marker': 'o', 'c': 'tab:orange', 'ls': '--'},
        },
        'vac': {
            'energy_true': {'label': 'Vac. True', 'marker': 'o', 'c': 'black'},
            'energy_pred0': {'label': 'Vac. No pos. change', 'marker': 'o', 'c': 'purple', 'ms': 12},
            'energy_hom_pred': {'label': 'Vac. Hom.', 'marker': 'o', 'c': 'goldenrod'},
            'energy_inhom_pred': {'label': 'Vac. Inhom.', 'marker': 'o', 'c': 'blue', 'ms': 12},
            'energy_full_pred': {'label': 'Vac. Hom. + Inhom.', 'marker': 'o', 'c': 'red'},
            'volume_pred': {'label': 'From dP/dV', 'marker': 'o', 'c': 'tab:blue'},
            'volume_pred_DT': {'label': 'From D@T', 'marker': 'o', 'c': 'tab:orange'},
        }
    }

    #
    # Plotting physical data
    #
    #sample = 1
    sample = 80
    #sample = 62

    #plot_coordinate_error = True
    plot_coordinate_error = False
    if plot_coordinate_error:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plot_coord_error(ax, run_dict, method_plot_dict, sample)
        fig.savefig(os.path.join(plot_dir, f'coord_error_sample_{sample:03d}.pdf'))
        plt.show()

    #
    # Layout 2x2
    #
    plot_2x2 = False
    if plot_2x2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(left=0.07, right=0.96, bottom=0.09, top=0.97, wspace=0.22, hspace=0.01)

        # Lattice constant
        plot_lattice_constant(axes[0, 0], run_dict, sample)
        axes[0, 0].set_xticklabels([])
        ax_inset = axes[0, 0].inset_axes([0.14, 0.14, 0.4, 0.4])
        plot_energy_volume_deltas(ax_inset, run_dict, sample, fsize_bar=12)

        # Formation energies
        plot_formation_energy(axes[1, 0], run_dict, method_plot_dict, sample)

        # Absolute volume
        plot_absolute_volume(axes[0, 1], run_dict, sample)
        axes[0, 1].set_xticklabels([])

        # Formation volume
        plot_formation_volume(axes[1, 1], run_dict, sample)

    #fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #plot_energy_error(ax, run_dict, sample, error_type='abs')
    #plot_energy_error(ax, run_dict, sample, error_type='rel')
    #plot_absolute_volume(ax, run_dict, sample)
    #plot_absolute_energy(ax, run_dict, sample)

    #
    # Layout 1x3
    #
    #plot_1x3 = True
    plot_1x3 = False
    if plot_1x3:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        plt.subplots_adjust(left=0.05, right=0.96, bottom=0.12, top=0.92, wspace=0.38, hspace=0.01)

        # Energy-volume
        plot_energy_volume_deltas(axes[0], run_dict, sample)

        # Formation energies
        plot_formation_energy(axes[1], run_dict, method_plot_dict, sample)
        axes[1].set_title(f'ncell_x={run_dict["ncell_x"]}; Natom={run_dict["Natom pure"]}; sample {sample}')

        # Formation volume
        plot_formation_volume(axes[2], run_dict, sample)

        plt.show()

    #
    # Layout 2x2
    #
    plot_2x2_new = True
    if plot_2x2_new:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(left=0.08, right=0.93, bottom=0.07, top=0.90, wspace=0.2, hspace=0.2)

        # Energy-volume
        plot_energy_volume_deltas(axes[0, 1], run_dict, sample, label_pad=0, second_xaxis=True, cmap_name='Paired') # 'coolwarm')

        # Formation volume
        plot_formation_volume(axes[1, 1], run_dict, sample)

        # Formation energies
        plot_formation_energy(axes[0, 0], run_dict, method_plot_dict, sample)
        axes[0, 1].set_title(f'ncell_x={run_dict["ncell_x"]}; Natom={run_dict["Natom pure"]}; sample {sample}', y=1.15, x=-0.2, fontsize=25)

        # Formation energies error
        plot_formation_energy_error(axes[1, 0], run_dict, method_plot_dict, sample, legend=False)

        plt.show()

    plot_samples = False
    #plot_samples = True
    if plot_samples:

        for sample in tqdm(run_dict['sample_list']):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            plt.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.90, wspace=0.2, hspace=0.2)

            # Energy-volume
            plot_energy_volume_deltas(axes[0, 1], run_dict, sample, label_pad=0, second_xaxis=True, cmap_name='Paired') # jet

            # Formation volume
            plot_formation_volume(axes[1, 1], run_dict, sample)

            # Formation energies
            plot_formation_energy(axes[0, 0], run_dict, method_plot_dict, sample)
            axes[0, 1].set_title(f'ncell_x={run_dict["ncell_x"]}; Natom={run_dict["Natom pure"]}; sample {sample}',
                                 y=1.15, x=-0.2, fontsize=25)

            # Formation energies error
            plot_formation_energy_error(axes[1, 0], run_dict, method_plot_dict, sample, legend=False, error_type='rel')

            plt.savefig(os.path.join(plot_dir, f'sample_{sample:03d}.pdf'))
            plt.close()

            # coord error
            plot_coordinate_error = True
            if plot_coordinate_error:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                plot_coord_error(ax, run_dict, method_plot_dict, sample)
                ax.set_title(f'ncell_x={run_dict["ncell_x"]}; Natom={run_dict["Natom pure"]}; sample {sample}', fontsize=20)
                fig.savefig(os.path.join(plot_dir, f'coord_error_sample_{sample:03d}.pdf'))
                plt.close()

    #plot_av_data = True
    plot_av_data = False
    if plot_av_data:
        average_dict = average_data(run_dict)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.95, wspace=0.22, hspace=0.01)

        plot_energy_error_average(axes[0, 0], average_dict, method_plot_dict)
        plot_volume_error_average(axes[0, 1], average_dict, method_plot_dict)
        plot_formation_energy_error_average(axes[1, 0], average_dict, method_plot_dict)
        plot_formation_volume_error_average(axes[1, 1], average_dict, method_plot_dict)

        axes[0, 0].set_xticklabels([])
        axes[0, 1].set_xticklabels([])

        for ax in axes.flatten():
            ax.legend()
        axes[0, 1].set_title(f'AVERAGE OVER {len(run_dict["sample_list"])} samples; ncell_x={run_dict["ncell_x"]}; Natom={run_dict["Natom pure"]}',
                             y=1.01, x=-0.2, fontsize=25)

        fig.savefig(os.path.join(plot_dir, 'average_data.pdf'))

        plt.show()

    plot_av_data_bins = True
    #plot_av_data_bins = False
    if plot_av_data_bins:
        average_dict = average_data(run_dict)
        bin_dict = compute_error_bins(run_dict, E_range=(-1, 1), V_range=(-3, 3))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        #plt.subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.95, wspace=0.22, hspace=0.01)
        plt.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.90, wspace=0.2, hspace=0.2)

        sample_list = run_dict['sample_list']

        plot_formation_energy(axes[0, 0], run_dict, method_plot_dict, sample)
        plot_formation_volume(axes[0, 1], run_dict, sample)
        plot_formation_energy_error_bins(axes[1, 0], bin_dict, method_plot_dict, spline_fill=True)
        plot_formation_volume_error_bins(axes[1, 1], bin_dict, method_plot_dict, spline_fill=True)

        for ax in axes.flatten():
            ax.legend()

        #axes[0, 1].set_title(f'AVERAGE OVER {len(sample_list)} samples; ncell_x={run_dict["ncell_x"]};'
        #                     f' Natom={run_dict["Natom pure"]}', y=1.01, x=-0.2, fontsize=25)

        #axes[0, 1].set_xticklabels([])
        #axes[1, 1].set_xticklabels([])

        fig.savefig(os.path.join(plot_dir, 'average_data_bins.pdf'))

        plt.show()


if __name__ == '__main__':
    main()
