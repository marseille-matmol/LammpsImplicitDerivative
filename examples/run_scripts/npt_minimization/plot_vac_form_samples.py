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
from lammps_implicit_der.systems import BCC_VACANCY, BCC
#from lammps_implicit_der.tools.error_tools import coord_error

from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

plotparams = plot_tools.plotparams.copy()
plotparams['figure.figsize'] = (9, 6)
plotparams['font.size'] = 16
plotparams['figure.subplot.wspace'] = 0.2
plotparams['axes.labelsize'] = 20 #22

#plotparams['text.usetex'] = True
#plotparams['text.latex.preamble'] = r'\usepackage{amsmath}'

plt.rcParams.update(plotparams)


def setup_method_plot_dict():

    method_plot_dict = {
        'formation': {
            'energy_true': {'label': 'True', 'marker': '', 'c': 'black', 'lw': 5.0},
            'energy_pred0': {'label': 'Constant', 'marker': 'o', 'c': 'purple'},
            'energy_hom_pred': {'label': 'Homogeneous', 'marker': 's', 'c': 'blue'}, # goldenrod
            'energy_inhom_pred': {'label': 'Inhom.', 'marker': 'o', 'c': 'tab:blue'},
            'energy_full_pred': {'label': 'Hom. + Inhom.', 'marker': '^', 'c': 'red', 'ms': 12},
            'volume_true': {'label': 'True', 'marker': '', 'c': 'black', 'lw': 5.0},
            'volume_pred0': {'label': 'Constant', 'marker': 'o', 'c': 'purple'},
            'volume_pred': {'label': 'Homogeneous', 'marker': 's', 'c': 'blue', 'ms': 13}, # goldenrod
            'volume_pred_full': {'label': 'Hom. + Inhom.', 'marker': '^', 'c': 'red', 'ms': 8},
        },
        'pure': {
            'energy_true': {'label': 'Bulk True', 'marker': '', 'c': 'black', 'ls': '--'},
            'energy_pred0': {'label': 'Bulk Constant', 'marker': 's', 'c': 'purple', 'ms': 12, 'ls': '--'},
            'energy_hom_pred': {'label': 'Bulk Hom.', 'marker': 's', 'c': 'blue', 'ls': '--'}, # goldenrod
            'energy_inhom_pred': {'label': 'Bulk Inhom.', 'marker': 's', 'c': 'tab:blue', 'ms': 12, 'ls': '--'},
            'energy_full_pred': {'label': 'Bulk Hom. + Inhom.', 'marker': '^', 'c': 'red', 'ls': '--'},
            'volume_pred': {'label': 'Homogeneous', 'marker': 'o', 'c': 'blue', 'ls': '--'}, # goldenrod
        },
        'vac': {
            'energy_true': {'label': 'Vac. True', 'marker': 'o', 'c': 'black'},
            'energy_pred0': {'label': 'Vac. Constant', 'marker': 'o', 'c': 'purple', 'ms': 12},
            'energy_hom_pred': {'label': 'Vac. Hom.', 'marker': 'o', 'c': 'blue'}, # goldenrod
            'energy_inhom_pred': {'label': 'Vac. Inhom.', 'marker': 'o', 'c': 'tab:blue', 'ms': 12},
            'energy_full_pred': {'label': 'Vac. Hom. + Inhom.', 'marker': 'o', 'c': 'red'},
            'volume_pred': {'label': 'Homogeneous', 'marker': 'o', 'c': 'blue'}, # goldenrod
        }
    }
    return method_plot_dict


def setup_method_plot_dict2():

    method_plot_dict2 = setup_method_plot_dict()

    for k in method_plot_dict2:
        for kk in method_plot_dict2[k]:

            if 'full_pred' in kk:
                method_plot_dict2[k][kk]['ms'] = 10
                method_plot_dict2[k][kk]['lw'] = 8
            elif kk == 'volume_pred':
                method_plot_dict2[k][kk]['ms'] = 11
                method_plot_dict2[k][kk]['lw'] = 9
            else:
                method_plot_dict2[k][kk]['ms'] = 8
                method_plot_dict2[k][kk]['lw'] = 6

    return method_plot_dict2


def compute_formation_property(run_dict, sample, property_name_pure, property_name_vac=None, delta_range=None):

    if property_name_vac is None:
        property_name_vac = property_name_pure

    Natom_pure = run_dict['Natom pure']
    Natom_vac = run_dict['Natom vac']

    s_str = f'sample_{sample}'

    if delta_range is None:
        delta_list = run_dict[s_str]['conv_idelta_list']
    else:
        delta_array = run_dict['delta_array']
        idx = np.where((delta_array >= delta_range[0]) & (delta_array <= delta_range[1]))[0]
        delta_list = [i for i in run_dict[s_str]['conv_idelta_list'] if i in idx]

    prop_pure = np.array([run_dict[s_str][f'delta_{i}']['pure']['npt'][property_name_pure] for i in delta_list])
    prop_vac = np.array([run_dict[s_str][f'delta_{i}']['vac']['npt'][property_name_vac] for i in delta_list])

    prop_formation = prop_vac - prop_pure * Natom_vac / Natom_pure

    return prop_formation


def get_success_matrix(run_dict):

    delta_array = run_dict['delta_array']
    delta_step = delta_array[1] - delta_array[0]
    num_deltas = len(delta_array)
    num_samples = len(run_dict['sample_list'])
    # Get deltas for which the calculations were successful
    success_matrix = np.zeros((len(run_dict['sample_list']), len(run_dict['delta_array'])), dtype=bool)
    success_matrix[:, :] = False
    for isample, sample in enumerate(run_dict['sample_list']):
        s_str = f'sample_{sample}'
        for idelta in run_dict[s_str]['conv_idelta_list']:
            success_matrix[isample, idelta] = True

    num_total = success_matrix.size
    num_converged = np.sum(success_matrix)

    print('')
    print(f'{"Run Stats":=^60}')
    print(f'{"Total number of runs":>30}: {num_total}')
    print(f'{"Number of converged runs":>30}: {num_converged} ({num_converged/num_total:.1%})')
    print(f'{"Number of samples":>30}: {num_samples}')
    print(f'{"Number of deltas":>30}: {num_deltas}')
    print(f'{"Delta step":>30}: {delta_step:.2f}')
    print('='*60)
    print('')

    return success_matrix


def plot_success_matrix(ax, success_matrix, delta_array):

    #cmap = plt.get_cmap('coolwarm')
    #cmap = plt.get_cmap('Dark2_r')
    cmap = plt.get_cmap('Accent_r')
    ax.imshow(success_matrix, cmap=cmap, aspect='auto')
    nstep = 5
    ax.set_xticks(nstep * np.array(range(len(delta_array[::nstep]))))
    ax.set_xticklabels([f'{delta:.0f}' for delta in delta_array[::nstep]])
    ax.set_xlabel('Perturbation Magnitude $\lambda$')
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

    ax.set_xlabel('Perturbation Magnitude $\lambda$')
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
    try:
        epsilon_array_en_vol = run_dict['epsilon_array_en_vol_pure']
    except KeyError:
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
        cbar.set_label('Perturbation Magnitude $\lambda$', labelpad=label_pad)

    if second_xaxis:
        ax2 = ax.twiny()
        d_str = f'delta_{run_dict[s_str]["conv_idelta_list"][0]}'
        volume0 = run_dict[s_str][d_str]['pure']['npt']['volume0']
        volume_array = [volume0 * (1.0 + epsilon)**3 for epsilon in epsilon_array_en_vol]
        ax2.set_xlim(volume_array[0], volume_array[-1])
        ax2.set_xlabel('Volume ($\mathrm{\AA}^3$)')


def plot_formation_energy(ax, run_dict, sample, method_plot_dict):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']
    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_full_pred', 'energy_true']

    for i, en_key in enumerate(en_key_list):
        E_form = compute_formation_property(run_dict, sample, en_key)

        kwargs = method_plot_dict['formation'][en_key].copy()
        kwargs.pop('marker', None)
        kwargs['lw'] = 6.0

        if en_key == 'energy_full_pred':
            # custom ls
            kwargs['ls'] = (0, (3, 2))
            kwargs['zorder'] = 3

        ax.plot(delta_array_sample, E_form, **kwargs)

    ax.set_xlabel('Perturbation Magnitude $\lambda$')
    ax.set_ylabel('Formation Energy (eV)')


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

    ax.set_xlabel('Perturbation Magnitude $\lambda$')

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


def plot_absolute_volume(ax, run_dict, sample, method_plot_dict):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']

    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'volume_true'), label='Pure True', marker='s', c='black')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'pure', 'volume_pred'), label='Pure Pred. from dP/dV', marker='s')

    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'volume_true'), label='Vac. True', marker='o', c='black')
    ax.plot(delta_array_sample, get_prop_array(run_dict, sample, 'vac', 'volume_pred'), label='Vac. Pred. from dP/dV', marker='o')

    ax.set_xlabel('Perturbation Magnitude $\lambda$')
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

    ax.set_xlabel('Perturbation Magnitude $\lambda$')
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

    ax.set_xlabel('Perturbation Magnitude $\lambda$')
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

    ax.set_xlabel('Perturbation Magnitude $\lambda$')

    if error_type == 'abs':
        ax.set_ylabel('Energy Error (eV)')
    elif error_type == 'rel':
        ax.set_ylabel('Energy Error (%)')

    ax.legend()


def plot_formation_volume(ax, run_dict, sample, method_plot_dict, plot_no_change=True):

    s_str = f'sample_{sample}'
    delta_array = run_dict['delta_array']
    delta_array_sample = [delta_array[i] for i in run_dict[s_str]['conv_idelta_list']]

    # One for pure, another for vacacny
    #vol_form_pred_DT = compute_formation_property(run_dict, sample, 'volume_pred_DT', 'volume_pred')

    vol_key_list = ['volume_pred', 'volume_pred_full', 'volume_true']
    lw = 6.0
    if plot_no_change:
        vol_form = compute_formation_property(run_dict, sample, 'volume_true')
        delta_array = run_dict['delta_array']
        idelta0 = np.argmin(np.abs(delta_array))

        vol_pure_true0 = run_dict['sample_0'][f'delta_{idelta0}']['pure']['npt']['volume_true']
        vol_vac_true0 = run_dict['sample_0'][f'delta_{idelta0}']['vac']['npt']['volume_true']
        vol_form_true0 = vol_vac_true0 - vol_pure_true0 * run_dict['Natom vac'] / run_dict['Natom pure']

        vol_no_change_array = np.zeros_like(vol_form)
        vol_no_change_array[:] = vol_form_true0
        kwargs = method_plot_dict['formation']['energy_pred0'].copy()
        kwargs.pop('marker', None)
        kwargs['lw'] = lw

        ax.plot(delta_array_sample, vol_no_change_array, zorder=-1, **kwargs)

    for i, vol_key in enumerate(vol_key_list):
        vol_form = compute_formation_property(run_dict, sample, vol_key)
        kwargs = method_plot_dict['formation'][vol_key].copy()
        kwargs.pop('marker', None)
        kwargs['lw'] = lw

        if vol_key == 'volume_pred_full':
            # custom ls
            kwargs['ls'] = (0, (3, 2))
            kwargs['zorder'] = 3

        ax.plot(delta_array_sample, vol_form, **kwargs)

    ax.set_xlabel('Perturbation Magnitude $\lambda$')
    ax.set_ylabel('Formation Volume ($\mathrm{\AA}^3$)')


def plot_formation_volume_scatter(ax, run_dict, method_plot_dict, label_fsize=20, tick_fsize=16,
                                  V_form_range=None, delta_range=None, plot_no_change=True):

    vol_key_list = ['volume_pred', 'volume_pred_full']

    # iterate over samples
    for sample in run_dict['sample_list']:

        # True formation volume
        vol_form_true = compute_formation_property(run_dict, sample, 'volume_true', delta_range=delta_range)

        if V_form_range is not None:
            mask = (vol_form_true > V_form_range[0]) & (vol_form_true < V_form_range[1])
        else:
            mask = np.ones_like(vol_form_true, dtype=bool)

        if plot_no_change:
            delta_array = run_dict['delta_array']
            idelta0 = np.argmin(np.abs(delta_array))

            vol_pure_true0 = run_dict['sample_0'][f'delta_{idelta0}']['pure']['npt']['volume_true']
            vol_vac_true0 = run_dict['sample_0'][f'delta_{idelta0}']['vac']['npt']['volume_true']
            vol_form_true0 = vol_vac_true0 - vol_pure_true0 * run_dict['Natom vac'] / run_dict['Natom pure']

            vol_no_change_array = np.zeros_like(vol_form_true)
            vol_no_change_array[:] = vol_form_true0
            color = method_plot_dict['formation']['volume_pred0']['c']
            ax.scatter(vol_form_true[mask], vol_no_change_array[mask], c=color, s=30, marker='o')

        for i, vol_key in enumerate(vol_key_list):
            vol_form = compute_formation_property(run_dict, sample, vol_key, delta_range=delta_range)
            color = method_plot_dict['formation'][vol_key]['c']
            #size = method_plot_dict['formation'][vol_key]['ms'] * 2
            size = method_plot_dict['formation'][vol_key]['ms'] * 5
            marker = method_plot_dict['formation'][vol_key]['marker']

            ax.scatter(vol_form_true[mask], vol_form[mask], color=color, s=size, marker=marker)

    # Create the legend manually
    ax.scatter([], [], c=method_plot_dict['formation']['volume_pred0']['c'], s=100, marker='o', label='Constant')
    for i, vol_key in enumerate(vol_key_list):

        s = 100 if i == 0 else 150

        ax.scatter([], [], color=method_plot_dict['formation'][vol_key]['c'], s=s,
                   marker=method_plot_dict['formation'][vol_key]['marker'], label=method_plot_dict['formation'][vol_key]['label'])

    # Plot diagonal line
    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [xmin, xmax], c='black', lw=3.0, ls='--', zorder=0)

    #for i, vol_key in enumerate(vol_key_list):
    #    ax.plot([0, 100], [0, 100], c='black', lw=1.0)

    # Setup the legend manually for vol_key from vol_key_list
    #for i, vol_key in enumerate(vol_key_list):
        #line

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fsize)

    ax.set_xlabel('True Formation Volume ($\mathrm{\AA}^3$)', fontsize=label_fsize)
    ax.set_ylabel('Formation Volume ($\mathrm{\AA}^3$)', fontsize=label_fsize)


def plot_formation_energy_error_bins(ax, bin_error_dict, method_plot_dict, fill=True, spline_fill=True, print_labels=True, label_fsize=20, tick_fsize=16):

    # Energies
    # en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']
    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_full_pred']

    dE_bin_centers = bin_error_dict['energy']['bin_centers']

    for i, en_key in enumerate(en_key_list):

        bin_average = bin_error_dict['energy'][en_key]['average']
        bin_perc_16 = bin_error_dict['energy'][en_key]['perc_16']
        bin_perc_84 = bin_error_dict['energy'][en_key]['perc_84']
        bin_std = bin_error_dict['energy'][en_key]['std']

        ax.plot(dE_bin_centers, bin_average, **method_plot_dict['formation'][en_key])

        if fill:
            if spline_fill:
                dE_grid, bin_16_interp = interpolate(dE_bin_centers, bin_perc_16)
                dE_grid, bin_84_interp = interpolate(dE_bin_centers, bin_perc_84)
                ax.fill_between(dE_grid, bin_16_interp, bin_84_interp, alpha0=0.2,
                                color=method_plot_dict['formation'][en_key]['c'])
            else:
                ax.fill_between(dE_bin_centers, bin_perc_16, bin_perc_84, alpha0=0.2,
                                color=method_plot_dict['formation'][en_key]['c'])

    # axes ticklabel font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fsize)

    if print_labels:
        ax.set_ylabel('Energy Error (eV)', fontsize=label_fsize)

        ax.set_xlabel('Formation Energy Change (eV)', fontsize=label_fsize)
        ax.set_ylabel('Formation Energy Error (eV)', fontsize=label_fsize)


def plot_formation_volume_error_bins(ax, bin_error_dict, method_plot_dict, spline_fill=True, av_suffix=''):

    vol_key_list = ['volume_pred0', 'volume_pred', 'volume_pred_full']#, 'volume_pred_d2Desc']

    dV_bin_centers = bin_error_dict['volume']['bin_centers']

    for i, vol_key in enumerate(vol_key_list):

        bin_average = bin_error_dict['volume'][vol_key][f'average{av_suffix}']
        bin_perc_16 = bin_error_dict['volume'][vol_key][f'perc_16{av_suffix}']
        bin_perc_84 = bin_error_dict['volume'][vol_key][f'perc_84{av_suffix}']

        ax.plot(dV_bin_centers, bin_average, **method_plot_dict['formation'][vol_key])

        if vol_key == 'volume_pred_full':
            continue

        if spline_fill:
            dV_grid, bin_16_interp = interpolate(dV_bin_centers, bin_perc_16)
            dV_grid, bin_84_interp = interpolate(dV_bin_centers, bin_perc_84)
            ax.fill_between(dV_grid, bin_16_interp, bin_84_interp, alpha0=0.2, color=method_plot_dict['formation'][vol_key]['c'])
        else:
            ax.fill_between(dV_bin_centers, bin_perc_16, bin_perc_84, color=method_plot_dict['formation'][vol_key]['c'], alpha0=0.2)

    ax.set_xlabel('Formation Volume Change ($\mathrm{\AA}^3$)')
    ax.set_ylabel('Formation Volume Error ($\mathrm{\AA}^3$)')


def plot_formation_energy_bins(ax, run_dict, bin_energy_dict, method_plot_dict, spline_fill=True, plot_diff=False):

    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']

    delta_array = run_dict['delta_array']
    idelta0 = np.argmin(np.abs(delta_array))

    en_vac_true0 = run_dict['sample_0'][f'delta_{idelta0}']['vac']['npt']['energy_true']
    en_pure_true0 = run_dict['sample_0'][f'delta_{idelta0}']['pure']['npt']['energy_true']
    E_form_true0 = en_vac_true0 - en_pure_true0 * run_dict['Natom vac'] / run_dict['Natom pure']

    E_form_bin_centers = bin_energy_dict['E_form_bin_centers']

    if plot_diff:
        dE = E_form_true0
    else:
        dE = 0.0

    for i, en_key in enumerate(en_key_list):

        bin_average = bin_energy_dict[en_key]['median']
        bin_perc_16 = bin_energy_dict[en_key]['perc_16']
        bin_perc_84 = bin_energy_dict[en_key]['perc_84']

        plot_kwargs = method_plot_dict['formation'][en_key].copy()
        plot_kwargs['ms'] = 14
        plot_kwargs['lw'] = 5.0

        ax.plot(E_form_bin_centers-dE, bin_average-dE, **plot_kwargs)

        if spline_fill:
            dE_grid, bin_16_interp = interpolate(E_form_bin_centers, bin_perc_16)
            dE_grid, bin_84_interp = interpolate(E_form_bin_centers, bin_perc_84)
            ax.fill_between(dE_grid-dE, bin_16_interp-dE, bin_84_interp-dE, alpha0=0.2, color=method_plot_dict['formation'][en_key]['c'])
        else:
            ax.fill_between(E_form_bin_centers-dE, bin_perc_16-dE, bin_perc_84-dE, color=method_plot_dict['formation'][en_key]['c'], alpha0=0.2)

    if plot_diff:
        ax.set_xlabel('Formation Energy Change (eV)')
        ax.set_ylabel('Pred. Form. Energy Change(eV)')
    else:
        ax.set_xlabel('True Formation Energy (eV)')
        ax.set_ylabel('Predicted Formation Energy (eV)')


def plot_formation_volume_bins(ax, run_dict, bin_vol_dict, method_plot_dict, fill=True, spline_fill=True, label_fsize=20, tick_fsize=16, atomic_V_range=None):

    vol_key_list = ['volume_pred', 'volume_pred_full', 'volume_true']

    V_bin_centers = bin_vol_dict['V_bin_centers'].copy()
    atomic_volume = V_bin_centers / run_dict['Natom pure']

    # No pos change
    delta_array = run_dict['delta_array']
    idelta0 = np.argmin(np.abs(delta_array))

    vol_pure_true0 = run_dict['sample_0'][f'delta_{idelta0}']['pure']['npt']['volume_true']
    vol_vac_true0 = run_dict['sample_0'][f'delta_{idelta0}']['vac']['npt']['volume_true']
    vol_form_true0 = vol_vac_true0 - vol_pure_true0 * run_dict['Natom vac'] / run_dict['Natom pure']

    vol_no_change_array = np.zeros_like(V_bin_centers)
    vol_no_change_array[:] = vol_form_true0
    kwargs = method_plot_dict['formation']['energy_pred0'].copy()
    kwargs['lw'] = 5.0
    #kwargs.pop('marker', None)

    if atomic_V_range is not None:
        mask = (atomic_volume > atomic_V_range[0]) & (atomic_volume < atomic_V_range[1])
        atomic_volume = atomic_volume[mask]
        vol_no_change_array = vol_no_change_array[mask]

    ax.plot(atomic_volume, vol_no_change_array, zorder=-1, **kwargs)

    for i, vol_key in enumerate(vol_key_list):

        bin_average = bin_vol_dict[vol_key]['median'].copy()
        bin_perc_16 = bin_vol_dict[vol_key]['perc_16'].copy()
        bin_perc_84 = bin_vol_dict[vol_key]['perc_84'].copy()

        if atomic_V_range is not None:
            bin_average = bin_average[mask]
            bin_perc_16 = bin_perc_16[mask]
            bin_perc_84 = bin_perc_84[mask]

        ax.plot(atomic_volume, bin_average, **method_plot_dict['formation'][vol_key])

        if vol_key == 'volume_pred':
            continue

        if fill:
            if spline_fill:
                dV_grid, bin_16_interp = interpolate(atomic_volume, bin_perc_16)
                dV_grid, bin_84_interp = interpolate(atomic_volume, bin_perc_84)
                ax.fill_between(dV_grid, bin_16_interp, bin_84_interp, alpha0=0.2, color=method_plot_dict['formation'][vol_key]['c'])
            else:
                ax.fill_between(atomic_volume, bin_perc_16, bin_perc_84, color=method_plot_dict['formation'][vol_key]['c'], alpha0=0.2)

    # x-axis multiple locator of 0.05
    ax.xaxis.set_major_locator(MultipleLocator(0.02))

    ax.tick_params(axis='both', which='major', labelsize=tick_fsize)

    ax.set_xlabel('Atomic Volume ($\mathrm{\AA}^3$)', fontsize=label_fsize)
    ax.set_ylabel('Formation Volume ($\mathrm{\AA}^3$)', fontsize=label_fsize)


def plot_formation_energy_error_average(ax, average_dict, method_plot_dict):

    delta_array = average_dict['detla_array']
    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']

    for i, en_key in enumerate(en_key_list):

        form_error_array = np.array([average_dict['error']['formation'][en_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        form_error_std = np.array([average_dict['error']['formation'][en_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        ax.plot(delta_array, form_error_array, **method_plot_dict['formation'][en_key])
        ax.fill_between(delta_array, form_error_array - form_error_std, form_error_array + form_error_std,
                        color=method_plot_dict['formation'][en_key]['c'], alpha0=0.2)

    ax.set_xlabel('Perturbation Magnitude $\lambda$')
    ax.set_ylabel('Formation Energy Error (eV)')


def plot_formation_volume_error_average(ax, average_dict, method_plot_dict, av_suffix='', spline_fill=True, no_markers=True):

    delta_array = average_dict['detla_array']

    if av_suffix == '':
        vol_key_list = ['volume_pred', 'volume_pred_full']
    else:
        vol_key_list = ['volume_pred0', 'volume_pred', 'volume_pred_full']

    for i, vol_key in enumerate(vol_key_list):

        form_error_array = np.array([average_dict['error']['formation'][vol_key][f'delta_{i}'][f'average{av_suffix}'] for i in range(len(delta_array))])

        form_error_16 = np.array([average_dict['error']['formation'][vol_key][f'delta_{i}'][f'perc_16{av_suffix}'] for i in range(len(delta_array))])
        form_error_84 = np.array([average_dict['error']['formation'][vol_key][f'delta_{i}'][f'perc_84{av_suffix}'] for i in range(len(delta_array))])

        plot_kwargs = method_plot_dict['formation'][vol_key].copy()
        plot_kwargs['lw'] = 5
        if vol_key == 'volume_pred_full':
            plot_kwargs['ls'] = '--'

        if no_markers:
            plot_kwargs.pop('marker', None)
        ax.plot(delta_array, form_error_array, **plot_kwargs)

        if spline_fill and vol_key == 'volume_pred':
            dE_grid, form_error_16_interp = interpolate(delta_array, form_error_16)
            dE_grid, form_error_84_interp = interpolate(delta_array, form_error_84)
            ax.fill_between(dE_grid, form_error_16_interp, form_error_84_interp, alpha0=0.2,
                            color=method_plot_dict['formation'][vol_key]['c'])

        elif vol_key == 'volume_pred':
            ax.fill_between(delta_array, form_error_16_interp, form_error_84_interp, alpha0=0.2,
                            color=method_plot_dict['formation'][vol_key]['c'])

    ax.set_xlabel('Perturbation Magnitude $\lambda$')

    if av_suffix == '_eq':
        ax.set_ylabel('Formation Volume Error ($\mathrm{\AA}^6$)')
    else:
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
        #ax.fill_between(delta_array, pure_error_array - pure_error_std, pure_error_array + pure_error_std, color=color_list[i], alpha0=0.2)

        ax.plot(delta_array, vac_error_array, **method_plot_dict['vac'][en_key])
        #ax.fill_between(delta_array, vac_error_array - vac_error_std, vac_error_array + vac_error_std, color=color_list[i], alpha0=0.2)

    ax.set_ylabel('Energy Error (eV)')


def plot_volume_error_average(ax, average_dict, method_plot_dict):
    delta_array = average_dict['detla_array']

    vol_key_list = ['volume_pred']#, 'volume_pred_DT']
    color_list = ['tab:blue', 'tab:orange']

    for i, vol_key in enumerate(vol_key_list):

        pure_error_array = np.array([average_dict['error']['pure'][vol_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        pure_error_std = np.array([average_dict['error']['pure'][vol_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        vac_error_array = np.array([average_dict['error']['vac'][vol_key][f'delta_{i}'][0] for i in range(len(delta_array))])
        vac_error_std = np.array([average_dict['error']['vac'][vol_key][f'delta_{i}'][1] for i in range(len(delta_array))])

        ax.plot(delta_array, pure_error_array, **method_plot_dict['pure'][vol_key])
        ax.fill_between(delta_array, pure_error_array - pure_error_std, pure_error_array + pure_error_std, color=color_list[i], alpha0=0.2)

        ax.plot(delta_array, vac_error_array, **method_plot_dict['vac'][vol_key])
        ax.fill_between(delta_array, vac_error_array - vac_error_std, vac_error_array + vac_error_std, color=color_list[i], alpha0=0.2)

    ax.set_ylabel('Volume Error ($\mathrm{\AA}^3$)')


def plot_LJ_error(ax, LJ_dict, method_plot_dict):

    LJ_to_method = {
                    'Constant': 'energy_pred0',
                    'Inhomogeneous': 'energy_full_pred',
                    }

    label_dict = {'Constant': 'Constant / Homogeneous',
                  'Inhomogeneous': 'Inhomogeneous',
                  }

    sigma_AB = LJ_dict['sigma_AB']

    for key in LJ_to_method:
        error_array = LJ_dict['error'][key]

        kwargs = method_plot_dict['formation'][LJ_to_method[key]].copy()
        kwargs.pop('marker', None)
        kwargs['lw'] = 7
        kwargs['label'] = label_dict[key]

        ax.plot(sigma_AB, error_array, **kwargs)

    fsize = 24
    ax.set_xlabel(r"Lennard-Jones Parameter $\sigma_{\rm AB}$", fontsize=fsize)
    #eps_unicode = '\U+03F5'
    # \U+03F5 is the unicode for the greek letter epsilon
    eps_unicode = '\u03F5'

    ax.set_ylabel('Energy error'+f' ({eps_unicode}'+r'$^{\rm LJ}$)', fontsize=fsize)
    #ax.set_ylabel(r'Error ($\epsilon^{\rm LJ}$)', fontsize=fsize)


def plot_LJ_distortion(ax, LJ_dict, method_plot_dict):

    LJ_to_method = {
                    'Constant': 'energy_pred0',
                    'Inhomogeneous': 'energy_full_pred',
                    'True': 'energy_true',
                    }

    label_dict = {'Constant': 'Constant / Homogeneous',
                  'Inhomogeneous': 'Inhomogeneous',
                  'True': 'True (Const. Vol.)',
                  }

    sigma_AB = LJ_dict['sigma_AB']

    for key in LJ_to_method:

        kwargs = method_plot_dict['formation'][LJ_to_method[key]].copy()
        kwargs.pop('marker', None)
        kwargs['lw'] = 7
        kwargs['label'] = label_dict[key]

        distortion_array = LJ_dict['distortion'][key]
        ax.plot(sigma_AB, distortion_array, **kwargs)

    fsize = 24
    ax.set_xlabel(r"Lennard-Jones Parameter $\sigma_{\rm AB}$", fontsize=fsize)
    ax.set_ylabel(r'Lattice distortion ($a^{\rm LJ}$)', fontsize=fsize)


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
    vol_key_list = ['volume_pred0', 'volume_pred', 'volume_pred_full']#, 'volume_pred_DT'] #, 'volume_pred_d2Desc']
    prop_key_list = en_key_list + vol_key_list

    # No pos change volumes
    idelta0 = np.argmin(np.abs(delta_array))
    vol_pure_true0 = run_dict['sample_0'][f'delta_{idelta0}']['pure']['npt']['volume_true']
    vol_vac_true0 = run_dict['sample_0'][f'delta_{idelta0}']['vac']['npt']['volume_true']
    vol_form_true0 = vol_vac_true0 - vol_pure_true0 * run_dict['Natom vac'] / run_dict['Natom pure']

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
                if vol_key == 'volume_pred0':
                    average_dict['error']['pure'][vol_key][d_str].append(vol_pure_true0 - vol_pure_true)
                    average_dict['error']['vac'][vol_key][d_str].append(vol_vac_true0 - vol_vac_true)
                    average_dict['error']['formation'][vol_key][d_str].append(vol_form_true0 - vol_form_true)

                else:
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
            for key, array in zip(['pure', 'vac', 'formation'], [pure_array, vac_array, form_array]):
                average_dict['error'][key][prop_key][d_str] = {}
                average_dict['error'][key][prop_key][d_str]['average'] = np.mean(array)
                average_dict['error'][key][prop_key][d_str]['perc_16'] = np.percentile(array, 16)
                average_dict['error'][key][prop_key][d_str]['perc_84'] = np.percentile(array, 84)

                average_dict['error'][key][prop_key][d_str]['average_sq'] = np.mean(array**2)
                average_dict['error'][key][prop_key][d_str]['perc_16_sq'] = np.percentile(array**2, 16)
                average_dict['error'][key][prop_key][d_str]['perc_84_sq'] = np.percentile(array**2, 84)

                average_dict['error'][key][prop_key][d_str]['average_abs'] = np.mean(np.abs(array))
                average_dict['error'][key][prop_key][d_str]['perc_16_abs'] = np.percentile(np.abs(array), 16)
                average_dict['error'][key][prop_key][d_str]['perc_84_abs'] = np.percentile(np.abs(array), 84)

    return average_dict


def compute_error_bins(run_dict, E_range=None, V_range=None, dE_num_bins=50, dV_num_bins=50):
    """
    Form. energy and form. volume ERRORS as a function of form. energy and form. volume
    """

    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']
    vol_key_list = ['volume_pred0', 'volume_pred', 'volume_pred_full']
    prop_key_list = en_key_list + vol_key_list

    bin_error_dict = {}
    bin_error_dict['energy'] = {}
    bin_error_dict['volume'] = {}

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

    print(f'Formation energy at delta=0: {en_form_true0:.3f} eV; range of {emin:.3f} to {emax:.3f} eV')
    print(f'Formation volume at delta=0: {vol_form_true0:.3f} A^3; range of {vmin:.3f} to {vmax:.3f} A^3')

    dE_bins = np.linspace(emin, emax, dE_num_bins + 1)
    dE_bin_centers = 0.5 * (dE_bins[:-1] + dE_bins[1:])
    bin_error_dict['energy']['bins'] = dE_bins
    bin_error_dict['energy']['bin_centers'] = dE_bin_centers
    bin_error_dict['energy']['at delta 0'] = en_form_true0

    dV_bins = np.linspace(vmin, vmax, dV_num_bins + 1)
    dV_bin_centers = 0.5 * (dV_bins[:-1] + dV_bins[1:])
    bin_error_dict['volume']['bins'] = dV_bins
    bin_error_dict['volume']['bin_centers'] = dV_bin_centers
    bin_error_dict['volume']['at delta 0'] = vol_form_true0

    for en_key in en_key_list:
        bin_error_dict['energy'][en_key] = {}
        bin_error_dict['energy'][en_key]['bin_errors'] = [[] for _ in range(dE_num_bins)]
    for vol_key in vol_key_list:
        bin_error_dict['volume'][vol_key] = {}
        bin_error_dict['volume'][vol_key]['bin_errors'] = [[] for _ in range(dV_num_bins)]

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
                    bin_error_dict['energy'][en_key]['bin_errors'][dE_bin_index].append(en_form_error)

            if 0 <= dV_bin_index < dV_num_bins:
                for vol_key in vol_key_list:

                    if vol_key == 'volume_pred0':
                        bin_error_dict['volume'][vol_key]['bin_errors'][dV_bin_index].append(vol_form_true0 - vol_form_true)

                    else:
                        vol_pure = run_dict[s_str][d_str]['pure']['npt'][vol_key]
                        vol_vac = run_dict[s_str][d_str]['vac']['npt'][vol_key]
                        vol_form = vol_vac - vol_pure * run_dict['Natom vac'] / run_dict['Natom pure']
                        vol_form_error = vol_form - vol_form_true
                        bin_error_dict['volume'][vol_key]['bin_errors'][dV_bin_index].append(vol_form_error)

    # Compute the average and percentiles
    for en_key in en_key_list:
        bin_errors = bin_error_dict['energy'][en_key]['bin_errors']
        bin_error_dict['energy'][en_key]['average'] = np.array([np.percentile(errors, 50) if errors else np.nan for errors in bin_errors])
        bin_error_dict['energy'][en_key]['perc_16'] = np.array([np.percentile(errors, 16) if errors else np.nan for errors in bin_errors])
        bin_error_dict['energy'][en_key]['perc_84'] = np.array([np.percentile(errors, 84) if errors else np.nan for errors in bin_errors])

        bin_error_dict['energy'][en_key]['average_abs'] = np.array([np.mean(np.abs(errors)) if errors else np.nan for errors in bin_errors])
        bin_error_dict['energy'][en_key]['perc_16_abs'] = np.array([np.percentile(np.abs(errors), 16) if errors else np.nan for errors in bin_errors])
        bin_error_dict['energy'][en_key]['perc_84_abs'] = np.array([np.percentile(np.abs(errors), 84) if errors else np.nan for errors in bin_errors])

        bin_error_dict['energy'][en_key]['std'] = np.array([np.std(errors) if errors else np.nan for errors in bin_errors])

    for vol_key in vol_key_list:
        bin_errors = bin_error_dict['volume'][vol_key]['bin_errors']
        bin_error_dict['volume'][vol_key]['average'] = np.array([np.percentile(errors, 50) if errors else np.nan for errors in bin_errors])
        bin_error_dict['volume'][vol_key]['perc_16'] = np.array([np.percentile(errors, 16) if errors else np.nan for errors in bin_errors])
        bin_error_dict['volume'][vol_key]['perc_84'] = np.array([np.percentile(errors, 84) if errors else np.nan for errors in bin_errors])

        bin_error_dict['volume'][vol_key]['average_abs'] = np.array([np.mean(np.abs(errors)) if errors else np.nan for errors in bin_errors])
        bin_error_dict['volume'][vol_key]['perc_16_abs'] = np.array([np.percentile(np.abs(errors), 16) if errors else np.nan for errors in bin_errors])
        bin_error_dict['volume'][vol_key]['perc_84_abs'] = np.array([np.percentile(np.abs(errors), 84) if errors else np.nan for errors in bin_errors])

        bin_error_dict['volume'][vol_key]['std'] = np.array([np.std(errors) if errors else np.nan for errors in bin_errors])

    return bin_error_dict


def compute_form_volume_bins(run_dict, V_range=None, V_num_bins=50):
    """
    Predicted and true formation volume as a function of bulk volume
    """

    bin_vol_dict = {}

    vol_key_list = ['volume_true', 'volume_pred', 'volume_pred_full']

    delta_array = run_dict['delta_array']
    idelta0 = np.argmin(np.abs(delta_array))
    V_pure_true0 = run_dict['sample_0'][f'delta_{idelta0}']['pure']['npt']['volume_true']

    # Find the volume range
    if V_range is None:
        V_min, V_max = V_pure_true0, V_pure_true0

        for sample in run_dict['sample_list']:
            s_str = f'sample_{sample}'
            # Only the filtered deltas
            delta_sample_list = run_dict[s_str]['conv_idelta_list']
            for delta in delta_sample_list:
                d_str = f'delta_{delta}'
                V_pure_true = run_dict[s_str][d_str]['pure']['npt']['volume_true']

                if V_pure_true < V_min:
                    V_min = V_pure_true
                elif V_pure_true > V_max:
                    V_max = V_pure_true
    else:
        V_min, V_max = V_range

    print(f'Bulk volume at delta=0: {V_pure_true0:.3f} A^3. Range: from {V_min:.3f} to {V_max:.3f} A^3')

    V_bins = np.linspace(V_min, V_max, V_num_bins + 1)
    V_bin_centers = 0.5 * (V_bins[:-1] + V_bins[1:])

    bin_vol_dict['V_bins'] = V_bins
    bin_vol_dict['V_bin_centers'] = V_bin_centers
    bin_vol_dict['at delta 0'] = V_pure_true0

    for vol_key in vol_key_list:
        bin_vol_dict[vol_key] = {}
        bin_vol_dict[vol_key]['bin_vals'] = [[] for _ in range(V_num_bins)]

    for sample in run_dict['sample_list']:
        s_str = f'sample_{sample}'
        # Only the filtered deltas
        delta_sample_list = run_dict[s_str]['conv_idelta_list']
        for delta in delta_sample_list:
            d_str = f'delta_{delta}'

            vol_pure_true = run_dict[s_str][d_str]['pure']['npt']['volume_true']
            V_bin_index = np.digitize(vol_pure_true, V_bins) - 1

            if 0 <= V_bin_index < V_num_bins:
                for vol_key in vol_key_list:
                    vol_pure = run_dict[s_str][d_str]['pure']['npt'][vol_key]
                    vol_vac = run_dict[s_str][d_str]['vac']['npt'][vol_key]
                    vol_form = vol_vac - vol_pure * run_dict['Natom vac'] / run_dict['Natom pure']

                    bin_vol_dict[vol_key]['bin_vals'][V_bin_index].append(vol_form)

    for vol_key in vol_key_list:
        bin_vals = bin_vol_dict[vol_key]['bin_vals']
        # To compute percentiles, bin_vals should not be empty, vals should not be empty
        bin_vol_dict[vol_key]['median'] = np.array([np.percentile(vals, 50) if bin_vals else np.nan for vals in bin_vals if vals])
        bin_vol_dict[vol_key]['perc_16'] = np.array([np.percentile(vals, 16) if bin_vals else np.nan for vals in bin_vals if vals])
        bin_vol_dict[vol_key]['perc_84'] = np.array([np.percentile(vals, 84) if bin_vals else np.nan for vals in bin_vals if vals])

    return bin_vol_dict


def compute_formation_energy_bins(run_dict, E_form_range=None, E_form_num_bins=50):
    """
    Predicted and true formation volume as a function of bulk volume
    """

    bin_en_dict = {}

    en_key_list = ['energy_pred0', 'energy_hom_pred', 'energy_inhom_pred', 'energy_full_pred']

    delta_array = run_dict['delta_array']
    idelta0 = np.argmin(np.abs(delta_array))

    en_vac_true0 = run_dict['sample_0'][f'delta_{idelta0}']['vac']['npt']['energy_true']
    en_pure_true0 = run_dict['sample_0'][f'delta_{idelta0}']['pure']['npt']['energy_true']
    E_form_true0 = en_vac_true0 - en_pure_true0 * run_dict['Natom vac'] / run_dict['Natom pure']

    # Find the volume range
    if E_form_range is None:
        E_form_min, E_form_max = E_form_true0, E_form_true0

        for sample in run_dict['sample_list']:
            s_str = f'sample_{sample}'
            # Only the filtered deltas
            delta_sample_list = run_dict[s_str]['conv_idelta_list']
            for delta in delta_sample_list:
                d_str = f'delta_{delta}'
                en_pure_true = run_dict[s_str][d_str]['pure']['npt']['energy_true']
                en_vac_true = run_dict[s_str][d_str]['vac']['npt']['energy_true']
                E_form_true = en_vac_true - en_pure_true * run_dict['Natom vac'] / run_dict['Natom pure']

                if E_form_true < E_form_min:
                    E_form_min = E_form_true
                elif E_form_true > E_form_max:
                    E_form_max = E_form_true

    else:
        E_form_min, E_form_max = E_form_range

    print(f'Formation energy at delta=0: {E_form_true0:.3f} eV. Range: from {E_form_min:.3f} to {E_form_max:.3f} eV')

    E_form_bins = np.linspace(E_form_min, E_form_max, E_form_num_bins + 1)
    E_form_bin_centers = 0.5 * (E_form_bins[:-1] + E_form_bins[1:])

    bin_en_dict['E_form_bins'] = E_form_bins
    bin_en_dict['E_form_bin_centers'] = E_form_bin_centers
    bin_en_dict['at delta 0'] = E_form_true0

    for en_key in en_key_list:
        bin_en_dict[en_key] = {}
        bin_en_dict[en_key]['bin_vals'] = [[] for _ in range(E_form_num_bins)]

    for sample in run_dict['sample_list']:
        s_str = f'sample_{sample}'
        # Only the filtered deltas
        delta_sample_list = run_dict[s_str]['conv_idelta_list']
        for delta in delta_sample_list:
            d_str = f'delta_{delta}'

            en_pure_true = run_dict[s_str][d_str]['pure']['npt']['energy_true']
            en_vac_true = run_dict[s_str][d_str]['vac']['npt']['energy_true']
            E_form_true = en_vac_true - en_pure_true * run_dict['Natom vac'] / run_dict['Natom pure']

            E_form_bin_index = np.digitize(E_form_true, E_form_bins) - 1

            if 0 <= E_form_bin_index < E_form_num_bins:
                for en_key in en_key_list:
                    en_pure = run_dict[s_str][d_str]['pure']['npt'][en_key]
                    en_vac = run_dict[s_str][d_str]['vac']['npt'][en_key]
                    en_form = en_vac - en_pure * run_dict['Natom vac'] / run_dict['Natom pure']

                    bin_en_dict[en_key]['bin_vals'][E_form_bin_index].append(en_form)

    for en_key in en_key_list:
        bin_vals = bin_en_dict[en_key]['bin_vals']
        # To compute percentiles, bin_vals should not be empty, vals should not be empty
        bin_en_dict[en_key]['median'] = np.array([np.percentile(vals, 50) if bin_vals else np.nan for vals in bin_vals if vals])
        bin_en_dict[en_key]['perc_16'] = np.array([np.percentile(vals, 16) if bin_vals else np.nan for vals in bin_vals if vals])
        bin_en_dict[en_key]['perc_84'] = np.array([np.percentile(vals, 84) if bin_vals else np.nan for vals in bin_vals if vals])

    return bin_en_dict


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


def get_run_dict(from_samples='False', save_run_dict=True):

    print('='*60)
    print('Reading the pickle files')
    print('='*60)
    if from_samples:
        run_dict = {}
        s_read_list = []
        for fname in os.listdir():
            if fname.startswith('sample_') and fname.endswith('.pkl'):
                s_str = fname.split('.')[0]
                s_read_list.append(int(s_str.split('_')[-1]))
                with open(fname, 'rb') as fstream:
                    sample_dict = pickle.load(fstream)

                run_dict[s_str] = sample_dict

        if len(s_read_list) == 0:
            raise RuntimeError('No sample_... .pkl files in the directory.')

        s_str0 = list(run_dict.keys())[0]
        for k, v in run_dict[s_str0]['system_info'].items():
            run_dict[k] = v

        s_read_list = sorted(s_read_list)

        run_dict['sample_list'] = s_read_list
        samples_missing = False
        print(f'Samples from {s_read_list[0]} to {s_read_list[-1]}')
        for isample in range(s_read_list[0], s_read_list[-1]):
            if isample not in s_read_list:
                print(f'!MISSING SAMPLE! {isample}')
                samples_missing = True
        if not samples_missing:
            print('No missing samples in the range')

        if save_run_dict:
            with open('run_dict.pkl', 'wb') as file:
                pickle.dump(run_dict, file)

    else:
        pickle_filename = 'run_dict.pkl'
        print(f'Reading {pickle_filename}...')
        with open(pickle_filename, 'rb') as f:
            run_dict = pickle.load(f)

    print('='*60+'\n')

    return run_dict


def main():

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    #from_samples = True
    from_samples = False
    run_dict = get_run_dict(from_samples=from_samples)
    method_plot_dict = setup_method_plot_dict()
    method_plot_dict2 = setup_method_plot_dict2()

    print(f'Number of atoms in pure system: {run_dict["Natom pure"]}')
    print(f'Number of atoms in vacancy system: {run_dict["Natom vac"]}')

    # filter data
    run_dict = filter_data(run_dict)
    run_dict = filter_data_energy_volume(run_dict, atol=1e-2, rtol=50.0)

    # Hard-remove deltas from -50.0 to 50.0
    #run_dict = cut_data(run_dict, delta_min=-50.0, delta_max=50.0)

    average_dict = average_data(run_dict)
    bin_error_dict = compute_error_bins(run_dict, E_range=(-1.5, 1.5), V_range=(-1.5, 1.5), dE_num_bins=25, dV_num_bins=15)

    success_matrix = get_success_matrix(run_dict)

    plot_succ_matrix = False
    #plot_succ_matrix = True
    if plot_succ_matrix:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_success_matrix(ax, success_matrix, run_dict['delta_array'])
        fig.savefig(os.path.join('plots', 'success_matrix.pdf'))

    #
    # Plotting physical data
    #
    sample = 0
    #sample = 20
    #sample = 62

    #V_range_bins = (2056, 2076)
    V_range_bins = None
    bin_vol_dict = compute_form_volume_bins(run_dict, V_range=V_range_bins, V_num_bins=40)
    bin_en_dict = compute_formation_energy_bins(run_dict, E_form_range=None, E_form_num_bins=20)

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
    #plot_2x2 = True
    if plot_2x2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(left=0.07, right=0.96, bottom=0.09, top=0.97, wspace=0.22, hspace=0.01)

        # Lattice constant
        plot_lattice_constant(axes[0, 0], run_dict, sample)
        axes[0, 0].set_xticklabels([])
        ax_inset = axes[0, 0].inset_axes([0.14, 0.14, 0.4, 0.4])
        plot_energy_volume_deltas(ax_inset, run_dict, sample, fsize_bar=12)

        # Formation energies
        plot_formation_energy(axes[1, 0], run_dict, sample, method_plot_dict)

        # Absolute volume
        plot_absolute_volume(axes[0, 1], run_dict, sample, method_plot_dict)
        axes[0, 1].set_xticklabels([])

        # Formation volume
        plot_formation_volume(axes[1, 1], run_dict, sample, method_plot_dict)

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
        plot_formation_energy(axes[1], run_dict, sample, method_plot_dict)
        axes[1].set_title(f'ncell_x={run_dict["ncell_x"]}; Natom={run_dict["Natom pure"]}; sample {sample}')

        # Formation volume
        plot_formation_volume(axes[2], run_dict, sample, method_plot_dict)

        plt.show()

    #
    # Layout 2x2
    #
    plot_2x2_new = True
    #plot_2x2_new = False
    if plot_2x2_new:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(left=0.08, right=0.93, bottom=0.07, top=0.90, wspace=0.2, hspace=0.2)

        # Energy-volume
        plot_energy_volume_deltas(axes[0, 1], run_dict, sample, label_pad=0, second_xaxis=True, cmap_name='Paired') # 'coolwarm')

        # Formation volume
        plot_formation_volume(axes[1, 1], run_dict, sample, method_plot_dict)

        # Formation energies
        plot_formation_energy(axes[0, 0], run_dict, sample, method_plot_dict)
        axes[0, 1].set_title(f'ncell_x={run_dict["ncell_x"]}; Natom={run_dict["Natom pure"]}; sample {sample}', y=1.15, x=-0.2, fontsize=25)

        # Formation energies error
        plot_formation_energy_error(axes[1, 0], run_dict, method_plot_dict, sample, legend=False)

        axes[0, 0].legend()
        axes[1, 0].legend()
        axes[1, 1].legend()

        plt.show()

    plot_2x2_two_samples = True
    #plot_2x2_two_samples = False
    if plot_2x2_two_samples:

        sample1 = 20
        #sample1 = 44
        sample2 = 80
        #sample2 = 44

        #sample1 = 0
        #sample2 = 1

        #fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex='col', sharey='row')
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex='col', sharey='row')
        #plt.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.97, wspace=0.2, hspace=0.01)
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.97, wspace=0.01, hspace=0.01)
        plot_formation_volume(axes[0, 0], run_dict, sample1, method_plot_dict2)
        plot_formation_energy(axes[1, 0], run_dict, sample1, method_plot_dict2)

        #axes[0, 0].set_xticklabels([])
        #axes[0, 1].set_xticklabels([])

        plot_formation_volume(axes[0, 1], run_dict, sample2, method_plot_dict2)
        plot_formation_energy(axes[1, 1], run_dict, sample2, method_plot_dict2)

        axes[0, 0].legend(fontsize=18, frameon=False)

        # The last line of the legend must be the first one, but the rest in the same order
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles = [handles[-1]] + handles[:-1]
        labels = [labels[-1]] + labels[:-1]
        axes[0, 0].legend(handles, labels, fontsize=18, frameon=False)

        axes[0, 1].set_ylabel('')
        axes[1, 1].set_ylabel('')

        # iterate over the lines of axes[0, 0] and set the linewidth to 6
        for line in axes[0, 0].get_legend().get_lines():
            line.set_linewidth(7)
            line.set_markersize(10)

        id_list = [1, 2, 1, 2]
        for i, idx in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            #text = rf'$\Theta^{(0)} + \lambda\, \delta \Theta^{{({id_list[i]})}}$'
            text = rf'$\Theta(\lambda, m={({id_list[i]})})$'
            axes[idx].text(0.96, 0.05, text, transform=axes[idx].transAxes, fontsize=45,
                           color='silver', verticalalignment='bottom', horizontalalignment='right')

        fig.savefig(os.path.join(plot_dir, f'two_samples_2x2_{sample1}_{sample2}.pdf'))

        plt.show()

    #plot_2x1_one_sample = True
    plot_2x1_one_sample = False
    if plot_2x1_one_sample:

        sample1 = 20

        fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex='col')
        plt.subplots_adjust(left=0.12, right=0.98, bottom=0.07, top=0.97, wspace=0.01, hspace=0.01)
        plot_formation_volume(axes[0], run_dict, sample1, method_plot_dict2)
        plot_formation_energy(axes[1], run_dict, sample1, method_plot_dict2)

        axes[0].text(-0.12, 1.05, 'a', transform=axes[0].transAxes, fontsize=28, fontweight='bold', va='top', ha='left')
        axes[1].text(-0.12, 1.0, 'b', transform=axes[1].transAxes, fontsize=28, fontweight='bold', va='top', ha='left')

        axes[0].legend(fontsize=18, frameon=False)

        # iterate over the lines of axes[0, 0] and set the linewidth to 6
        for line in axes[0].get_legend().get_lines():
            line.set_linewidth(5)
            line.set_markersize(7)

        fig.savefig(os.path.join(plot_dir, f'one_sample_2x1_{sample1}.pdf'))

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
            plot_formation_volume(axes[1, 1], run_dict, sample, method_plot_dict)
            axes[1, 1].legend()

            # Formation energies
            plot_formation_energy(axes[0, 0], run_dict, sample, method_plot_dict)
            axes[0, 1].set_title(f'ncell_x={run_dict["ncell_x"]}; Natom={run_dict["Natom pure"]}; sample {sample}',
                                 y=1.15, x=-0.2, fontsize=25)
            axes[0, 0].legend()

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

    #plot_av_data_bins = True
    plot_av_data_bins = False
    if plot_av_data_bins:

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.97, wspace=0.2, hspace=0.2)
        plot_formation_energy(axes[0, 0], run_dict, sample, method_plot_dict)
        plot_formation_volume(axes[0, 1], run_dict, sample, method_plot_dict)
        spline_fill = True
        #spline_fill = False
        plot_formation_energy_error_bins(axes[1, 0], bin_error_dict, method_plot_dict, spline_fill=spline_fill)
        plot_formation_volume_error_bins(axes[1, 1], bin_error_dict, method_plot_dict, spline_fill=spline_fill)

        for ax_idx in [(0, 0), (1, 0)]:
            axes[ax_idx].legend()

        fig.savefig(os.path.join(plot_dir, 'average_data_bins.pdf'))

        plt.show()

    #plot_av_data_bins_1x3 = True
    plot_av_data_bins_1x3 = False
    if plot_av_data_bins_1x3:

        # for 3x1
        #fig, axes = plt.subplots(1, 3, figsize=(7, 13))
        #plt.subplots_adjust(left=0.145, right=0.97, bottom=0.05, top=0.995, wspace=0.2, hspace=0.27)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(left=0.05, right=0.985, bottom=0.13, top=0.96, wspace=0.24, hspace=0.27)

        spline_fill = True
        #spline_fill = False
        plot_formation_volume_bins(axes[0], run_dict, bin_vol_dict, method_plot_dict, spline_fill=spline_fill)
        plot_formation_volume_error_bins(axes[1], bin_error_dict, method_plot_dict, spline_fill=spline_fill)
        plot_formation_energy_error_bins(axes[2], bin_error_dict, method_plot_dict, spline_fill=spline_fill)

        # no legend box around it
        axes[0].legend(loc='upper left', fontsize=18, frameon=False)

        # iterate over the lines of the legend and set the marker size to 10
        for line in axes[0].get_legend().get_lines():
            line.set_markersize(10)

        # a,b, c labels
        label_list = ['a', 'b', 'c']
        pos_list = [(-0.15, 1.0), (-0.15, 1.0), (-0.15, 1.0)]
        for i, ax in enumerate(axes):
            ax.text(pos_list[i][0], pos_list[i][1], label_list[i], transform=ax.transAxes, fontsize=30, fontweight='bold', va='top', ha='left')

        fig.savefig(os.path.join(plot_dir, 'average_data_bins_1x3.pdf'))

        plt.show()

    #plot_av_data_bins_1x4 = True
    plot_av_data_bins_1x4 = False
    if plot_av_data_bins_1x4:

        sample1 = 20

        #fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        #plt.subplots_adjust(left=0.05, right=0.985, bottom=0.13, top=0.96, wspace=0.24, hspace=0.27)
        fig = plt.figure(figsize=(22, 6))
        gs0 = gridspec.GridSpec(2, 4, figure=fig)

        # with this grid, split the first plot into two vertically
        # and for 2,3,4 create axes
        ax0_top = fig.add_subplot(gs0[0, 0])
        ax0_bot = fig.add_subplot(gs0[1, 0])

        ax1 = fig.add_subplot(gs0[0:2, 1])
        ax2 = fig.add_subplot(gs0[0:2, 2])
        ax3 = fig.add_subplot(gs0[0:2, 3])

        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.13, top=0.96, wspace=0.24, hspace=0.02)

        # AXIS 0
        plot_formation_volume(ax0_top, run_dict, sample1, method_plot_dict)
        plot_formation_energy(ax0_bot, run_dict, sample1, method_plot_dict)

        ax0_top.set_ylabel('Formation\nVolume ($\mathrm{\AA}^3$)')
        ax0_bot.set_ylabel('Formation\nEnergy (eV)')

        ax0_top.set_xticklabels([])

        spline_fill = True
        #spline_fill = False

        # AXIS 1
        plot_formation_volume_bins(ax1, run_dict, bin_vol_dict, method_plot_dict, spline_fill=spline_fill)

        # AXIS 2
        plot_formation_energy_bins(ax2, run_dict, bin_en_dict, method_plot_dict, spline_fill=spline_fill, plot_diff=True)

        # Get the max and mix of ax2
        x_min, x_max = ax2.get_xlim()
        # plot the diagonal line
        l = ax2.plot([x_min, x_max], [x_min, x_max], ls='--', color='black', lw=4)

        # legend only for dashed line - ideal
        ax2.legend([l[0]], ['Ideal'], fontsize=20, frameon=False)

        #ax2.set_ylim(-0.3, 0.6)
        #ax2.set_xlim(5, 15)
        #ax2.set_ylim(5, 15)

        #plot_formation_volume_error_bins(ax2, bin_error_dict, method_plot_dict, spline_fill=spline_fill)
        #plot_formation_volume_scatter(ax2, run_dict, method_plot_dict)

        # AXIS 3
        plot_formation_energy_error_bins(ax3, bin_error_dict, method_plot_dict, spline_fill=spline_fill)

        ax1.set_ylim(5, 17)

        ax3.set_ylim(ymax=0.175)
        # For ax3 multiple locator by 0.05
        ax3.yaxis.set_major_locator(MultipleLocator(0.05))

        # no legend box around it
        ax0_top.legend(loc='upper left', fontsize=16, frameon=False)

        # iterate over the lines of the legend and set the marker size to 10
        for line in ax0_top.get_legend().get_lines():
            line.set_markersize(10)
            line.set_linewidth(3)

        # a,b, c labels
        label_list = ['a', 'b', 'c', 'd', 'e']
        pos_list = [(-0.18, 1.05), (-0.18, 1.02), (-0.15, 1.02), (-0.17, 1.02), (-0.17, 1.02)]
        for i, ax in enumerate([ax0_top, ax0_bot, ax1, ax2, ax3]):
            ax.text(pos_list[i][0], pos_list[i][1], label_list[i], transform=ax.transAxes, fontsize=30, fontweight='bold', va='top', ha='left')

        fig.savefig(os.path.join(plot_dir, 'average_data_bins_1x4.pdf'))

        plt.show()

    plot_av_data_bins_1x3_NEW = True
    #plot_av_data_bins_1x3_NEW = False
    if plot_av_data_bins_1x3_NEW:

        sample1 = 20
        #sample1 = 0

        #fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        #plt.subplots_adjust(left=0.05, right=0.985, bottom=0.13, top=0.96, wspace=0.24, hspace=0.27)
        fig = plt.figure(figsize=(20, 6))
        gs0 = gridspec.GridSpec(2, 3, figure=fig)

        # with this grid, split the first plot into two vertically
        # and for 2,3,4 create axes
        ax0_top = fig.add_subplot(gs0[0, 0])
        ax0_bot = fig.add_subplot(gs0[1, 0])

        ax1 = fig.add_subplot(gs0[0:2, 1])
        ax2 = fig.add_subplot(gs0[0:2, 2])
        #ax3 = fig.add_subplot(gs0[0:2, 3])

        plt.subplots_adjust(left=0.055, right=0.99, bottom=0.13, top=0.96, wspace=0.24, hspace=0.02)

        # AXIS 0
        plot_formation_volume(ax0_top, run_dict, sample1, method_plot_dict)
        plot_formation_energy(ax0_bot, run_dict, sample1, method_plot_dict)

        ax0_top.set_ylabel('Formation\nVolume ($\mathrm{\AA}^3$)')
        ax0_bot.set_ylabel('Formation\nEnergy (eV)')

        ax0_top.set_xticklabels([])
        ax0_top.set_ylim(6.0, 21.0)
        ax0_top.legend(loc='upper left', fontsize=16, frameon=False)
        # iterate over the lines of the legend and set the marker size to 10
        for line in ax0_top.get_legend().get_lines():
            line.set_markersize(10)
            #line.set_markersize(0)
            line.set_linewidth(4)
            #line.set_linestyle('-')

        spline_fill = True
        #spline_fill = False

        # AXIS 1
        #ax1_data = 'form_volume_bins'
        #ax1_data = 'form_volume_error_bins'
        ax1_data = 'form_volume_error_average'

        if ax1_data == 'form_volume_bins':
            plot_formation_volume_bins(ax1, run_dict, bin_vol_dict, method_plot_dict, spline_fill=spline_fill, atomic_V_range=(16.06, 16.19))

            ax1.set_xlim(16.06, 16.194)
            ax1.set_ylim(10.35, 12.35)
        elif ax1_data == 'form_volume_error_bins':
            plot_formation_volume_error_bins(ax1, bin_error_dict, method_plot_dict, spline_fill=spline_fill, av_suffix='_abs')
        elif ax1_data == 'form_volume_error_average':

            #av_suffix = ''
            #av_suffix = '_sq'
            av_suffix = '_abs'
            plot_formation_volume_error_average(ax1, average_dict, method_plot_dict, spline_fill=spline_fill, av_suffix=av_suffix)

            if av_suffix == '_abs':
                ax1.set_xlim(-10, 10)
                ax1.set_ylim(-0.05, 1.2)
                xmin, xmax = ax1.get_xlim()
                ax1.plot([xmin, xmax], [0.0, 0.0], ls='-', color='black', lw=2, zorder=0)
            elif av_suffix == '':
                ax1.set_xlim(-20, 20)
                ax1.set_ylim(-0.5, 0.5)

        # FOR PRESENTATION, COMMENT OUT
        #ax1.plot([0.0, 0.0], [0.0, 0.0], ls='-', color='black', lw=5, zorder=0, label='True')
        ax1.legend(loc='upper left', fontsize=20,
                   facecolor='white', edgecolor='white', frameon=True, framealpha=1.0)


        # Inset plot in ax1
        inset_plot = None
        #inset_plot = 'scatter'
        #inset_plot = 'form_volume_bins'
        if inset_plot == 'scatter':
            #ax_inset = ax1.inset_axes([0.6, 0.1, 0.35, 0.35])
            ax_inset = ax1.inset_axes([0.32, 0.57, 0.4, 0.4])
            plot_formation_volume_scatter(ax_inset, run_dict, method_plot_dict, label_fsize=10, tick_fsize=10, V_form_range=(11.0, 12.0), delta_range=(-1.0,1.0))

            # multiple locator by 0.5
            ax_inset.xaxis.set_major_locator(MultipleLocator(0.5))
            ax_inset.yaxis.set_major_locator(MultipleLocator(0.5))

            fsize = 14
            ax_inset.set_xlabel(r'True $V_f$ ($\mathrm{\AA^3}$)', fontsize=fsize)
            ax_inset.set_ylabel(r'Predicted $V_f$ ($\mathrm{\AA^3}$)', fontsize=fsize)

        elif inset_plot == 'form_volume_bins':
            #ax_inset = ax1.inset_axes([0.6, 0.1, 0.35, 0.35])
            ax_inset = ax1.inset_axes([0.3, 0.3, 0.4, 0.4])
            plot_formation_volume_bins(ax_inset, run_dict, bin_vol_dict, method_plot_dict, fill=False, label_fsize=12, tick_fsize=10, atomic_V_range=(16.06, 16.21))
            ax_inset.set_xlim(16.055, 16.215)
            ax_inset.set_ylim(10.35, 12.35)

        # AXIS 2
        plot_formation_energy_error_bins(ax2, bin_error_dict, method_plot_dict, spline_fill=spline_fill)

        ax2.set_ylim(ymin=-0.005, ymax=0.175)
        # For ax3 multiple locator by 0.05
        ax2.yaxis.set_major_locator(MultipleLocator(0.05))
        # Get the last line of ax2
        l = ax2.lines[-1]
        xmin, xmax = l.get_xdata().min(), l.get_xdata().max()
        ax2.plot([xmin, xmax], [0.0, 0.0], ls='-', color='black', lw=2, zorder=0)

        # FOR PRESENTATION, COMMENT OUT
        ax2.legend(loc='upper center', fontsize=20,
                   facecolor='white', edgecolor='white', frameon=True, framealpha=1.0)

        inset_plot_ax2 = None
        #inset_plot_ax2 = 'zoom'
        if inset_plot_ax2 == 'zoom':
            ax2_inset = ax2.inset_axes([0.3, 0.57, 0.4, 0.4])

            plot_formation_energy_error_bins(ax2_inset, bin_error_dict, method_plot_dict,
                                             spline_fill=spline_fill, label_fsize=12, tick_fsize=10, print_labels=False, fill=False)
            # y axis major locator by 0.005
            ax2_inset.yaxis.set_major_locator(MultipleLocator(0.005))
            ax2_inset.axhline(0.0, ls='-', color='black', lw=2, zorder=0)
            ax2_inset.set_xlim(-0.4, 0.4)
            ax2_inset.set_ylim(-0.001, 0.01)

        # a,b, c labels
        label_list = ['a', 'b', 'c', 'd']
        pos_list = [(-0.18, 1.05), (-0.18, 1.02), (-0.15, 1.02), (-0.17, 1.02)]
        for i, ax in enumerate([ax0_top, ax0_bot, ax1, ax2]):
            ax.text(pos_list[i][0], pos_list[i][1], label_list[i], transform=ax.transAxes, fontsize=30, fontweight='bold', va='top', ha='left')

        fig.savefig(os.path.join(plot_dir, 'average_data_bins_1x3.pdf'))

        plt.show()

    #plot_LJ_2x1 = True
    plot_LJ_2x1 = False
    if plot_LJ_2x1:

        LJ_dict_path = '/Users/imaliyov/Papers/Potential-Perturbation/LJ_notebook/LATTICE-DIST/LJ_dict.pkl'

        with open(LJ_dict_path, 'rb') as f:
            LJ_dict = pickle.load(f)

        fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex='col')
        plt.subplots_adjust(left=0.14, right=0.97, bottom=0.1, top=0.97, wspace=0.2, hspace=0.01)

        plot_LJ_error(axes[0], LJ_dict, method_plot_dict)
        axes[0].set_xlabel('')

        plot_LJ_distortion(axes[1], LJ_dict, method_plot_dict)

        # LEGEND
        axes[1].legend(fontsize=20, frameon=False)
        # iterate over the lines of legend and assign them to the axes[0] legend
        line_list = []
        for line in axes[1].get_legend().get_lines():
            line_list.append(line)
        legend = axes[0].legend(handles=line_list, loc='upper left', fontsize=24, frameon=False)
        # remove legend from axes[1]
        axes[1].get_legend().remove()

        # axes[1] x multiple of 0.05
        axes[1].xaxis.set_major_locator(MultipleLocator(0.05))

        fig.savefig(os.path.join(plot_dir, 'LJ_2x1.pdf'))

        plt.show()

    plot_scatter_form_volume = True
    if plot_scatter_form_volume:

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.97)
        ax.set_aspect('equal', 'box')

        plot_formation_volume_scatter(ax, run_dict, method_plot_dict, V_form_range=(11.0, 12.0), delta_range=(-1.0,1.0))

        # multiple locator by 0.5
        #ax.xaxis.set_major_locator(MultipleLocator(0.5))
        #ax.yaxis.set_major_locator(MultipleLocator(0.5))

        fsize = 24
        #ax.set_xlabel(r'True $V_f$ ($\mathrm{\AA^3}$)', fontsize=fsize)
        #ax.set_ylabel(r'Predicted $V_f$ ($\mathrm{\AA^3}$)', fontsize=fsize)
        ax.set_xlabel(r'True Formation Volume ($\mathrm{\AA^3}$)', fontsize=fsize)
        ax.set_ylabel(r'Predicted Formation Volume ($\mathrm{\AA^3}$)', fontsize=fsize)

        ax.legend(loc='upper left', fontsize=25,
                  facecolor='white', edgecolor='white', frameon=True, framealpha=1.0)

        ax.grid()

        # equal aspect ratio


        fig.savefig(os.path.join(plot_dir, 'scatter_form_volume.pdf'))

        plt.show()

if __name__ == '__main__':
    main()

