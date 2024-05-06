#!/usr/bin/env python3
"""
Tools for NPT calculations: energy-volume curves, prediction of volume change with the change in potetial parameters.
"""

import numpy as np
from .error_tools import coord_error


def compute_energy_volume(system, epsilon_array):

    energy_array = np.zeros_like(epsilon_array)
    volume_array = np.zeros_like(epsilon_array)

    virial_array = np.zeros((epsilon_array.shape[0], 6, system.Ndesc))
    pressure_array = np.zeros_like(epsilon_array)

    system.scatter_coord()
    system.gather_D_dD()

    system.compute_virial()

    system.get_cell()
    initial_cell = system.cell

    for i, epsilon in enumerate(epsilon_array):

        #M = np.diag([epsilon, epsilon, -2.0 * epsilon])
        M = np.diag([epsilon, epsilon, epsilon])

        cell = np.dot(initial_cell, (np.eye(3) + M))

        system.apply_strain(cell)

        volume = np.linalg.det(cell)
        volume_array[i] = volume

        system.gather_D_dD()

        # extract_compute("thermo_pe")
        energy = system.energy

        energy_array[i] = energy

        system.gather_virial()
        virial_array[i, :, :] = np.sum(system.virial, axis=0)

        # Compute pressure
        system.get_pressure_from_virial()
        pressure_array[i] = system.pressure

    # Reapply the original cell
    system.apply_strain(initial_cell)

    energy_array /= system.ncell_x**3

    energy_array -= energy_array.min()

    return energy_array, volume_array, virial_array, pressure_array


def create_perturbed_system(Theta_ens, delta, LammpsClass, snapcoeff_filename, snapparam_filename=None,
                            data_path=None, sample=1, alat=3.185, ncell_x=2, logname='perturb.log', fix_box_relax=False, minimize=True, verbose=False):

    # system_tmp is created only to save the SNAP potential files
    system_tmp = LammpsClass(data_path=data_path, snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename,
                             ncell_x=ncell_x, alat=alat, logname='tmp.log', minimize=False, verbose=False)

    if len(system_tmp.pot.elem_list) > 1:
        raise RuntimeError('Implemented for single element systems only')

    element = system_tmp.pot.elem_list[0]

    Theta_perturb = Theta_ens['Theta_mean'] + delta * (Theta_ens['Theta_ens_list'][sample] - Theta_ens['Theta_mean'])

    # Set the perturbed parameters
    system_tmp.pot.Theta_dict[element]['Theta'] = Theta_perturb
    system_tmp.pot.to_files(path='.', overwrite=True, snapcoeff_filename='perturb.snapcoeff', snapparam_filename='perturb.snapparam', verbose=verbose)

    # Create the perturbed system with the new potential
    system_perturb = LammpsClass(ncell_x=ncell_x, alat=alat, logname=logname, minimize=minimize, verbose=verbose,
                                 snapcoeff_filename='perturb.snapcoeff', snapparam_filename='perturb.snapparam', fix_box_relax=fix_box_relax,
                                 data_path='.')

    return system_perturb


def run_npt_implicit_derivative(LammpsClass, alat, ncell_x, Theta_ens, delta, sample,
                                data_path, snapcoeff_filename, snapparam_filename,
                                virial_trace, virial_der0,
                                dX_dTheta_inhom, dX_dTheta_full=None):
    """
    TODO: avoid multiple minimizations of the same system
    """

    # For the ground truth - fix box/relax. Theta1
    s_box_relax = create_perturbed_system(Theta_ens, delta, LammpsClass, logname='s_box_relax.log',
                                          data_path=data_path, snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename,
                                          sample=sample, alat=alat, ncell_x=ncell_x, fix_box_relax=True, minimize=True, verbose=False)

    # For full implicit derivative. Theta0
    s_pred_full = LammpsClass(alat=alat, ncell_x=ncell_x, minimize=True, logname='s_pred_full.log',
                              data_path=data_path, snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename, verbose=False)

    # For the homogeneous contribution only. Theta0
    s_pred_hom = LammpsClass(alat=alat, ncell_x=ncell_x, minimize=True, logname='s_pred_hom.log',
                             data_path=data_path, snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename, verbose=False)

    # Parameter perturbation
    Theta0 = s_pred_full.Theta.copy()
    Theta_pert = s_box_relax.Theta.copy()
    dTheta = Theta_pert - Theta0

    # Initial system parameters
    volume0 = s_pred_full.volume
    cell0 = s_pred_full.cell.copy()
    X_coord0 = s_pred_full.X_coord.copy()
    energy0 = s_pred_full.energy
    dU_dTheta0 = s_pred_full.dU_dTheta.copy()
    energy_pred0 = dU_dTheta0 @ Theta_pert

    # Gound truth - fix box/relax
    volume_true = s_box_relax.volume
    X_coord_true = s_box_relax.X_coord.copy()
    s_box_relax.gather_D_dD()
    energy_true = s_box_relax.energy

    # Inhomogeneous contribution
    dX_inhom_pred = dTheta @ dX_dTheta_inhom
    s_pred_full.X_coord += dX_inhom_pred
    coord_error_inhom = coord_error(X_coord_true, s_pred_full.X_coord)
    # Update the descriptors
    s_pred_full.scatter_coord()
    s_pred_full.gather_D_dD()

    # s_pred_full has the non-perturbed parameters
    # to get the energy from s_pred_full using .energy, one would need to update the potential parameters
    # like so:
    # s_pred_full.pot.snapcoeff_path = './perturb.snapcoeff'
    # s_pred_full.pot.snapparam_path = './perturb.snapparam'
    # s_pred_full.setup_snap_potential()
    # energy_inhom_pred = s_pred_full.energy
    # Or, one can compute the energy as D@Theta_new:
    energy_inhom_pred = s_pred_full.dU_dTheta @ Theta_pert

    # Predict the volume change
    dV_pred = - np.dot(virial_trace, dTheta) / np.dot(virial_der0, Theta0)
    strain_pred = ((volume0 + dV_pred) / volume0)**(1/3)
    cell_pred = np.dot(cell0, np.eye(3) * strain_pred)

    # Apply the strain to s_pred_full and s_pred_hom
    # Compute energy as D@Theta_pert
    s_pred_full.apply_strain(cell_pred)
    s_pred_hom.apply_strain(cell_pred)
    s_pred_full.gather_D_dD()
    s_pred_hom.gather_D_dD()
    energy_hom_pred = s_pred_hom.dU_dTheta @ Theta_pert
    energy_full_pred = s_pred_full.dU_dTheta @ Theta_pert

    # Coordinate errors
    coord_error_full = coord_error(X_coord_true, s_pred_full.X_coord)
    coord_error_hom = coord_error(X_coord_true, s_pred_hom.X_coord)
    coord_error_idle = coord_error(X_coord_true, X_coord0)

    result_dict = {
        # Volumes
        'volume0': volume0,
        'volume_true': volume_true,
        'volume_pred': volume0 + dV_pred,
        # Energies
        'energy0': energy0,
        'energy_true': energy_true,
        'energy_pred0': energy_pred0,
        'energy_hom_pred': energy_hom_pred,
        'energy_inhom_pred': energy_inhom_pred,
        'energy_full_pred': energy_full_pred,
        # Coord. errors
        'coord_error_full': coord_error_full,
        'coord_error_inhom': coord_error_inhom,
        'coord_error_hom': coord_error_hom,
        'coord_error_idle': coord_error_idle,
    }

    return result_dict