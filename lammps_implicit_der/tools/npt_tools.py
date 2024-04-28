#!/usr/bin/env python3
"""
Tools for NPT calculations: energy-volume curves, prediction of volume change with the change in potetial parameters.
"""

import numpy as np


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
                            data_path=None, sample=1, alat=3.185, ncell_x=2, logname='perturb.log', fix_box_relax=False, minimize=True):

    system_tmp = LammpsClass(data_path=data_path, snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename,
                             ncell_x=ncell_x, alat=alat, logname='tmp.log', minimize=minimize, verbose=False)

    if len(system_tmp.pot.elem_list) > 1:
        raise RuntimeError('Implemented for single element systems only')

    element = system_tmp.pot.elem_list[0]

    Theta_perturb = Theta_ens['Theta_mean'] + delta * (Theta_ens['Theta_ens_list'][sample] - Theta_ens['Theta_mean'])

    # Set the perturbed parameters
    system_tmp.pot.Theta_dict[element]['Theta'] = Theta_perturb
    system_tmp.pot.to_files(path='.', overwrite=True, snapcoeff_filename='perturb.snapcoeff', snapparam_filename='perturb.snapparam', verbose=False)

    system_perturb = LammpsClass(ncell_x=ncell_x, alat=alat, logname=logname, minimize=False, verbose=False,
                                 snapcoeff_filename='perturb.snapcoeff', snapparam_filename='perturb.snapparam', fix_box_relax=fix_box_relax,
                                 data_path='.')

    return system_perturb
