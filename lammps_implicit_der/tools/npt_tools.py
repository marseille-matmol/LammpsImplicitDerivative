#!/usr/bin/env python3
"""
Tools for NPT calculations: energy-volume curves, prediction of volume change with the change in potetial parameters.
"""

import os
import uuid
import numpy as np
from .error_tools import coord_error
from .timing import TimingGroup
from .utils import mpi_print


def compute_energy_volume(system, epsilon_array, compute_forces=False):

    energy_array = np.zeros_like(epsilon_array)
    volume_array = np.zeros_like(epsilon_array)

    virial_array = np.zeros((epsilon_array.shape[0], 6, system.Ndesc))
    pressure_array = np.zeros_like(epsilon_array)

    descriptor_array = np.zeros((epsilon_array.shape[0], system.Ndesc))

    if compute_forces:
        force_array = np.zeros((epsilon_array.shape[0], system.Natom * 3))

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
        descriptor_array[i, :] = system.dU_dTheta

        # extract_compute("thermo_pe")
        energy = system.energy

        energy_array[i] = energy

        system.gather_virial()
        virial_array[i, :, :] = np.sum(system.virial, axis=0)

        # Compute pressure
        system.get_pressure_from_virial()
        pressure_array[i] = system.pressure

        if compute_forces:
            force_array[i, :] = system.compute_forces(dx=np.zeros(system.Natom*3))

    # Reapply the original cell
    system.apply_strain(initial_cell)

    energy_array /= system.ncell_x**3

    energy_array -= energy_array.min()

    result_dict = {
        'energy_array': energy_array,
        'volume_array': volume_array,
        'virial_array': virial_array,
        'pressure_array': pressure_array,
        'descriptor_array': descriptor_array
    }

    if compute_forces:
        result_dict['force_array'] = force_array

    return result_dict


def create_perturbed_system(Theta_perturb, LammpsClass, snapcoeff_filename, snapparam_filename=None,
                            data_path=None, alat=3.185, ncell_x=2, logname='perturb.log', fix_box_relax=False, minimize=True, verbose=False, comm=None):
    """
    Create an instance of LammpsClass with  potential parameters from Theta_perturb numpy array.
    """

    # system_tmp is created only to save the SNAP potential files
    system_tmp = LammpsClass(data_path=data_path, snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename,
                             ncell_x=ncell_x, alat=alat, logname='tmp.log', minimize=False, verbose=False, comm=comm)

    if len(system_tmp.pot.elem_list) > 1:
        raise RuntimeError('Implemented for single element systems only')

    element = system_tmp.pot.elem_list[0]

    #Theta_perturb = Theta_ens['Theta_mean'] + delta * (Theta_ens['Theta_ens_list'][sample] - Theta_ens['Theta_mean'])

    # Set the perturbed parameters
    system_tmp.pot.Theta_dict[element]['Theta'] = Theta_perturb

    if comm is None or comm.Get_rank() == 0:
        prefix = f'perturb_{uuid.uuid4()}'
        system_tmp.pot.to_files(path='.', overwrite=True, snapcoeff_filename=f'{prefix}.snapcoeff', snapparam_filename=f'{prefix}.snapparam', verbose=verbose)
    else:
        prefix = None

    if comm is not None:
        prefix = comm.bcast(prefix, root=0)

    # Create the perturbed system with the new potential
    system_perturb = LammpsClass(ncell_x=ncell_x, alat=alat, logname=logname, minimize=False, verbose=verbose, minimize_maxiter=500,
                                 snapcoeff_filename=f'{prefix}.snapcoeff', snapparam_filename=f'{prefix}.snapparam', fix_box_relax=fix_box_relax,
                                 data_path='.', comm=comm)

    if comm is not None:
        comm.Barrier()

    if comm is None or comm.Get_rank() == 0:
        os.remove(f'{prefix}.snapcoeff')
        os.remove(f'{prefix}.snapparam')

    if minimize:
        try:
            system_perturb.minimize_energy()
        except Exception as e:
            mpi_print(f'Error in minimization of system_perturb: {e}', comm=comm)
            return None

    if system_perturb.not_converged:
        return None

    return system_perturb


def run_npt_implicit_derivative(LammpsClass, alat, ncell_x, Theta_perturb,
                                snapcoeff_filename, snapparam_filename,
                                virial_der0, dX_dTheta_inhom, force_der0=None,
                                data_path=None, comm=None, trun=None):

    if trun is None:
        trun = TimingGroup('NPT implicit derivative')
        total_tag = 'total'
    else:
        total_tag = 'NPT RUN'
    trun.add(total_tag, level=2).start()

    with trun.add('NPT minimization'):
        # For the ground truth - fix box/relax. Theta1
        s_box_relax = create_perturbed_system(Theta_perturb, LammpsClass, logname='s_box_relax.log',
                                              data_path=data_path, snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename,
                                              alat=alat, ncell_x=ncell_x, fix_box_relax=True, minimize=True, verbose=False, comm=comm)

        if comm is not None:
            comm.Barrier()

        if s_box_relax is None:
            trun.timings[total_tag].stop()
            mpi_print('Minimization did not converge for s_box_relax', comm=comm)
            return None

        #mpi_print(f'   box/relax steps: {s_box_relax.minimization_nstep}', comm=comm)

    with trun.add('NVT minimization'):
        # For full implicit derivative. Theta0
        s_pred = LammpsClass(alat=alat, ncell_x=ncell_x, minimize=True, logname='s_pred.log',
                             data_path=data_path, snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename, verbose=False, comm=comm)
        if comm is not None:
            comm.Barrier()

    # Parameter perturbation
    with trun.add('initial and true'):
        Theta0 = s_pred.Theta.copy()
        Theta_pert = s_box_relax.Theta.copy()
        dTheta = Theta_pert - Theta0

        # Initial system parameters
        volume0 = s_pred.volume
        cell0 = s_pred.cell.copy()
        X_coord0 = s_pred.X_coord.copy()
        energy0 = s_pred.energy
        dU_dTheta0 = s_pred.dU_dTheta.copy()
        energy_pred0 = dU_dTheta0 @ Theta_pert

        # Gound truth - fix box/relax
        volume_true = s_box_relax.volume
        X_coord_true = s_box_relax.X_coord.copy()
        energy_true = s_box_relax.energy
        coord_error0 = coord_error(X_coord_true, X_coord0)

    # Predict the volume change
    with trun.add('volume prediction'):

        # Finite diff Desc prediction
        """
        dL_dTheta = s_pred.implicit_derivative_hom(method='d2Desc')
        dL_pred = dTheta @ dL_dTheta
        L0 = volume0**(1/3)
        V_pred_d2Desc = (L0 + dL_pred)**3
        strain_pred_d2Desc = ((V_pred_d2Desc) / volume0)**(1/3)
        cell_pred_d2Desc = np.dot(cell0, np.eye(3) * strain_pred_d2Desc)
        """

        # Finite diff Virial presdiction
        dL_dTheta = s_pred.implicit_derivative_hom(method='dVirial')
        dL_pred = dTheta @ dL_dTheta
        L0 = volume0**(1.0/3.0)
        V_pred = (L0 + dL_pred)**3
        strain_pred = ((V_pred) / volume0)**(1.0/3.0)
        cell_pred = np.dot(cell0, np.eye(3) * strain_pred)

        # Inhomogeneous correction to volume that requires the derivative of the forces
        if force_der0 is not None:
            dot_prod = -np.einsum('ji,i->j', dX_dTheta_inhom, force_der0)
            dV_pred_inhom = np.dot(dot_prod, dTheta) / np.dot(virial_der0, Theta0)

    # Homogeneous contribution: scale the system with the predicted volume change
    # And then return to the original volume with cell0
    with trun.add('homogeneous'):
        s_pred.apply_strain(cell_pred, update_system=True)
        energy_hom_pred = s_pred.dU_dTheta @ Theta_pert
        coord_error_hom = coord_error(X_coord_true, s_pred.X_coord)
        s_pred.apply_strain(cell0, update_system=True)

    # Inhomogeneous contribution
    with trun.add('inhomogeneous'):

        # s_pred has the non-perturbed parameters
        # to get the energy from s_pred using .energy, one would need to update the potential parameters
        # like so:
        # s_pred.pot.snapcoeff_path = './perturb.snapcoeff'
        # s_pred.pot.snapparam_path = './perturb.snapparam'
        # s_pred.setup_snap_potential()
        # energy_inhom_pred = s_pred.energy
        # Or, one can compute the energy as D@Theta_new:
        dX_inhom_pred = dTheta @ dX_dTheta_inhom
        s_pred.X_coord += dX_inhom_pred
        s_pred.scatter_coord()
        s_pred.gather_D_dD()
        energy_inhom_pred = s_pred.dU_dTheta @ Theta_pert
        coord_error_inhom = coord_error(X_coord_true, s_pred.X_coord)

    # Full prediction
    with trun.add('full prediction'):
        # virial volume prediction
        s_pred.apply_strain(cell_pred, update_system=True)
        energy_full_pred = s_pred.dU_dTheta @ Theta_pert
        coord_error_full = coord_error(X_coord_true, s_pred.X_coord)
        s_pred.apply_strain(cell0, update_system=True)

    trun.timings[total_tag].stop()

    result_dict = {
        'trun': trun,
        'trun_s_pred': s_pred.timings,
        'trun_s_box_relax': s_box_relax.timings,
        # Parameters
        'Theta_pert': Theta_pert,
        # Volumes
        'volume0': volume0,
        'volume_true': volume_true,
        'volume_pred': V_pred,
        'volume_pred_full': V_pred if force_der0 is None else V_pred + dV_pred_inhom,
        # Energies
        'energy0': energy0,
        'energy_true': energy_true,
        'energy_pred0': energy_pred0,
        'energy_hom_pred': energy_hom_pred,
        'energy_inhom_pred': energy_inhom_pred,
        'energy_full_pred': energy_full_pred,
        # Coord. errors
        'coord_error_full': coord_error_full,
        'coord_error_hom': coord_error_hom,
        'coord_error_inhom': coord_error_inhom,
        'coord_error0': coord_error0,
    }

    return result_dict