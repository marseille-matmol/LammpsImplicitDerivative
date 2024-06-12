#!/usr/bin/env python3
import pickle
import os
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

# Local imports
from lammps_implicit_der import LammpsImplicitDer, SNAP
from lammps_implicit_der.tools import mpi_print, initialize_mpi, TimingGroup, plot_tools, \
                                      compute_energy_volume, create_perturbed_system, run_npt_implicit_derivative
from lammps_implicit_der.systems import BccVacancy, Bcc


def get_min(en_vol_dict, ncell_x):

    energy_array = en_vol_dict['energy_array']
    volume_array = en_vol_dict['volume_array']
    pressure_array = en_vol_dict['pressure_array']

    min_idx = np.argmin(energy_array)

    energy_min = energy_array[min_idx]
    volume_min = volume_array[min_idx]
    pressure_min = pressure_array[min_idx]

    alat_min = volume_min**(1/3) / ncell_x

    return energy_min, volume_min, pressure_min, alat_min


def main():

    trun = TimingGroup('RUN GLOBAL', sort=False)
    trun.add('total', level=2).start()

    comm, rank = initialize_mpi()

    ncell_x = int(sys.argv[1])
    impl_der_method = sys.argv[2]

    sample_first = int(sys.argv[3])
    sample_last = int(sys.argv[4])

    alat0 = 3.18
    snapcoeff_filename = 'W_NEW.snapcoeff'
    snapparam_filename = 'W_NEW.snapparam'

    mpi_print(f'{ncell_x = }', comm=comm)

    run_dict = {}
    run_dict['ncell_x'] = ncell_x

    mpi_print('Computing lattice parameters for Bcc and BccVacancy systems...', comm=comm)
    with trun.add('find alat min'):

        # Find alat from fix box/relax

        bcc_pure_box_relax = Bcc(alat=alat0, ncell_x=ncell_x, minimize=True, logname='bcc.log',
                                 snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename,
                                 verbose=False, fix_box_relax=True)

        bcc_vac_box_relax = BccVacancy(alat=alat0, ncell_x=ncell_x, minimize=True, logname='bcc_vac.log',
                                       snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename,
                                       verbose=False, fix_box_relax=True)

        volume_pure_true = bcc_pure_box_relax.volume
        volume_vac_true = bcc_vac_box_relax.volume

        alat = volume_pure_true**(1/3) / ncell_x
        alat_vac = volume_vac_true**(1/3) / ncell_x

        bcc_pure = Bcc(alat=alat, ncell_x=ncell_x, minimize=True, logname='bcc.log', # alat=alat0
                       snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename,
                       verbose=False)

        bcc_vac = BccVacancy(alat=alat_vac, ncell_x=ncell_x, minimize=True, logname='bcc_vac.log', # alat=alat0
                             snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename,
                             verbose=False)

        epsilon_array = np.linspace(-0.03, 0.03, 101)
        en_vol_pure_dict = compute_energy_volume(bcc_pure, epsilon_array)
        en_vol_vac_dict = compute_energy_volume(bcc_vac, epsilon_array, compute_forces=True)

        # We keep alat computed from fix box/relax, alat from E-V curves is not used
        energy_pure_min, volume_pure_min, pressure_pure_min, alat_tmp = \
            get_min(en_vol_pure_dict, ncell_x)

        energy_vac_min, volume_vac_min, pressure_vac_min, alat_vac_tmp = \
            get_min(en_vol_vac_dict, ncell_x)

        run_dict['alat'] = alat
        run_dict['alat_vac'] = alat_vac
        run_dict['Natom pure'] = bcc_pure.Natom
        run_dict['Natom vac'] = bcc_vac.Natom

    with trun.add('virial derivatives'):
        spline_virial_list_pure = []
        spline_virial_list_vac = []
        spline_force_list_vac = []

        volume_array_pure = en_vol_pure_dict['volume_array']
        volume_array_vac = en_vol_vac_dict['volume_array']

        virial_array_pure = en_vol_pure_dict['virial_array']
        virial_array_vac = en_vol_vac_dict['virial_array']
        force_array_vac = en_vol_vac_dict['force_array']

        for idesc in range(bcc_pure.Ndesc):

            virial_trace_array_pure = np.sum(virial_array_pure[:, :3, :], axis=1) / 3.0
            spline_virial_list_pure.append(CubicSpline(volume_array_pure, virial_trace_array_pure[:, idesc]))

            virial_trace_array_vac = np.sum(virial_array_vac[:, :3, :], axis=1) / 3.0
            spline_virial_list_vac.append(CubicSpline(volume_array_vac, virial_trace_array_vac[:, idesc]))

        for icoord in range(bcc_vac.Natom * 3):
            spline_force_list_vac.append(CubicSpline(volume_array_pure, force_array_vac[:, icoord]))

        virial_der_pure0 = np.array([spline_virial_list_pure[idesc](volume_pure_min, nu=1) for idesc in range(bcc_pure.Ndesc)])
        virial_der_vac0 = np.array([spline_virial_list_vac[idesc](volume_vac_min, nu=1) for idesc in range(bcc_vac.Ndesc)])
        force_der_vac0 = np.array([spline_force_list_vac[icoord](volume_vac_min, nu=1) for icoord in range(bcc_vac.Natom * 3)])

        bcc_pure.compute_virial()
        bcc_pure.gather_virial()
        virial_pure = np.sum(bcc_pure.virial, axis=0)
        virial_trace_pure = np.sum(virial_pure[:3, :], axis=0) / 3.0

        bcc_vac.compute_virial()
        bcc_vac.gather_virial()
        virial_vac = np.sum(bcc_vac.virial, axis=0)
        virial_trace_vac = np.sum(virial_vac[:3, :], axis=0) / 3.0

    with trun.add('dX_dTheta inhom'):
        dX_dTheta_pure_inhom = bcc_pure.implicit_derivative(method=impl_der_method)
        dX_dTheta_vac_inhom = bcc_vac.implicit_derivative(method=impl_der_method)

    with trun.add('sample loop', level=1):

        with open('Theta_ens.pkl', 'rb') as file:
            Theta_ens = pickle.load(file)

        #delta_array = np.linspace(-100.0, 100.0, 11)
        #delta_array = np.linspace(-10.0, 10.0, 3)
        delta_array = np.linspace(-50.0, 50.0, 101)
        # For energy-volume curves
        #epsilon_array_en_vol = np.linspace(-0.05, 0.05, 15)
        #epsilon_array_en_vol = np.linspace(-0.05, 0.05, 25)
        epsilon_array_en_vol = np.linspace(-0.05, 0.05, 61)

        run_dict['delta_array'] = delta_array

        run_dict['epsilon_array_en_vol'] = epsilon_array_en_vol

        descriptor_array_pure = en_vol_pure_dict['descriptor_array']
        descriptor_array_vac = en_vol_vac_dict['descriptor_array']
        volume_array_pure = en_vol_pure_dict['volume_array']
        volume_array_vac = en_vol_vac_dict['volume_array']

        trun_npt = TimingGroup('NPT implicit derivative')

        sample_list = np.arange(sample_first, sample_last + 1)
        run_dict['sample_list'] = sample_list

        for sample in sample_list:

            trun.add('sample').start()
            mpi_print('\n'+'*'*80+f'\n{sample = }'+'\n'+'*'*80, comm=comm)

            s_str = f'sample_{sample}'
            sample_dict = {}
            sample_dict['system_info'] = {}
            # Copy all the system info into sample_dict
            for k, v in run_dict.items():
                if 'sample' not in k or k == 'sample_list':
                    sample_dict['system_info'][k] = v

            # delta values for converged box/relax
            conv_idelta_list = []

            for idelta, delta in enumerate(delta_array):
                mpi_print(f'\n{idelta+1}/{len(delta_array)}, {delta=:.1f}', comm=comm)
                d_str = f'delta_{idelta}'
                sample_dict[d_str] = {}
                sample_dict[d_str]['pure'] = {}
                sample_dict[d_str]['vac'] = {}

                mpi_print('   Eergy-volume curves...', comm=comm)
                with trun.add('en.-vol. curves'):

                    bcc_pure_tmp = create_perturbed_system(Theta_ens, delta, Bcc, logname='vac_tmp.log',
                                                           snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename, comm=comm,
                                                           sample=sample, alat=alat, ncell_x=ncell_x, fix_box_relax=False, minimize=True, verbose=False)

                    bcc_vac_tmp = create_perturbed_system(Theta_ens, delta, BccVacancy, logname='vac_tmp.log',
                                                          snapcoeff_filename=snapcoeff_filename, snapparam_filename=snapparam_filename, comm=comm,
                                                          sample=sample, alat=alat_vac, ncell_x=ncell_x, fix_box_relax=False, minimize=True, verbose=False)

                    if bcc_pure_tmp is None or bcc_vac_tmp is None:
                        mpi_print('   Error in creating perturbed systems.', comm=comm)
                        sample_dict[d_str]['pure'] = None
                        sample_dict[d_str]['vac'] = None
                        continue

                    en_vol_pure_dict = compute_energy_volume(bcc_pure_tmp, epsilon_array_en_vol)
                    en_vol_vac_dict = compute_energy_volume(bcc_vac_tmp, epsilon_array_en_vol)

                    sample_dict[d_str]['pure']['en_vol'] = en_vol_pure_dict
                    sample_dict[d_str]['vac']['en_vol'] = en_vol_vac_dict

                mpi_print('   NPT minimization...', comm=comm)
                with trun.add('npt'):
                    pure_dict = run_npt_implicit_derivative(Bcc, alat, ncell_x, Theta_ens, delta, sample,
                                                            snapcoeff_filename, snapparam_filename,
                                                            virial_trace_pure, virial_der_pure0, descriptor_array_pure, volume_array_pure,
                                                            dX_dTheta_pure_inhom, comm=comm, trun=trun_npt)

                    if comm is not None:
                        comm.Barrier()

                    vac_dict = run_npt_implicit_derivative(BccVacancy, alat_vac, ncell_x, Theta_ens, delta, sample,
                                                           snapcoeff_filename, snapparam_filename,
                                                           virial_trace_vac, virial_der_vac0, descriptor_array_vac, volume_array_vac,
                                                           dX_dTheta_vac_inhom, force_der0=force_der_vac0, comm=comm, trun=trun_npt)

                    if comm is not None:
                        comm.Barrier()

                    sample_dict[d_str]['pure']['npt'] = pure_dict
                    sample_dict[d_str]['vac']['npt'] = vac_dict

                    if pure_dict is not None and vac_dict is not None:
                        conv_idelta_list.append(idelta)

                sample_dict['conv_idelta_list'] = conv_idelta_list
                run_dict[s_str] = copy.deepcopy(sample_dict)

                if rank == 0:
                    with open(f'{s_str}.pkl', 'wb') as file:
                        pickle.dump(sample_dict, file)

            trun.timings['sample'].stop()
            if rank == 0:
                print(trun.timings['sample'])

    mpi_print(trun_npt, comm=comm)

    # Save run_dict to pickle
    if rank == 0:
        with open('run_dict.pkl', 'wb') as file:
            pickle.dump(run_dict, file)

    trun.timings['total'].stop()
    mpi_print(trun, comm=comm)


if __name__ == "__main__":
    main()

# Exception: ERROR on proc 0: Neighbor list overflow, boost neigh_modify one (src/npair_full_bin_atomonly.cpp:87