"""
Tests of NPT minimization tools.
These tests are integration tests since they include many steps of the workflow.
"""

import pytest
import numpy as np

from scipy.interpolate import CubicSpline
from lammps_implicit_der import SNAP
from lammps_implicit_der.systems import BccVacancy
from lammps_implicit_der.tools import compute_energy_volume, run_npt_implicit_derivative

# To save time, compute th BccVacancy object only once
@pytest.fixture(scope="module")
def bcc_vacancy(comm):
    return BccVacancy(alat=3.163, ncell_x=2, minimize=True, logname=None, del_coord=[0.0, 0.0, 0.0],
                      data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)


def test_energy_volume(bcc_vacancy):

    epsilon_array = np.linspace(-0.05, 0.05, 3)
    en_vol_dict = compute_energy_volume(bcc_vacancy, epsilon_array)

    energy_array_desired = np.array([0.8362907116, 0.0000000000, 0.6414094626])
    volume_array_desired = np.array([217.0492945327, 253.1556139760, 293.0592676290])
    pressure_array_desired = np.array([88.0973712299, 0.0812326755, -63.8315221331])

    energy_array = en_vol_dict['energy_array']
    volume_array = en_vol_dict['volume_array']
    pressure_array = en_vol_dict['pressure_array']

    np.testing.assert_allclose(energy_array, energy_array_desired, atol=1e-7)
    np.testing.assert_allclose(volume_array, volume_array_desired, atol=1e-7)
    np.testing.assert_allclose(pressure_array, pressure_array_desired, atol=1e-7)


def test_run_npt(comm):

    alat = 3.163
    ncell_x = 2
    snapcoeff_filename = 'W.snapcoeff'
    snapparam_filename = 'W.snapparam'
    pot_perturb = SNAP.from_files('W_perturb.snapcoeff', data_path='./refs', comm=comm)
    Theta_perturb = pot_perturb.Theta_dict['W']['Theta'].copy()

    bcc_vac = BccVacancy(alat=alat, ncell_x=ncell_x, minimize=True, logname=None, data_path='./refs/',
                         snapcoeff_filename='W_perturb.snapcoeff', verbose=False, comm=comm)

    dX_dTheta_vac_inhom = bcc_vac.implicit_derivative(method='dense')

    epsilon_array = np.linspace(-0.05, 0.05, 3)
    en_vol_dict = compute_energy_volume(bcc_vac, epsilon_array)

    # Quantities as a function of strain
    volume_array_vac = en_vol_dict['volume_array']
    virial_array_vac = en_vol_dict['virial_array']
    energy_array = en_vol_dict['energy_array']

    # virial
    bcc_vac.compute_virial()
    bcc_vac.gather_virial()

    # virial derivative
    spline_list_vac = []
    virial_trace_array_vac = np.sum(virial_array_vac[:, :3, :], axis=1) / 3.0
    for idesc in range(bcc_vac.Ndesc):
        spline_list_vac.append(CubicSpline(volume_array_vac, virial_trace_array_vac[:, idesc]))

    # Virial derivative at minimum energy point
    min_idx = np.argmin(energy_array)
    volume_vac_min = volume_array_vac[min_idx]
    virial_der_vac0 = np.array([spline_list_vac[idesc](volume_vac_min, nu=1) for idesc in range(bcc_vac.Ndesc)])

    res_dict = run_npt_implicit_derivative(BccVacancy, alat, ncell_x, Theta_perturb,
                                           snapcoeff_filename, snapparam_filename,
                                           virial_der_vac0, dX_dTheta_vac_inhom, data_path='./refs')

    volume0_desired = 253.1556139760
    volume_true_desired = 234.1690107103
    volume_pred_desired = 192.0956114462
    volume_pred_full_desired = 192.0956114462
    energy0_desired = -80.3652746790
    energy_true_desired = -298.0019653030
    energy_pred0_desired = -291.8781171140
    energy_hom_pred_desired = -261.7611209474
    energy_inhom_pred_desired = -291.9562001894
    energy_full_pred_desired = -262.9750024118
    coord_error_full_desired = 0.7668165511
    coord_error_hom_desired = 0.7898012372
    coord_error_inhom_desired = 0.2970626250
    coord_error0_desired = 0.3126601018

    np.testing.assert_allclose(res_dict['volume0'], volume0_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['volume_true'], volume_true_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['volume_pred'], volume_pred_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['volume_pred_full'], volume_pred_full_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy0'], energy0_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy_true'], energy_true_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy_pred0'], energy_pred0_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy_hom_pred'], energy_hom_pred_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy_inhom_pred'], energy_inhom_pred_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy_full_pred'], energy_full_pred_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['coord_error_full'], coord_error_full_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['coord_error_hom'], coord_error_hom_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['coord_error_inhom'], coord_error_inhom_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['coord_error0'], coord_error0_desired, atol=1e-8)