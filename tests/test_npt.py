"""
Tests of NPT minimization tools.
These tests are integration tests since they include many steps of the workflow.
"""

import pytest
import numpy as np

from scipy.interpolate import CubicSpline
from lammps_implicit_der import SNAP
from lammps_implicit_der.systems import BCC_VACANCY
from lammps_implicit_der.tools import compute_energy_volume, run_npt_implicit_derivative

# To save time, compute th BCC_VACANCY object only once
@pytest.fixture(scope="module")
def bcc_vacancy(comm):
    return BCC_VACANCY(alat=3.163, ncell_x=2, minimize=True, logname=None, del_coord=[0.0, 0.0, 0.0],
                      data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)


# Helper function, will not be tested
def sort_coord(X_coord):

    X_round = np.round(X_coord, decimals=8)
    X_3D = X_round.reshape(-1, 3)

    # Sort based on x, y, z
    idx_sort = np.lexsort((X_3D[:, 2], X_3D[:, 1], X_3D[:, 0]))
    flat_indices = np.concatenate([np.arange(i*3, i*3+3) for i in idx_sort])

    return flat_indices


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
    pot_perturb = SNAP.from_files('W_perturb3.snapcoeff', snapparam_filename=snapparam_filename, data_path='./refs', comm=comm)
    Theta_perturb = pot_perturb.Theta_dict['W']['Theta'].copy()

    bcc_vac = BCC_VACANCY(alat=alat, ncell_x=ncell_x, minimize=True, logname=None, data_path='./refs/',
                          del_coord=[0.0, 0.0, 0.0],
                          snapcoeff_filename=snapcoeff_filename, verbose=False, comm=comm)

    dX_dTheta_vac_inhom = bcc_vac.implicit_derivative(method='dense')
    dX_dTheta_vac_inhom = dX_dTheta_vac_inhom[:, sort_coord(bcc_vac.X_coord)]
    dStrain_dTheta = bcc_vac.implicit_derivative_hom_iso(delta_Strain=1e-5)

    res_dict = run_npt_implicit_derivative(BCC_VACANCY, alat, ncell_x, Theta_perturb,
                                           snapcoeff_filename, snapparam_filename,
                                           dX_dTheta_vac_inhom, dStrain_dTheta, data_path='./refs',
                                           log_box_relax=None, log_pred=None, comm=comm, del_coord=[0.0, 0.0, 0.0])

    dStrain_dTheta_0_desired = 5.7560175907647
    volume0_desired = 253.1556139760
    volume_true_desired = 248.1193905314
    volume_pred_desired = 248.3463734820
    volume_pred_full_desired = 248.3463734820
    energy0_desired = -80.3652746790
    energy_true_desired = -151.1767062076
    energy_pred0_desired = -151.0473247287
    energy_hom_pred_desired = -151.1509332695
    energy_inhom_pred_desired = -151.0344359304
    energy_full_pred_desired = -151.1369099993

    np.testing.assert_allclose(dStrain_dTheta[0], dStrain_dTheta_0_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['volume0'], volume0_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['volume_true'], volume_true_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['volume_pred'], volume_pred_desired, atol=1e-2)
    np.testing.assert_allclose(res_dict['volume_pred_full'], volume_pred_full_desired, atol=1e-2)
    np.testing.assert_allclose(res_dict['energy0'], energy0_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy_true'], energy_true_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy_pred0'], energy_pred0_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy_hom_pred'], energy_hom_pred_desired, atol=1e-8)
    np.testing.assert_allclose(res_dict['energy_inhom_pred'], energy_inhom_pred_desired, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(res_dict['energy_full_pred'], energy_full_pred_desired, atol=1e-4, rtol=1e-4)
