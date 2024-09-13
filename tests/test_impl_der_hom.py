"""
Tests of the homogeneous implicit derivative.
Since the homogeneous implicit derivative is used for the strain/volume predictions,
we do not need to store position or the impl. der. matrix for these tests in contrast to test_impl_der.py.
"""
import pytest
import numpy as np

from lammps_implicit_der.systems import BCC_VACANCY, HCP

# To save time, compute the BCC_VACANCY object only once
@pytest.fixture(scope="module")
def bcc_vacancy(comm):
    return BCC_VACANCY(alat=3.163, ncell_x=2, minimize=True, logname=None, del_coord=[0.0, 0.0, 0.0],
                      data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)


@pytest.fixture(scope="module")
def bcc_vacancy_perturb(comm):
    return BCC_VACANCY(alat=3.163, ncell_x=2, minimize=True, fix_box_relax=True,
                      logname=None, del_coord=[0.0, 0.0, 0.0],
                      data_path='./refs', snapcoeff_filename='W_perturb3.snapcoeff', snapparam_filename='W.snapparam', verbose=False, comm=comm)


def test_impl_der_hom_iso(bcc_vacancy, bcc_vacancy_perturb):

    volume0 = bcc_vacancy.volume
    L0 = volume0**(1.0/3.0)
    cell0 = bcc_vacancy.cell.copy()

    volume_true = bcc_vacancy_perturb.volume
    L_true = volume_true**(1.0/3.0)

    Strain_true = (L_true - L0) / L0

    dTheta = bcc_vacancy_perturb.Theta - bcc_vacancy.Theta

    dStrain_dTheta = bcc_vacancy.implicit_derivative_hom_iso(delta_Strain=1e-5)
    Strain_pred = dTheta @ dStrain_dTheta
    cell_pred = cell0 @ (np.eye(3) * (1.0 + Strain_pred))
    volume_pred = np.linalg.det(cell_pred)

    dStrain_dTheta_0_desired = 5.756017590783412

    volume0_desired = 253.15561398
    volume_true_desired = 248.11939053
    Strain_true_desired = -0.00667573

    volume_pred_desired = 248.34636447
    Strain_pred_desired = -0.00637293

    np.testing.assert_allclose(dStrain_dTheta[0], dStrain_dTheta_0_desired, atol=1e-8)
    np.testing.assert_allclose(volume0, volume0_desired, atol=1e-8)
    np.testing.assert_allclose(volume_true, volume_true_desired, atol=1e-8)
    np.testing.assert_allclose(Strain_true, Strain_true_desired, atol=1e-8)
    np.testing.assert_allclose(volume_pred, volume_pred_desired, atol=1e-8)
    np.testing.assert_allclose(Strain_pred, Strain_pred_desired, atol=1e-8)


def test_impl_der_hom_iso_hcp(comm):

    alat_hcp = 2.89082627

    hcp_W = HCP(alat=alat_hcp, ncell_x=1,
                minimize=True, fix_box_relax=False, logname=None,
                data_path='./refs/', snapcoeff_filename='W.snapcoeff',
                verbose=False, comm=comm)

    hcp_W_perturb = HCP(alat=alat_hcp, ncell_x=1,
                        minimize=True, fix_box_relax=False, logname=None,
                        data_path='./refs/', snapcoeff_filename='W_perturb2.snapcoeff', snapparam_filename='W.snapparam',
                        verbose=False, comm=comm)

    cell0 = hcp_W.cell.copy()
    volume0 = hcp_W.volume

    dTheta = hcp_W_perturb.Theta - hcp_W.Theta

    dStrain_dTheta = hcp_W.implicit_derivative_hom_iso()

    Strain_pred = dTheta @ dStrain_dTheta
    cell_pred = cell0 @ (np.eye(3) * (1.0 + Strain_pred))
    volume_pred = np.linalg.det(cell_pred)

    Strain_pred_desired = 0.00011244071585598834
    volume0_desired = 68.32992931626956
    volume_pred_desired = 68.35298110653517

    np.testing.assert_allclose(Strain_pred, Strain_pred_desired, atol=1e-8)
    np.testing.assert_allclose(volume0, volume0_desired, atol=1e-8)
    np.testing.assert_allclose(volume_pred, volume_pred_desired, atol=1e-8)