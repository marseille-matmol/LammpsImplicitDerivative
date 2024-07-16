"""
Tests of the homogeneous implicit derivative.
Since the homogeneous implicit derivative is used for the strain/volume predictions,
we do not need to store position or the impl. der. matrix for these tests in contrast to test_impl_der.py.
"""
import pytest
import numpy as np

from lammps_implicit_der.systems import BccVacancy

# To save time, compute the BccVacancy object only once
@pytest.fixture(scope="module")
def bcc_vacancy(comm):
    return BccVacancy(alat=3.163, ncell_x=2, minimize=True, logname=None, del_coord=[0.0, 0.0, 0.0],
                      data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)


@pytest.fixture(scope="module")
def bcc_vacancy_perturb(comm):
    return BccVacancy(alat=3.163, ncell_x=2, minimize=True, fix_box_relax=True,
                      logname=None, del_coord=[0.0, 0.0, 0.0],
                      data_path='./refs', snapcoeff_filename='W_perturb3.snapcoeff', snapparam_filename='W.snapparam', verbose=False, comm=comm)


def test_impl_der_hom_dVirial(bcc_vacancy, bcc_vacancy_perturb):

    volume0 = bcc_vacancy.volume
    L0 = volume0**(1.0/3.0)

    volume_true = bcc_vacancy_perturb.volume

    dTheta = bcc_vacancy_perturb.Theta - bcc_vacancy.Theta

    dL_dTheta = bcc_vacancy.implicit_derivative_hom(method='dVirial')
    dL_pred = dTheta @ dL_dTheta
    volume_pred = (L0 + dL_pred)**3

    volume0_desired = 253.15561398
    volume_true_desired = 248.11939053
    volume_pred_desired = 248.34648965

    np.testing.assert_allclose(volume0, volume0_desired, atol=1e-8)
    np.testing.assert_allclose(volume_true, volume_true_desired, atol=1e-8)
    np.testing.assert_allclose(volume_pred, volume_pred_desired, atol=1e-8)


def test_impl_der_hom_d2Desc(bcc_vacancy, bcc_vacancy_perturb):

    volume0 = bcc_vacancy.volume
    L0 = volume0**(1.0/3.0)

    dTheta = bcc_vacancy_perturb.Theta - bcc_vacancy.Theta

    dL_dTheta = bcc_vacancy.implicit_derivative_hom(method='d2Desc')
    dL_pred = dTheta @ dL_dTheta
    volume_pred = (L0 + dL_pred)**3

    volume_pred_desired = 248.34677882

    np.testing.assert_allclose(volume_pred, volume_pred_desired, atol=1e-8)


