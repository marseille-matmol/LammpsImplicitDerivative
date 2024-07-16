"""
Tests of the inhomogeneous implicit derivative.
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
    return BccVacancy(alat=3.163, ncell_x=2, minimize=True, logname=None,
                      data_path='./refs/', snapcoeff_filename='W_perturb.snapcoeff', snapparam_filename='W.snapparam', verbose=False, comm=comm)


# Helper function, will not be tested
def sort_coord(X_coord):

    X_round = np.round(X_coord, decimals=8)
    X_3D = X_round.reshape(-1, 3)

    # Sort based on x, y, z
    idx_sort = np.lexsort((X_3D[:, 2], X_3D[:, 1], X_3D[:, 0]))

    flat_indices = np.concatenate([np.arange(i*3, i*3+3) for i in idx_sort])

    return flat_indices


def test_impl_der_inverse(bcc_vacancy):

    dX_dTheta_desired = np.load('./refs/test_impl_der_inverse.npy')
    dX_dTheta = bcc_vacancy.implicit_derivative(method='inverse')

    # The LAMMPS atom indexing can be different for MPI runs, hence the sorting
    dX_dTheta = dX_dTheta[:, sort_coord(bcc_vacancy.X_coord)]

    np.testing.assert_allclose(dX_dTheta, dX_dTheta_desired, atol=1e-7)


def test_impl_der_dense(bcc_vacancy):

    dX_dTheta_desired = np.load('./refs/test_impl_der_dense.npy')
    dX_dTheta = bcc_vacancy.implicit_derivative(method='dense')

    dX_dTheta = dX_dTheta[:, sort_coord(bcc_vacancy.X_coord)]

    np.testing.assert_allclose(dX_dTheta, dX_dTheta_desired, atol=1e-7)


def test_impl_der_sparse(bcc_vacancy):

    dX_dTheta_desired = np.load('./refs/test_impl_der_sparse.npy')
    dX_dTheta = bcc_vacancy.implicit_derivative(method='sparse', alpha=1e-4, adaptive_alpha=False, maxiter=20)

    dX_dTheta = dX_dTheta[:, sort_coord(bcc_vacancy.X_coord)]

    np.testing.assert_allclose(dX_dTheta, dX_dTheta_desired, atol=1e-7)


def test_impl_der_energy_sd(bcc_vacancy):

    dX_dTheta_desired = np.load('./refs/test_impl_der_energy_sd.npy')
    dX_dTheta = bcc_vacancy.implicit_derivative(method='energy', adaptive_alpha=True, min_style='sd', alpha=1e-6, ftol=1e-10, maxiter=200)

    dX_dTheta = dX_dTheta[:, sort_coord(bcc_vacancy.X_coord)]

    np.testing.assert_allclose(dX_dTheta, dX_dTheta_desired, atol=1e-7)


def test_impl_der_energy_cg(bcc_vacancy):

    dX_dTheta_desired = np.load('./refs/test_impl_der_energy_cg.npy')
    dX_dTheta = bcc_vacancy.implicit_derivative(method='energy', adaptive_alpha=True, min_style='cg', alpha=1e-6, ftol=1e-10, maxiter=200)

    dX_dTheta = dX_dTheta[:, sort_coord(bcc_vacancy.X_coord)]

    np.testing.assert_allclose(dX_dTheta, dX_dTheta_desired, atol=1e-7)


def test_impl_der_energy_fire(comm):

    # Fire algo is slow, therefore, we test it on the smallest system possible, 3 atoms

    bcc_vacancy_211 = BccVacancy(alat=3.163, ncell_x=2, ncell_y=1, ncell_z=1, minimize=True, logname=None,  del_coord=[0.0, 0.0, 0.0],
                                 data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False)

    dX_dTheta_desired = np.load('./refs/test_impl_der_energy_fire.npy')
    dX_dTheta = bcc_vacancy_211.implicit_derivative(method='energy', adaptive_alpha=True, min_style='fire', alpha=1e-3, ftol=1e-7, maxiter=50)

    dX_dTheta = dX_dTheta[:, sort_coord(bcc_vacancy_211.X_coord)]

    np.testing.assert_allclose(dX_dTheta, dX_dTheta_desired, atol=1e-7)


def test_impl_der_energy_dX_sd(bcc_vacancy, bcc_vacancy_perturb):

    dX_dTheta = bcc_vacancy.implicit_derivative(method='energy', adaptive_alpha=True, min_style='sd', alpha=1e-6, ftol=1e-10, maxiter=200)

    dTheta = bcc_vacancy_perturb.Theta - bcc_vacancy.Theta

    dX_pred = dTheta @ dX_dTheta

    dX_pred -= dX_pred.mean()

    dX_pred = dX_pred[sort_coord(bcc_vacancy.X_coord)]
    dX_pred_desired = np.load('./refs/test_impl_der_energy_dX_sd.npy')

    np.testing.assert_allclose(dX_pred, dX_pred_desired, atol=1e-7)