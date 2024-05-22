"""
Test the properties of the LammpsImplicitDer child classes.
"""
import pytest
import numpy as np

from lammps_implicit_der.systems import BccVacancy, BccBinary, Bcc


# To save time, compute th BccVacancy object only once
@pytest.fixture(scope="module")
def bcc_system(comm):
    return Bcc(alat=3.163, ncell_x=1, minimize=False, logname=None,
               data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)


def test_energy(bcc_system):

    energy = bcc_system.energy
    energy_DT = bcc_system.dU_dTheta @ bcc_system.Theta

    np.testing.assert_allclose(energy, -11.115719916006501)
    np.testing.assert_allclose(energy_DT, -11.1157199160065)


def test_X_coord(bcc_system):

    X_coord_desired = np.array([0.03163,  0.03163,  0.03163, -1.54987, -1.54987, -1.54987])

    np.testing.assert_allclose(bcc_system.X_coord, X_coord_desired)


def test_cell(bcc_system):

    cell_desired = np.eye(3) * 3.163
    inv_cell_desired = np.eye(3) / 3.163

    np.testing.assert_allclose(bcc_system.cell, cell_desired)
    np.testing.assert_allclose(bcc_system.inv_cell, inv_cell_desired)


def test_minimimal_image(bcc_system):

    X_coord0 = bcc_system.X_coord.copy()
    X_coord_test = bcc_system.X_coord.copy()

    X_coord_test[0] = 20.0
    X_coord_test[3] = 50.0
    # Minimal image convention is applied:
    bcc_system.X_coord = X_coord_test

    X_coord_desired = np.array([1.022,  0.03163,  0.03163, -0.608, -1.54987, -1.54987])

    np.testing.assert_allclose(bcc_system.X_coord, X_coord_desired)

    bcc_system.X_coord = X_coord0


def test_volume(bcc_system):

    volume_desired = 3.163**3

    np.testing.assert_allclose(bcc_system.volume, volume_desired)


def test_apply_strain(bcc_system):

    cell0 = bcc_system.cell.copy()

    strain = 1.5
    cell_test = np.dot(cell0, np.eye(3) * strain)
    bcc_system.apply_strain(cell_test, update_system=True)

    cell_desired = np.eye(3) * 3.163 * strain

    np.testing.assert_allclose(bcc_system.cell, cell_desired)

    bcc_system.apply_strain(cell0, update_system=True)


def test_D_unary(comm):

    bcc_system_tmp = Bcc(alat=3.163, ncell_x=1, minimize=False, logname=None,
                         data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    X_coord_test = bcc_system_tmp.X_coord.copy()
    X_coord_test[4] = 0.5

    bcc_system_tmp.scatter_coord(X_coord=X_coord_test)
    bcc_system_tmp.compute_D_dD()
    bcc_system_tmp.gather_D_dD()

    dU_dTheta_desired = np.load('./refs/test_system_props_dU_dTheta.npy')

    np.testing.assert_allclose(bcc_system_tmp.dU_dTheta, dU_dTheta_desired, atol=1e-8)


def test_dD_unary(comm):
    """
    Does not work with ntasks 4 and 8, works with others!
    Maybe, the system is too small.
    """

    bcc_system_tmp = Bcc(alat=3.163, ncell_x=1, minimize=False, logname=None,
                         data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    X_coord_test = bcc_system_tmp.X_coord.copy()
    X_coord_test[4] = 0.5

    bcc_system_tmp.scatter_coord(X_coord=X_coord_test)
    bcc_system_tmp.compute_D_dD()
    bcc_system_tmp.gather_D_dD()

    mixed_hessian_desired = np.load('./refs/test_system_props_mixed_hessian.npy')

    np.testing.assert_allclose(bcc_system_tmp.mixed_hessian, mixed_hessian_desired, atol=1e-8)


def test_D_dD_binary(comm):

    #if comm is not None and comm.Get_size() > 1:
    #    pytest.skip("Test is disabled when run with MPI. Wrong species generation.")

    bcc_binary = BccBinary(alat=3.13, ncell_x=1, minimize=True, logname=None,
                           data_path='./refs/', snapcoeff_filename='NiMo.snapcoeff', verbose=False, comm=comm)

    X_coord_test = bcc_binary.X_coord.copy()
    X_coord_test[4] = 0.5
    bcc_binary.scatter_coord(X_coord=X_coord_test)
    bcc_binary.compute_D_dD()
    bcc_binary.gather_D_dD()

    dU_dTheta_desired = np.load('./refs/test_system_props_dU_dTheta_binary.npy')
    mixed_hessian_desired = np.load('./refs/test_system_props_mixed_hessian_binary.npy')

    np.testing.assert_allclose(bcc_binary.dU_dTheta, dU_dTheta_desired, atol=1e-8)
    np.testing.assert_allclose(bcc_binary.mixed_hessian, mixed_hessian_desired, atol=1e-8)


def test_forces(comm):

    bcc_system_tmp = Bcc(alat=3.163, ncell_x=1, minimize=False, logname=None,
                         data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    dX_vector = np.zeros(bcc_system_tmp.Natom*3)
    dX_vector[0] = 0.1

    forces_desired = np.array(
     [-0.05229752523716957, 9.575673587391975e-16, 2.847605337023395e-15, 0.05229752523716957, -1.0269562977782698e-15, -2.9211995299477888e-15]
    )

    np.testing.assert_allclose(bcc_system_tmp.forces(dx=dX_vector), forces_desired, atol=1e-12)


def test_hessian(comm):

    bcc_system_tmp = Bcc(alat=3.163, ncell_x=1, minimize=False, logname=None,
                         data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    desired_hessian = np.load('./refs/test_system_props_hessian.npy')

    np.testing.assert_allclose(bcc_system_tmp.hessian(), desired_hessian, atol=1e-12)