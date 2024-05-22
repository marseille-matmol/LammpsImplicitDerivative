"""
Basic tests of the systems module.
"""
import numpy as np
import pytest

from lammps_implicit_der.systems import Bcc, BccBinary, BccBinaryVacancy, BccVacancy, BccSIA


def test_bcc_no_minimization(comm):
    system = Bcc(alat=3.18, ncell_x=2, minimize=False, logname=None,
                 data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(system.energy, -89.0555278168474)
    np.testing.assert_equal(system.Natom, 16)
    np.testing.assert_equal(system.Ndesc, 55)


def test_bcc(comm):

    system = Bcc(alat=3.18427, ncell_x=2, minimize=True, logname=None,
                 data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(system.energy, -89.060889667491)
    np.testing.assert_equal(system.Natom, 16)


def test_bcc_vacancy(comm):

    system = BccVacancy(alat=3.163, ncell_x=2, minimize=True, logname=None,
                        data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(system.energy, -80.36527467896518)
    np.testing.assert_equal(system.Natom, 15)


def test_bcc_binary(comm):

    #if comm is not None and comm.Get_size() > 1:
    #    pytest.skip("Test is disabled when run with MPI. Wrong species generation.")

    system = BccBinary(alat=3.13, ncell_x=2, minimize=True, logname=None,
                       data_path='./refs/', snapcoeff_filename='NiMo.snapcoeff', verbose=False, comm=comm)

    species_desired = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1]

    np.testing.assert_equal(system.species, species_desired)
    np.testing.assert_allclose(system.energy, -122.76839711519715)
    np.testing.assert_equal(system.Natom, 16)
    np.testing.assert_equal(system.Ndesc, 30)


def test_bcc_binary_vacancy(comm):

    #if comm is not None and comm.Get_size() > 1:
    #    pytest.skip("Test is disabled when run with MPI. Wrong species generation.")

    system = BccBinaryVacancy(alat=3.13, ncell_x=2, minimize=True, logname=None,
                              data_path='./refs/', snapcoeff_filename='NiMo.snapcoeff', verbose=False, comm=comm)

    species_desired = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1]

    np.testing.assert_equal(system.species, species_desired)
    np.testing.assert_allclose(system.energy, -116.12471812826522)
    np.testing.assert_equal(system.Natom, 15)


def test_bcc_SIA(comm):

    system = BccSIA(alat=3.18, ncell_x=2, minimize=False, logname=None,
                    data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(system.energy, -53.04187280340517)
    np.testing.assert_equal(system.Natom, 17)


def test_box_relax(comm):

    alat0 = 3.175
    ncell_x = 2
    system = BccVacancy(alat=alat0, ncell_x=ncell_x, minimize=True, logname=None,
                        data_path='./refs/', snapcoeff_filename='W.snapcoeff', fix_box_relax=True, verbose=False, comm=comm)

    alat = system.volume**(1/3) / ncell_x

    np.testing.assert_allclose(alat, 3.163163264038606)
    np.testing.assert_allclose(system.energy, -80.3652809680334)