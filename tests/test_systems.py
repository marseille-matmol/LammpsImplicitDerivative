import pytest
import numpy as np

from lammps_implicit_der.systems import Bcc, BccBinary, BccBinaryVacancy, BccVacancy, BccSIA


def test_bcc():

    system = Bcc(alat=3.18427, ncell_x=2, minimize=True, logname=None, data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False)

    np.testing.assert_allclose(system.energy, -89.060889667491)
    np.testing.assert_equal(system.Natom, 16)


def test_bcc_vacancy():

    system = BccVacancy(alat=3.163, ncell_x=2, minimize=True, logname=None, data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False)

    np.testing.assert_allclose(system.energy, -80.36527467896518)
    np.testing.assert_equal(system.Natom, 15)


def test_bcc_binary():

    system = BccBinary(alat=3.13, ncell_x=2, minimize=True, logname=None, data_path='./refs/', snapcoeff_filename='NiMo.snapcoeff', verbose=False)

    species_target = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1]

    np.testing.assert_allclose(system.energy, -122.76839711519715)
    np.testing.assert_equal(system.Natom, 16)
    np.testing.assert_equal(system.species, species_target)