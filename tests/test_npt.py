"""
Tests of NPT minimization tools.
"""

import pytest
import numpy as np

from lammps_implicit_der.systems import BccVacancy
from lammps_implicit_der.tools import compute_energy_volume

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