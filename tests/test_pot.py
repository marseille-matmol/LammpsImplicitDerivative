"""
Test the methods and attributes of the SNAP potential class
"""
import os
import numpy as np
import pytest

from lammps_implicit_der.systems import BCC, BCC_BINARY


def test_pot_keys(comm):

    bcc_binary = BCC_BINARY(alat=3.13, specie_B_concentration=0.5, ncell_x=1,
                           minimize=False, logname=None, data_path='./refs/', snapcoeff_filename='NiMo.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_equal(bcc_binary.pot.elem_list, ['Ni', 'Mo'])
    assert all(k in bcc_binary.pot.Theta_dict for k in ['Ni', 'Mo', 'radii', 'weights'])


def test_pot_params(comm):

    bcc_binary = BCC_BINARY(alat=3.13, specie_B_concentration=0.5, ncell_x=1,
                           minimize=False, logname=None, data_path='./refs/', snapcoeff_filename='NiMo.snapcoeff', verbose=False, comm=comm)

    radii_desired = '0.575 0.575'
    weights_desired = '0.5 1.0'

    np.testing.assert_string_equal(bcc_binary.pot.Theta_dict['radii'], radii_desired)
    np.testing.assert_string_equal(bcc_binary.pot.Theta_dict['weights'], weights_desired)

    np.testing.assert_allclose(bcc_binary.pot.Theta_dict['Ni']['elem_params']['radius'], 0.575)
    np.testing.assert_allclose(bcc_binary.pot.Theta_dict['Ni']['elem_params']['weight'], 0.5)
    np.testing.assert_allclose(bcc_binary.pot.Theta_dict['Ni']['beta0'], -5.74137597148)

    np.testing.assert_allclose(bcc_binary.pot.Theta_dict['Mo']['elem_params']['radius'], 0.575)
    np.testing.assert_allclose(bcc_binary.pot.Theta_dict['Mo']['elem_params']['weight'], 1.0)
    np.testing.assert_allclose(bcc_binary.pot.Theta_dict['Mo']['beta0'], -11.1413071988)


def test_pot_coeff(comm):

    bcc_pure = BCC(alat=3.18427, ncell_x=2, minimize=False, logname=None,
                   data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)
    coeffs = np.loadtxt('./refs/W.snapcoeff', skiprows=7)

    np.testing.assert_allclose(bcc_pure.pot.Theta_dict['W']['Theta'], coeffs)


def test_write(comm):

    bcc_binary = BCC_BINARY(alat=3.13, specie_B_concentration=0.5, ncell_x=1,
                           minimize=False, logname=None, data_path='./refs/', snapcoeff_filename='NiMo.snapcoeff', verbose=False, comm=comm)

    Theta0 = bcc_binary.pot.Theta_dict['Ni']['Theta'].copy()

    Theta_test = np.arange(Theta0.shape[0])
    bcc_binary.pot.Theta_dict['Ni']['Theta'] = Theta_test

    bcc_binary.pot.to_files(path='.', overwrite=True, snapcoeff_filename='test.snapcoeff', snapparam_filename='test.snapparam', verbose=False)

    # check if the files were written
    assert os.path.isfile('test.snapcoeff')
    assert os.path.isfile('test.snapparam')

    bcc_test = BCC_BINARY(alat=3.13, specie_B_concentration=0.2, ncell_x=2,
                         minimize=False, logname=None, data_path='.', snapcoeff_filename='test.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(bcc_test.pot.Theta_dict['Ni']['Theta'], Theta_test)

    # remove the files
    if comm is None or comm.Get_rank() == 0:
        os.remove('test.snapcoeff')
        os.remove('test.snapparam')