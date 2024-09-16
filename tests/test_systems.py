"""
Basic tests of the systems module.
"""
import os
import numpy as np
import pytest

from lammps_implicit_der.systems import BCC, BCC_BINARY, BCC_BINARY_VACANCY, BCC_VACANCY, BCC_SIA, FromData, HCP


def test_bcc_no_minimization(comm):
    system = BCC(alat=3.18, ncell_x=2, minimize=False, logname=None,
                 data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(system.energy, -89.0555278168474)
    np.testing.assert_equal(system.Natom, 16)
    np.testing.assert_equal(system.Ndesc, 55)


def test_bcc(comm):

    system = BCC(alat=3.18427, ncell_x=2, minimize=True, logname=None,
                 data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(system.energy, -89.060889667491)
    np.testing.assert_equal(system.Natom, 16)


def test_bcc_vacancy(comm):

    system = BCC_VACANCY(alat=3.163, ncell_x=2, minimize=True, logname=None,
                         data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(system.energy, -80.36527467896518)
    np.testing.assert_equal(system.Natom, 15)


def test_bcc_binary(comm):

    if comm is not None and comm.Get_size() > 1:
        pytest.skip("Test is disabled when run with MPI. Wrong species generation.")

    system = BCC_BINARY(alat=3.13, ncell_x=2, minimize=True, logname=None,
                        data_path='./refs/', snapcoeff_filename='NiMo.snapcoeff', verbose=False, comm=comm)

    species_desired = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1]

    np.testing.assert_equal(system.species, species_desired)
    np.testing.assert_allclose(system.energy, -122.76839711519715)
    np.testing.assert_equal(system.Natom, 16)
    np.testing.assert_equal(system.Ndesc, 30)


def test_bcc_binary_vacancy(comm):

    if comm is not None and comm.Get_size() > 1:
        pytest.skip("Test is disabled when run with MPI. Wrong species generation.")

    system = BCC_BINARY_VACANCY(alat=3.13, ncell_x=2, minimize=True, logname=None,
                                data_path='./refs/', snapcoeff_filename='NiMo.snapcoeff', verbose=False, comm=comm)

    species_desired = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1]

    np.testing.assert_equal(system.species, species_desired)
    np.testing.assert_allclose(system.energy, -116.12471812826522)
    np.testing.assert_equal(system.Natom, 15)


def test_bcc_SIA(comm):

    system = BCC_SIA(alat=3.18, ncell_x=2, minimize=False, logname=None,
                     data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(system.energy, -53.04187280340517)
    np.testing.assert_equal(system.Natom, 17)


def test_hcp(comm):

    alat_hcp = 2.84752278

    hcp_W = HCP(alat=alat_hcp, ncell_x=1,
                minimize=True, fix_box_relax=False, logname=None,
                data_path='./refs/', snapcoeff_filename='W.snapcoeff',
                verbose=False, comm=comm)

    np.testing.assert_allclose(hcp_W.energy, -20.19323450167738)
    np.testing.assert_allclose(hcp_W.cell[0, 0], 2.84752278)
    # Should be equal to alat_hcp * np.sqrt(3)
    np.testing.assert_allclose(hcp_W.cell[1, 1], 4.932054130669774)
    # Should be equal to alat_hcp * np.sqrt(8/3)
    np.testing.assert_allclose(hcp_W.cell[2, 2], 4.649985227967626)


def test_hcp_box_relax(comm):

    alat_hcp = 2.84752278

    hcp_W_box_relax = HCP(alat=alat_hcp, ncell_x=1,
                          minimize=True, fix_box_relax=True, minimize_maxiter=1000, box_relax_iso=False,
                          data_path='./refs/', snapcoeff_filename='W.snapcoeff',
                          verbose=False, comm=comm)

    np.testing.assert_allclose(hcp_W_box_relax.energy, -20.554598258667664)
    np.testing.assert_allclose(hcp_W_box_relax.cell[0, 0], 2.8054015915512567)
    np.testing.assert_allclose(hcp_W_box_relax.cell[1, 1], 4.85909809220136)
    np.testing.assert_allclose(hcp_W_box_relax.cell[2, 2], 4.958918214223956)


def test_box_relax(comm):

    alat0 = 3.175
    ncell_x = 2
    system = BCC_VACANCY(alat=alat0, ncell_x=ncell_x, minimize=True, logname=None,
                         data_path='./refs/', snapcoeff_filename='W.snapcoeff', fix_box_relax=True, verbose=False, comm=comm)

    alat = system.volume**(1/3) / ncell_x

    np.testing.assert_allclose(alat, 3.163163264038606)
    np.testing.assert_allclose(system.energy, -80.3652809680334)


def test_from_data(comm):

    data = """LAMMPS data file via write_data, version 2 Aug 2023, timestep = 50, units = metal

    3 atoms
    1 atom types

    0.12887883656057453 6.221121163439436 xlo xhi
    0.06443941828028726 3.110560581719718 ylo yhi
    0.06443941828028726 3.110560581719718 zlo zhi

    Masses

    1 184

    Atoms # atomic

    1 1 4.932172063178125 1.6179612116343973 1.6179612116343973 -1 -1 -1
    2 1 1.4787503600906773 1.6179612116343973 1.6179612116343973 0 -1 -1
    3 1 3.205461211634401 0.09490062991468183 0.09490062991468182 -1 0 0

    Velocities

    1 0 0 0
    2 0 0 0
    3 0 0 0
    """

    if comm is None or comm.Get_rank() == 0:
        with open('test.data', 'w') as f:
            f.write(data)

    system = FromData(datafile='test.data', data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    if comm is not None:
        comm.Barrier()

    if comm is None or comm.Get_rank() == 0:
        os.remove('test.data')

    np.testing.assert_allclose(system.energy, -14.215395402648571)
    np.testing.assert_equal(system.Natom, 3)


def test_from_data_input_script(comm):

    input_script = """
        atom_modify map array sort 0 0.0
        units metal
        boundary p p p
        lattice bcc 3.175 origin 0.01 0.01 0.01
        region C block 0 2 0 1 0 1 units lattice
        create_box 1 C
        create_atoms 1 region C
        mass * 184.
    """

    system = FromData(input_script=input_script, data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    np.testing.assert_allclose(system.energy, -22.258870715899455)
    np.testing.assert_equal(system.Natom, 4)


