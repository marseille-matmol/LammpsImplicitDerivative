"""
Test of the inverse design routines.
Integration tests.
"""
import shutil
import pytest
import numpy as np

from lammps_implicit_der.systems import BCC_VACANCY
from lammps_implicit_der.tools import error_tools

# To save time, compute the BCC_VACANCY object only once
@pytest.fixture(scope="module")
def bcc_vacancy(comm):
    return BCC_VACANCY(alat=3.163, ncell_x=2, minimize=True, logname=None, del_coord=[0.0, 0.0, 0.0],
                      data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)


@pytest.fixture(scope="module")
def bcc_vacancy_target(comm):
    return BCC_VACANCY(alat=3.163, ncell_x=2, minimize=True, fix_box_relax=True,
                                 logname=None, del_coord=[0.0, 0.0, 0.0], comm=comm,
                                 data_path='./refs', snapcoeff_filename='W_perturb4.snapcoeff', snapparam_filename='W.snapparam', verbose=False)


def test_minimze(bcc_vacancy, bcc_vacancy_target, comm):

    X_start = bcc_vacancy.X_coord.copy()
    X_target = bcc_vacancy_target.X_coord.copy()
    X_target = X_start.copy() + bcc_vacancy.minimum_image(X_target-X_start)

    bcc_start = BCC_VACANCY(alat=3.163, ncell_x=2, minimize=True, logname=None, del_coord=[0.0, 0.0, 0.0],
                           data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    bcc_vacancy_final, minim_dict = error_tools.minimize_loss(
                                        bcc_start,
                                        X_target,
                                        'W',
                                        comm=comm,
                                        fixed_step=2e-3,
                                        adaptive_step=True,
                                        maxiter=10,
                                        error_tol=5e-2,
                                        der_method='dense',
                                        verbosity=0,
                                        minimize_at_iters=False,
                                        apply_hard_constraints=False,
                                        )

    comm.Barrier()

    if comm is None or comm.rank == 0:
        shutil.rmtree('minim_output')

    numiter_desired = 2
    error_array_desired = np.array([0.0325912346, 0.0256618410])
    loop_completed_desired = True
    converged_desired = True
    step_desired = 0.0011412791

    np.testing.assert_allclose(minim_dict['numiter'], numiter_desired)
    np.testing.assert_allclose(minim_dict['error_array'], error_array_desired, atol=1e-8)
    np.testing.assert_allclose(minim_dict['loop_completed'], loop_completed_desired)
    np.testing.assert_allclose(minim_dict['converged'], converged_desired)
    np.testing.assert_allclose(minim_dict["iter"][1]["step"], step_desired, atol=1e-8)
