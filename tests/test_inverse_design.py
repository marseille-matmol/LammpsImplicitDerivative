"""
Test of the inverse design routines.
Integration tests.
"""
import pytest
import numpy as np

from lammps_implicit_der.systems import BccVacancy
from lammps_implicit_der.tools import error_tools

# To save time, compute the BccVacancy object only once
@pytest.fixture(scope="module")
def bcc_vacancy(comm):
    return BccVacancy(alat=3.163, ncell_x=2, minimize=True, logname=None, del_coord=[0.0, 0.0, 0.0],
                      data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)


@pytest.fixture(scope="module")
def bcc_vacancy_target(comm):
    return BccVacancy(alat=3.163, ncell_x=2, minimize=True, fix_box_relax=True,
                                 logname=None, del_coord=[0.0, 0.0, 0.0], comm=comm,
                                 data_path='./refs', snapcoeff_filename='W_perturb4.snapcoeff', snapparam_filename='W.snapparam', verbose=False)


def test_minimze(bcc_vacancy, bcc_vacancy_target, comm):

    X_start = bcc_vacancy.X_coord.copy()
    X_target = bcc_vacancy_target.X_coord.copy()
    X_target = X_start.copy() + bcc_vacancy.minimum_image(X_target-X_start)

    bcc_start = BccVacancy(alat=3.163, ncell_x=2, minimize=True, logname=None, del_coord=[0.0, 0.0, 0.0],
                        data_path='./refs/', snapcoeff_filename='W.snapcoeff', verbose=False, comm=comm)

    bcc_vacancy_final, minim_dict = error_tools.minimize_loss(
                                        bcc_start,
                                        X_target,
                                        'W',
                                        comm=comm,
                                        step=2e-3,
                                        adaptive_step=True,
                                        maxiter=10,
                                        error_tol=5e-2,
                                        der_method='dense',
                                        verbosity=0,
                                        minimize_at_iters=False,
                                        apply_hard_constraints=False,
                                        )

    numiter_desired = 1
    error_array_desired = np.array([0.0325912346, 0.0256618410])
    loop_completed_desired = True
    converged_desired = True
    step_desired = 0.0011412791

    np.testing.assert_allclose(minim_dict['numiter'], numiter_desired)
    np.testing.assert_allclose(minim_dict['error_array'], error_array_desired, atol=1e-8)
    np.testing.assert_allclose(minim_dict['loop_completed'], loop_completed_desired)
    np.testing.assert_allclose(minim_dict['converged'], converged_desired)
    np.testing.assert_allclose(minim_dict["iter"][1]["step"], step_desired, atol=1e-8)