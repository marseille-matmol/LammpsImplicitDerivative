#!/usr/bin/env python3
# coding: utf-8

import os
import time
from tqdm import tqdm
import copy
import numpy as np
import pickle

# local imports
from .utils import mpi_print, save_snap_coeff, get_projection
from .timing import TimingGroup
from ..systems.bcc_vacancy import BccVacancy
# from ase.io import read, write


def compute_dX_dTheta_dict(S_non_perturb, alpha=0.001, comm=None):
    """Compute the dictionary of dX_dTheta arrays for multiple methods.

    Parameters
    ----------

    S_non_perturb : BccVacancy
        BccVacancy object with unperturbed parameters.

    alpha : float
        Finite difference parameter.

    Returns
    -------

    dX_dTheta_dict : dictionary
        Dictionary of dX_dTheta arrays.
    """

    dX_dTheta_dict = {}

    mpi_print('Computing dX/dTheta with sparse method...', comm=comm)
    t0 = time.perf_counter()
    dX_dTheta_dict['sparse'] = S_non_perturb.implicit_derivative_sparse(alpha=alpha, atol=1e-7, maxiter=1000)['dX_dTheta']
    mpi_print(f'...done in {time.perf_counter()-t0:.3f} s', comm=comm)

    mpi_print('Computing dX/dTheta with energy fix method...', comm=comm)
    t0 = time.perf_counter()
    dX_dTheta_dict['energy'] = S_non_perturb.implicit_derivative_energy(alpha=alpha, maxiter=1000, ftol=1e-8)['dX_dTheta']
    mpi_print(f'...done in {time.perf_counter()-t0:.3f} s', comm=comm)

    mpi_print('Computing dX/dTheta with inverse method...', comm=comm)
    t0 = time.perf_counter()
    dX_dTheta_dict['inverse'] = S_non_perturb.implicit_derivative_inverse()
    mpi_print(f'...done in {time.perf_counter()-t0:.3f} s', comm=comm)

    mpi_print('Computing dX/dTheta with dense method...', comm=comm)
    t0 = time.perf_counter()
    dX_dTheta_dict['dense'] = S_non_perturb.implicit_derivative_dense()
    mpi_print(f'...done in {time.perf_counter()-t0:.3f} s', comm=comm)

    return dX_dTheta_dict


def compute_dX_error(S_non_perturb, dX_dTheta_dict, delta, numiter=100, verbosity=2, store_dX=False, comm=None):
    """Compute the error in displacements from a derivative.

    Parameters
    ----------

    S_non_perturb : LammpsImplicitDer
        LammpsImplicitDer object with unperturbed parameters.

    dX_dTheta_dict : dictionary
        Dictionary of dX_dTheta arrays.

    delta : float
        Perturbation parameter.

    numiter : int, optional
        Number of iterations, by default 100.

    verbosity : int, optional
        Verbosity level, by default 2.

    store_dX : bool, optional
        Store the displacements for each iteration, by default False.

    Returns
    -------

    error_dict : dictionary
        Dictionary of errors for each method.
        Structure:
            error_dict[method] : array of errors for each iteration

    trun_dict : dictionary
        Dictionary of timing results.
    """

    trun = TimingGroup("compute_dX_error")

    # Error dictionary, error defined as ||dX_true - dX_approx|| / ||dX_true||
    error_dict = {method: [] for method in dX_dTheta_dict.keys()}

    # Displacement dictionary |dX|^2 / Natom, for each method and the ground truth
    dX_error_dict = {method: [] for method in ['true'] + list(dX_dTheta_dict.keys())}

    if store_dX:
        dX_dict = {method: [] for method in ['true'] + list(dX_dTheta_dict.keys())}
    else:
        dX_dict = None

    Theta = S_non_perturb.Theta

    trun.add('total', level=2).start()

    if verbosity >= 1:
        iterator = tqdm(range(numiter))
    else:
        iterator = range(numiter)

    for i in iterator:

        # Generate a random perturbation of potential parameters
        with trun.add('init dTheta') as t:
            dTheta = Theta * np.random.uniform(-1.0, 1.0, size=Theta.shape) * delta

        # Save the parameters to a file
        with trun.add('save_snap_coeff') as t:
            save_snap_coeff("TEMP.snapcoeff", Theta+dTheta)

        # Initialize a new BccVacancy object with perturbed parameters
        with trun.add('minimize for Theta+dTheta') as t:
            S_perturb = BccVacancy(num_cells=S_non_perturb.num_cells, comm=S_non_perturb.comm, snapcoeff_path="TEMP.snapcoeff", verbose=False)

        # Compute the ground truth for displacements
        dX_true = S_perturb.X_coord - S_non_perturb.X_coord

        dX_error_dict['true'].append(np.linalg.norm(dX_true)**2 / S_non_perturb.Natom)

        if store_dX:
            dX_dict['true'].append(dX_true)

        # Compute the displacements from the derivatives
        with trun.add('Compute dX_approx') as t:

            for method, dX_dTheta in dX_dTheta_dict.items():
                dX_approx = dTheta @ dX_dTheta

                error_dict[method].append(np.linalg.norm(dX_true - dX_approx) / np.linalg.norm(dX_true))

                dX_error_dict[method].append(np.linalg.norm(dX_approx)**2 / S_non_perturb.Natom)

                if store_dX:
                    dX_dict[method].append(dX_approx)

    error_dict = {method: np.array(error_dict[method]) for method in error_dict.keys()}
    dX_error_dict = {method: np.array(dX_error_dict[method]) for method in dX_error_dict.keys()}

    if store_dX:
        dX_dict = {method: np.array(dX_dict[method]) for method in dX_dict.keys()}

    trun.timings['total'].stop()

    mpi_print(trun, comm=comm)

    return error_dict, dX_dict, dX_error_dict, trun.to_dict()


def compute_dX_error_delta(S_non_perturb, dX_dTheta_dict, delta_array, numiter=100, comm=None):
    """Compute the error in displacements from a derivative for multiple values of delta.

    Parameters
    ----------
    S_non_perturb : LammpsImplicitDer
        LammpsImplicitDer object with unperturbed parameters.

    dX_dTheta_dict : dictionary
        Dictionary of dX_dTheta arrays.

    delta_array : numpy array
        Array of delta values.

    numiter : int, optional
        Number of iterations, by default 100.

    Returns
    -------

    error_dict : dictionary
        Dictionary of errors for each method.
        Structure:
            error_dict[method]['error']['mean'] : list of mean errors (over iterations) for each delta
            error_dict[method]['error']['iterations'] : list of arrays of errors for each iteration
            error_dict[method]['dx']['mean'] : list of mean displacements (over iterations) for each delta
            error_dict[method]['dx']['iterations'] : list of arrays of displacements for each iteration
    """

    error_delta_dict = {method: {'error': {'mean': [], 'iterations': []}, 'dx': {'mean': [], 'iterations': []}} for method in dX_dTheta_dict.keys()}

    # For dx, we also have the ground truth
    error_delta_dict['true'] = {'dx': {'mean': [], 'iterations': []}}

    for i, delta in enumerate(delta_array):

        mpi_print(f'{i+1:>3} / {delta_array.size}    delta = {delta:9.4f}', comm=comm)

        error_dict, dX_dict, dX_error_dict, trun_dict = compute_dX_error(S_non_perturb, dX_dTheta_dict, delta, numiter=numiter, verbosity=1)

        error_delta_dict['true']['dx']['mean'].append(dX_error_dict['true'].mean())
        error_delta_dict['true']['dx']['iterations'].append(dX_error_dict['true'])

        for method in error_dict.keys():
            error_delta_dict[method]['error']['mean'].append(error_dict[method].mean())
            error_delta_dict[method]['error']['iterations'].append(error_dict[method])

            error_delta_dict[method]['dx']['mean'].append(dX_error_dict[method].mean())
            error_delta_dict[method]['dx']['iterations'].append(dX_error_dict[method].mean())

        mpi_print(f'{trun_dict["total"]["runtime"]:.3f} s', comm=comm)
        mpi_print('', comm=comm)

    return error_delta_dict


def compute_dX_error_alpha(S_non_perturb, delta, alpha_array, numiter=100, comm=None):
    """Compute the error in displacements from a derivative for multiple values of alpha.

    Parameters
    ----------

    S_non_perturb : LammpsImplicitDer
        LammpsImplicitDer object with unperturbed parameters.

    delta : float
        Perturbation parameter.

    alpha_array : numpy array
        Array of alpha values.

    numiter : int, optional
        Number of iterations, by default 100.

    Returns
    -------

    error_alpha_dict : dictionary
        Dictionary of errors for each method.
        Structure:
            error_alpha_dict[method]['mean'] : list of mean errors (over iterations) for each alpha
            error_alpha_dict[method]['iterations'] : list of array of errors for each iteration
    """

    # dict of solvers and functions to call, information if it depends on alpha
    solver_dict = {
        'sparse': {'alpha-dependent': True,
                   'function': lambda alpha: S_non_perturb.implicit_derivative_sparse(alpha=alpha,
                                                                                       maxiter=1000,
                                                                                       atol=1e-7)},
        'energy': {'alpha-dependent': True,
                   'function': lambda alpha: S_non_perturb.implicit_derivative_energy(alpha=alpha,
                                                                                       maxiter=1000,
                                                                                       adaptive_alpha=False,
                                                                                       ftol=1e-8)},
        'inverse': {'alpha-dependent': False,
                    'function': lambda alpha: S_non_perturb.implicit_derivative_inverse()},
        'dense': {'alpha-dependent': False,
                  'function': lambda alpha: S_non_perturb.implicit_derivative_dense()}
    }

    error_alpha_dict = {method: {'mean': [], 'iterations': []} for method in solver_dict.keys()}
    dX_dTheta_dict = {}

    for i, alpha in enumerate(alpha_array):

        mpi_print(f'{i+1:>3} / {alpha_array.size}    alpha = {alpha:9.4f}', comm=comm)

        # Compute the dX_dTheta_dict for fixed alpha
        for method, solver in solver_dict.items():

            # no calculation needed if i>0 and solver does not depend on alpha, reuse the result from the dX_dTheta_dict
            if solver['alpha-dependent'] or i == 0:
                mpi_print(f'Computing dX/dTheta with {method} method...')
                t0 = time.perf_counter()
                dX_dTheta = solver['function'](alpha)
                mpi_print(f'...done in {time.perf_counter()-t0:.3f} s')

                if type(dX_dTheta) is dict:
                    dX_dTheta = dX_dTheta['dX_dTheta']

                dX_dTheta_dict[method] = dX_dTheta

        mpi_print(f'Computing errors with {numiter} iterations ...', comm=comm)
        t0 = time.perf_counter()
        error_dict, _, _, _ = compute_dX_error(S_non_perturb, dX_dTheta_dict, delta, numiter=numiter, verbosity=1)
        mpi_print(f'...done in {time.perf_counter()-t0:.3f} s', comm=comm)

        for method in solver_dict:
            error_alpha_dict[method]['mean'].append(error_dict[method].mean())
            error_alpha_dict[method]['iterations'].append(error_dict[method])

    return error_alpha_dict


def coord_diff(min_image_func, X1, X2):
    """Compute the difference between two sets of coordinates.
    Apply the minimum image convention and remove the mean.

    Parameters
    ----------

    min_image_func : function
        Function to compute the minimum image.

    X1 : numpy array
        First set of coordinates.

    X2 : numpy array
        Second set of coordinates.

    Returns
    -------

    dX : numpy array
        Difference between the two sets of coordinates.
    """

    dX = min_image_func(X1 - X2 - (X1 - X2).mean(0))
    dX -= dX.mean(0)

    #dX = min_image_func(X1 - X2)

    return dX


def loss_function(min_image_func, X1, X2, **kwargs):
    """Loss function for the minimization algorithm."""

    return 0.5 * (coord_diff(min_image_func, X1, X2)**2).sum()


def minimize_loss(  sim,
                    X_target,
                    comm=None,
                    step=0.01,
                    adaptive_step=True,
                    error_tol=1e-6,
                    maxiter=100,
                    der_method='inverse',
                    der_ftol=1e-8,
                    der_alpha=0.5,
                    der_adaptive_alpha=True,
                    der_maxiter=500,
                    der_atol=1e-5,
                    verbosity=2,
                    pickle_name=None,
                    minimize_at_iters=True,
                    apply_hard_constraints=False,
                    binary=False,
                    output_folder='minim_output',
                    ):
    """
    Optimize the potential parameters to get the X_target
    as stationary point of the potential.

    This is done using the gradient descent method of
    minimizing the loss function:

    .. math::
        L(\Theta) = 1/2 (X(\Theta) - X_{target})^2

    Parameters
    ----------

    sim : simulation object
        Instance of the ImplicitDerivative class (child classes: BccVacancy, Dislo, ...)

    X_target : numpy array
        Target positions.

    step : float, optional
        Gradient descent step size, by default 0.01.
        If adaptive_step is True, step is ignored.

    adaptive_step : bool, optional
        Use adaptive step size, by default True.

    error_tol : float, optional
        Error tolerance for the minimization algorithm, by default 1e-6.

    maxiter : int, optional
        Maximum number of iterations, by default 100.

    der_ftol : float, optional
        Force tolerance for the implicit derivative, by default 1e-8.

    der_alpha : float, optional
        Finite difference parameter for the implicit derivative, by default 0.001.

    der_maxiter : int, optional
        Maximum number of iterations for the implicit derivative, by default 500.

    verbosity : int, optional
        Verbosity level, by default 2.

    Returns
    -------

    X_final : numpy array
        Final positions.

    Theta_final : numpy array
        Final potential parameters.
    """

    os.makedirs(output_folder, exist_ok=True)

    trun = TimingGroup("minimize_loss")
    trun.add('total', level=2).start()

    rank = comm.Get_rank() if comm is not None else 0

    if verbosity < 2:
        sim.verbose = False

    mpi_print('\n'+'='*80, comm=comm)
    mpi_print('='*23 + 'Running the parameter optimization' + '='*23, comm=comm)
    mpi_print('='*80 + '\n', comm=comm)

    if apply_hard_constraints:

        if not hasattr(sim, 'A_hard'):
            raise ValueError('The system does not have hard constraints')

        P_matrix = get_projection(sim.A_hard, sim.Ndesc)

    minim_dict = {}

    minim_dict['converged'] = False

    minim_dict['sim_init'] = copy.deepcopy(sim.to_dict())
    minim_dict['X_target'] = X_target
    minim_dict['cell'] = sim.cell
    minim_dict['params'] = {
        'step': step,
        'adaptive_step': adaptive_step,
        'error_tol': error_tol,
        'maxiter': maxiter,
        'der_method': der_method,
        'der_ftol': der_ftol,
        'der_alpha': der_alpha,
        'der_maxiter': der_maxiter,
        'der_atol': der_atol,
        'der_adaptive_alpha': der_adaptive_alpha,
        'minimize_at_iters': minimize_at_iters,
        'apply_hard_constraints': apply_hard_constraints,
    }

    error_array = np.zeros(maxiter + 1)
    step_array = np.zeros(maxiter)

    # Compute the initial error
    error_array[0] = loss_function(sim.minimum_image, sim.X_coord, X_target)
    mpi_print(f'{"Initial error":>30}: {error_array[0]:.3e}\n', comm=comm)

    minim_dict['iter'] = {}

    min_error = error_array[0]
    min_X = sim.X_coord.copy()
    min_Theta = sim.Theta.copy()

    for i in range(maxiter):

        if verbosity > 0:
            mpi_print('Iteration', i+1, '/', maxiter, comm=comm)

        minim_dict['iter'][i] = {}

        # Compute the implicit derivative
        if verbosity > 1:
            mpi_print(f'Computing dX/dTheta using {der_method} method...', comm=comm)
        with trun.add('dX_dTheta') as t:

            try:
                dX_dTheta = sim.implicit_derivative(
                                            method=der_method,
                                            alpha=der_alpha,
                                            adaptive_alpha=der_adaptive_alpha,
                                            maxiter=der_maxiter,
                                            atol=der_atol,
                                            ftol=der_ftol)
            except Exception as e:
                mpi_print(f'Iteration {i+1}, LAMMPS error at dX_dTheta: {e}', comm=comm)
                minim_dict['LAMMPS error'] = e
                break

        # minim_dict['iter'][i]['dX_dTheta'] = dX_dTheta
        if hasattr(sim, 'alpha_array'):
            minim_dict['iter'][i]['alpha_array'] = sim.alpha_array

        # Compute the change in parameters and positions
        """
        L(T) = 1/2 * (X(T) - X_target)^2
        dL/dT = (X(T) - X_target) * dX/dT
        """

        # Potential parameters change given by the loss function gradient
        dTheta = - dX_dTheta @ coord_diff(sim.minimum_image, sim.X_coord, X_target)
        if apply_hard_constraints:
            dTheta = P_matrix @ dTheta

        # Displacements change from the parameters change and implicit derivative
        dX = dTheta @ dX_dTheta
        dX -= dX.mean(0)

        # Step size
        if adaptive_step:
            """
            dL(step) = step^2 * dX^2 + 2*step dX.dX_2
            dL/dstep = 2dX^2 + 2*dX.dX_2
            """
            dX_2 = coord_diff(sim.minimum_image, sim.X_coord, X_target)
            dX_2 -= dX_2.mean(0)
            dot_prod = dX @ dX_2
            step = - dot_prod / (dX**2).sum()

        dTheta *= step
        dX *= step

        step_array[i] = step

        if verbosity > 2:
            mpi_print(f'\n{"Step size":>30}: {step:.3e}', comm=comm)
            mpi_print('\n'+' '*11+'-'*13+'Params'+'-'*13, comm=comm)
            mpi_print(f'{"Largest dX/dTheta":>30}: {np.max(np.abs(dX_dTheta)):.3e}', comm=comm)
            mpi_print(f'{"Largest dTheta":>30}: {np.max(np.abs(dTheta)):.3e}', comm=comm)

            mpi_print('\n'+' '*11+'-'*11+'Positions'+'-'*12, comm=comm)
            mpi_print(f'{"Largest dX":>30}: {np.max(np.abs(dX)):.3e}', comm=comm)
            mpi_print(f'{"Std Dev of dX":>30}: {np.std(dX):.3e}', comm=comm)


        # Update the LAMMPS system
        try:
            sim.X_coord += dX
            sim.lmp.scatter("x", 1, 3, np.ctypeslib.as_ctypes(sim.X_coord))
            sim.lmp.command("run 0")

            # Update the potential
            sim.Theta += dTheta
            sim.gather_D_dD()

            # Update the parameters in the pot object
            # !!! HARDCODED: update the Mo parameters if binary, W otherwise
            elem = 'Mo' if binary else 'W'
            sim.pot.Theta_dict[elem]['Theta'] = sim.Theta

            # Update the potential parameters
            if rank == 0:
                sim.pot.to_files(path='.',
                                 snapcoeff_filename='tmp.snapcoeff',
                                 snapparam_filename='tmp.snapparam',
                                 overwrite=True, verbose=False)

                # save parameters to output_folder
                sim.pot.to_files(path=output_folder,
                                 snapcoeff_filename=f'{sim.pot.elmnts}_step_{i:04d}.snapcoeff',
                                 snapparam_filename=f'{sim.pot.elmnts}_step_{i:04d}.snapparam',
                                 overwrite=True, verbose=False)

                # save coordinates to output_folder
                sim.write_xyz_file(filename=os.path.join(output_folder, f'coords_step_{i:04d}.xyz'))

            #sim.setup_snap_potential()
            sim.lmp.commands_string(f"""
            pair_coeff * * tmp.snapcoeff tmp.snapparam {sim.pot.elements}
            run 0
            """)

            # Compute the force
            force = np.ctypeslib.as_array(sim.lmp.gather("f", 1, 3)).flatten()
            force_max_pre_min = np.abs(force).max()
            force_norm_pre_min = np.linalg.norm(force)
            minim_dict['iter'][i]['force_max_pre_min'] = force_max_pre_min
            minim_dict['iter'][i]['force_norm_pre_min'] = force_norm_pre_min

            if verbosity > 2:
                mpi_print('\n'+' '*11+'-'*13+'Forces'+'-'*13, comm=comm)
                mpi_print(f'{"Largest force value":>30}: {force_max_pre_min:.3e}; '
                      f'Norm: {force_norm_pre_min:.3e}', comm=comm)
                mpi_print(f'{"Energy":>30}: {sim.energy:.10e}', comm=comm)

            if minimize_at_iters:

                with trun.add('minimize at iters') as t:
                    if verbosity > 2:
                        mpi_print('\n'+' '*11+'-'*4+'Forces after minimization'+'-'*3, comm=comm)

                    sim.lmp.command(f"minimize 0 {sim.minimize_ftol} 30 30")
                    sim.lmp.command("run 0")

                    # Compute the force
                    force = np.ctypeslib.as_array(sim.lmp.gather("f", 1, 3)).flatten()
                    force_max_post_min = np.abs(force).max()
                    force_norm_post_min = np.linalg.norm(force)
                    minim_dict['iter'][i]['force_max_post_min'] = force_max_post_min
                    minim_dict['iter'][i]['force_norm_post_min'] = force_norm_post_min

                    # Update the coordinates
                    sim.X_coord = np.ctypeslib.as_array(sim.lmp.gather("x", 1, 3)).flatten()
                    if verbosity > 2:
                        mpi_print(f'{"Largest force value":>30}: {force_max_post_min:.3e}; '
                                f'Norm: {force_norm_post_min:.3e}', comm=comm)
                        mpi_print(f'{"Energy":>30}: {sim.energy:.10e}', comm=comm)

        except Exception as e:
            mpi_print(f'Iteration {i+1}, LAMMPS error at update: {e}', comm=comm)
            minim_dict['LAMMPS error'] = e
            break

        # Evaluate the error
        error_array[i+1] = loss_function(sim.minimum_image, sim.X_coord, X_target)

        if verbosity > 0:
            # Predicted change in the loss function
            # Without minimization at iters, pred_change and real_change are the same
            dot_prod = dX @ coord_diff(sim.minimum_image, sim.X_coord, X_target)
            pred_change = dot_prod + (dX**2).sum() / 2.0

            # ||dX+X-Y||-||X-Y|| = ||dX|| + 2*dot_prod + ||X-Y|| - ||X-Y|| = ||dX|| + 2*dot_prod

            # Actual change in the loss function
            real_change = error_array[i] - error_array[i+1]

            mpi_print('', comm=comm)
            mpi_print('\n'+' '*11+'-'*13+'Errors'+'-'*13, comm=comm)
            mpi_print(f'{"Current error":>30}: {error_array[i+1]:.3e}', comm=comm)
            #mpi_print(f'{"Predicted change":>30}: {pred_change:.3e}', comm=comm)
            mpi_print(f'{"Actual change":>30}: {real_change:.3e}', comm=comm)
            #mpi_print('', comm=comm)

        if error_array[i+1] < min_error:
            min_error = error_array[i+1]
            min_X = sim.X_coord.copy()
            min_Theta = sim.Theta.copy()

        if error_array[i+1] < error_tol:
            mpi_print('Convergence reached!', comm=comm)
            minim_dict['converged'] = True
            break

        # Sync the MPI processes once per iteration
        if comm is not None:
            comm.Barrier()

    mpi_print('='*80+'\n', comm=comm)

    minim_dict['numiter'] = i+1
    minim_dict['error_array'] = error_array[:i+1]
    minim_dict['sim_final'] = sim.to_dict()

    # delete the tmp files
    if rank == 0:
        if os.path.exists('tmp.snapcoeff'):
            os.remove('tmp.snapcoeff')
        if os.path.exists('tmp.snapparam'):
            os.remove('tmp.snapparam')

    if rank == 0:
        # Save the final potential parameters
        if rank == 0:
            sim.pot.to_files(path='.',
                             snapcoeff_filename=f'minimized_{sim.pot.elmnts}.snapcoeff',
                             snapparam_filename=f'minimized_{sim.pot.elmnts}.snapparam',
                             overwrite=True, verbose=False)

        # Save into a pickle file
        if pickle_name is None:
            pickle_name = f'minim_dict_{der_method}.pkl'
        with open(pickle_name, 'wb') as f:
            pickle.dump(minim_dict, f)

    trun.timings['total'].stop()

    if verbosity > 0:
        # Align the lines for printing
        mpi_print('Number of iterations:', i+1, comm=comm)
        mpi_print('Converged:', minim_dict['converged'], comm=comm)
        mpi_print('Final error:', error_array[i+1], comm=comm)
        mpi_print('\n', trun, comm=comm)

    return sim, error_array, min_X, min_Theta
