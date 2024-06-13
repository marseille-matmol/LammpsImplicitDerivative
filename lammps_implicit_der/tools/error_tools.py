#!/usr/bin/env python3
# coding: utf-8

import os
import time
from tqdm import tqdm
import copy
import numpy as np
import pickle

# local imports
from .utils import mpi_print, get_projection
from .timing import TimingGroup


def coord_error(X1, X2, remove_mean=True):
    """
    Error = ||X1 - X2|| / ||X1|| / N
    """
    #return np.linalg.norm(X1 - X2) / np.linalg.norm(X1) / X1.shape[0]
    X1_tmp = X1.copy()
    X2_tmp = X2.copy()

    if remove_mean:
        X1_tmp -= np.mean(X1, axis=0)
        X2_tmp -= np.mean(X2, axis=0)

    return np.linalg.norm(X1_tmp - X2_tmp)


def coord_error_old(X1, X2):
    """
    Error = ||X1 - X2|| / ||X1|| / N
    """
    return np.linalg.norm(X1 - X2) / np.linalg.norm(X1) / X1.shape[0]


def compute_histogram(data, bins, density=False):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return hist, bin_centers


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
    """
    Loss function for the minimization algorithm.

    Possible change: add the Theta distance as a penalty term.
    + 0.5 * lambda_param * ((Theta - Theta0)**2).sum()

    Then, the gradient of the loss function is:
    dL/dTheta = (X - X_target) * dX/dTheta + lambda_param * (Theta - Theta0)
    """

    return 0.5 * (coord_diff(min_image_func, X1, X2)**2).sum()


def minimize_loss(sim,
                  X_target,
                  sub_element,
                  # Implicit derivative parameters
                  der_method='inverse',
                  der_min_style='cg',
                  der_adaptive_alpha=True,
                  der_alpha=1e-4,
                  der_ftol=1e-8,
                  der_atol=1e-5,
                  der_maxiter=500,
                  der_hess_mask=None,
                  # Minimization parameters
                  maxiter=100,
                  step=0.01,
                  adaptive_step=True,
                  error_tol=1e-6,
                  minimize_at_iters=True,
                  apply_hard_constraints=False,
                  # io parameters
                  verbosity=2,
                  pickle_name=None,
                  binary=False,
                  output_folder='minim_output',
                  comm=None,
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

    sub_element : str
        Element name for which the potential parameters will be optimized.

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

    minim_dict = {
        'converged': False,
        'sim_init': copy.deepcopy(sim.to_dict()),
        'X_target': X_target,
        'cell': sim.cell
        }

    minim_dict['params'] = {
        # Implicit derivative parameters
        'der_method': der_method,
        'der_min_style': der_min_style,
        'der_adaptive_alpha': der_adaptive_alpha,
        'der_alpha': der_alpha,
        'der_ftol': der_ftol,
        'der_atol': der_atol,
        'der_maxiter': der_maxiter,
        # Minimization parameters
        'maxiter': maxiter,
        'step': step,
        'adaptive_step': adaptive_step,
        'error_tol': error_tol,
        'minimize_at_iters': minimize_at_iters,
        'apply_hard_constraints': apply_hard_constraints,
    }

    error_array = np.zeros(maxiter + 1)
    step_array = np.zeros(maxiter + 1)

    # Compute the initial error
    error_array[0] = loss_function(sim.minimum_image, sim.X_coord, X_target)
    mpi_print(f'{"Initial error":>30}: {error_array[0]:.3e}\n', comm=comm)

    minim_dict['iter'] = {}

    min_error = error_array[0]
    min_X = sim.X_coord.copy()
    min_Theta = sim.Theta.copy()

    # Write the initial lammps data and potential files
    sim.write_data(filename=os.path.join(output_folder, 'data_step_0000.lammps-data'))
    if rank == 0:
        sim.pot.to_files(path=output_folder,
                         snapcoeff_filename=f'{sim.pot.elmnts}_step_0000.snapcoeff',
                         snapparam_filename=f'{sim.pot.elmnts}_step_0000.snapparam',
                         overwrite=True, verbose=False)

    for i in range(1, maxiter+1):

        if verbosity > 0:
            mpi_print('Iteration', i, '/', maxiter, comm=comm)

        minim_dict['iter'][i] = {}

        # Compute the implicit derivative
        if verbosity > 1:
            mpi_print(f'Computing dX/dTheta using {der_method} method.', comm=comm)
            if der_method == 'energy':
                mpi_print(f'{der_min_style=}, {der_adaptive_alpha=}, {der_alpha=:.3e}, {der_ftol=:.3e}', comm=comm)
        with trun.add('dX_dTheta') as t:

            try:
                dX_dTheta = sim.implicit_derivative(
                                            method=der_method,
                                            min_style=der_min_style,
                                            alpha=der_alpha,
                                            adaptive_alpha=der_adaptive_alpha,
                                            maxiter=der_maxiter,
                                            atol=der_atol,
                                            ftol=der_ftol,
                                            hess_mask=der_hess_mask)
            except Exception as e:
                mpi_print(f'Iteration {i}, error at dX_dTheta: {e}', comm=comm)
                minim_dict['iter'][i]['error'] = e
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

        minim_dict['iter'][i]['dTheta'] = dTheta
        minim_dict['iter'][i]['X_coord'] = dX

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

            dX_target = coord_diff(sim.minimum_image, sim.X_coord + dX, X_target)
            mpi_print(f'{"Largest dX wrt target":>30}: {np.max(np.abs(dX_target)):.3e}', comm=comm)

        # Update the LAMMPS system
        try:
            sim.X_coord += dX
            minim_dict['iter'][i]['X_coord'] = sim.X_coord

            sim.scatter_coord()

            # Update the potential
            sim.Theta += dTheta
            sim.gather_D_dD()

            # Update the parameters in the pot object
            mpi_print(f'\n  >>Updating the potential parameters for {sub_element}', comm=comm)
            sim.pot.Theta_dict[sub_element]['Theta'] = sim.Theta

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
                # sim.write_xyz_file(filename=os.path.join(output_folder, f'coords_step_{i:04d}.xyz'))

            sim.write_data(filename=os.path.join(output_folder, f'data_step_{i:04d}.lammps-data'))

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
            mpi_print(f'Iteration {i}, error at update: {e}', comm=comm)
            minim_dict['iter'][i]['error'] = e
            break

        # Evaluate the error
        error_array[i] = loss_function(sim.minimum_image, sim.X_coord, X_target)

        if verbosity > 0:
            # Predicted change in the loss function
            # Without minimization at iters, pred_change and real_change are the same
            dot_prod = dX @ coord_diff(sim.minimum_image, sim.X_coord, X_target)
            pred_change = dot_prod + (dX**2).sum() / 2.0

            # ||dX+X-Y||-||X-Y|| = ||dX|| + 2*dot_prod + ||X-Y|| - ||X-Y|| = ||dX|| + 2*dot_prod

            # Actual change in the loss function
            real_change = error_array[i-1] - error_array[i]

            mpi_print('', comm=comm)
            mpi_print('\n'+' '*11+'-'*13+'Errors'+'-'*13, comm=comm)
            mpi_print(f'{"Current error":>30}: {error_array[i]:.3e}', comm=comm)
            #mpi_print(f'{"Predicted change":>30}: {pred_change:.3e}', comm=comm)
            mpi_print(f'{"Error change":>30}: {real_change:.3e}', comm=comm)
            #mpi_print('', comm=comm)

        if error_array[i] < min_error:
            min_error = error_array[i]
            min_X = sim.X_coord.copy()
            min_Theta = sim.Theta.copy()

        if error_array[i] < error_tol:
            mpi_print('Convergence reached!', comm=comm)
            minim_dict['converged'] = True
            break

        # Sync the MPI processes once per iteration
        if comm is not None:
            comm.Barrier()

    mpi_print('='*80+'\n', comm=comm)

    minim_dict['numiter'] = i
    minim_dict['error_array'] = error_array[:i]
    minim_dict['sim_final'] = sim.to_dict()
    minim_dict['X_final'] = min_X
    minim_dict['Theta_final'] = min_Theta
    trun.timings['total'].stop()

    minim_dict['sim_timings'] = sim.timings

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

    if verbosity > 0:
        mpi_print('Number of iterations:', i, comm=comm)
        mpi_print('Converged:', minim_dict['converged'], comm=comm)
        mpi_print(f'Final error: {error_array[i]:.3e}', comm=comm)
        mpi_print('\n', trun, comm=comm)
        mpi_print(sim.timings, comm=comm)

    return sim, minim_dict
