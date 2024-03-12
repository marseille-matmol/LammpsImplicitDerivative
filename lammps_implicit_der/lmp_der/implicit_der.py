#!/usr/bin/env python3
# coding: utf-8

import os
from tqdm import tqdm

import copy

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lgmres
import numpy as np

from lammps import lammps
from lammps import LMP_TYPE_SCALAR, LMP_STYLE_GLOBAL

# local imports
from ..tools.utils import mpi_print, get_default_data_path
from ..tools.timing import TimingGroup, measure_runtime_and_calls


class LammpsImplicitDer:
    @measure_runtime_and_calls
    def __init__(self,
                 snapcoeff_filename=None,
                 snapparam_filename=None,
                 datafile=None,
                 data_path=None,
                 minimize=True,
                 minimize_algo='cg',
                 minimize_ftol=1e-8,
                 minimize_maxiter=1000,
                 minimize_maxeval=1000,
                 comm=None,
                 logname='none',
                 fix_sel='all',
                 verbose=True):
        """Set of methods for implicit derivative. Parent class.

        Attributes
        ----------

        _X_coord : numpy array
            atomic coordinates with pbc applied

        """

        self.comm = comm
        self.rank = 0 if comm is None else comm.Get_rank()

        # Minimization parameters
        self.minimize = minimize
        self.minimize_algo = minimize_algo
        self.minimize_ftol = minimize_ftol
        self.minimize_maxiter = minimize_maxiter
        self.minimize_maxeval = minimize_maxeval

        self.snapcoeff_filename = snapcoeff_filename
        self.snapparam_filename = snapparam_filename
        self.datafile = datafile

        self.logname = logname

        self.verbose = verbose

        if hasattr(self, 'timings'):
            self.timings.name = 'ImplicitDer-LAMMPS'
            self.timings.sort = True
        else:
            self.timings = TimingGroup('ImplicitDer-LAMMPS')
        #self.timings.levelup_that_t('__init__')

        self.step = 0
        self.dX_dTheta = None

        self.lmp = None
        self.pot = None
        self.cell = None
        self.Natom = None

        # Assume a unary system by default, change in the child class if needed
        self.binary = False

        # Positions with pbc applied
        self._X_coord = None

        # Energy of the system
        self._energy = None

        if data_path is None:
            self.data_path = get_default_data_path()
        else:
            self.data_path = data_path

        # For energy method, selection for addforce command
        self.fix_sel = fix_sel

        # Command line arguments for LAMMPS
        self.cmdargs = ['-screen', 'none', '-log', self.logname]

        self.print_run_info()
        # Initialize LAMMPS simulation
        self.lmp = lammps(cmdargs=self.cmdargs, comm=self.comm)

    def __del__(self):
        """Destructor"""
        pass
        #self.lmp.close()

    def copy(self):
        """Return a copy of the object.
        Does not work because of the LAMMPS object.
        """
        return copy.deepcopy(self)

    def print_run_info(self):
        mpi_print('\n'+'-'*80, comm=self.comm)
        mpi_print('Running LAMMPS with the following arguments:', verbose=self.verbose,
                  comm=self.comm)
        mpi_print(' '.join(self.cmdargs)+'\n', verbose=self.verbose, comm=self.comm)

    def print_system_info(self):
        mpi_print(f'Number of atoms: {self.Natom}, largest force value: {np.abs(self.f0).max():.3e}, '
                  f'force norm: {np.linalg.norm(self.f0):.3e}', verbose=self.verbose, comm=self.comm)

    def to_dict(self):

        out_dict = {
            'X_coord': self._X_coord,
            'dX_dTheta': self.dX_dTheta,
            'dU_dTheta': self.dU_dTheta,
            'mixed_hessian': self.mixed_hessian,
            'Natoms': self.Natom,
            'Theta': self.Theta,
        }

        if 'alpha_array' in self.__dict__:
            out_dict['alpha_array'] = self.alpha_array

        return out_dict

    @property
    def X_coord(self):
        """Coordinates getter"""
        return self._X_coord

    @X_coord.setter
    def X_coord(self, input_coord):
        """Coordinates setter.
        Apply pbcs to the coordinates and store them internally."""

        if self.cell is None:
            self.get_cell()

        self._X_coord = self.minimum_image(input_coord)

    @property
    def energy(self):
        """Energy getter.
        Every time energy is requested, it is computed in LAMMPS and stored internally.
        """
        self._energy = self.lmp.numpy.extract_compute("thermo_pe",
                                                      LMP_STYLE_GLOBAL,
                                                      LMP_TYPE_SCALAR)
        return self._energy

    @measure_runtime_and_calls
    def minimize_energy(self, ftol=None, maxiter=None, maxeval=None, algo=None, verbose=True):
        """Minimize the energy of the system"""

        # Set the minimization parameters
        ftol = self.minimize_ftol if ftol is None else ftol
        maxiter = self.minimize_maxiter if maxiter is None else maxiter
        maxeval = self.minimize_maxeval if maxeval is None else maxeval
        algo = self.minimize_algo if algo is None else algo

        if self.lmp is None:
            raise RuntimeError('LAMMPS object lmp must be defined for minimization')

        if verbose:
            mpi_print(f'Minimizing energy with the following parameters:', comm=self.comm)
            mpi_print(f'ftol: {ftol}, maxiter: {maxiter}, maxeval: {maxeval}, algo: {algo} \n', comm=self.comm)

        self.lmp.commands_string(f"""
        min_style {algo}
        minimize 0 {ftol} {maxiter} {maxeval}
        """)

        self.X_coord = np.ctypeslib.as_array(self.lmp.gather("x", 1, 3)).flatten()

    def write_data(self, filename):
        """Write the current configuration to a data file"""
        self.lmp.command(f'write_data {filename}')

    def setup_snap_potential(self):
        """Set up the potential in LAMMPS
        Currently implemented only for SNAP"""

        # Check that pot must be defined
        if self.pot is None:
            raise RuntimeError('Potential must be defined')

        mpi_print(f'Setting SNAP potential', comm=self.comm)

        self.lmp.commands_string(f"""
        pair_style snap

        # Take coefficients from files W.snapcoeff and W.snapparam
        pair_coeff * * {self.pot.snapcoeff_path} {self.pot.snapparam_path} {self.pot.elements}
        run 0
        """)

        self.Ndesc = self.pot.num_param

        mpi_print(self.pot, verbose=self.verbose, comm=self.comm)

    def scatter_coord(self):
        """Send the coordinates to LAMMPS"""
        self.lmp.scatter("x", 1, 3, np.ctypeslib.as_ctypes(self._X_coord))
        self.lmp.command("run 0")

    @measure_runtime_and_calls
    def compute_D_dD(self):
        """Set up the potential in LAMMPS
        Currently implemented only for SNAP"""

        # Check that pot must be defined
        if self.pot is None:
            raise RuntimeError('Potential must be defined')

        # Read the potential parameters from the potential object
        rcutfac, twojmax, rfac0 = \
            self.pot.snapparam_dict['rcutfac'], \
            self.pot.snapparam_dict['twojmax'], \
            self.pot.snapparam_dict['rfac0']

        radii, weights = \
            self.pot.Theta_dict['radii'], \
            self.pot.Theta_dict['weights']

        self.lmp.commands_string(f"""
        # descriptors
        compute D all sna/atom {rcutfac} {rfac0} {twojmax} {radii} {weights}

        # derivatives of descriptors
        compute dD all snad/atom {rcutfac} {rfac0} {twojmax} {radii} {weights}

        # potential energy per atom
        compute E all pe/atom

        run 0
        """)

    @measure_runtime_and_calls
    def gather_D_dD(self):
        """Compute descriptors and their derivatives in LAMMPS and store them internally.
        Wrapper for unary or binary systems.
        TODO: general implementation for multi-component systems
        """

        if self.binary:
            self.gather_D_dD_binary()

        else:
            self.gather_D_dD_unary()

    def gather_D_dD_unary(self):
        """Compute descriptors and their derivatives in LAMMPS and store them internally"""

        dU_dTheta = self.lmp.gather("c_D", 1, self.Ndesc)
        mixed_hessian = self.lmp.gather("c_dD", 1, 3*self.Ndesc)
        self.dU_dTheta = np.ctypeslib.as_array(dU_dTheta).reshape((-1, self.Ndesc)).sum(axis=0)
        self.mixed_hessian = np.ctypeslib.as_array(mixed_hessian).reshape((-1, self.Ndesc)).T

    def gather_D_dD_binary(self):
        """Compute descriptors and their derivatives in LAMMPS and store them internally, only for specie B
        """

        dU_dTheta = np.ctypeslib.as_array(self.lmp.gather("c_D", 1, self.Ndesc)).reshape((-1, self.Ndesc))

        self.dU_dTheta = dU_dTheta[self.species == 2].sum(0)

        dD = np.ctypeslib.as_array(
                self.lmp.gather("c_dD", 1, 3*2*self.Ndesc)
            ).reshape((-1, 2, 3, self.Ndesc))

        self.mixed_hessian = dD[:, 1, :, :].reshape((-1, self.Ndesc)).T

    @measure_runtime_and_calls
    def run_init(self, setup_snap=True):
        """
        Initial LAMMPS run and initialization of basic properties
        """

        self.lmp.command("run 0")

        if setup_snap:
            self.setup_snap_potential()

        if self.minimize:
            self.minimize_energy()

        self.compute_D_dD()

        self.f0 = np.ctypeslib.as_array(self.lmp.gather("f", 1, 3)).flatten()

        # Gather coordinates and apply minimum image
        self.X_coord = np.ctypeslib.as_array(self.lmp.gather("x", 1, 3)).flatten()

        self.species = np.ctypeslib.as_array(self.lmp.gather("type", 0, 1)).flatten()

        self.atom_name_list = list(np.array(self.pot.elem_list)[self.species-1])

        # Send the coordinates back to LAMMPS
        self.lmp.scatter("x", 1, 3, np.ctypeslib.as_ctypes(self._X_coord))
        self.lmp.command("run 0")

        # Number of atoms x 3
        self.N = self._X_coord.size
        self.Natom = self.N//3

        # Initialize the simulation cell
        self.get_cell()

        self.gather_D_dD()

        # Setup hard constraints
        hard_constraints_path = os.path.join(self.data_path, f'{self.pot.elmnts}_constraints.txt')
        if os.path.exists(hard_constraints_path):
            self.A_hard = np.loadtxt(hard_constraints_path)

        self.print_system_info()

    def get_cell(self):
        """Get the simulation cell properties from LAMMPS"""

        boxlo, boxhi, xy, yz, xz, periodicity, box_change = \
            self.lmp.extract_box()

        self.periodicity = periodicity  # or define  e.g np.ones(3,bool)

        self.cell = np.zeros((3,3))

        for cell_j in range(3):
            self.cell[cell_j][cell_j] = boxhi[cell_j]-boxlo[cell_j]

        self.cell[0][1] = xy
        self.cell[0][2] = xz
        self.cell[1][2] = yz
        self.inv_cell = np.linalg.inv(self.cell)

    def minimum_image(self, X_vector):
        """Compute the minimum image of a vector X_vector (applying pbc)"""

        # Convert to (Natom, 3) shape
        X_3D = X_vector.reshape((-1, 3))

        correction = np.floor(X_3D @ self.inv_cell + 0.5) @ self.cell

        return X_vector - (correction * self.periodicity).flatten()

    @measure_runtime_and_calls
    def forces(self, dx, alpha=0.05):
        """
            Evaluate forces for given position
            Uses [F(X+alpha * dX_dTheta)-F(X) ] /alpha -> H.dX_dTheta as alpha -> 0
        """
        # update positions
        x = self._X_coord + alpha * dx.flatten()

        # apply pbc
        x = self.minimum_image(x)

        # send new positions to LAMMPS
        self.lmp.scatter("x", 1, 3, np.ctypeslib.as_ctypes(x))
        self.lmp.command("run 0")

        self.step += 1

        # get the forces from LAMMPS: "f" - forces, 1 - type, LAMMPS_DOUBLE, 3 - values per atom
        f = np.ctypeslib.as_array(self.lmp.gather("f", 1, 3)).flatten()
        return f

    @measure_runtime_and_calls
    def hessian(self, dx=0.001, return_sparse=False, sparse_tol=1e-6):
        """Naive calculation of the hessian

        Use the finite difference:

        H_ij = -[F_i(X + dx_j) - F_i(X - dx_j)]/(2*dx_j)

        Parameters
        ----------
        dx : float, optional
            finite difference, by default 0.001
        """
        H = np.zeros((self.N,self.N))

        # We need to define a vector of displacements because the forces function
        # expects a vector of displacements
        dx_vector = np.zeros_like(self._X_coord.flatten())
        # iterate over 3N

        if self.verbose and self.rank == 0:
            iterator = tqdm(range(self.N), desc='Hessian')
        else:
            iterator = range(self.N)

        for i in iterator:

            # displace posotion i by dx
            dx_vector[i] = dx

            # compute forces: F(X + alpha * dX_dTheta)
            H[i, :] = -self.forces(dx_vector, alpha=1.0)

            # compute forces: F(X - alpha * dX_dTheta) and subtract
            dx_vector[i] = -dx
            H[i, :] -= -self.forces(dx_vector, alpha=1.0)

            dx_vector[i] = 0.0

        H /= 2.0*dx

        # Add a small identity matrix to the Hessian to avoid singularities
        epsilon = 1.0e-3
        Natom = self.N//3
        T = np.array([np.array([1,0,0]*Natom),np.array([0,1,0]*Natom),np.array([0,0,1]*Natom)])
        T = T.T@T * epsilon
        H = np.add(H,T)

        if return_sparse:
            H[np.abs(H)<sparse_tol] = 0.0
            H_sparse = sparse.csr_matrix(H)
            mpi_print(f"Total number of elements: {H.size}, "
                      f"number of non-zero elements: {H_sparse.nnz} ({H_sparse.nnz/H.size*100:.2f}%)",
                      comm=self.comm)
            return H_sparse
        else:
            return H

    @measure_runtime_and_calls
    def implicit_derivative(self,
                            method='energy',
                            #alpha=0.001,
                            alpha=0.5,
                            adaptive_alpha=True,
                            atol=1e-5,
                            ftol=1e-10,
                            maxiter=500):
        """A wrapper for implicit derivative calculation

        Returns
        -------

        dX_dTheta : numpy array
            implicit derivative
        """

        if method == 'energy':
            res_dict = self.implicit_derivative_energy(alpha=alpha,
                                                       adaptive_alpha=adaptive_alpha,
                                                       ftol=ftol,
                                                       maxiter=maxiter)
            return res_dict['dX_dTheta']

        elif method == 'sparse':
            res_dict = self.implicit_derivative_sparse(alpha=alpha,
                                                       adaptive_alpha=adaptive_alpha,
                                                       atol=atol,
                                                       maxiter=maxiter)
            return res_dict['dX_dTheta']

        elif method == 'inverse':
            return self.implicit_derivative_inverse()

        elif method == 'dense':
            return self.implicit_derivative_dense()

        else:
            raise ValueError(f'Unknown method for implicit derivative: {method}')

    def implicit_derivative_sparse(self,
                                   alpha=0.01,
                                   atol=1e-5,
                                   maxiter=100,
                                   adaptive_alpha=True):
        """
        Evaluation of implicit position derivative
        via sparse linear methods.

        Parameters
        ----------
        alpha : float, optional
            Numerical parameter in expansion [F(X+alpha*dX_dTheta)-F(X)]/alpha->H.dX_dTheta,
            exact as alpha->0, but too small causes finite difference errors.
            Default 0.05
        atol : float, optional
            tolerance for least squares minimization, by default 1e-5
        maxiter : int, optional
            maximum iteration for , by default 100
        return_values : bool, optional
            return value or only store internally, by default True

        Returns
        -------
        res_dict: dictionary
            `dX_dTheta' : numpy array (Ndesc,3N) implicit derivative
            `calls' : number of force calls during iteration
            `err' : residue from lstsq fit
        """
        self.lmp.scatter("x", 1, 3, np.ctypeslib.as_ctypes(self._X_coord))

        # Compute the force at the initial position,
        # Analytically, it must be zero, but for numerical reasons, it is small
        dX0 = np.zeros_like(self._X_coord)
        F0 = self.forces(dX0, alpha=0.0)

        # result holder
        res_dict = {'dX_dTheta': [], 'err': [], 'calls': []}

        if self.verbose and self.rank == 0:
            iterator = tqdm(self.mixed_hessian, desc='Impl. Der. Sparse')
        else:
            iterator = self.mixed_hessian

        # One linear solutions per parameter
        for iCl, Cl in enumerate(iterator):
            # initialize step counter
            self.step = 0

            # determine the alpha_factor
            if adaptive_alpha:
                alpha_factor = alpha / np.max(np.abs(Cl))
            else:
                alpha_factor = alpha

            # define linear operator with matrix-vector product matvec()
            matvec = lambda dx: (F0-self.forces(dx, alpha_factor)) / alpha_factor
            linop = LinearOperator((self.N, self.N), matvec=matvec, rmatvec=matvec)

            # perform iterative linear solution routine LGMRES: Ax = b, solve for x
            # linop - linear operator which can produce Ax
            dX_dTheta = lgmres(linop, Cl, x0=dX0, atol=atol, maxiter=maxiter)[0]

            # log results
            res_dict['dX_dTheta'] += [dX_dTheta]
            res_dict['err'] += [np.linalg.norm(matvec(dX_dTheta) - Cl)]
            res_dict['calls'] += [self.step]

        # convert to dictionary of numpy arrays
        res_dict = {k: np.array(res_dict[k]) for k in res_dict}

        # store internally
        self.dX_dTheta = res_dict['dX_dTheta'].copy()

        # return values
        return res_dict

    def implicit_derivative_energy(self,
                                   alpha=0.5,
                                   adaptive_alpha=True,
                                   ftol=1e-10,
                                   maxiter=200):
        """Evaluation of implicit position derivative
        via sparse linear methods.

        Parameters
        ----------
        alpha : float, optional
            Numerical parameter in expansion [F(X+alpha*dX_dTheta)-F(X)]/alpha->H.dX_dTheta,
            exact as alpha->0, but too small causes finite difference errors.
            Default 0.01 (0.5 for adaptive)
        ftol : float, optional
            force tolerance for  minimization, by default 1e-10
        maxiter : int, optional
            maximum iteration for , by default 200
        return_values : bool, optional
            return value or only store internally, by default True

        LAMMPS terminology:
        compute: run a simulation
        fix: do something to the system

        Returns
        -------
        res_dict: dictionary
            `dX_dTheta' : numpy array (Ndesc,3N) implicit derivative
            `calls' : number of force calls during iteration
            `err' : residue from lstsq fit
        """
        # declare an array of size N x 3 to be stored in the simulation
        # fix x all ... : add per atom property "x" to all atoms
        # "d_" means it is a float
        # alphaCl = alpha * d/dT_l (dU/dX), N x 3 vector
        # alphaClx = alpha * d/dT_l (dU/dx), N vector
        # alphaCly = alpha * d/dT_l (dU/dy)
        # The alphaClx, alphaCly, and alphaClz names are defined here
        self.lmp.commands_string("""
            fix mixedhessianrow all property/atom d_alphaClx d_alphaCly d_alphaClz

            #compute deltaX all displace/atom

            # FIRE minimization algorithm
            min_style fire
            #min_style hftn
            #min_style sd
            #min_style cg
        """)

        res_dict = {'dX_dTheta': [], 'err': [], 'calls': [], 'alpha_array': []}

        # iterate over columns of the mixed hessian obtained from LAMMPS SNAP potential
        # d/dT_i (dU/dX)

        # For debugging purposes
        # print_iCl = True
        print_iCl = False

        if print_iCl:
            iterator = self.mixed_hessian
        else:
            if self.verbose and self.rank == 0:
                iterator = tqdm(self.mixed_hessian, desc='Impl. Der. Energy')
            else:
                iterator = self.mixed_hessian

        for iCl, Cl in enumerate(iterator):

            if print_iCl:
                mpi_print(f'{"Parameter index iCl":>30}: {iCl:3d}', comm=self.comm)

            # reset the positions to the initial minimum configuration
            self.lmp.scatter("x", 1, 3, np.ctypeslib.as_ctypes(self._X_coord))

            # count the number of force calls as the number of time steps
            self.lmp.command("reset_timestep 0")

            # compute the 3N vector of displacements X - X0
            self.lmp.command("compute deltaX all displace/atom")

            if adaptive_alpha:
                alpha_factor = alpha / np.max(np.abs(Cl))
            else:
                alpha_factor = alpha

            res_dict['alpha_array'] += [alpha_factor]

            # set up the C vector:
            # fix x, y, or z
            # The names alphaClx must be the same as above in "fix"
            for i, LAMMPS_aCl_i in enumerate(["d_alphaClx", "d_alphaCly", "d_alphaClz"]):
                # i is x, y, or z
                # alpha C_l = alpha * d/dT_l (dU/dX)
                # Take only the i-th component of l-th vector
                alphaCl_i = Cl.reshape((-1, 3))[:, i].astype(np.float64)

                # Multiply by a small parameter alpha
                alphaCl_i *= alpha_factor

                # send the vector to LAMMPS
                self.lmp.scatter(LAMMPS_aCl_i, 1, 1, np.ctypeslib.as_ctypes(alphaCl_i))

            # Now, we have a column of mixed Hessian passed and positions passed to LAMMPS

            # build the energy and force function
            self.lmp.commands_string(f"""

                # Retrieve alpha * C_l
                compute alphaCl all property/atom d_alphaClx d_alphaCly d_alphaClz

                # Turn alpha * C_l into variables
                # c_ means compute

                # alpha * d/dT_l dU/dx
                variable aClx atom c_alphaCl[1]

                # alpha * d/dT_l dU/dy
                variable aCly atom c_alphaCl[2]

                # alpha * d/dT_l dU/dz
                variable aClz atom c_alphaCl[3]

                # additional term in energy:
                # dot procuct C_l * dX
                # v_ is variable
                # NO SPACES ALLOWED in the expression
                variable addE atom v_aClx*c_deltaX[1]+v_aCly*c_deltaX[2]+v_aClz*c_deltaX[3]

                # additional force to the system
                # fx = alpha * d/dT_l dU/dx, etc
                fix f {self.fix_sel} addforce v_aClx v_aCly v_aClz energy v_addE

                # blank iteration to make sure everything works
                run 0

                # minimize the energy
                minimize 0. {ftol} {maxiter} {maxiter}
            """)

            # extract the final positions
            X_new = np.ctypeslib.as_array(self.lmp.gather("x", 1, 3)).flatten()

            # apply pbc
            #dX_tmp = x_new - self._X_coord
            dX_tmp = self.minimum_image(X_new - self._X_coord)

            #mpi_print(iCl, Cl, np.max(np.abs(dX_tmp)), alpha_factor)

            res_dict['dX_dTheta'] += [dX_tmp / alpha_factor]

            res_dict['err'] += [ftol]

            # number of force calls
            res_dict['calls'] += [int(self.lmp.get_thermo("step"))]

            # disable commands
            self.lmp.commands_string("""
                uncompute alphaCl
                unfix f
                uncompute deltaX
            """)

        self.lmp.commands_string("""
            variable aClx delete
            variable aCly delete
            variable aClz delete
            unfix mixedhessianrow
        """)
        res_dict = {k: np.array(res_dict[k]) for k in res_dict}

        # store internally
        self.dX_dTheta = res_dict['dX_dTheta'].copy()

        self.alpha_array = res_dict['alpha_array'].copy()

        return res_dict

    def implicit_derivative_inverse(self):
        """Compute implicit derivative from Hessian inverse

        Returns
        -------
        dX_dTheta : numpy array
            implicit derivative
        """

        # Compute the Moore-Penrose inverse of the Hessian
        H = self.hessian()
        H += np.eye(H.shape[0]) * 0.01 * np.diag(H).min()
        H_inv = np.linalg.pinv(H)

        # Matrix multiplication to get dX_dTheta
        return (H_inv @ self.mixed_hessian.T).T

    def implicit_derivative_dense(self):
        """Compute implicit derivative from Hessian inverse

        Returns
        -------
        dX_dTheta : numpy array
            implicit derivative
        """

        # Use linalg.solve to find dX_dTheta in H.dX_dTheta = C
        dX_dTheta = np.linalg.solve(self.hessian(), self.mixed_hessian.T).T

        return dX_dTheta

    def implicit_derivative_direct_sparse(self):
        """Compute implicit derivative from Hessian inverse
        by solving dX_dT @ H = -C
        using sparse solver
        """

        dX_dTheta = sparse.linalg.spsolve(self.hessian(return_sparse=True),self.mixed_hessian.T).T

        return dX_dTheta

    @measure_runtime_and_calls
    def perturb(self, dTheta,return_X=True):
        """Evaluate energy and position perturbations

        Parameters
        ----------
        dTheta : vector of parameter perturbations
        return_X : bool, return position perturbations or not

        Returns dE and dX_dTheta
        """

        # Check if implicit derivative has been computed
        if self.dX_dTheta is None:
            mpi_print("Evaluating implicit derivative with default parameters", comm=self.comm)
            self.implicit_derivative_sparse(return_values=False)

        # check if dThta dimension is correct
        assert dTheta.size == self.Ndesc
        dX_dTheta = dTheta @ self.dX_dTheta

        # compute energy perturbation
        dE = self.dU_dTheta @ dTheta - 0.5 * np.dot(dTheta @ self.mixed_hessian, dX_dTheta)

        if return_X:
            return dE, dX_dTheta
        else:
            return dE

    @measure_runtime_and_calls
    def write_xyz_file(self, filename="coordinates.xyz", verbose=False):
        """Writes atomic coordinates to an .xyz file."""
        # Number of atoms
        Natoms = self._X_coord.size // 3

        with open(filename, 'w') as f:
            # First line: Number of atoms
            f.write(f"{Natoms}\n")

            # Second line: Comment or blank
            f.write("Atomic coordinates from LammpsImplicitDer\n")

            # Extract and write the atomic coordinates
            iatom = 0
            for i in range(0, self._X_coord.size, 3):
                f.write(f"{self.atom_name_list[iatom]} {self._X_coord[i]:.6f} {self._X_coord[i+1]:.6f} {self._X_coord[i+2]:.6f}\n")
                iatom += 1

        if verbose:
            mpi_print(f"Coordinates saved to {filename}", comm=self.comm)

