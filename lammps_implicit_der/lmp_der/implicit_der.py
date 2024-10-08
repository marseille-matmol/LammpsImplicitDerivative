#!/usr/bin/env python3
# coding: utf-8

import os
from tqdm import tqdm

import copy

from scipy.sparse.linalg import LinearOperator, lgmres, spsolve
import numpy as np

from lammps import lammps
from lammps import LMP_TYPE_SCALAR, LMP_STYLE_GLOBAL

# local imports
from ..tools.utils import mpi_print, get_default_data_path, get_matrix_basis, matrix_3x3_from_Voigt, \
                          matrices_3x3_from_Voigt
from ..tools.timing import TimingGroup, measure_runtime_and_calls


class LammpsImplicitDer:
    @measure_runtime_and_calls
    def __init__(self,
                 snapcoeff_filename=None,
                 snapparam_filename=None,
                 datafile=None,
                 input_script=None,
                 data_path=None,
                 minimize=True,
                 minimize_algo='cg',
                 minimize_ftol=1e-8,
                 minimize_maxiter=1000,
                 minimize_maxeval=1000,
                 fix_box_relax=False,
                 box_relax_vmax=0.001,
                 box_relax_iso=True,
                 zbl_dict=None,
                 logname='none',
                 fix_sel='all',
                 dump_lmp_cmd=True,
                 lmp_cmd_filename='commands.lammps',
                 comm=None,
                 verbose=True):
        """Set of methods for implicit derivative. Parent class.

        Parameters
        ----------

        snapcoeff_filename : str, optional
            SNAP coefficients filename, e.g. 'W.snapcoeff'.

        snapparam_filename : str, optional
            SNAP parameters filename, e.g. 'W.snapparam'.

        datafile : str, optional
            Data file for LAMMPS. Default None.

        input_script : str, optional
            Input script for LAMMPS. Default None.

        data_path : str, optional
            Path to the potential files.

        minimize : bool, optional
            Minimize the energy of the system with the fixed box.

        minimize_algo : str, optional
            Minimization algorithm.

        minimize_ftol : float, optional
            Minimization force tolerance.

        minimize_maxiter : int, optional
            Minimization maximum number of iterations.

        minimize_maxeval : int, optional
            Minimization maximum number of force evaluations.

        fix_box_relax : bool, optional
            Relax the box during minimization.

        box_relax_vmax : float, optional
            Maximum volume change during box relaxation (LAMMPS parameter).

        box_relax_iso : bool, optional
            Isotropic box relaxation.

        zbl_dict : dict, optional
            Paramaters for the ZBL potential. EXPERIMENTAL.

        logname : str, optional
            LAMMPS log file name.

        fix_sel : str, optional
            Selection for the addforce command. EXPERIMENTAL.

        dump_lmp_cmd : bool, optional
            Dump LAMMPS commands to a file.

        lmp_cmd_filename : str, optional
            Filename containing all the LAMMPS commands sent to LAMMPS for an object.

        comm : MPI communicator, optional
            mpi4py MPI communicator.
       """

        if logname is None:
            logname = 'none'

        self.comm = comm
        self.rank = 0 if comm is None else comm.Get_rank()

        # Minimization parameters
        self.minimize = minimize
        self.minimize_algo = minimize_algo
        self.minimize_ftol = minimize_ftol
        self.minimize_maxiter = minimize_maxiter
        self.minimize_maxeval = minimize_maxeval
        self.not_converged = False

        # Implicit derivative parameters
        self.dX_dTheta = None
        self.Theta = None
        self.virial = None
        self.mixed_hessian = None
        self.impl_der_stats = {}

        # Apply an external pressure tensor with the "fix box/relax" command
        self.fix_box_relax = fix_box_relax
        self.box_relax_vmax = box_relax_vmax
        self.box_relax_iso = box_relax_iso

        self.snapcoeff_filename = snapcoeff_filename
        self.snapparam_filename = snapparam_filename
        self.datafile = datafile
        self.input_script = input_script

        self.logname = logname

        self.verbose = verbose

        if hasattr(self, 'timings'):
            self.timings.name = 'ImplicitDer-LAMMPS'
            self.timings.sort = True
        else:
            self.timings = TimingGroup('ImplicitDer-LAMMPS')
        #self.timings.levelup_that_t('__init__')

        self.force_call_counter = 0
        self.lmp = None
        self.cell = None
        self.Natom = None

        self.dump_lmp_cmd = dump_lmp_cmd
        self.lmp_cmd_filename = lmp_cmd_filename

        # Potential attrs
        self.pot = None
        self.zbl_dict = zbl_dict

        # Assume a unary system by default, change in the child class if needed
        self.binary = False

        # Positions with pbc applied
        self._X_coord = None

        # Energy of the system
        self._energy = None

        # Volume
        self._volume = None

        # Pressure tensor and scalar
        self.pressure_tensor = None
        self.pressure = None

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
        self.check_lmp_cmd_file()
        self.lmp_commands_string("""
        # INITIALIZATION
        clear
        # Configure how atoms are mapped to store and access
        atom_modify map array sort 0 0.0
        # metal units: lenfth: Angstroms, energy: eV, time: ps, force: eV/Angstrom, temperture: K
        units metal
        """)

    def __del__(self):
        """Destructor"""
        self.lmp.close()

    def __str__(self):
        """Print out the object information."""

        out_str = 'LAMMPSImplicitDer object:\n'
        if self.Natom is not None:
            out_str += f'Number of atoms: {self.Natom}\n'
        if self.cell is not None:
            out_str += f'Cell:\n{self.cell}\n'
        if self.pot is not None:
            out_str += 'Potential:\n'
            out_str += f'{self.pot}\n'

        return out_str

    def lmp_commands_string(self, commands):
        """Send a string of LAMMPS commands to LAMMPS"""

        self.lmp.commands_string(commands)

        if self.dump_lmp_cmd and self.rank == 0:
            with open(self.lmp_cmd_filename, 'a') as f:
                # Split a multiline string into a list of strings
                for line in commands.splitlines():
                    f.write(line.strip()+'\n')

    def check_lmp_cmd_file(self):
        """Check if lmp_cmd_filename exists, rename the old one if necessary"""

        if self.dump_lmp_cmd and self.rank == 0:
            if os.path.exists(self.lmp_cmd_filename):
                if self.verbose:
                    mpi_print(f'WARNING: {self.lmp_cmd_filename} already exists, renaming it to {self.lmp_cmd_filename}.old', comm=self.comm)
                try:
                    os.rename(self.lmp_cmd_filename, f'{self.lmp_cmd_filename}.old')
                except FileNotFoundError as e:
                    mpi_print(f'Error in check_lmp_cmd_file: {e}', comm=self.comm)
                    mpi_print('This might happen when a system is initialized with an MPI communicator, but ran with mpirun.', comm=self.comm)

            # Touch lmp_cmd_filename
            with open(self.lmp_cmd_filename, 'w') as f:
                pass

    def print_run_info(self):
        mpi_print('\n'+'-'*80, comm=self.comm, verbose=self.verbose)
        mpi_print('Running LAMMPS with the following arguments:', verbose=self.verbose,
                  comm=self.comm)
        mpi_print(' '.join(self.cmdargs)+'\n', verbose=self.verbose, comm=self.comm)

    def print_system_info(self):
        mpi_print(f'Number of atoms: {self.Natom}, largest force value: {np.abs(self.force0).max():.3e}, '
                  f'force norm: {np.linalg.norm(self.force0):.3e}', verbose=self.verbose, comm=self.comm)

    def to_dict(self):
        """Return some of the object attributes as a dictionary. EXPERIMENTAL."""

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

        # Check if lmp is defined
        if self.lmp is None:
            raise RuntimeError('LAMMPS object lmp must be defined for energy calculation')

        self._energy = self.lmp.numpy.extract_compute("thermo_pe",
                                                      LMP_STYLE_GLOBAL,
                                                      LMP_TYPE_SCALAR)
        return self._energy

    @property
    def volume(self):
        """Volume getter.
        Every time volume is requested, it is computed from the self.cell.
        """

        self.get_cell()
        self._volume = np.linalg.det(self.cell)

        return self._volume

    @measure_runtime_and_calls
    def minimize_energy(self, ftol=None, maxiter=None, maxeval=None, algo=None, update_system=True, verbose=True):
        """Minimize the energy of the system"""

        if self.lmp is None:
            raise RuntimeError('LAMMPS object lmp must be defined for minimization')

        # Set the minimization parameters
        ftol = self.minimize_ftol if ftol is None else ftol
        maxiter = self.minimize_maxiter if maxiter is None else maxiter
        maxeval = self.minimize_maxeval if maxeval is None else maxeval
        algo = self.minimize_algo if algo is None else algo

        mpi_print(f'Minimizing energy with the following parameters:', verbose=self.verbose, comm=self.comm)
        mpi_print(f'ftol: {ftol}, maxiter: {maxiter}, maxeval: {maxeval}, algo: {algo}, fix_box_relax: {self.fix_box_relax} \n',
                  verbose=self.verbose, comm=self.comm)

        self.lmp_commands_string("reset_timestep 0")

        #e0 = self.lmp.get_thermo("pe")
        f0_max, f0_norm = self.lmp.get_thermo("fmax"), self.lmp.get_thermo("fnorm")

        # Fix box relax if specified
        # {'fix 1 all box/relax aniso 0.0 dilate partial' if False else ''}
        if self.fix_box_relax:
            if self.box_relax_iso:
                self.lmp_commands_string(f"""
                fix boxrelax all box/relax iso 0.0 vmax {self.box_relax_vmax}
                """)
            else:
                self.lmp_commands_string(f"""
                fix boxrelax all box/relax aniso 0.0 dilate partial
                """)

        # Minimization
        self.lmp_commands_string(f"""
        min_style {algo}
        minimize 0 {ftol} {maxiter} {maxeval}
        """)

        #e1 = self.lmp.get_thermo("pe")
        f1_max, f1_norm = self.lmp.get_thermo("fmax"), self.lmp.get_thermo("fnorm")
        nstep = int(self.lmp.get_thermo("step"))

        mpi_print(f'Minimization finished in {nstep} steps', verbose=self.verbose, comm=self.comm)
        #mpi_print(f'Initial energy: {e0:.3e}, final energy: {e1:.3e}', verbose=self.verbose, comm=self.comm)
        mpi_print(f'Initial fmax: {f0_max:.3e}, final fmax: {f1_max:.3e}', verbose=self.verbose, comm=self.comm)
        mpi_print(f'Initial fnorm: {f0_norm:.3e}, final fnorm: {f1_norm:.3e}', verbose=self.verbose, comm=self.comm)

        self.minimization_nstep = nstep

        if nstep >= maxiter - 1:
            mpi_print(f'WARNING: Minimization maxed out at {nstep} steps', comm=self.comm)
            self.not_converged = True

        if update_system:
            self.gather_D_dD()
            self.get_cell()
            self.X_coord = np.ctypeslib.as_array(self.lmp.gather("x", 1, 3)).flatten()

    def write_data(self, filename):
        """Write the current configuration to a data file"""
        self.lmp_commands_string(f'write_data {filename}')

    def setup_snap_potential(self):
        """Set up the potential in LAMMPS
        Currently implemented only for SNAP"""

        # Check that pot must be defined
        if self.pot is None:
            raise RuntimeError('Potential must be defined')

        mpi_print('Setting SNAP potential', comm=self.comm, verbose=self.verbose)

        if self.pot.set_zbl:
            mpi_print('Setting ZBL', comm=self.comm, verbose=self.verbose)
            mpi_print(' '*10 + f'{self.pot.zbl_rcut1=} {self.pot.zbl_rcut2=}', comm=self.comm)
            mpi_print(' '*10 + f'{self.pot.zbl_charge1=} {self.pot.zbl_charge2=}', comm=self.comm)
            self.lmp_commands_string(f"""
                pair_style hybrid/overlay zbl {self.pot.zbl_rcut1} {self.pot.zbl_rcut2} snap
                pair_coeff 1 1 zbl {self.pot.zbl_charge1} {self.pot.zbl_charge1}
                pair_coeff 1 2 zbl {self.pot.zbl_charge1} {self.pot.zbl_charge2}
                pair_coeff 2 2 zbl {self.pot.zbl_charge2} {self.pot.zbl_charge2}
            """)

        self.lmp_commands_string(f"""
        pair_style snap

        # Take coefficients from files W.snapcoeff and W.snapparam
        pair_coeff * * {self.pot.snapcoeff_path} {self.pot.snapparam_path} {self.pot.elements}
        run 0
        """)

        self.Ndesc = self.pot.num_param

        mpi_print(self.pot, verbose=self.verbose, comm=self.comm)

    def scatter_coord(self, X_coord=None, verbose=False):
        """
        Send the coordinates to LAMMPS

        Parameters
        ----------

        X_coord : numpy array, optional
            Coordinates to scatter. If None, the internal coordinates self.X_coord are used.

        verbose : bool, optional
            Print the scatter message as a comment to lmp_cmd_filename.
        """

        if X_coord is not None:
            X_scatter = X_coord
        else:
            X_scatter = self.X_coord

        try:
            self.lmp.scatter("x", 1, 3, np.ctypeslib.as_ctypes(X_scatter))
            self.lmp_commands_string("run 0")

            if verbose:
                self.lmp_commands_string("# Scattered X_coord to LAMMPS")

        except Exception as e:
            mpi_print(f'Error in scatter_coord: {e}', verbose=self.verbose, comm=self.comm)

    def gather_coord(self):
        """Gather the coordinates from LAMMPS and store them internally in self.X_coord.
        Minimum image convention is applied automatically to self.X_coord.
        """

        self.X_coord = np.ctypeslib.as_array(self.lmp.gather("x", 1, 3)).flatten()

    @measure_runtime_and_calls
    def compute_D_dD(self):
        """Compute descriptors and their derivatives in LAMMPS and store them internally"""

        # Check that pot must be defined
        if self.pot is None:
            raise RuntimeError('Potential must be defined')

        # Read the potential parameters from the potential object
        rcutfac, twojmax, rfac0, quadraticflag = [self.pot.snapparam_dict[k] for k in ['rcutfac', 'twojmax', 'rfac0', 'quadraticflag']]

        radii, weights = \
            self.pot.Theta_dict['radii'], \
            self.pot.Theta_dict['weights']

        try:
            self.lmp_commands_string(f"""
            # descriptors
            compute D all sna/atom {rcutfac} {rfac0} {twojmax} {radii} {weights} quadraticflag {quadraticflag}

            # derivatives of descriptors
            compute dD all snad/atom {rcutfac} {rfac0} {twojmax} {radii} {weights} quadraticflag {quadraticflag}

            # potential energy per atom
            compute E all pe/atom

            run 0
            """)
        except Exception as e:
            mpi_print(f'Error in compute_D_dD: {e}', verbose=self.verbose, comm=self.comm)

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
            ).reshape((-1, 2, 3, self.Ndesc)) # n atom types, 3 coordinates, n descriptors

        self.mixed_hessian = dD[:, 1, :, :].reshape((-1, self.Ndesc)).T

    @measure_runtime_and_calls
    def compute_virial(self):
        """Compute virial"""

        # Check that potential is defined
        if self.pot is None:
            raise RuntimeError('Potential must be defined')

        # Unpack the potential parameters
        rcutfac, twojmax, rfac0, quadraticflag = [self.pot.snapparam_dict[k] for k in ['rcutfac', 'twojmax', 'rfac0', 'quadraticflag']]

        radii, weights = \
            self.pot.Theta_dict['radii'], \
            self.pot.Theta_dict['weights']

        try:
            self.lmp_commands_string(f"""
            # descriptors
            compute virial all snav/atom {rcutfac} {rfac0} {twojmax} {radii} {weights} quadraticflag {quadraticflag}
            run 0
            """)
        except Exception as e:
            mpi_print(f'Error in compute_virial: {e}', verbose=self.verbose, comm=self.comm)

    @measure_runtime_and_calls
    def gather_virial(self):
        """Gather virial"""

        virial = self.lmp.gather("c_virial", 1, 6 * self.Ndesc)
        self.virial = np.ctypeslib.as_array(virial).reshape(-1, 6, self.Ndesc)

    def get_pressure_tensor_from_virial(self):
        """
            Get the pressure tensor from virial.
        """

        if self.Theta is None or self.virial is None:
            raise RuntimeError('Theta and virial must be set to calculate pressure')

        self.pressure_tensor = np.dot(np.sum(self.virial, axis=0), self.Theta)

    def get_pressure_from_virial(self):
        """
            Get the pressure from virial as the trace of the virial tensor contracted with the potential parameters.
        """

        self.get_pressure_tensor_from_virial()
        self.pressure = np.sum(self.pressure_tensor[:3]) / 3.0

    @measure_runtime_and_calls
    def run_init(self, setup_snap=True):
        """
        Initial LAMMPS run and initialization of basic properties
        """

        self.lmp_commands_string("run 0")
        self.species = np.ctypeslib.as_array(self.lmp.gather("type", 0, 1)).flatten()

        if setup_snap:
            self.setup_snap_potential()

        if self.minimize:
            self.minimize_energy()

        self.compute_D_dD()

        self.force0 = self.compute_forces()

        # Gather coordinates and apply minimum image
        self.gather_coord()
        # Send the coordinates back to LAMMPS with minimum image applied
        self.scatter_coord()

        self.atom_name_list = list(np.array(self.pot.elem_list)[self.species-1])

        # Number of atoms x 3
        self.N = self._X_coord.size
        self.Natom = self.N//3

        # Initialize the cell
        self.get_cell()

        # Compute the descriptors and their derivatives
        self.gather_D_dD()

        # Virial
        self.compute_virial()
        self.gather_virial()

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

        self.cell = np.zeros((3, 3))

        for cell_j in range(3):
            self.cell[cell_j][cell_j] = boxhi[cell_j]-boxlo[cell_j]

        self.cell[0][1] = xy
        self.cell[0][2] = xz
        self.cell[1][2] = yz

        self.cell[1][0] = xy
        self.cell[2][0] = xz
        self.cell[2][1] = yz

        self.inv_cell = np.linalg.inv(self.cell)

    @measure_runtime_and_calls
    def change_box(self, cell, update_system=True):
        """
        Apply strain to the system
        """

        C = cell.copy()

        try:
            self.lmp_commands_string(f"""
            change_box all triclinic
            change_box all x final 0.0 {C[0,0]} y final 0.0 {C[1,1]} z final 0.0 {C[2,2]} xy final {C[0,1]} xz final {C[0,2]} yz final {C[1,2]} remap units box
            run 0
            """)
        except Exception as e:
            mpi_print(f'Error in change_box: {e}', verbose=self.verbose, comm=self.comm)

        if update_system:
            self.gather_D_dD()
            self.get_cell()
            self.gather_virial()
            self.get_pressure_tensor_from_virial()
            self.get_pressure_from_virial()
            self.gather_coord()

    def minimum_image(self, X_vector):
        """Compute the minimum image of a vector X_vector (applying pbc)"""

        # Convert to (Natom, 3) shape
        X_3D = X_vector.reshape((-1, 3))

        correction = np.floor(X_3D @ self.inv_cell + 0.5) @ self.cell

        return X_vector - (correction * self.periodicity).flatten()

    @measure_runtime_and_calls
    def compute_forces(self, dX_vector=None, alpha0=0.05):
        """
        Evaluate forces for given position
        Uses [F(X+alpha0 * dX_dTheta)-F(X) ] /alpha0 -> hessian.dX_dTheta as alpha0 -> 0
        """

        if dX_vector is not None:
            # update positions
            X_tmp = self.X_coord.copy() #+ alpha0 * dX_vector.flatten()
            X_tmp += alpha0 * dX_vector.flatten()
            # send new positions to LAMMPS
            self.scatter_coord(X_coord=X_tmp, verbose=False)

        self.force_call_counter += 1

        # get the forces from LAMMPS: "f" - forces, 1 - type, LAMMPS_DOUBLE, 3 - values per atom
        force_array = np.ctypeslib.as_array(self.lmp.gather("f", 1, 3)).flatten()

        # Scatter the initial coordinates back to LAMMPS
        if dX_vector is not None:
            self.scatter_coord(verbose=False)

        return force_array

    @measure_runtime_and_calls
    def compute_hessian(self, dx=0.001, hess_mask=None, store_internally=True):
        """
        Finite difference Hessian calculation

        H_ij = -[F_i(X + dx_j) - F_i(X - dx_j)]/(2*dx_j)

        Parameters
        ----------
        dx : float, optional
            Finite difference.

        hess_mask : numpy array, optional
            Mask for the Hessian calculation.

        store_internally : bool, optional
            Store the Hessian internally in the object.

        Returns
        -------

        hessian : numpy array
            Hessian matrix. Shape: (3N, 3N).
        """

        mpi_print('Computing the Hessian...', comm=self.comm, verbose=self.verbose)

        hessian = np.zeros((self.N, self.N))

        # Memory-efficient
        # H = np.zeros((self.mask.sum(),self.mask.sum()))

        # We need to define a vector of displacements because the forces function
        # expects a vector of displacements
        dX_vector = np.zeros_like(self._X_coord.flatten())

        # Iterate over the atoms
        if hess_mask is None:
            idx_move = np.arange(self.N, dtype=int)
            desc = 'Hessian (full)'
        else:
            if (hess_mask.size != self.N):
                raise ValueError('hess_mask must have the same size as the number of atoms')
            idx_move = np.arange(self.N, dtype=int)
            idx_move = idx_move[hess_mask]
            desc = 'Hessian (masked)'

        if self.verbose and self.rank == 0:
            iterator = tqdm(idx_move, desc=desc)
        else:
            iterator = idx_move

        for i in iterator:

            # displace posotion i by dx
            dX_vector[i] = dx

            # compute forces: F(X + alpha0 * dX_dTheta)
            hessian[i, :] = -self.compute_forces(dX_vector=dX_vector, alpha0=1.0)
            # Memory-efficient
            # hessian[i,:] = -self.compute_forces(dX_vector, alpha0=1.0)[self.mask]

            # compute forces: F(X - alpha0 * dX_dTheta) and subtract
            dX_vector[i] = -dx
            hessian[i, :] -= -self.compute_forces(dX_vector=dX_vector, alpha0=1.0)
            # Memory-efficient
            # hessian[i,:] -= -self.compute_forces(dX_vector, alpha0=1.0)[self.mask]

            dX_vector[i] = 0.0

        hessian /= 2.0*dx

        # Add a small identity matrix to the Hessian to avoid singularities
        epsilon = 1.0e-3
        T = np.array([np.array([1, 0, 0] * self.Natom), np.array([0, 1, 0]*self.Natom), np.array([0, 0, 1]*self.Natom)])
        T = T.T@T * epsilon
        hessian = np.add(hessian, T)

        if store_internally:
            self.hessian = hessian

        return hessian

    def implicit_derivative(self,
                            method='energy',
                            min_style='fire',
                            alpha0=1e-2,
                            adaptive_alpha=True,
                            atol=1e-5,
                            ftol=1e-8,
                            maxiter=500,
                            hess_mask=None):
        """A wrapper for implicit derivative calculation

        General expression for the implicit derivative:
        impl_der = - mixed_hessian @ pseudoinverse(position_hessian)

        Parameters
        ----------

        method: str
            Method for implicit derivative calculation.
            Options:
                - inverse: straightforward Moore-Penrose inverse of the Hessian matrix, `np.linalg.pinv()`
                - dense: pseudoinverse from the system of linear equations, `np.linalg.solve(hessian, mixed_hessian)`
                - sparse: sparse linear method, solve for each potential parameter $\Theta_l$ separately
                - energy: constraint energy minimization in LAMMPS with additional force and energy terms
                          that correspond to a given parameter $\Theta_l$

        min_style: str
            LAMMPS minimization algorithm. Applies to the energy method only.
            Options: fire, cg, sd, htfn

        adaptive_alpha: bool
            Adaptive scaling factor alpha0 for implicit derivative calculation with sparse and energy methods.

        alpha0: float
            Scaling factor alpha0 for implicit derivative calculation with sparse and energy methods.
            If adaptive_alpha is True, alpha0 is the prefactor for the adaptive scaling.

        atol: float
            Absolute tolerance for implicit derivative calculation with sparse method for the lgmres solver.

        ftol: float
            Force tolerance for constrained LAMMPS minimization with energy method.

        maxiter: int
            Maximum number of iterations for implicit derivative calculation with sparse and energy methods.

        hess_mask: numpy array
            Mask for the Hessian calculation. Currently, applies only to the dense method.
            None for the full Hessian. Can be generated with tools.generate_masks() utility.
            Shape: (3 * Natom,)

        Returns
        -------
        dX_dTheta: numpy array
            Implicit derivative. Shape: (Ndesc, 3 * Natom)
        """

        if self.mixed_hessian is None:
            mpi_print('WARNING: mixed_hessian is not set, computing it now', comm=self.comm)
            self.compute_D_dD()
            self.gather_D_dD()

        # TODO: adjust sparse return format similarly to energy
        if method == 'energy':
            dX_dTheta = self.implicit_derivative_energy(alpha0=alpha0,
                                                        adaptive_alpha=adaptive_alpha,
                                                        ftol=ftol,
                                                        maxiter=maxiter,
                                                        min_style=min_style)

        elif method == 'sparse':
            dX_dTheta = self.implicit_derivative_sparse(alpha0=alpha0,
                                                        adaptive_alpha=adaptive_alpha,
                                                        atol=atol,
                                                        maxiter=maxiter)

        elif method == 'inverse':
            if hess_mask is not None:
                mpi_print('WARNING: hess_mask is not yet implemented in the inverse method', comm=self.comm)

            dX_dTheta = self.implicit_derivative_inverse()

        elif method == 'dense':
            dX_dTheta = self.implicit_derivative_dense(hess_mask=hess_mask)

        else:
            raise ValueError(f'Unknown method for implicit derivative: {method}')

        self.dX_dTheta = dX_dTheta.copy()

        return dX_dTheta

    @measure_runtime_and_calls
    def implicit_derivative_sparse(self,
                                   alpha0=0.01,
                                   atol=1e-5,
                                   maxiter=100,
                                   adaptive_alpha=True):
        """
        Evaluation of implicit position derivative via sparse linear methods.
        See the description of the parameters in the implicit_derivative() method.
        """
        # Compute the force at the initial position,
        # Analytically, it must be zero, but for numerical reasons, it is small
        dX0 = np.zeros_like(self._X_coord)
        F0 = self.compute_forces()

        # result holder
        dX_dTheta = np.zeros((self.Ndesc, self.Natom * 3))

        if self.verbose and self.rank == 0:
            iterator = tqdm(self.mixed_hessian, desc='Impl. Der. Sparse')
        else:
            iterator = self.mixed_hessian

        # One linear solutions per parameter
        for idesc, Cl in enumerate(iterator):

            # determine the alpha_factor
            if adaptive_alpha:
                alpha_factor = alpha0 / np.max(np.abs(Cl))
            else:
                alpha_factor = alpha0

            # define linear operator with matrix-vector product matvec()
            matvec = lambda dX: (F0-self.compute_forces(dX_vector=dX, alpha0=alpha_factor)) / alpha_factor
            linop = LinearOperator((self.N, self.N), matvec=matvec, rmatvec=matvec)

            # perform iterative linear solution routine LGMRES: Ax = b, solve for x
            # linop - linear operator which can produce Ax
            dX_dTheta[idesc, :] = lgmres(linop, Cl, x0=dX0, atol=atol, maxiter=maxiter)[0]

        return dX_dTheta

    @measure_runtime_and_calls
    def implicit_derivative_energy(self,
                                   alpha0=0.5,
                                   adaptive_alpha=True,
                                   min_style='fire',
                                   ftol=1e-10,
                                   maxiter=200):
        """Evaluation of implicit position derivative
        via sparse linear methods.

        Parameters
        ----------
        alpha0 : float, optional
            Numerical parameter in expansion [F(X+alpha0*dX_dTheta)-F(X)]/alpha0->hessian.dX_dTheta,
            exact as alpha0->0, but too small causes finite difference errors.
            Default 0.01 (0.5 for adaptive)
        ftol : float, optional
            force tolerance for  minimization, by default 1e-10
        maxiter : int, optional
            maximum iteration for , by default 200

        LAMMPS terminology:
        compute: run a simulation
        fix: do something to the system
        """

        # declare an array of size N x 3 to be stored in the simulation
        # fix x all ... : add per atom property "x" to all atoms
        # "d_" means it is a float
        # alphaCl = alpha0 * d/dT_l (dU/dX), N x 3 vector
        # alphaClx = alpha0 * d/dT_l (dU/dx), N vector
        # alphaCly = alpha0 * d/dT_l (dU/dy)
        # The alphaClx, alphaCly, and alphaClz names are defined here
        self.lmp_commands_string(f"""
            fix mixedHessianRow all property/atom d_alphaClx d_alphaCly d_alphaClz

            #compute deltaX all displace/atom

            # minimization algorithm. Options: fire, cg, sd, htfn
            min_style {min_style}
        """)

        dX_dTheta = np.zeros((self.Ndesc, self.Natom * 3))

        self.impl_der_stats['energy'] = {}
        self.impl_der_stats['energy']['alpha_array'] = np.zeros(self.Ndesc)
        self.impl_der_stats['energy']['calls'] = np.zeros(self.Ndesc)
        self.impl_der_stats['energy']['f0_max'] = np.zeros((self.Ndesc))
        self.impl_der_stats['energy']['f0_norm'] = np.zeros((self.Ndesc))
        self.impl_der_stats['energy']['f1_max'] = np.zeros((self.Ndesc))
        self.impl_der_stats['energy']['f1_norm'] = np.zeros((self.Ndesc))

        # iterate over columns of the mixed hessian obtained from LAMMPS SNAP potential
        # d/dT_i (dU/dX)

        # For debugging purposes
        # print_iCl = True
        print_iCl = False

        if print_iCl:
            iterator = self.mixed_hessian
        else:
            if self.verbose and self.rank == 0:
                iterator = tqdm(self.mixed_hessian, desc=f'Impl. Der. Energy {min_style.upper()}')
            else:
                iterator = self.mixed_hessian

        for idesc, Cl in enumerate(iterator):

            if print_iCl:
                mpi_print(f'{"Parameter index idesc":>30}: {idesc:3d}', comm=self.comm)

            # reset the positions to the initial minimum configuration
            self.scatter_coord(verbose=False)

            # count the number of force calls as the number of time steps
            self.lmp_commands_string("reset_timestep 0")

            # compute the 3N vector of displacements X - X0 taking into account pbc
            self.lmp_commands_string("compute deltaX all displace/atom")

            if adaptive_alpha:
                alpha_factor = alpha0 / np.max(np.abs(Cl))
            else:
                alpha_factor = alpha0

            self.impl_der_stats['energy']['alpha_array'] += [alpha_factor]

            # set up the C vector:
            # fix x, y, or z
            # The names alphaClx must be the same as above in "fix"
            for i, LAMMPS_aCl_i in enumerate(["d_alphaClx", "d_alphaCly", "d_alphaClz"]):
                # i is x, y, or z
                # alpha0 C_l = alpha0 * d/dT_l (dU/dX)
                # Take only the i-th component of l-th vector
                alphaCl_i = Cl.reshape((-1, 3))[:, i].astype(np.float64)

                # Multiply by a small parameter alpha0
                alphaCl_i *= alpha_factor

                # send the vector to LAMMPS
                self.lmp.scatter(LAMMPS_aCl_i, 1, 1, np.ctypeslib.as_ctypes(alphaCl_i))

            # Now, we have a column of mixed Hessian passed and positions passed to LAMMPS

            # build the energy and force function
            self.lmp_commands_string(f"""

                # Retrieve alpha0 * C_l
                compute alphaCl all property/atom d_alphaClx d_alphaCly d_alphaClz

                # Turn alpha0 * C_l into variables
                # c_ means compute

                # alpha0 * d/dT_l dU/dx
                variable aClx atom c_alphaCl[1]

                # alpha0 * d/dT_l dU/dy
                variable aCly atom c_alphaCl[2]

                # alpha0 * d/dT_l dU/dz
                variable aClz atom c_alphaCl[3]

                # additional term in energy:
                # dot procuct C_l * dX
                # v_ is variable
                # NO SPACES ALLOWED in the expression
                variable addEnergy atom v_aClx*c_deltaX[1]+v_aCly*c_deltaX[2]+v_aClz*c_deltaX[3]

                # additional force to the system
                # fx = alpha0 * d/dT_l dU/dx, etc
                fix addForce {self.fix_sel} addforce v_aClx v_aCly v_aClz energy v_addEnergy

                # blank iteration to make sure everything works
                run 0
            """)

            self.impl_der_stats['energy']['f0_max'][idesc] = self.lmp.get_thermo("fmax")
            self.impl_der_stats['energy']['f0_norm'][idesc] = self.lmp.get_thermo("fnorm")

            if self.fix_box_relax:
                self.lmp_commands_string(f"""
                    fix boxrelax all box/relax iso 0.0 vmax {self.box_relax_vmax}
                """)

            self.lmp_commands_string(f"""
                minimize 0. {ftol} {maxiter} {maxiter}
            """)

            f1_max, f1_norm = self.lmp.get_thermo("fmax"), self.lmp.get_thermo("fnorm")

            # extract the final positions
            X_new = np.ctypeslib.as_array(self.lmp.gather("x", 1, 3)).flatten()

            # apply pbc
            #dX_tmp = X_new - self._X_coord
            dX_tmp = self.minimum_image(X_new - self._X_coord)

            #mpi_print(idesc, Cl, np.max(np.abs(dX_tmp)), alpha_factor)

            # Store to idesc row of dX_dTheta
            dX_dTheta[idesc, :] = dX_tmp / alpha_factor

            # Stats
            self.impl_der_stats['energy']['calls'][idesc] = int(self.lmp.get_thermo("step"))
            self.impl_der_stats['energy']['f1_max'][idesc] = f1_max
            self.impl_der_stats['energy']['f1_norm'][idesc] = f1_norm

            # disable commands
            self.lmp_commands_string("""
                uncompute alphaCl
                unfix addForce
                uncompute deltaX
            """)

        self.lmp_commands_string("""
            variable aClx delete
            variable aCly delete
            variable aClz delete
            unfix mixedHessianRow
        """)

        return dX_dTheta

    @measure_runtime_and_calls
    def implicit_derivative_inverse(self):
        """Compute implicit derivative from Hessian inverse

        Returns
        -------
        dX_dTheta : numpy array
            implicit derivative
        """

        # Compute the Moore-Penrose inverse of the Hessian
        hessian = self.compute_hessian()
        hessian += np.eye(hessian.shape[0]) * 0.01 * np.diag(hessian).min()
        H_inv = np.linalg.pinv(hessian)

        # Matrix multiplication to get dX_dTheta
        dX_dTheta = (H_inv @ self.mixed_hessian.T).T

        return dX_dTheta

    @measure_runtime_and_calls
    def implicit_derivative_dense(self, hessian=None, hess_mask=None):
        """Compute implicit derivative from Hessian inverse

        Returns
        -------
        dX_dTheta : numpy array
            implicit derivative
        """

        if hessian is None:
            hessian = self.compute_hessian(hess_mask=hess_mask)

            # Lift the diagonal of the Hessian to avoid singularities
            hessian += np.eye(hessian.shape[0]) * 0.01 * np.diag(hessian).min()

        #return np.zeros((self.Ndesc, self.Natom * 3))

        # Use linalg.solve to find dX_dTheta in H.dX_dTheta = C
        if hess_mask is not None:
            dX_dTheta = np.zeros_like(self.mixed_hessian)

            # hessian with rows and columns corresponding to hess_mask
            hessian_mask = hessian[hess_mask, :][:, hess_mask]

            mpi_print(f'Computing dX_dTheta with linalg.solve, Hessian shape (masked): {hessian_mask.shape}', comm=self.comm, verbose=self.verbose)
            with self.timings.add('linalg.solve') as t:
                dX_dTheta[:, hess_mask] = np.linalg.solve(hessian_mask, self.mixed_hessian.T[hess_mask, :]).T

        else:
            mpi_print(f'Computing dX_dTheta with linalg.solve, Hessian shape: {hessian.shape}', comm=self.comm, verbose=self.verbose)
            with self.timings.add('linalg.solve') as t:
                dX_dTheta = np.linalg.solve(hessian, self.mixed_hessian.T).T

        return dX_dTheta

    @measure_runtime_and_calls
    def implicit_derivative_hom_iso(self, delta_Strain=1e-5):
        """
        Compute the homogenous implicit derivative with finite differences applied
        to the virial derivative.
        Isotropic strain.

        dStrain_dTheta_j = - Virial_j / [ (dVirial/dStrain) @ Theta ]
        """
        cell0 = self.cell.copy()

        self.gather_virial()

        virial_trace0 = np.sum(self.virial, axis=0)
        virial_trace0 = np.sum(virial_trace0[:3, :], axis=0) / 3.0

        # Compute the virial derivative
        strain_matrix = np.eye(3) * (1.0 + delta_Strain)
        # Check the ORDER: cell0 @strain_matrix OR strain_matrix @ cell0
        cell_plus = cell0 @ strain_matrix

        # change box RENAME
        self.change_box(cell_plus, update_system=True)
        # Sum over atoms
        virial_trace_plus = np.sum(self.virial, axis=0)
        # Take first 3 elements of the virial tensor
        virial_trace_plus = np.sum(virial_trace_plus[:3, :], axis=0) / 3.0

        dVirial_dStrain = (virial_trace_plus - virial_trace0) / delta_Strain

        # Compute the homogenous implicit derivative
        dStrain_dTheta = - virial_trace0 / np.dot(self.Theta, dVirial_dStrain)

        # Set to the original cell
        self.change_box(cell0)

        return dStrain_dTheta

    def implicit_derivative_hom_dVirial_dL(self, delta_L=1e-3):
        """
        Compute the homogenous implicit derivative with finite differences applied
        to the virial derivative.

        dL_dTheta_j = - Virial_j / [ (dVirial/dL) @ Theta ]
        """
        cell0 = self.cell.copy()

        self.gather_virial()

        virial_trace0 = np.sum(self.virial, axis=0)
        virial_trace0 = np.sum(virial_trace0[:3, :], axis=0) / 3.0

        # Compute the virial derivative
        cell_plus = cell0 + np.eye(3) * delta_L

        self.change_box(cell_plus, update_system=True)
        # Sum over atoms
        virial_trace_plus = np.sum(self.virial, axis=0)
        # Take first 3 elements of the virial tensor
        virial_trace_plus = np.sum(virial_trace_plus[:3, :], axis=0) / 3.0

        dVirial_dL = (virial_trace_plus - virial_trace0) / delta_L

        # Compute the homogenous implicit derivative
        dL_dTheta = - virial_trace0 / np.dot(self.Theta, dVirial_dL)

        # Set to the original cell
        self.change_box(cell0)

        return dL_dTheta

    def implicit_derivative_hom_aniso(self, delta_L=1e-3):
        """
        Homogeneous implicit derivative, general case: fully anisotropic system.
        """

        # Cell and pressure tensor at zero strain
        cell0 = self.cell.copy()

        self.gather_virial()

        self.get_pressure_tensor_from_virial()
        pressure_tensor0 = matrix_3x3_from_Voigt(self.pressure_tensor)

        # Matrix-valued basis vectors
        basis_E = get_matrix_basis()

        # Virial, summed over atoms, shape: (Ndesc, 6)
        virial_Nd_6 = np.sum(self.virial, axis=0).T
        # Convert to shape (Ndesc, 3, 3) using the Voigt notation
        virial_Nd_3x3 = matrices_3x3_from_Voigt(virial_Nd_6[:, :])

        # Compute the double dot product between virial and 6 matrix-valued basis vectors
        # (Ndesc, 3, 3) : (Ndesc, 3, 3, 6) -> (Ndesc, 6)
        virial_basis_prod = np.zeros((self.Ndesc, 6), dtype=float)
        for iTheta in range(self.Ndesc):
            for idx_n in range(6):
                virial_basis_prod[iTheta, idx_n] = np.tensordot(virial_Nd_3x3[iTheta, :, :], basis_E[idx_n, :, :])

        # Compute the S-matrix: S_mn = E_m : d sigma / dC : E_n, shape: (6, 6)
        S_matrix = np.zeros((6, 6), dtype=float)

        weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

        for idx_m in range(6):
            cell_perturb = cell0 + delta_L * basis_E[idx_m]

            self.change_box(cell_perturb)
            pressure_tensor_3x3 = matrix_3x3_from_Voigt(self.pressure_tensor)
            #perturb_Em = 1.0 / ( weights[idx_m] * delta_L ) * (pressure_tensor_3x3 - pressure_tensor0)
            perturb_Em = 1.0 / delta_L * (pressure_tensor_3x3 - pressure_tensor0)

            for idx_n in range(6):
                S_matrix[idx_m, idx_n] = np.tensordot(perturb_Em, basis_E[idx_n])

        self.change_box(cell0)

        # Pseudoinverse of the S-matrix
        S_matrix_inv = np.linalg.pinv(S_matrix)

        # Compute the matrix product between virial_basis_prod and S_matrix_inv, shape: (Ndesc, 6)
        # l: Ndesc, m: 6, n: 6
        virial_S_prod = np.einsum('lm, mn -> ln', virial_basis_prod, S_matrix_inv)

        # The final expression dC/dTheta = - sum_{m=1}^{6} virial_S_prod_m * E_m
        # DIVIDE by weights for m
        virial_S_prod_weighted = np.einsum('lm, m -> lm', virial_S_prod, 1.0 / weights)
        #dCell_dTheta = - np.einsum('lm, mab -> lab', virial_S_prod[:, :], basis_E[:, :, :])
        dCell_dTheta = - np.einsum('lm, mab -> lab', virial_S_prod_weighted[:, :], basis_E[:, :, :])

        return dCell_dTheta

    def energy_expansion(self, dTheta):
        """
        Taylor energy expansion up to the second order in dX
        """

        raise NotImplemented('Analytic energy expansion is not Implemented')

        if self.dU_dTheta is None:
            self.compute_D_dD()
            self.gather_D_dD()

        # First order in dTheta (first order in dX is zero)
        #dU = np.dot(self.dU_dTheta, dTheta)

        print(self.dU_dTheta.shape)

        return None

    @measure_runtime_and_calls
    def write_xyz_file(self, filename="coordinates.xyz", verbose=False):
        """Writes atomic coordinates to an .xyz file."""
        # Number of atoms
        Natoms = self._X_coord.size // 3

        with open(filename, 'w') as f:
            # First line: Number of atoms
            f.write(f"{Natoms}\n")

            # Second line: Comment or blank
            f.write("Atomic coordinates generated by LammpsImplicitDer\n")

            # Extract and write the atomic coordinates
            iatom = 0
            for i in range(0, self._X_coord.size, 3):
                f.write(f"{self.atom_name_list[iatom]} {self._X_coord[i]:.6f} {self._X_coord[i+1]:.6f} {self._X_coord[i+2]:.6f}\n")
                iatom += 1

        if verbose:
            mpi_print(f"Coordinates saved to {filename}", comm=self.comm)

    """
    # LAMMPS: compute D all sna/atom **params**
    Ndesc = 55 # e.g.
    Nspec = 2

    # shape of (N,Ndesc)
    D_all = np.ctypeslib.as_array(\
        lmp.gather("c_D", 1, Ndesc)).reshape((-1, Ndesc))

    # 1: W 2: C
    specie = np.ctypeslib.as_array(lmp.gather("type",0,1)).astype(int)

    # note shape: (N_spec,Ndesc)
    D = np.asarray([D_all[specie==s+1].sum(0) for s in range(Nspec)])

    # LAMMPS: compute dD all snad/atom **params**
    # note shape: (N,Nspec,3,Ndesc) (could use np.swapaxes)

    dD = np.ctypeslib.as_array(\
        lmp.gather("c_dD", 1, Nspec*3*Ndesc)
        ).reshape((-1, Nspec, 3, Ndesc))

    # LAMMPS: compute ddD all snav/atom **params**
    # note shape: (N,Nspec,6,Ndesc)
    V = np.ctypeslib.as_array(\
        lmp.gather("c_ddD", 1, Nspec*6*Ndesc)
        ).reshape((-1, Nspec, 6, Ndesc))
    """
