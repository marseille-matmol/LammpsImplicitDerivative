#!/usr/bin/env python3
"""
Bcc vacancy systems.
Chlid classes of LammpsImplicitDer.
"""

import os
import numpy as np

from lammps import lammps

# local imports
from utils import mpi_print
from timing import measure_runtime_and_calls
from potential_tools import SNAP
from implicit_der import LammpsImplicitDer


class BccVacancy(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 num_cells=3,
                 alat=3.1855,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.num_cells = num_cells
        self.alat = alat

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for BccVacancy')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename)

        # Get Ndesc from the pot object, maybe later use it directly from pot
        self.Ndesc = self.pot.num_param

        # Load the hard constrains file, if present
        hard_constraints_path = os.path.join(self.data_path, f'{self.pot.elmnts}_constraints.txt')
        if os.path.exists(hard_constraints_path):
            self.A_hard = np.loadtxt(hard_constraints_path)

        # Potential parameters
        # Hardcoded for tungsten
        self.Theta = self.pot.Theta_dict['W']['Theta']

        self.lmp.commands_string(f"""
        clear

        atom_modify map array sort 0 0.0

        # Initialize simulation
        units metal
        lattice bcc {self.alat} origin 0.01 0.01 0.01
        """)

        # Setup the coordinates from scratch
        if self.datafile is None:

            self.lmp.commands_string(f"""
            # create a block of atoms
            region C block 0 {self.num_cells} 0 {self.num_cells} 0 {self.num_cells} units lattice
            create_box 1 C

            # add atoms
            create_atoms 1 region C

            # delete one atom
            group del id 10
            delete_atoms group del
            """)

        # Read from a datafile
        else:

            self.lmp.commands_string(f"""
            read_data {self.datafile}
            """)

        # W mass in a.m.u.
        self.lmp.commands_string(f'mass * 184.')

        self.setup_snap_potential()

        self.lmp.commands_string(f"""
        # Minimization algorithm
        min_style {self.minimize_algo}

        # minimize energy until ftol is reached
        {'minimize 0 '+str(self.minimize_ftol)+' 1000 1000' if self.minimize else ''}
        """)

        self.compute_D_dD()

        self.gather_D_dD()

        self.run_init()


class BccBinaryVacancy(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 num_cells=3,
                 alat=3.13,
                 custom_create_script=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.num_cells = num_cells
        self.alat = alat

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for BccBinaryVacancy')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename)

        # Load the hard constrains file, if present
        hard_constraints_path = os.path.join(self.data_path, f'{self.pot.elmnts}_constraints.txt')
        if os.path.exists(hard_constraints_path):
            self.A_hard = np.loadtxt(hard_constraints_path)

        # Potential parameters: ONLY MOLYBDENUM
        self.Theta = self.pot.Theta_dict['Mo']['Theta']
        self.Ndesc = self.pot.num_param

        self.lmp.commands_string(f"""
        clear

        atom_modify map array sort 0 0.0
        boundary p p p

        # Initialize simulation
        units metal
        lattice bcc {self.alat} origin 0.01 0.01 0.01
        """)

        # Setup the coordinates from scratch
        if self.datafile is None:

            self.lmp.commands_string(f"""
            # create a block of atoms
            region C block 0 {num_cells} 0 {num_cells} 0 {num_cells} units lattice
            create_box 2 C
            create_atoms 1 region C
            """)
            if custom_create_script is None:
                self.lmp.commands_string(f"""
                set group all type/fraction 2 0.5 12393
                # delete one atom
                group del id 10
                delete_atoms group del
                """)
            else:
                self.lmp.commands_string(custom_create_script)


        # Read from a datafile
        else:
            mpi_print(f'Reading datafile {self.datafile}', verbose=self.verbose, comm=self.comm)
            self.lmp.commands_string(f"""
            read_data {self.datafile}
            """)

        self.lmp.commands_string(f'mass * 45.')

        self.setup_snap_potential()

        self.lmp.commands_string(f"""
        # Minimization algorithm
        min_style {self.minimize_algo}

        # minimize energy until ftol is reached
        {'minimize 0 '+str(self.minimize_ftol)+' 1000 1000' if self.minimize else ''}
        """)

        self.compute_D_dD()

        self.num_cells = num_cells
        self.alat = alat

        self.run_init()

        self.gather_D_dD()

    def gather_D_dD(self):
        """Compute descriptors and their derivatives in LAMMPS and store them internally, only for specie B
        """
        dU_dTheta = np.ctypeslib.as_array(self.lmp.gather("c_D", 1, self.Ndesc)).reshape((-1, self.Ndesc))

        self.dU_dTheta = dU_dTheta[self.species == 2].sum(0)

        dD = np.ctypeslib.as_array(
                self.lmp.gather("c_dD", 1, 3*2*self.Ndesc)
            ).reshape((-1, 2, 3, self.Ndesc))

        self.mixed_hessian = dD[:, 1, :, :].reshape((-1, self.Ndesc)).T