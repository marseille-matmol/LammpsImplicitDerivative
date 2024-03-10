#!/usr/bin/env python3
"""
Bcc vacancy systems.
Chlid classes of LammpsImplicitDer.
"""

import os
import numpy as np

# local imports
from ..tools.utils import mpi_print
from ..tools.timing import measure_runtime_and_calls
from ..lmp_der.snap import SNAP
from ..lmp_der.implicit_der import LammpsImplicitDer


class BccBinary(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 num_cells=3,
                 alat=3.13,
                 specie_B_concentration=0.5,
                 setup_snap=True,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.binary = True

        self.num_cells = num_cells
        self.alat = alat

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for BccBinaryVacancy')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)


        # Potential parameters: ONLY MOLYBDENUM
        self.Theta = self.pot.Theta_dict['Mo']['Theta']

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

            set group all type/fraction 2 {specie_B_concentration} 12393
            """)

        # Read from a datafile
        else:
            mpi_print(f'Reading datafile {self.datafile}', verbose=self.verbose, comm=self.comm)
            self.lmp.commands_string(f"""
            read_data {self.datafile}
            """)

        self.lmp.commands_string(f'mass * 45.')

        self.run_init(setup_snap=setup_snap)


class BccVacancy(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 num_cells=3,
                 #alat=3.13,
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
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        # Potential parameters, hardcoded for tungsten
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

        self.run_init()


class BccBinaryVacancy(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 num_cells=3,
                 alat=3.13,
                 custom_create_script=None,
                 specie_B_concentration=0.5,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.binary = True
        self.num_cells = num_cells
        self.alat = alat

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for BccBinaryVacancy')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        # Potential parameters: ONLY MOLYBDENUM
        self.Theta = self.pot.Theta_dict['Mo']['Theta']

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
                set group all type/fraction 2 {specie_B_concentration} 12393
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

        self.run_init()