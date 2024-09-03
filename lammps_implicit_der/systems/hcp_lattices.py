#!/usr/bin/env python3
"""
Hex systems.
Chlid classes of LammpsImplicitDer.

hcp has a1 = 1 0 0, a2 = 0 sqrt(3) 0, and a3 = 0 0 sqrt(8/3).
"""

import os
import numpy as np

# local imports
from ..tools.utils import mpi_print
from ..tools.timing import measure_runtime_and_calls
from ..lmp_der.snap import SNAP
from ..lmp_der.implicit_der import LammpsImplicitDer


class Hcp(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 ncell_x=3,
                 ncell_y=None,
                 ncell_z=None,
                 alat=2.46,
                 setup_snap=True,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.binary = False
        self.ncell_x = ncell_x
        self.alat = alat

        self.ncell_y = ncell_y if ncell_y is not None else ncell_x
        self.ncell_z = ncell_z if ncell_z is not None else ncell_x

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified.')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        if len(self.pot.elem_list) > 1:
            raise RuntimeError('Must be single-element')

        self.element = self.pot.elem_list[0]

        self.Theta = self.pot.Theta_dict[self.element]['Theta']

        self.lmp_commands_string(f"""
        boundary p p p
        lattice hcp {self.alat} origin 0.01 0.01 0.01
        """)

        # Setup the coordinates from scratch
        if self.datafile is None:

            self.lmp_commands_string(f"""
            # create a block of atoms
            region C block 0 {self.ncell_x} 0 {self.ncell_y} 0 {self.ncell_z} units lattice
            create_box 1 C
            create_atoms 1 region C
            """)

        # Read from a datafile
        else:
            mpi_print(f'Reading datafile {self.datafile}', verbose=self.verbose, comm=self.comm)
            self.lmp_commands_string(f"""
            read_data {self.datafile}
            """)

        self.lmp_commands_string(f'mass * 45.')

        self.run_init(setup_snap=setup_snap)