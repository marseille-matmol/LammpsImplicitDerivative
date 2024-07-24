#!/usr/bin/env python3
"""
Initialize from lammps data files or scripts
Chlid classes of LammpsImplicitDer.
"""

import os
import numpy as np

# local imports
from ..tools.utils import mpi_print
from ..tools.timing import measure_runtime_and_calls
from ..lmp_der.snap import SNAP
from ..lmp_der.implicit_der import LammpsImplicitDer


class FromData(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self, setup_snap=True,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for FromData class')

        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        if len(self.pot.elem_list) > 2:
            raise RuntimeError('Not implemented for number of elements > 2')

        self.binary = len(self.pot.elem_list) == 2

        self.element = self.pot.elem_list[0]

        self.Theta = self.pot.Theta_dict[self.element]['Theta']

        if self.datafile is None and self.input_script is None:
            raise RuntimeError('datafile or input_script must be proivded to FromData class')

        if self.input_script is not None:
            mpi_print(f'Loading the input_script', verbose=self.verbose, comm=self.comm)
            self.lmp_commands_string(self.input_script)

        if self.datafile is not None:
            mpi_print(f'Reading input file {self.datafile}', verbose=self.verbose, comm=self.comm)
            self.lmp_commands_string(f"""
            read_data {self.datafile}
            """)

        self.run_init(setup_snap=setup_snap)