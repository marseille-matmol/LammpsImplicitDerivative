#!/usr/bin/env python3
"""
Miscellaneous systems.
Chlid classes of LammpsImplicitDer.
"""

import os
import numpy as np

# local imports
from ..tools.utils import mpi_print
from ..tools.timing import measure_runtime_and_calls
from ..lmp_der.snap import SNAP
from ..lmp_der.implicit_der import LammpsImplicitDer


class VacW(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for VacW')

        if self.datafile is None:
            raise ValueError('datafile must be specified for VacW')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        # Potential parameters
        self.Theta = self.pot.Theta_dict['W']['Theta']

        self.lmp.commands_string(f"""
        clear

        units metal
        atom_modify map array sort 0 0.0

        boundary p p p

        # Read the coordinates in .lmp format
        read_data {self.datafile}

        # Tungsten mass
        mass 1 183.839999952708211594654130749404
        """)

        self.run_init()