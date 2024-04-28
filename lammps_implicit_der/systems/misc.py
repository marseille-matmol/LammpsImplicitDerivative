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
        mass 1 184.0
        """)

        self.run_init()


class BccVacancyConcentration(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 ncell_x=3,
                 alat=3.1855,
                 vac_conc=0.2,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ncell_x = ncell_x
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
        atom_style atomic
        atom_modify map array sort 0 0.0
        units metal

        # generate the box and atom positions using a BCC lattice
        boundary p p p
        lattice bcc {alat}
        region box block 0 {ncell_x} 0 {ncell_x} 0 {ncell_x}
        """)

        # Setup the coordinates from scratch
        if self.datafile is None:

            self.lmp.commands_string(f"""
            create_box 1 box
            create_atoms 1 region box

            # Tungsten mass
            mass 1 184.0

            # Create *very* large vac concentration (20%!) "123" is a random number seed
            delete_atoms random fraction {vac_conc} yes all box 123

            # Setup output
            thermo          1
            thermo_modify norm no
            neighbor 2.0 bin
            neigh_modify once no every 1 delay 0 check yes
            """)

        # Read from a datafile
        else:
            self.lmp.commands_string(f"""
            read_data {self.datafile}
            """)

        self.run_init(setup_snap=True)
