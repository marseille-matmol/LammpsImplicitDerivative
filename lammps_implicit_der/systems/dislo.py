#!/usr/bin/env python3
"""
Dislocation systems.
Child classes of LammpsImplicitDer.
"""

import os
from tqdm import tqdm

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lgmres, lsqr
import numpy as np

# local imports
from ..tools.timing import measure_runtime_and_calls
from ..lmp_der.snap import SNAP
from ..lmp_der.implicit_der import LammpsImplicitDer


class SCREW_DISLO(LammpsImplicitDer):
    """
    Screw dislocation system. By default, tungsten. Allows interstitial or substitutional defects.
    Requires a LAMMPS data file for coordinates ONLY.
    """
    @measure_runtime_and_calls
    def __init__(self,
                 element='W',
                 element_mass=183.84,
                 fix_border_atoms=True,
                 fixed_cyl_axis='z',
                 fixed_cyl_x1='70.2259',
                 fixed_cyl_x2='72.0799',
                 fixed_cyl_r='49.9',
                 fixed_cyl_lo='0.0',
                 fixed_cyl_hi='2.7587',
                 sub_element=None,
                 sub_mass=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        if sub_element is not None:
            self.binary = True

            if sub_mass is None:
                sub_mass = element_mass

        # Selection for addforce command
        if fix_border_atoms:
            self.fix_sel = 'moving_atoms'
        else:
            self.fix_sel = 'all'

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for DISLO')

        if self.datafile is None:
            raise ValueError('datafile must be specified for DISLO')
        elif not os.path.exists(self.datafile):
            raise FileNotFoundError(f'SCREW_DISLO: datafile {self.datafile} does not exist.')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename,
                                   zbl_dict=self.zbl_dict,
                                   comm=self.comm)

        # Potential parameters: if binary, the substitutional element parameters ONLY
        if self.binary:
            self.Theta = self.pot.Theta_dict[sub_element]['Theta']
        else:
            self.Theta = self.pot.Theta_dict[element]['Theta']

        self.lmp_commands_string(f"""
        clear

        units metal
        atom_modify map array sort 0 0.0

        boundary f f p

        # Read the coordinates in lammps-data format
        read_data {self.datafile}

        mass 1 {element_mass}
        {f'mass 2 {sub_mass}' if self.binary else ''}
        """)

        self.setup_snap_potential()

        if fix_border_atoms:

            self.lmp_commands_string(f"""
            # Fix the border atoms
            # Define the cylinder region
            region fixed_cyl cylinder {fixed_cyl_axis} {fixed_cyl_x1} {fixed_cyl_x2} {fixed_cyl_r} {fixed_cyl_lo} {fixed_cyl_hi} units box side out

            # Define the group of atoms in the cylinder
            group fixed_atoms region fixed_cyl

            # All the rest
            group {self.fix_sel} subtract all fixed_atoms

            # Zero force on the fixed atoms
            fix freeze fixed_atoms setforce 0.0 0.0 0.0
            """)

        self.lmp_commands_string(f"""
        # Neighbors
        # Add skin distance of 2 A to the cutoff radius to create a buffer zone?
        # bin - the method to build the neighbor list (binning)
        neighbor 2.0 bin

        # rebuild the neighbor list at every step
        # no delay
        # check if any atom moved more than half of the skin distance
        # if yes, rebuild the list
        neigh_modify every 1 delay 0 check yes

        # output of thermodynamic information every 10 steps
        thermo 10
        run 0
        """)

        self.run_init(setup_snap=False)
