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


class Dislo(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 fix_sel='moving_atoms',
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Selection for addforce command
        self.fix_sel = fix_sel

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for Dislo')

        if self.datafile is None:
            raise ValueError('datafile must be specified for Dislo')

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

        boundary f f p

        # Read the easy core coordinates in .lmp format
        read_data {self.datafile}

        # Tungsten mass
        mass 1 183.839999952708211594654130749404
        """)

        self.setup_snap_potential()

        self.lmp.commands_string(f"""

        # Fix the border atoms
        # Define the cylinder region
        region fixed_cyl cylinder z 70.22590967109964 72.07990729368126 49.9 0.0 2.7587342746349166 units lattice side out

        # Define the group of atoms in the cylinder
        group fixed_atoms region fixed_cyl

        # All the rest
        group moving_atoms subtract all fixed_atoms

        # Zero force on the fixed atoms
        fix freeze fixed_atoms setforce 0.0 0.0 0.0

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
        """)

        self.run_init()


class DisloSub(LammpsImplicitDer):
    """
    Dislocaiton with substitutional defect
    """
    @measure_runtime_and_calls
    def __init__(self,
                 fix_sel='moving_atoms',
                 sub_element='Be',
                 sub_mass=9.012182,
                 *args, **kwargs):
        """Simple LAMMPS simulator

        Parameters
        ----------
        Ndesc: number of descriptor components
        ftol : force minimization threshold
        """

        super().__init__(*args, **kwargs)

        self.binary = True

        W_mass = 183.84
        if sub_mass is None:
            sub_mass = W_mass

        # Selection for addforce command
        self.fix_sel = fix_sel

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for Dislo')

        if self.datafile is None:
            raise ValueError('datafile must be specified for Dislo')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        # Potential parameters
        self.Theta = self.pot.Theta_dict[sub_element]['Theta']

        self.lmp.commands_string(f"""
        boundary f f p

        # Read the easy core coordinates in .lmp format
        read_data {self.datafile}

        # Tungsten mass
        mass 1 183.839999952708211594654130749404
        mass 2 {sub_mass}
        """)

        self.setup_snap_potential()

        self.lmp.commands_string(f"""
        # Fix the border atoms
        # Define the cylinder region
        region fixed_cyl cylinder z 70.22590967109964 72.07990729368126 49.9 0.0 2.7587342746349166 units lattice side out

        # Define the group of atoms in the cylinder
        group fixed_atoms region fixed_cyl

        # All the rest
        group moving_atoms subtract all fixed_atoms

        # Zero force on the fixed atoms
        fix freeze fixed_atoms setforce 0.0 0.0 0.0

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