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
from timing import measure_runtime_and_calls
from potential_tools import SNAP
from implicit_der import LammpsImplicitDer


class Dislo(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 fix_sel='moving_atoms',
                 *args, **kwargs):
        """Simple LAMMPS simulator

        Parameters
        ----------
        Ndesc: number of descriptor components
        ftol : force minimization threshold
        """

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
                                   snapparam_filename=self.snapparam_filename)

        # Get Ndesc from the pot object, maybe later use it directly from pot
        self.Ndesc = self.pot.num_param

        # Load the hard constrains file, if present
        hard_constraints_path = os.path.join(self.data_path, f'{self.pot.elmnts}_constraints.txt')
        if os.path.exists(hard_constraints_path):
            self.A_hard = np.loadtxt(hard_constraints_path)

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

        min_style cg
        #min_style fire

        #       1) energy tol 2) force tol 3) max iterations 4) max force eval.
        # minimize 0.0           0.01         10000             10000
        {'minimize 0 '+str(self.minimize_ftol)+' 1000 1000' if self.minimize else ''}
        """)

        self.compute_D_dD()

        self.gather_D_dD()

        self.lmp.commands_string(f"""
        # output of thermodynamic information every 10 steps
        thermo 10
        #write_dump all custom relaxed_easy_core.lammpstrj id type x y z
        """)

        self.run_init()


class DisloSub(LammpsImplicitDer):
    """
    Dislocaiton with substitutional defect
    """
    @measure_runtime_and_calls
    def __init__(self,
                 fix_sel='moving_atoms',
                 *args, **kwargs):
        """Simple LAMMPS simulator

        Parameters
        ----------
        Ndesc: number of descriptor components
        ftol : force minimization threshold
        """

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
                                   snapparam_filename=self.snapparam_filename)

        # Get Ndesc from the pot object, maybe later use it directly from pot
        self.Ndesc = self.pot.num_param

        # Load the hard constrains file, if present
        hard_constraints_path = os.path.join(self.data_path, f'{self.pot.elmnts}_constraints.txt')
        if os.path.exists(hard_constraints_path):
            self.A_hard = np.loadtxt(hard_constraints_path)

        # Potential parameters
        self.Theta = self.pot.Theta_dict['X']['Theta']

        self.lmp.commands_string(f"""
        clear

        units metal
        atom_modify map array sort 0 0.0

        boundary f f p

        # Read the easy core coordinates in .lmp format
        read_data {self.datafile}

        # Tungsten mass
        mass 1 183.839999952708211594654130749404
        mass 2 183.839999952708211594654130749404
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

        min_style cg
        #min_style fire

        #       1) energy tol 2) force tol 3) max iterations 4) max force eval.
        # minimize 0.0           0.01         10000             10000
        {'minimize 0 '+str(self.minimize_ftol)+' 1000 1000' if self.minimize else ''}
        """)

        self.compute_D_dD()

        self.lmp.commands_string(f"""
        # output of thermodynamic information every 10 steps
        thermo 10
        run 0
        #write_dump all custom relaxed_easy_core.lammpstrj id type x y z
        """)

        self.run_init()

        self.gather_D_dD()

    def gather_D_dD(self):
        """Compute descriptors and their derivatives in LAMMPS and store them internally, only for specie B
        """
        dU_dTheta = np.ctypeslib.as_array(self.lmp.gather("c_D", 1, self.Ndesc)).reshape((-1, self.Ndesc))

        self.dU_dTheta = dU_dTheta[self.species == 2].sum(0)

        # self.lmp.numpy.gather("c_dD", 1, 3*2*self.Ndesc)

        dD = np.ctypeslib.as_array(
                self.lmp.gather("c_dD", 1, 3*2*self.Ndesc)
            ).reshape((-1, 2, 3, self.Ndesc))

        self.mixed_hessian = dD[:, 1, :, :].reshape((-1, self.Ndesc)).T
