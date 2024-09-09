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

    def implicit_derivative_hom_dVirial_HCP(self, delta_eps=1e-3):
        """
        Compute the homogenous implicit derivative with finite differences applied
        to the virial derivative.

        dL_dTheta_j = - Virial_j / [ (dVirial/dL) @ Theta ]
        """
        cell0 = self.cell.copy()

        self.compute_virial()
        self.gather_virial()

        virial_trace0 = np.sum(self.virial, axis=0)
        virial_trace_XY0 = np.sum(virial_trace0[:2, :], axis=0) / 2.0
        virial_trace_Z0 = virial_trace0[2, :]

        # Compute the virial derivative
        strain_matrix_XY = np.eye(3)
        strain_matrix_XY[0, 0] *= 1.0 + delta_eps
        strain_matrix_XY[1, 1] *= 1.0 + delta_eps

        strain_matrix_Z = np.eye(3)
        strain_matrix_Z[2, 2] *= 1.0 + delta_eps

        dStrain_dTheta_Nd_2 = np.zeros((self.Ndesc, 2), dtype=float)

        cell_plus_XY = cell0 @ strain_matrix_XY
        cell_plus_Z = cell0 @ strain_matrix_Z

        # === XY plane ===
        self.change_box(cell_plus_XY, update_system=True)
        # Sum over atoms
        virial_trace_plus_XY = np.sum(self.virial, axis=0)

        # two minima conditions: P along z = 0; P in XY = 0

        virial_trace_plus_XY = np.sum(virial_trace_plus_XY[:2, :], axis=0) / 2.0

        # === Z direction ===
        self.change_box(cell_plus_Z, update_system=True)
        # Sum over atoms
        virial_trace_plus_Z = np.sum(self.virial, axis=0)
        virial_trace_plus_Z = virial_trace_plus_Z[2, :]

        dVirial_dStrain_XY = (virial_trace_plus_XY - virial_trace_XY0) / delta_eps
        dVirial_dStrain_Z = (virial_trace_plus_Z - virial_trace_Z0) / delta_eps

        dStrain_dTheta_Nd_2[:, 0] = - virial_trace_XY0 / np.dot(self.Theta, dVirial_dStrain_XY)
        dStrain_dTheta_Nd_2[:, 1] = - virial_trace_Z0 / np.dot(self.Theta, dVirial_dStrain_Z)

        # Set to the original cell
        self.change_box(cell0)

        return dStrain_dTheta_Nd_2

    def compute_energy_volume_eps_a(self, epsilon_array):

        cell0 = self.cell.copy()

        energy_array = np.zeros_like(epsilon_array)
        volume_array = np.zeros_like(epsilon_array)

        for i, epsilon in enumerate(epsilon_array):
            cell_perturb = cell0.copy()
            cell_perturb[0, 0] *= 1.0 + epsilon
            cell_perturb[1, 1] *= 1.0 + epsilon

            self.change_box(cell_perturb, update_system=True)

            #energy = self.dU_dTheta @ self.Theta
            energy_array[i] = self.energy
            volume_array[i] = self.volume

        self.change_box(cell0)

        return energy_array, volume_array

    def compute_energy_volume_eps_c(self, epsilon_array):

        cell0 = self.cell.copy()

        energy_array = np.zeros_like(epsilon_array)
        volume_array = np.zeros_like(epsilon_array)

        for i, epsilon in enumerate(epsilon_array):
            cell_perturb = cell0.copy()
            cell_perturb[2, 2] *= 1.0 + epsilon

            self.change_box(cell_perturb, update_system=True)

            energy_array[i] = self.energy
            volume_array[i] = self.volume

        self.change_box(cell0)

        return energy_array, volume_array

    def compute_energy_volume_iso(self, epsilon_array):

        cell0 = self.cell.copy()

        energy_array = np.zeros_like(epsilon_array)
        volume_array = np.zeros_like(epsilon_array)

        for i, epsilon in enumerate(epsilon_array):
            cell_perturb = cell0.copy()
            cell_perturb *= 1.0 + epsilon

            self.change_box(cell_perturb, update_system=True)

            energy_array[i] = self.energy
            volume_array[i] = self.volume

        self.change_box(cell0)

        return energy_array, volume_array