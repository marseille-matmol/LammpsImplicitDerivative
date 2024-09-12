#!/usr/bin/env python3
"""
BCC systems.
Chlid classes of LammpsImplicitDer.
"""

import os
import numpy as np

# local imports
from ..tools.utils import mpi_print
from ..tools.timing import measure_runtime_and_calls
from ..lmp_der.snap import SNAP
from ..lmp_der.implicit_der import LammpsImplicitDer


class BCC(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 ncell_x=3,
                 ncell_y=None,
                 ncell_z=None,
                 alat=3.1855,
                 setup_snap=True,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.binary = False
        self.ncell_x = ncell_x
        self.alat = alat

        self.ncell_y = ncell_y if ncell_y is not None else ncell_x
        self.ncell_z = ncell_z if ncell_z is not None else ncell_x

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for BCC class')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        if len(self.pot.elem_list) > 1:
            raise RuntimeError('BCC system must be a single element')

        self.element = self.pot.elem_list[0]

        self.Theta = self.pot.Theta_dict[self.element]['Theta']

        self.lmp_commands_string(f"""
        boundary p p p
        lattice bcc {self.alat} origin 0.01 0.01 0.01
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


class BCC_BINARY(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 ncell_x=3,
                 ncell_y=None,
                 ncell_z=None,
                 alat=3.13,
                 specie_B_concentration=0.5,
                 setup_snap=True,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.binary = True

        self.ncell_x = ncell_x
        self.alat = alat

        self.ncell_y = ncell_y if ncell_y is not None else ncell_x
        self.ncell_z = ncell_z if ncell_z is not None else ncell_x

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for BCC_BINARY')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)


        # Potential parameters: ONLY MOLYBDENUM
        self.Theta = self.pot.Theta_dict['Mo']['Theta']

        self.lmp_commands_string(f"""
        boundary p p p
        lattice bcc {self.alat} origin 0.01 0.01 0.01
        """)

        # Setup the coordinates from scratch
        if self.datafile is None:

            self.lmp_commands_string(f"""
            # create a block of atoms
            region C block 0 {self.ncell_x} 0 {self.ncell_y} 0 {self.ncell_z} units lattice
            create_box 2 C
            create_atoms 1 region C

            set group all type/fraction 2 {specie_B_concentration} 12393
            """)

        # Read from a datafile
        else:
            mpi_print(f'Reading datafile {self.datafile}', verbose=self.verbose, comm=self.comm)
            self.lmp_commands_string(f"""
            read_data {self.datafile}
            """)

        self.lmp_commands_string(f'mass * 45.')

        self.run_init(setup_snap=setup_snap)


class BCC_VACANCY(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 ncell_x=3,
                 ncell_y=None,
                 ncell_z=None,
                 alat=3.1855,
                 del_coord=None,
                 id_del=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ncell_x = ncell_x
        self.alat = alat

        self.ncell_y = ncell_y if ncell_y is not None else ncell_x
        self.ncell_z = ncell_z if ncell_z is not None else ncell_x

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for BCC_VACANCY')

        if del_coord is not None and id_del is not None:
            raise RuntimeError('BCC_VACANCY: id_del and del_coord cannot be both specified')


        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        # Potential parameters, hardcoded for tungsten
        self.Theta = self.pot.Theta_dict['W']['Theta']

        self.lmp_commands_string(f"""
        boundary p p p
        lattice bcc {self.alat} origin 0.01 0.01 0.01
        """)

        self.lmp_commands_string(f"""
        # create a block of atoms
        region C block 0 {self.ncell_x} 0 {self.ncell_y} 0 {self.ncell_z} units lattice
        create_box 1 C

        # add atoms
        create_atoms 1 region C
        """)

        if del_coord is not None:
            assert len(del_coord) == 3
            X_3D = np.ctypeslib.as_array(self.lmp.gather("x", 1, 3)).reshape(-1, 3)
            id_del = np.argmin(np.linalg.norm(X_3D-del_coord, axis=1))+1
        else:
            id_del = 10

        self.id_del = id_del
        self.lmp_commands_string(f"""
        # delete one atom
        # Create a group called 'del' with the atom to be deleted
        group del id {id_del}
        delete_atoms group del
        """)

        # W mass in a.m.u.
        self.lmp_commands_string(f'mass * 184.')

        self.run_init()


class BCC_BINARY_VACANCY(LammpsImplicitDer):
    @measure_runtime_and_calls
    def __init__(self,
                 ncell_x=3,
                 ncell_y=None,
                 ncell_z=None,
                 alat=3.13,
                 custom_create_script=None,
                 specie_B_concentration=0.5,
                 id_del=10,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.binary = True
        self.ncell_x = ncell_x
        self.alat = alat

        self.ncell_y = ncell_y if ncell_y is not None else ncell_x
        self.ncell_z = ncell_z if ncell_z is not None else ncell_x

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for BCC_BINARY_VACANCY')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        # Potential parameters: ONLY MOLYBDENUM
        self.Theta = self.pot.Theta_dict['Mo']['Theta']

        self.lmp_commands_string(f"""
        boundary p p p
        lattice bcc {self.alat} origin 0.01 0.01 0.01
        """)

        # Setup the coordinates from scratch
        if self.datafile is None:

            self.lmp_commands_string(f"""
            # create a block of atoms
            region C block 0 {self.ncell_x} 0 {self.ncell_y} 0 {self.ncell_z} units lattice
            create_box 2 C
            create_atoms 1 region C
            """)
            if custom_create_script is None:
                self.lmp_commands_string(f"""
                set group all type/fraction 2 {specie_B_concentration} 12393
                # delete one atom
                # Create a group called 'del' with the atom to be deleted
                group del id {id_del}
                delete_atoms group del
                """)
            else:
                self.lmp_commands_string(custom_create_script)

        # Read from a datafile
        else:
            mpi_print(f'Reading datafile {self.datafile}', verbose=self.verbose, comm=self.comm)
            self.lmp_commands_string(f"""
            read_data {self.datafile}
            """)

        self.lmp_commands_string(f'mass * 45.')

        self.run_init()


class BCC_SIA(LammpsImplicitDer):
    """
    Self-interstitial atom in BCC lattice.
    """
    @measure_runtime_and_calls
    def __init__(self,
                 ncell_x=3,
                 ncell_y=None,
                 ncell_z=None,
                 alat=3.1855,
                 SIA_pos=None,
                 origin_pos=0.01,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ncell_x = ncell_x
        self.alat = alat

        self.ncell_y = ncell_y if ncell_y is not None else ncell_x
        self.ncell_z = ncell_z if ncell_z is not None else ncell_x

        if self.snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be specified for BCC_VACANCY')

        # Load the SNAP potential instance
        self.pot = SNAP.from_files(self.snapcoeff_filename,
                                   data_path=self.data_path,
                                   snapparam_filename=self.snapparam_filename, comm=self.comm)

        # Potential parameters, hardcoded for tungsten
        self.Theta = self.pot.Theta_dict['W']['Theta']

        self.lmp_commands_string(f"""
        boundary p p p
        lattice bcc {self.alat} origin {origin_pos} {origin_pos} {origin_pos}
        """)

        if SIA_pos is None:
            SIA_pos = 0.25 + origin_pos

        # Setup the coordinates from scratch
        if self.datafile is None:

            self.lmp_commands_string(f"""
            # create a block of atoms
            region C block 0 {self.ncell_x} 0 {self.ncell_y} 0 {self.ncell_z} units lattice
            create_box 1 C

            # add atoms
            create_atoms 1 region C

            # create a self-interstitial atom
            create_atoms 1 single {SIA_pos} {SIA_pos} {SIA_pos} units lattice
            """)

        # Read from a datafile
        else:

            self.lmp_commands_string(f"""
            read_data {self.datafile}
            """)

        # W mass in a.m.u.
        self.lmp_commands_string(f'mass * 184.')

        self.run_init()