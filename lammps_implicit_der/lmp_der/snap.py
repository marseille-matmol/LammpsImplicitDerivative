#!/usr/bin/env python3
"""
SNAP potential and related classes.
"""

import os
import numpy as np
import datetime

# local imports
from ..tools.utils import mpi_print


class SNAP():

    def __init__(self,
                 elem_list,
                 Theta_dict,
                 snapparam_dict,
                 snapcoeff_path=None,
                 snapparam_path=None,
                 comm=None,
                 zbl_dict=None,
                 ):

        self.snapcoeff_path = snapcoeff_path
        self.snapparam_path = snapparam_path
        self.comm = comm

        if isinstance(elem_list, str):
            elem_list = [elem_list]

        self.elem_list = elem_list
        self.elements = ' '.join(elem_list)
        self.elmnts = ''.join(elem_list)
        self.num_el = len(elem_list)

        check_num_el_list = []
        for elem in elem_list:
            if elem not in Theta_dict:
                raise RuntimeError(f'Element {elem} not found in Theta_dict')
            num_param = Theta_dict[elem]['Theta'].shape[0]
            check_num_el_list.append(num_param)

        if len(set(check_num_el_list)) != 1:
            raise RuntimeError(f'Number of parameters for each element must be the same')

        # Number of parameters -1 (beta0)
        self.num_param = check_num_el_list[0]

        self.Theta_dict = Theta_dict

        self.snapparam_dict = snapparam_dict

        # ZBL part of potential
        # Hard coded for two elements
        self.set_zbl = False
        if zbl_dict is not None:
            self.set_zbl = True
            self.zbl_charge1 = zbl_dict['charge1']
            self.zbl_charge2 = zbl_dict['charge2']
            self.zbl_rcut1 = zbl_dict['rcut1']
            self.zbl_rcut2 = zbl_dict['rcut2']

    @classmethod
    def from_files(cls,
                   snapcoeff_filename,
                   data_path=None,
                   snapparam_filename=None,
                   zbl_dict=None,
                   comm=None,
                   num_param_fix=None,
                   ):

        if data_path is None:
            script_path = os.path.abspath(__file__)
            script_dir = os.path.dirname(script_path)
            data_path = os.path.join(script_dir, 'data_files')

        if snapcoeff_filename is None:
            raise RuntimeError('snapcoeff_filename must be provided')

        snapcoeff_path = os.path.join(data_path, snapcoeff_filename)

        if snapparam_filename is None:
            snapparam_filename = snapcoeff_filename.replace('snapcoeff', 'snapparam')

        snapparam_path = os.path.join(data_path, snapparam_filename)

        if not os.path.exists(snapcoeff_path):
            raise RuntimeError(f'File {snapcoeff_path} not found')
        if not os.path.exists(snapparam_path):
            raise RuntimeError(f'File {snapparam_path} not found')

        # Read the potential coefficients
        Theta_dict = {}

        with open(snapcoeff_path, 'r') as f:
            # Ignore all lines starting with # and all empty lines
            lines = [line.strip() for line in f
                     if line.strip() and not line.strip().startswith('#')]

            # Remove all comments after the data
            lines = [line.split('#')[0] for line in lines]

            # First line is the number of elements
            num_el, num_param = [int(x) for x in lines[0].split()]

            # Total number of parameters including beta0, to get the line index correctly
            num_param_total = num_param
            if num_param_fix is not None:
                # Parameters to read
                num_param = num_param_fix

            line_index = 1
            elem_list = []
            # Iterate over elements
            for _ in range(num_el):

                # Element name and parameters
                elem_name, R_elem, w_elem = lines[line_index].split()
                Theta_dict[elem_name] = {}
                Theta_dict[elem_name]['elem_params'] = {}
                Theta_dict[elem_name]['elem_params']['radius'] = float(R_elem)
                Theta_dict[elem_name]['elem_params']['weight'] = float(w_elem)

                elem_list.append(elem_name)
                line_index += 1

                # beta0 value
                beta0 = float(lines[line_index])
                Theta_dict[elem_name]['beta0'] = beta0
                line_index += 1

                # Theta values
                # Subtract 1 since beta0 is already read
                Theta_list = [float(lines[line_index + i]) for i in range(num_param - 1)]
                Theta_dict[elem_name]['Theta'] = np.array(Theta_list)
                line_index += num_param_total - 1

        # Store radii and weights as strings
        r_list = [Theta_dict[elem]['elem_params']['radius'] for elem in elem_list]
        Theta_dict['radii'] = ' '.join(map(str, r_list))

        w_list = [Theta_dict[elem]['elem_params']['weight'] for elem in elem_list]
        Theta_dict['weights'] = ' '.join(map(str, w_list))

        # Read the snapparam file into a dict
        # Fill with default values
        snapparam_dict = {
            'quadraticflag': 0,
        }

        with open(snapparam_path, 'r') as f:
            # Ignore all lines starting with # and all empty lines
            lines = [line.strip() for line in f
                     if line.strip() and not line.strip().startswith('#')]

            # Remove all comments after the data
            lines = [line.split('#')[0] for line in lines]

            for line in lines:
                key, value = line.split()

                # Guess the type of the value: int or float
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)

                snapparam_dict[key] = value

        return cls(elem_list=elem_list,
                   Theta_dict=Theta_dict,
                   snapparam_dict=snapparam_dict,
                   snapcoeff_path=snapcoeff_path,
                   snapparam_path=snapparam_path,
                   zbl_dict=zbl_dict,
                   comm=comm,
                   )

    def to_files(self,
                 path='./',
                 snapcoeff_filename=None,
                 snapparam_filename=None,
                 overwrite=False,
                 verbose=True,
                 ):
        """
        Save the SNAP coefficients and parameters to .snapcoeff and .snapparam files.
        """

        if snapcoeff_filename is None:
            snapcoeff_filename = f"{''.join(self.elem_list)}.snapcoeff"
        if snapparam_filename is None:
            snapparam_filename = f"{''.join(self.elem_list)}.snapparam"

        snapcoeff_path = os.path.join(path, snapcoeff_filename)
        snapparam_path = os.path.join(path, snapparam_filename)

        if os.path.exists(snapcoeff_path):
            if overwrite:
                mpi_print(f'Overwriting {snapcoeff_path}', verbose=verbose, comm=self.comm)
            else:
                raise RuntimeError(f'File {snapcoeff_path} already exists. '
                                   'To overwrite, set overwrite=True.')

        with open(snapcoeff_path, 'w') as f:
            f.write(f"# SNAP coeffs for {' '.join(self.elem_list)}\n")
            f.write(f"# Generated on {datetime.datetime.now().strftime('%B %d, %Y, %H:%M')}"
                    f" by {os.path.basename(__file__)}, lammps_implicit_der\n")
            f.write(f"\n")
            # num_param does not include beta0
            f.write(f"{self.num_el} {self.num_param+1}\n")
            for elem in self.elem_list:
                R_elem = self.Theta_dict[elem]['elem_params']['radius']
                w_elem = self.Theta_dict[elem]['elem_params']['weight']
                f.write(f"{elem} {R_elem} {w_elem}\n")
                f.write(f"{self.Theta_dict[elem]['beta0']:30.20e}\n")
                for theta in self.Theta_dict[elem]['Theta']:
                    f.write(f"{theta:30.20e}\n")

        with open(snapparam_path, 'w') as f:
            f.write(f"# SNAP parameters for {' '.join(self.elem_list)}\n")
            f.write(f"# Generated on {datetime.datetime.now().strftime('%B %d, %Y, %H:%M')}"
                    f" by {os.path.basename(__file__)}, lammps_implicit_der\n")
            f.write(f"\n")
            for key, value in self.snapparam_dict.items():
                f.write(f"{key} {value}\n")

        if verbose:
            mpi_print(f'Saved SNAP coefficients to {snapcoeff_path}', comm=self.comm)
            mpi_print(f'Saved SNAP parameters to {snapparam_path}', comm=self.comm)

    def __str__(self):
        info = '\n'

        info += f'{"SNAP coefficients for:":>40} {" ".join(self.elem_list)}\n'
        # quadraticflag
        info += f'{"quadraticflag:":>40} {self.snapparam_dict["quadraticflag"]}\n'
        # info += f'{"Number of elements:":>40} {self.num_el}\n'
        # info += f'{"Path to SNAP coefficients:":>40}\n{self.snapcoeff_path}\n'
        # info += f'{"Path to SNAP parameters:":>40}\n{self.snapparam_path}\n'
        info += f'{"Number of parameters (excluding beta0):":>40} {self.num_param}\n'

        for elem in self.elem_list:
            info += f'{"Element:":>40} {elem:>2}  |  '\
                    f'R = {self.Theta_dict[elem]["elem_params"]["radius"]:>7.4f} '\
                    f'w = {self.Theta_dict[elem]["elem_params"]["weight"]:>7.4f}\n'

        #info += self.Theta_dict['radii'] + '\n'
        #info += self.Theta_dict['weights'] + '\n'

        return info
