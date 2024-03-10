#!/usr/bin/env python3
"""
Utils for the sparse solver.
"""
import pynvml
import psutil
import yaml
import numpy as np
import datetime
import os
from scipy.linalg import orth


def initialize_mpi():
    """
    Initialize MPI communicator.
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ModuleNotFoundError:
        print('mpi4py not found')
        comm = None
        rank = 0

    return comm, rank


def mpi_print(*args, comm=None, verbose=True, **kwargs):
    """
    Print function that only executes on the MPI task with rank 0.
    """

    if comm is None:
        rank = 0
    else:
        rank = comm.Get_rank()

    # if flush is not in kwargs, set it to True
    if 'flush' not in kwargs:
        kwargs['flush'] = True

    if rank == 0 and verbose:
        print(*args, **kwargs)


def get_default_data_path():

    # Directory of this file
    script_path = os.path.abspath(__file__)
    utils_dir = os.path.dirname(script_path)

    # One directory up
    package_dir = os.path.dirname(utils_dir)

    data_path = os.path.join(package_dir, 'data_files')

    return data_path


def get_memory_usage(yml_fname, runmode='cpu'):

    if runmode == 'cpu':
        info_dict = get_ram_memory_usage()
    elif runmode == 'gpu':
        info_dict = get_gpu_memory_usage()
    else:
        raise RuntimeError(f'Wrong runmode {runmode}')

    # Dump memory profile to yaml
    with open(yml_fname, 'w') as file:
        yaml.dump(info_dict, file)


def get_ram_memory_usage():

    memory_info = psutil.virtual_memory()

    # Memory details
    ram_info = {
        "total_memory": memory_info.total / (1024 ** 2),    # Convert bytes to MB
        "free_memory": memory_info.available / (1024 ** 2), # Convert bytes to MB
        "used_memory": memory_info.used / (1024 ** 2),      # Convert bytes to MB
        "percentage_used": memory_info.percent
    }

    return ram_info


def get_gpu_memory_usage():

    try:
        pynvml.nvmlInit()

    except pynvml.NVMLError_LibraryNotFound:
        print('nvml report was not created. NVML Shared Library was not found on the system.')
        return None

    # Get number of available GPU devices
    device_count = pynvml.nvmlDeviceGetCount()

    memory_info = {}

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            device_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        except:
            device_name = pynvml.nvmlDeviceGetName(handle)

        memory_info[i] = {
            "name": device_name,
            "total_memory": info.total / (1024 ** 2),  # Convert bytes to MB
            "free_memory": info.free / (1024 ** 2),    # Convert bytes to MB
            "used_memory": info.used / (1024 ** 2)     # Convert bytes to MB
        }

    pynvml.nvmlShutdown()

    return memory_info


def read_snap_coeff(filename,binary=False):
    """Read SNAP coefficients from a file.

    We skip the first parameter, which is an offset.
    Therefore, we read 55 parameters out of 56.

    Parameters
    ----------
    filename : str
        Name of the file containing SNAP coefficients.

    Returns
    -------
    Theta : numpy array
        SNAP coefficients.
    """
    if not binary:
        Theta = np.loadtxt(filename, skiprows=7)
    else:
        Theta = np.loadtxt(filename, skiprows=38)
    return Theta


def save_snap_coeff(filename, Theta, binary=False):
    """Save SNAP coefficients to a file.

    We save the first parameter (an offset) fixed, therefore it is in the header.

    Parameters
    ----------
    filename : str
        Name of the file to save SNAP coefficients to.

    Theta : numpy array
    """
    if not binary:
        header = f"""# Generated on {datetime.datetime.now().strftime("%B %d, %Y  %H:%M")}
#
# LAMMPS SNAP coefficients for W

1 56
W 0.5 1
-5.306025530340759744e+00"""

        np.savetxt(filename, Theta, header=header, comments="")
    else:
        header = f"""# Generated on {datetime.datetime.now().strftime("%B %d, %Y  %H:%M")}
#
# LAMMPS SNAP coefficients for NiMo

2 31
Ni 0.575 0.5
-5.74137597148
0.00261516042413
-0.009699168626
0.0818951233671
-0.0569299158728
0.300080068305
0.0752233805991
0.147857919016
0.0221085009743
0.131302557568
0.14467283511
0.0531194735283
-0.0708752194168
-0.0386225752369
0.0230915952873
0.0643797269545
-0.00309279910187
0.055335750844
-0.101194753284
-0.0788356620704
-0.01485153828
0.0144374653104
-0.00207986530203
0.0333887815654
-0.0639521013963
-0.0306755928606
0.000464150398115
0.00714699762365
0.0199617506567
-0.0108048397609
0.00604686131228
Mo 0.575 1.0
-11.1413071988"""
    np.savetxt(filename, Theta, header=header, comments="")
    #print(f"SNAP coefficients saved to {filename}")


def get_projection(matrix_A, Ndesc):
    """
    Adopred from LML_retrain.py
    Build projection of a constraint matrix using scipy.linalg.orth.
    Is used for getting projection matrix for hard constraints.

    Parameters
    ----------
    matrix_A : np.array
        Matrix to build projection.
    size : int
        Size of the matrix.
        Default is None and the size is defined by self.N_quad_desc

    Returns
    -------
    P : np.array
        Projection matrix.
    """

    if matrix_A.shape[0] == Ndesc:
        e = orth(matrix_A)
    else:
        e = orth(matrix_A.T)

    P = e @ e.T

    return P

