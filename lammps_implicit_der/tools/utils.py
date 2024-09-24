#!/usr/bin/env python3
"""
Utils for the sparse solver.
"""
import psutil
import yaml
import numpy as np
import os
from scipy.linalg import orth

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def initialize_mpi():
    """
    Initialize MPI communicator.
    """
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        #if comm.Get_size() == 1:
        #    print('WARNING: initialize_mpi() was called, but MPI communicator has only one task.')
    else:
        print('initialize_mpi() was called, but mpi4py package not found, MPI will not be used')
        comm = None
        rank = 0

    return comm, rank


def finalize_mpi():
    """
    Finalize MPI communicator.
    """
    if MPI is not None:
        MPI.Finalize()
    else:
        print('finalize_mpi() was called, but mpi4py package not found')


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
        import pynvml
    except ImportError:
        print('nvml package not found.')
        return None

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


def get_size(array, name='array', dump=True, comm=None):
    """
    Get size of an array in KB/MB/GB.

    Parameters
    ----------
    array : numpy array
        Array.

    name :
        Name of array, optional.

    dump :
        Specifies whether to print out or not the size

    Returns
    -------
    size : float
        Size of the array in KB/MB/GB.

    unit : str
        Unit of the size.
    """

    # Check if the array is a numpy array and calculate size accordingly
    if isinstance(array, np.ndarray):
        # Calculate size in bytes for numpy array
        size_bytes = array.size * array.itemsize
    else:
        # Use sys.getsizeof() for other types of arrays
        size_bytes = sys.getsizeof(array)

    size_kb = size_bytes / 1024.0

    if size_kb < 1024.0:

        size = size_kb
        unit = 'KB'

    elif size_kb / 1024.0 < 1024.0:

        size_mb = size_kb / 1024.0
        size = size_mb
        unit = 'MB'

    else:

        size_gb = size_kb / 1024.0**2
        size = size_gb
        unit = 'GB'

    if dump:
        mpi_print(f'===Size of {name:<10} {str(array.shape):<12} {str(array.dtype):<8}: {size:6.3f} {unit}', comm=comm)

    return size, unit


def get_matrix_basis():

    # Matrix-valued basis
    basis_E = np.zeros((6, 3, 3), dtype=int)

    # Cartesian basis vectors
    ortho_e1 = np.array([1, 0, 0], dtype=int)
    ortho_e2 = np.array([0, 1, 0], dtype=int)
    ortho_e3 = np.array([0, 0, 1], dtype=int)

    # Fill the matrix-valued basis
    basis_E[0] = np.outer(ortho_e1, ortho_e1)
    basis_E[1] = np.outer(ortho_e2, ortho_e2)
    basis_E[2] = np.outer(ortho_e3, ortho_e3)

    basis_E[3] = np.outer(ortho_e1, ortho_e2) + np.outer(ortho_e2, ortho_e1)
    basis_E[4] = np.outer(ortho_e2, ortho_e3) + np.outer(ortho_e3, ortho_e2)
    basis_E[5] = np.outer(ortho_e3, ortho_e1) + np.outer(ortho_e1, ortho_e3)

    print_matrices = False
    if print_matrices:
        for i in range(6):
            print(f'basis_E[{i}] = \n{basis_E[i]}')

    return basis_E


def matrix_3x3_from_Voigt(voigt_vector):
    """
    Convert Voigt vector to 3x3 matrix.

    Parameters
    ----------
    voigt_vector : np.array
        Voigt vector.

    Returns
    -------
    matrix : np.array
        3x3 matrix.
    """

    matrix = np.array([[voigt_vector[0], voigt_vector[5], voigt_vector[4]],
                       [voigt_vector[5], voigt_vector[1], voigt_vector[3]],
                       [voigt_vector[4], voigt_vector[3], voigt_vector[2]]])

    return matrix


def matrices_3x3_from_Voigt(voigt_vectors):
    """
    Convert an array of Voigt vectors to 3x3 matrices.

    Parameters
    ----------
    voigt_vectors : np.array
        Array of Voigt vectors, shape (N, 6)

    Returns
    -------
    matrices : np.array
        3x3 matrices, shape (N, 3, 3)
    """
    # Allocate array for the matrices
    matrices = np.zeros((voigt_vectors.shape[0], 3, 3), dtype=voigt_vectors.dtype)

    # Fill the diagonal
    matrices[:, 0, 0] = voigt_vectors[:, 0]
    matrices[:, 1, 1] = voigt_vectors[:, 1]
    matrices[:, 2, 2] = voigt_vectors[:, 2]

    # Fill off-diagonals
    matrices[:, 0, 1] = matrices[:, 1, 0] = voigt_vectors[:, 5]  # Vxy
    matrices[:, 0, 2] = matrices[:, 2, 0] = voigt_vectors[:, 4]  # Vxz
    matrices[:, 1, 2] = matrices[:, 2, 1] = voigt_vectors[:, 3]  # Vyz

    return matrices