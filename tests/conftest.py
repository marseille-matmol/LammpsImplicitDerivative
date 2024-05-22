import pytest
from lammps_implicit_der.tools import initialize_mpi

mpi_run = initialize_mpi()[0] is not None and initialize_mpi()[0].Get_size() > 1


# Setup MPI environment
@pytest.fixture(scope="session")
def comm():
    return initialize_mpi()[0]