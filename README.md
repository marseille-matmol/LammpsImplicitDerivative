# Implicit differentiation with LAMMPS

Compute the implicit derivative $\nabla_{\mathbf{\Theta}}\mathbf{X}^*$ of stationary atomic minima positions $\mathbf{X}^*$ with respect to interatomic potential parameters $\mathbf{\Theta}$ with LAMMPS.

![Implicit Diff](examples/images/implicit_diff.png)

For a detailed description of the formalism and derivations used in this project, please refer to the following paper:

[I. Maliyov, P. Grigorev, T.D. Swinburne, _Exploring parameter dependence of atomic minima with implicit differentiation_, arXiv:2407.02414](https://arxiv.org/abs/2407.02414)

# Installation

## Requirements

* Before installation of the package, install `lammps` as Python module. More details [here](https://docs.lammps.org/Python_install.html).

* Optionally, install `mpi4py` for parallel MPI runs.

## Download and Install

1. Clone the GitHub repository:
```bash
git clone git@github.com:marseille-matmol/LammpsImplicitDerivative.git
```

2. Navigate to the package folder:
```bash
cd LammpsImplicitDerivative
```

3. Install with `pip`
```bash
pip install .
```

4. Import as Python package. E.g., within a Python file:
```python
from lammps_implicit_der.systems import BCC
```

# Examples

In the `./examples` folder, you will find several Jupyter notebooks that demonstrate how to use the package to compute the inhomogeneous and homogeneous implicit derivatives in different atomic systems.
The files are arranged in the recommended order for reviewing:

| No. | Title              | Description                                                                 |
|-----|--------------------|-----------------------------------------------------------------------------|
| 01 | Getting Started     | Basic properties of the Lammps-Implicit Diff. wrapper classes (`LammpsImplicitDer`) for different systems (`BCC`, `HCP`, etc.) and basic atomic system properties (energy, force, pressure, etc.).    |
| 02 | Implicit Diff Inhom.| Inhomogeneous implicit derivative calculation - the **core functionality of the package**.|
| 03 | Implicit Diff Hom.  | Homogeneous implicit derivative calculation - the **core functionality of the package**.|
| 04 | Inverse Design      | Inverse design application of the inhomogeneous implicit derivative. |

# Tests

Running tests is recommended after installation and when new implementations are made.

Navigate to the `tests` folder and run `pytest`:
```bash
cd ./tests
pytest -s -v
```

Optionally, tests can be run with MPI:
```bash
mpirun -n 8 pytest -s -v
```

Reference values for the tests are obtained from the `./tests/tests.ipynb` notebook.
