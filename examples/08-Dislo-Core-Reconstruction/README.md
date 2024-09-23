# Inverse design: dislocation core reconstruction

Initial system: easy-core W dislocation
Target system: hard-core dislocation

Parameters will be changed on only one W atom, called atom `X`.

1. Set up the parameters in the `minimize_param.yml` file.
2. Run the `run_dislo.py` script with MPI (OpenMP parallelization is also supported by LAMMPS), recommended to run on a supercomputer with >64 cores.
3. Plot the simulation results with the `plot_core_reconstruction.ipynb` notebook (requires `ase` and `matscipy` packages).

For demonstration purposes, we perform only 5 iterations of the loss minimization. To improve the convergence increase the number of iterations in the `minimize_param.yml` file:

```python
minimization:
    maxiter: 10 # or more
```
