# Implicit derivative with LAMMPS

This folder contains scripts to compute the implicit derivative of the stationary point coordinates with respect to potential parameters. For now, only the SNAP potential is considered. Refer to this technical note for derivations: [Potential Perturbation: Technical Notes](https://www.overleaf.com/project/654a56375116627f65ceeefa).

# Structure of the code

## `LammpsImplicitDer` class, `implicit_der.py` file

This is the core parent class that contains the implicit derivative implementations and the LAMMPS simulator.
Any LAMMPS system class should inherit from this class.

### Important attributes:

* `lmp` - LAMMPS simulator
* `comm` - MPI commutator, passed as an argument for each instance
* `data_path` - path to potential and LAMMPS data files, default: `./data_files`, can be retrieved by calling `utils.get_default_data_path()`
* `snapcoeff_filename`, `snapparam_filename` - names of SNAP files
* `X_coord` - system coordinates, implemented with Python setter and getter, such that whenever `X_coord` is assigned to an array, the minimum image operation is automatically applied. Stored internally as `_X_coord`
* `energy` - getter of the energy of the system. Implemented as an attribute, every time `self.energy` is accessed, LAMMPS "thermo_pe" is called automatically, the value is stored internally as `_energy`.
* `mixed_hessian` - derivative of the potential energy with respect to the positions and potential parameters
* `pot` - SNAP potential data, instance of the [`SNAP`](#snap-class-potential_toolspy-file) class, described below

### Important methods:

* `forces()` - LAMMPS forces call, converts to `numpy` `ndarray`
* `hessian()` - Hessian of a system implemented using finite differences (`dx` default is 0.001, but should be checked and changed if needed).
* `implicit_derivative()` - implicit derivative wrapper, `method` keyword selects the derivative method, options are:

    * `inverse` - Moore-Penrose inverse of the Hessian matrix, `np.linalg.pinv()`
    * `dense` -  pseudoinverse from the system of linear equations, `np.linalg.solve(hessian, mixed_hessian)`
    * `sparse` - sparse linear method, solve for each potential parameter $\Theta_l$ separately
    * `energy` - find impl. der. from the LAMMPS minimization with additional force and energy terms that correspond to a given parameter $\Theta_l$

## `SNAP` class, `potential_tools.py` file

This class contains SNAP data for a given system, can be multielement.

### Important attributes:

* `element_list` - list of elements
* `Theta_dict` - dictionary of the potential parameters, read from the .snapcoeff file, has the following structure:
    ```
    Theta_dict[<elem name>]
                           ['radii'] - string of radii for all elements
                           ['weights'] - string of weights for all elements, useful for LAMMPS commands
                           ['elem_params']
                                          ['radius'] - per elements radius
                                          ['weight'] - per element weight
                           ['beta0'] - the beta0 coefficient (not included into the impl. der. calcs)
                           ['Theta'] - potential parameters excluding beta0
    ```
* `snapparam_dict` - dictionary of SNAP parameters read from the .snapparam file

### Important methods:

* `from_files()` - class method to initialize an instance from the .snapcoeff and .snapparam files.
    The `SNAP` class instance can be created by passing directly the parameter dicts, built elsewhere, e.g.
    ```
    snap_obj = SNAP(elem_list=..., Theta_dict=..., snapparam_dict=..., snapcoeff_path=..., snapparam_path=...)
    ```
    An easier way is to let the `SNAP` class build the dicts itself, using the `from_files()` class method like so:
    ```
    snap_obj = SNAP.from_files(snapcoeff_filename,
                               data_path=...,
                               snapparam_filename=...)
    ```
    As stated [above](#important-attributes), the `SNAP`instance is usually stored as `self.pot` of the `ImplicitDer` class.

* `to_files()` - write the .snapcoeff and .snapparam files with the current potential parameters

# Physical systems

Currently implemented systems are:

1. Bcc vacancy (W), `BccVacancy(LammpsImplicitDer)` class, `bcc_vacancy.py` file. Size of the systems can be changed with a `num_cells` parameter

2. Bcc binary vacancy (Ni Mo), `BccBinaryVacancy(LammpsImplicitDer)` class, `bcc_vacancy.py` file

3. Dislocation (W, easy core), `Dislo(LammpsImplicitDer)` class, `dislo.py` file

# Inverse design: minimization function

The machinery of inverse design is described in the [Technical Notes](https://www.overleaf.com/project/654a56375116627f65ceeefa), section II. Put simply, we start with some parameters $\Theta_0$ and stationary point positions $\mathbf{X}_{\Theta_0}$ and search new parameters $\tilde{\Theta}$ with new, target stationary positions $\mathbf{X}_{\tilde{\Theta}}$.

The minimization function, `minimize_loss()`, `error_tools.py` file, minimizes the loss of the following form:

$$
L(\Theta) = \frac{1}{2} ({\bf X}(\Theta) - {\bf X}^{\mathrm{target}})^2
$$

### Important parameters:

* `sim` - `BccVacancy`, `BccBinaryVacancy`, or `Dislo` instance, **non-perturbed**
* `X_target` - `ndarray`, target positions
* `step` - gradient descent step size, by default 0.01. If adaptive_step is `True`, step is ignored
* `adaptive_step` - Use adaptive step size, by default `True`
* `der_method` - implicit derivative method to be used for the minimization
* `der_...` - parameters for the impl. der. calculations
* `minimize_at_iters` - perform the force minimization after each parameter update

The minimization data will be written into a `minim_dict_<der_method>.pkl` file that can be used for postprocessing and plotting (useful when ran on a supercomputer).

### Running minimization

For the three systems implemented, `BccVacancy`, `BccBinaryVacancy`, and `Dislo`, there are three corresponding Python scripts for the minimization runs: `run_bcc_vacancy.py`, `run_bcc_binary_vacancy.py`, and `run_dislo.py`.

There, we create a non-perturbed system and a perturbed one, take the target positions from the perturbed system and call the minimize_loss function.

For bcc vacancy and binary vacancy, we perturb randomly the potential parameters and save them to files, this is NOT done at each run.
For the binary vacancy, only the Mo parameters are perturbed. 
In the minimization function, currently, we hardcoded to modify the Mo parameters if binary is True.

For the dislocation system, the easy and hard core configurations were prepared in advance and stored in the LAMMPS data format.

With the current settings, the systems minimize at:

* bcc vacancy: 15 iterations
* bcc binary vacancy: 72 iterations
* dislo: minimization is not efficient, error decreases very slowly

Example:

```
mpirun -n 8 ~/github/PotentialPerturbation/SparseLinearSolution/run_bcc_vacancy.py
```

Expected output:

```Running LAMMPS with the following arguments:
-screen none -log vac.log
Number of atoms: 53, largest force value: 1.998e-10, force norm: 1.088e-09
None
Running LAMMPS with the following arguments:
-screen none -log vac_perturbed.log
Number of atoms: 53, largest force value: 7.472e-10, force norm: 4.146e-09

================================================================================
=======================Running the parameter optimization=======================
================================================================================

                 Initial error: 1.497e-01

Iteration 1 / 30
Computing dX/dTheta using sparse method...
100%|██████████| 55/55 [00:01<00:00, 32.60it/s]

                     Step size: 1.222e-03

           -------------Params-------------
             Largest dX/dTheta: 4.361e+00
                Largest dTheta: 1.216e-02

           -----------Positions------------
                    Largest dX: 1.249e-01
                 Std Dev of dX: 4.173e-02

           -------------Forces-------------
           Largest force value: 2.841e+00; Norm: 9.400e+00
                        Energy: -7.2453888578e+02

           ----Forces after minimization---
           Largest force value: 1.043e-09; Norm: 8.071e-09
                        Energy: -7.2554780690e+02


           -------------Errors-------------
                 Current error: 2.807e-02
              Predicted change: 2.177e-02
                 Actual change: 1.216e-01

<15 iterations>

Number of iterations: 15
Converged: True
Final error: 0.0009807234193521077

 ======================================================================
==================== TIMING SUMMARY: minimize_loss ===================
======================================================================
                                     Tag  Time (s)     Calls
----------------------------------------------------------------------
      ▷----------------------------total  37.41231     1
                               dX_dTheta  36.23277     15
                       minimize at iters  0.97911      15
======================================================================
```

To make the minimization run faster for bcc vacancy:

1. **Delete** the `./data_files/bcc_vacancy.data` file
2. In the file `run_bcc_vacancy.py`, change `num_cells` from 3 to 2.
