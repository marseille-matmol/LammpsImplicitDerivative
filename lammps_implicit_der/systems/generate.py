#!/usr/bin/env python3

import numpy as np

# local imports
from . import BccBinary


def get_perturbed_Theta_alloy(Theta1, Theta2, delta):
    """
    Get perturbed potential parameters between two elements.

    Parameters
    ----------

    Theta1 : numpy.ndarray
        SNAP potential parameters for element 1.

    Theta2 : numpy.ndarray
        SNAP potential parameters for element 2.

    delta : float
        Perturbation parameter. 0.0 <= delta <= 1.0
    """

    # delta = 0.0 -> Theta1
    return delta * Theta1 + (1.0 - delta) * Theta2


def get_bcc_alloy_A_delta_B(delta, num_cells=2, minimize=False, datafile=None, specie_B_concentration=0.5, element_A='Ni', element_B='Mo'):
    """
    Create a perturbed bcc alloy of A and B species.
    delta = 0 => A-B alloy
    delta = 1 => A-(quasi-A)

    Quasi-A implies same SNAP theta coefficies as A but SNAP parameters from B.

    Parameters
    ----------

    delta : float
        Perturbation parameter. 0.0 <= delta <= 1.0

    num_cells : int, optional
        Number of unit cells in each direction.

    minimize : bool, optional
        Whether to minimize the alloy.

    datafile : str, optional
        Path to the LAMMPS data file.

    specie_B_concentration : float, optional
        Concentration of B species in the alloy. 0.0 <= specie_B_concentration <= 1.0

    element_A : str, optional
        Element A name.

    element_B : str, optional
        Element B name.

    Returns
    -------
    bcc_alloy_A_delta_B : BccBinary
        BccBinary instance of the perturbed alloy A-delta-B.
    """

    # Create a normal bcc alloy of A and B elements from AB.snapcoeff
    # No minimization at this stage
    bcc_alloy_A_B_tmp = BccBinary(datafile=datafile,
                                  snapcoeff_filename=f'{element_A}{element_B}.snapcoeff',
                                  num_cells=num_cells,
                                  specie_B_concentration=specie_B_concentration,
                                  minimize=False)

    # A-element Theta parameters
    Theta_A = bcc_alloy_A_B_tmp.pot.Theta_dict[element_A]['Theta'].copy()

    # B-element Theta parameters
    Theta_B = bcc_alloy_A_B_tmp.pot.Theta_dict[element_B]['Theta'].copy()

    # delta = 0 => A
    Theta_perturbed = get_perturbed_Theta_alloy(Theta_A, Theta_B, delta)

    # Set the perturbed Theta parameters
    bcc_alloy_A_B_tmp.pot.Theta_dict[element_B]['Theta'] = Theta_perturbed.copy()

    delta_snapcoeff_filename = f'{element_A}_delta_{element_B}.snapcoeff'
    delta_snapparam_filename = f'{element_A}_delta_{element_B}.snapparam'

    # Save the perturbed SNAP potential
    bcc_alloy_A_B_tmp.pot.to_files(path='./',
                                   snapcoeff_filename=delta_snapcoeff_filename,
                                   snapparam_filename=delta_snapparam_filename,
                                   overwrite=True,
                                   verbose=True)

    # Setup a new instance of BccBinary with the perturbed SNAP potential
    bcc_alloy_A_delta_B = BccBinary(datafile=datafile,
                                    logname='bcc_alloy_tmp.log',
                                    minimize_algo='cg', # 'sd', 'fire', 'hftn', 'cg'
                                    data_path='./',
                                    snapcoeff_filename=delta_snapcoeff_filename,
                                    snapparam_filename=delta_snapparam_filename,
                                    num_cells=num_cells,
                                    specie_B_concentration=specie_B_concentration,
                                    minimize=minimize)

    return bcc_alloy_A_delta_B