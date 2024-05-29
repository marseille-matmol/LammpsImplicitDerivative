#!/usr/bin/env python3
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from lammps_implicit_der.systems import Bcc


def main():

    path = '/Users/imaliyov/run/potential-perturbation/vacancy_formation/new_ensemble'

    Theta_ensemble = np.loadtxt(os.path.join(path, 'new_ensemble_Thetas_no_constant.dat'))
    Theta_average = np.loadtxt(os.path.join(path, 'new_average_Theta_no_constant.dat'))
    Fisher_distance = np.loadtxt(os.path.join(path, 'new_ensemble_Fisher_Distance.dat'))

    # print all shapes
    print(Theta_ensemble.shape)
    print(Theta_average.shape)
    print(Fisher_distance.shape)

    path_pickle = '/Users/imaliyov/run/potential-perturbation/vacancy_formation/ncell_x_4_dense'
    with open(os.path.join(path_pickle, 'Theta_ens.pkl'), 'rb') as file:
        Theta_ens = pickle.load(file)

    print(Theta_ens.keys())
    print(Theta_ens['Theta_ens_list'][0].shape)

    Theta_ens_NEW = {}
    Theta_ens_NEW['Theta_ens_list'] = []
    for i in range(Theta_ensemble.shape[0]):
        Theta_ens_NEW['Theta_ens_list'].append(Theta_ensemble[i])

    Theta_ens_NEW['Theta_mean'] = Theta_average

    Theta_ens_NEW['Fisher_distance'] = Fisher_distance

    with open(os.path.join(path, 'Theta_ens_NEW.pkl'), 'wb') as file:
        pickle.dump(Theta_ens_NEW, file)

    system_tmp = Bcc(alat=4.0, ncell_x=2, minimize=False, logname=None,
                         snapcoeff_filename='W_NEW.snapcoeff', verbose=False)

    system_tmp.pot.Theta_dict['W']['Theta'] = Theta_average
    system_tmp.pot.Theta_dict['W']['beta0'] = 0.0
    system_tmp.pot.to_files(path='.', overwrite=True, snapcoeff_filename='W_Fisher.snapcoeff', snapparam_filename='W_Fisher.snapparam', verbose=True)


if __name__ == "__main__":
    main()