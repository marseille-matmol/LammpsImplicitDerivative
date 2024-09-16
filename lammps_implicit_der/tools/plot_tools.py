#!/usr/bin/env python3
"""
Plotting tools.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Set common figure parameters
plotparams = {'figure.figsize': (10, 8),
                  'axes.grid': False,
                  'lines.linewidth': 2.8,
                  'axes.linewidth': 1.1,
                  'lines.markersize': 10,
                  'xtick.bottom': True,
                  'xtick.top': True,
                  'xtick.direction': 'in',
                  'xtick.minor.visible': True,
                  'ytick.left': True,
                  'ytick.right': True,
                  'ytick.direction': 'in',
                  'ytick.minor.visible': True,
                  'figure.autolayout': False,
                  'mathtext.fontset': 'dejavusans', # 'cm' 'stix'
                  'mathtext.default' : 'it',
                  'xtick.major.size': 4.5,
                  'ytick.major.size': 4.5,
                  'xtick.minor.size': 2.5,
                  'ytick.minor.size': 2.5,
                  'legend.handlelength': 3.0,
                  'legend.shadow'     : False,
                  'legend.markerscale': 1.0 ,
                  'font.size': 20}


def plot_coords(axes, X_coord, color_legend=None, **kwargs):

    # X_coords shape must be (N, 3)
    if X_coord.ndim != 2:
        raise ValueError('X_coord must be 2D')
    if X_coord.shape[1] != 3:
        raise ValueError('X_coord must have shape (N, 3)')

    # Check the size of axes
    if len(axes) != 3:
        raise ValueError('axes must be a list of 3 axes')

    ax_labels = ['x', 'y', 'z']
    ax_combs = [(0, 1), (0, 2), (1, 2)]

    for i, ax in enumerate(axes):
        ax.set_aspect('equal')
        ax.scatter(X_coord[:, ax_combs[i][0]], X_coord[:, ax_combs[i][1]], **kwargs)

        ax.set_xlabel(ax_labels[ax_combs[i][0]], fontsize=20)
        ax.set_ylabel(ax_labels[ax_combs[i][1]], fontsize=20)

    if 'label' in kwargs:
        ax.legend()


def plot_coords_3D(ax, X_coord, color_legend=None, **kwargs):
    """
    Plot 3D coordinates.

    Example:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_coords_3D(ax, X_coord, color='r', label='X_coord')
    plt.show()
    """

    # X_coords shape must be (N, 3)
    if X_coord.shape[1] != 3:
        raise ValueError('X_coord must have shape (N, 3)')
    if X_coord.ndim != 2:
        raise ValueError('X_coord must be 2D')

    ax.set_aspect('equal')
    ax.scatter(X_coord[:, 0], X_coord[:, 1], X_coord[:, 2], **kwargs)

    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('z', fontsize=20)

    if 'label' in kwargs:
        ax.legend()


def setup_color_legend(ax, color_list, label_list, loc='upper right', **kwargs):
    if len(color_list) != len(label_list):
        raise ValueError('color_list and label_list must have the same length')

    if 'ls' not in kwargs:
        kwargs['ls'] = 'none'
    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'

    for i, color in enumerate(color_list):
        ax.plot([], [], color=color, label=label_list[i], **kwargs)

    ax.legend(loc=loc)