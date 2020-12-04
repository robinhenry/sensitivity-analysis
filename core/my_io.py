"""
This file contains a series of functions to handle I/O operations.
"""
import os
from scipy.io import loadmat, savemat


def load_mat(p, var_names=None):
    """Load data from a .mat file."""
    return loadmat(p, mat_dtype=False, squeeze_me=True, variable_names=var_names)

def save_mat(p, data):
    """Save data to a .mat file."""
    savemat(p + '.mat', data, do_compression=True)

def save_figure(filepath, fig):
    """Save a figure with my default settings."""
    if filepath[-4:] not in ['.pdf', '.png']:
        filepath += '.png'

    fig.savefig(filepath, bbox_inches='tight')

