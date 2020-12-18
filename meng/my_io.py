"""
This file contains a series of functions to handle I/O operations.
"""
import os
from scipy.io import loadmat, savemat
import numpy as np
import torch

from meng import constants

def load_mat(p, var_names=None, squeeze_me=True):
    """Load data from a .mat file."""
    return loadmat(p, mat_dtype=False, squeeze_me=squeeze_me, variable_names=var_names)

def save_mat(p, data):
    """Save data to a .mat file."""
    savemat(p + '.mat', data, do_compression=True)

def save_figure(filepath, fig):
    """Save a figure with my default settings."""
    if filepath[-4:] not in ['.pdf', '.png']:
        filepath += '.png'

    fig.savefig(filepath, bbox_inches='tight')


class ExperimentLogger():

    def __init__(self, exp_name, config):
        """Create a folder for an experiment and save the experiment
        configuration."""

        # Append a number to the experiment folder to make it unique.
        folder = os.path.join(constants.EXPERIMENT_FOLDER, exp_name + '_%d')
        i = 0
        while os.path.exists(folder % i):
            i += 1
        folder = folder % i
        os.makedirs(folder)

        # Save the configuration file inside the folder.
        config_file = os.path.join(folder, 'config.txt')
        with open(config_file, 'w') as f:
            f.write(str(config))

        self.folder = folder

    def save_data(self, data, filename):

        # Convert None values to empty arrays.
        for k, v in data.items():
            if v is None:
                data[k] = np.empty(0)

        save_mat(os.path.join(self.folder, filename), data)

    def save_figure(self, fig, filename):
        save_figure(os.path.join(self.folder, filename), fig)


class NeuralNetLogger():

    def __init__(self):
        self.folder = constants.TRAINED_NETS_FOLDER

    def save_model(self, model, filename):
        path = os.path.join(self.folder, filename)
        torch.save(model, path)

    def load_model(self, filename):
        path = os.path.join(self.folder, filename)
        model = torch.load(path)

        return model


class EstimationLogger():

    def __init__(self):
        self.folder = constants.ESTIMATIONS_FOLDER

    def save_estimation(self, data, filename):
        path = os.path.join(self.folder, filename)
        save_mat(path, data)

    def load_estimation(self, filename):
        path = os.path.join(self.folder, filename)
        return load_mat(path, squeeze_me=False)


class ComparisonLogger():

    def __init__(self, folder_name):
        self.folder = os.path.join(constants.COMPARISONS_FOLDER, folder_name)
        os.makedirs(self.folder, exist_ok=True)

    def save_fig(self, fig, filename):
        path = os.path.join(self.folder, filename)
        save_figure(path, fig)


