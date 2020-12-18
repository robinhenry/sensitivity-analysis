import numpy as np


def prefilter(x, std):
    """
    Pre-filter measurements to reduce numerical stability issues.

    Measurement at time t is removed if dx < 3*std for all x_i variables.

    Parameters
    ----------
    x : (M, T) array_like
        The dependent variable measurements.
    std : float
        The std of the white Gaussian noise added to x.

    Returns
    -------
    mask : (m,) array_like
        A boolean mask that corresponds to the load flow timesteps to keep.
    """

    dx = x[:, 1:] - x[:, :-1]
    mask = np.abs(dx) >= 3 * std
    mask = np.any(mask, axis=0)
    mask = np.concatenate(([True], mask))

    return mask