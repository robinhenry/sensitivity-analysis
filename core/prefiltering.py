import numpy as np


def prefilter(x, H, std):
    """
    Pre-filter measurements to reduce numerical stability issues.

    Measurement at time t is removed if dx < 3*std for all x_i variables.

    Parameters
    ----------
    x : (M, T) array_like
        The dependent variable measurements.
    H : (N, T) array_like
        The independent variable measurements.
    std : float
        The std of the white Gaussian noise added to x.

    Returns
    -------
    x : (M, m) array_like
        The pre-filtered m (out of T) dependent variable measurements.
    H : (N, m) array_like
        The pre-filtered m (out of T) independent variable measurements.
    """

    dx = x[:, 1:] - x[:, :-1]
    mask = np.abs(dx) >= 3 * std
    mask = np.any(mask, axis=0)
    mask = np.concatenate(([True], mask))

    x = x[:, mask]
    H = H[:, mask]

    return x, H