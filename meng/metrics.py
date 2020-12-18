import numpy as np

def normalized_error(y_true, y_estimated):
    """
    Compute the normalized errors ```(est - true) / true```.

    Parameters
    ----------
    y_true : np.ndarray
        The true values
    y_estimated : np.ndarray
        The est values.

    Returns
    -------
    np.ndarray
        The normalized errors as an array of the same shape as the inputs.
    """
    return np.abs((y_estimated - y_true) / (y_true + 1e-10))
