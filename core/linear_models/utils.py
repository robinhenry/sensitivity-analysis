import numpy as np

def coeff_type_setup(x, coeff_type):
    """
    Prepare dependent variable measurements for the type of SC of interest.

    Parameters
    ----------
    x : array_like
        The dependent variable measurements.
    coeff_type : {'magn', 'real', 'imag'}
        Which type of sensitivity coefficient is considered.

    Returns
    -------
    array_like
        The transformed measurements.
    """
    if coeff_type == 'magn':
        return np.abs(x)
    if coeff_type == 'real':
        return np.real(x)
    if coeff_type == 'imag':
        return np.imag(x)

def unpack_which_i(which_i, M):
    """
    Make a list of which coefficients to estimate.

    Parameters
    ----------
    which_i : 'all' or array_like
        Which coefficients (ie dependent variables x_i) to estimate.
    M : int
        The total number of dependent variables.

    Returns
    -------
    array_like
        The indices of the coefficients to estimate.
    """
    if isinstance(which_i, str) and which_i == 'all':
        which_i = np.arange(0, M)
    elif isinstance(which_i, list):
        which_i = np.array(which_i)
    else:
        raise NotImplementedError()

    return which_i