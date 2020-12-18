import numpy as np

def dependent_vars_setup(X, coeff_type):
    """
    Prepare dependent variable measurements for the type of SC of interest.

    Parameters
    ----------
    X : array_like
        The dependent variable measurements.
    coeff_type : {'magn', 'real_imag'}
        Which type of sensitivity coefficient is considered.

    Returns
    -------
    X_org_a : array_like
        |X| if `coeff_type` is `magn`; else it is Real{X}.
    X_org_b : array_like
        None if `coeff_type` is `magn`; else it is Imag{X}.
    """
    if coeff_type == 'magn':
        X_org_a = np.abs(X)
        X_org_b = None
    elif coeff_type == 'real_imag':
        X_org_a = np.real(X)
        X_org_b = np.imag(X)
    else:
        raise NotImplementedError()

    return X_org_a, X_org_b

def coefficients_setup(coefficients, coeff_type, which_i):

    # Load true coefficients.
    if coeff_type == 'magn':
        Sp_a = coefficients['vmagn_p']
        Sq_a = coefficients['vmagn_q']
        S_a = np.hstack((Sp_a, Sq_a))[which_i - 1]
        S_b = None

    else:
        Sp_a = coefficients['vreal_p']
        Sq_a = coefficients['vreal_q']
        S_a = np.hstack((Sp_a, Sq_a))[which_i - 1]

        Sp_b_true = coefficients['vimag_p']
        Sq_b_true = coefficients['vimag_q']
        S_b = np.hstack((Sp_b_true, Sq_b_true))[which_i - 1]

    # Transform array to a dict, indexed by the dependent variable index.
    to_dict = lambda x: {i: y for y, i in zip(x, which_i)}
    S_a = to_dict(S_a)
    S_b = to_dict(S_b) if S_b is not None else None

    return S_a, S_b


def unpack_which_i(which_i, M):
    """
    Make a list of which coefficients to estimate.

    The indices are 1-indexed, and should be in {1,...,N}, where N is the
    number of buses. Bus 0 is assumed to be the slack bus.

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
    if (isinstance(which_i, str) and which_i == 'all') or which_i is None:
        which_i = np.arange(1, M+1)
    elif isinstance(which_i, list):
        which_i = np.array(which_i)
    else:
        raise NotImplementedError()

    return which_i