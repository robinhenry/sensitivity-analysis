import numpy as np

from core.linear_models import utils
from core.linear_models.core import linear_model


def linear_model_1(coeff_type, x, PQ, use_sigma, tau, freq, which_i, k_pcr, qr):
    """
    Linear Model 1: :math:`\Delta y = \Delta X \beta + e`.

    Parameters
    ----------
    See `core.linear_models.linear.py`

    Returns
    -------
    See `core.linear_models.linear.py`
    """

    # Transform x into either (|x|, Re{x}, or Im{x}) based on `coeff_type`.
    x = utils.coeff_type_setup(x, coeff_type)

    # Pre-filtering step.
    # x, PQ = _prefiltering_method1(x, PQ, sigma_magn)

    # Create measurement deviation matrices.
    y = x[:, 1:] - x[:, :-1]
    X = PQ[:, 1:] - PQ[:, :-1]

    # Extract the list of which coefficients to estimate.
    which_i = utils.unpack_which_i(which_i, x.shape[0])

    # Estimate the coefficients.
    results = linear_model(y, X, use_sigma, tau, freq, which_i,
                           k_pcr, qr)
    return results


def linear_model_2(coeff_type, x, PQ, tau, freq, which_i, k_pcr, qr):
    """
    Linear Model 2: :math:`y = x_0 + X \beta + e`.

    Parameters
    ----------
    See `core.linear_models.linear.py`.

    Returns
    -------
    S_dict : see `core.linear_models.linear.py`
    ts_linear : see `core.linear_models.linear.py`
    cond_nums : see `core.linear_models.linear.py`
    x0 : dict of {int : (T',) array_like}
        The bias terms for the estimated coefficients.
    """

    # Transform x into either (|x|, Re{x}, or Im{x}) based on `coeff_type`.
    x = utils.coeff_type_setup(x, coeff_type)

    # Pre-filtering step.
    # x, PQ = _prefiltering_method1(x, PQ, sigma_magn)

    # Add row of 1s for the bias terms.
    bias = np.ones(PQ.shape[1])
    PQ = np.vstack((bias, PQ))

    # Extract the list of which coefficients to estimate.
    which_i = utils.unpack_which_i(which_i, x.shape[0])

    # Estimate the coefficients.
    use_sigma = False
    S_dict, ts, cond_nums = linear_model(x, PQ, use_sigma, tau, freq, which_i,
                                         k_pcr, qr)

    # Extract the bias terms x0 from the coefficient estimations.
    x0 = {}
    for i in which_i:
        x0[i] = S_dict[i][0]
        S_dict[i] = S_dict[i][1:]

    return S_dict, ts, cond_nums, x0