import numpy as np

def lm_estimate_dVmagn(dPQ_meas, S_a, S_b, is_real_imag):
    """
    Estimate d|X| using {dP, dQ} and the sensitivity coefficients.

    Parameters
    ----------
    dPQ_meas : (2N, ts) array_like
        The power injection deviations at `ts` different timesteps.
    S_a : dict of {int : (2N, ts) array_like}
        The sensitivity coefficients estimated, indexed by their bus number, of
        either |.| or Re{.} coefficients.
    S_b : dict of {int : (2N, ts) array_like} or None
        The sensitivity coefficients estimated, indexed by their bus number, for
        Im{.}; or None if |.| coefficients are used.
    is_real_imag : bool
        True if |.| coefficients are used; False if (Re{.}, Im{.}) are used.

    Returns
    -------
    dict of {int : (ts,) array_like}
        The estimated d|X| at the different `ts` timesteps, for all buses in the
        input dicts.
    """
    dV_est = {}

    if not is_real_imag:
        for x_i, Si_a in S_a.items():
            dV_est[x_i] = np.sum(Si_a * dPQ_meas, axis=0)

    else:
        for (x_i, Si_a), (_, Si_b) in zip(S_a.items(), S_b.items()):
            dVi_real = np.sum(Si_a * dPQ_meas, axis=0)
            dVi_imag = np.sum(Si_b * dPQ_meas, axis=0)
            dV_est[x_i] = np.sqrt(dVi_real ** 2 + dVi_imag ** 2)

    return dV_est


def estimations_filename(dataset, sensor_class, x_i, N_exp):
    f = f'est_{dataset}_{sensor_class}_x{x_i}_Nexp{N_exp}'
    return f