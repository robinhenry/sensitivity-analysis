"""
This file contains several useful functions to load and simulate load flow data.

Functions
---------
load_full_loadflow(dataset)
    Load a full load flow from a .mat file.
load_true_data(dataset, data=None)
    Load only the loadflow measurements useful from the .mat file.
simulate_noisy_meas(sensor_class, V_org, I_org, current_idx)
    Add Gaussian noise to the loadflow to simulate measurement errors.
load_coefficients(dataset)
    Load the true sensitivity coefficients.
load_voltage_exp_data(dataset, sensor_class)
    Load all data required to run a voltage sensitivity coefficient estimation
    experiment.
"""
import numpy as np

from core import my_io, constants, noise


def load_full_loadflow(dataset):
    """
    Load the full .mat load flow simulation.

    Parameters
    ----------
    dataset : str
        The load flow ID (see `core.constants`).

    Returns
    -------
    dict
        The full load flow simulation loaded from the .mat file.
    """

    filepath = constants.DATASET_PATHS[dataset]
    data = my_io.load_mat(filepath)
    return data


def load_true_data(dataset, data=None):
    """
    Load the true (noiseless) load flow simulation data.

    Parameters
    ----------
    dataset : str
        The load flow ID (see `core.constants`).
    data : dict, optional
        The full load flow simulation, as returned by `load_full_loadflow`.

    Returns
    -------
    V_org : (N+1, T) array_like
        The voltage phasor measurements.
    I_org : (M, T) array_like
        The branch current phasor measurements.
    P_org : (N+1, T) array_like
        The nodal active power injection measurements.
    Q_org : (N+1, T) array_like
        The nodal reactive power injection measurements.
    current_list : list
        A list of indices mapping branch IDs to pairs of nodes (0 indexed).
    """

    if data is None:
        # Load data from .mat file.
        filepath = constants.DATASET_PATHS[dataset]
        vars = ['S', 'E', 'IB_big', 'I_YYT_save', 'current_idx']
        data = my_io.load_mat(filepath, vars)

    current_idx = data['current_idx'] - 1
    N, T = data['E'].shape

    # Compute true noise voltage phasors.
    V_org = data['E']

    # Compute true current phasors.
    I_org_magn = data['IB_big']
    I_big_arg = []
    for t in range(T):
        xyz = np.angle(data['I_YYT_save'][:, :, t])
        I_big_arg.append(xyz.flatten('F')[current_idx])
    I_big_arg = np.array(I_big_arg).T
    I_org = I_org_magn * np.exp(1j * I_big_arg)

    # Compute true power injections.
    P_org = data['S'].real
    Q_org = data['S'].imag

    return V_org, I_org, P_org, Q_org, current_idx

def simulate_noisy_meas(sensor_class, V_org, I_org, current_idx):
    """
    Add measurement noise to a load flow simulation.

    Parameters
    ----------
    sensor_class : see `constants.SENSOR_CLASSES`
        The measurement error sensor class
    V_org : (N+1, T) array_like
        The voltage phasor measurements.
    I_org : (M, T) array_like
        The branch current phasor measurements.
    current_list : list
        A list of indices mapping branch IDs to pairs of nodes (0 indexed).

    Returns
    -------
    V_meas : (N+1, T) array_like
        The noisy voltage phasor measurements.
    I_meas : (M, T) array_like
        The noisy branch current phasor measurements.
    P_meas : (N+1, T) array_like
        The noisy nodal active power injection measurements.
    Q_meas : (N+1, T) array_like
        The noisy nodal reactive power injection measurements.
    std_abs : float
        The std of the noise w.r.t. the absolute value of V and I.
    std_ang : float
        The std of the noise w.r.t. the angle of V and I.
    std_V_real : float
        The std of the noise w.r.t. the real part of V.
    std_V_imag : float
        The std of the noise w.r.t. the imaginary part of V.
    std_I_real : float
        The std of the noise w.r.t. the real part of I.
    std_I_imag : float
        The std of the noise w.r.t. the imaginary part of I.
    """
    N, T = V_org.shape

    std_abs = constants.SENSOR_STD_ABS[sensor_class]
    std_ang = constants.SENSOR_STD_ANG[sensor_class]

    # Add noise to voltage and current phasors.
    V_meas = noise.add_noise(V_org, std_abs, std_ang)
    I_meas = noise.add_noise(I_org, std_abs, std_ang)

    # Compute noisy nodal current injections.
    I_YYT_corrupt = np.zeros((N, N, T), dtype=np.complex)
    for t in range(T):
        abc = np.zeros(N ** 2, dtype=np.complex)
        abc[current_idx] = I_meas[:, t]
        I_YYT_corrupt[:, :, t] = abc.reshape((N, N)).T
    I_node_meas = np.sum(I_YYT_corrupt, axis=1)

    # Compute noisy power injections.
    S_meas = V_meas * np.conj(I_node_meas)
    P_meas = S_meas.real
    Q_meas = S_meas.imag

    # Project std from polar to rectangular coordinates.
    std_V_real, std_V_imag = noise.project(V_org, std_abs, std_ang)
    std_I_real, std_I_imag = noise.project(I_org, std_abs, std_ang)

    return V_meas, I_meas, P_meas, Q_meas, std_abs, std_ang, std_V_real, std_V_imag, \
        std_I_real, std_I_imag


def load_coefficients(dataset):
    """
    Load the true sensitivity coefficients from the load flow simulation.

    Parameters
    ----------
    dataset : str
        The load flow ID (see `constants`).

    Returns
    -------
    dict of {str : (N, N, T) array_like}
        The sensitivity coefficients.
    """
    filepath = constants.DATASET_PATHS[dataset]

    # Load data from .mat file.
    coeff_names = ['COEFF', 'COEFFq', 'COEFFcomplex', 'COEFFqcomplex']
    data = my_io.load_mat(filepath, var_names=coeff_names)
    coefficients = {
        'vmagn_p': np.stack(data['COEFF'], -1),
        'vmagn_q': np.stack(data['COEFFq'], -1),
        'vreal_p': np.stack(data['COEFFcomplex'], -1).real,
        'vimag_p': np.stack(data['COEFFcomplex'], -1).imag,
        'vreal_q': np.stack(data['COEFFqcomplex'], -1).real,
        'vimag_q': np.stack(data['COEFFqcomplex'], -1).imag,
    }

    return coefficients


def load_voltage_exp_data(dataset, sensor_class):
    """
    Load all data associated with a V sensitivity coefficient estimation problem.

    Parameters
    ----------
    dataset : str
        The load flow ID (see `constants`).
    sensor_class : float
        The error measurement sensor class (see `constants`).

    Returns
    -------
    P_meas : (N, T) array_like
        The noisy nodal active power injection measurements.
    Q_meas : (N, T) array_like
        The noisy nodal reactive power injection measurements.
    V_meas : (N, T) array_like
        The noisy complex voltage phasor measurements.
    P_org : (N, T) array_like
        The true nodal active power injection measurements.
    Q_org : (N, T) array_like
        The true nodal reactive power injection measurements.
    V_org : (N, T) array_like
        The true complex voltage phasor measurements.
    coefficients : dict of {str : (N, N, T) array_like}
        The sensitivity coefficients.
    std_abs : float
        The std of the white Gaussian noise added to |V|.
    std_V_real : float
        The std of the white Gaussian noise added to Re{V}.
    std_V_imag : float
        The std of the white Gaussian noise added to Im{V}.
    """

    # Load measurements.
    V_org, I_org, P_org, Q_org, current_idx = load_true_data(dataset)
    V_meas, I_meas, P_meas, Q_meas, std_abs, std_ang, std_V_real, std_V_imag, _, _ = \
        simulate_noisy_meas(sensor_class, V_org, I_org, current_idx)

    # Remove slack bus measurements.
    V_org, V_meas = V_org[1:], V_meas[1:]
    P_org, P_meas = P_org[1:], P_meas[1:]
    Q_org, Q_meas = Q_org[1:], Q_meas[1:]

    # Load true coefficients.
    coefficients = load_coefficients(dataset)

    return P_meas, Q_meas, V_meas, P_org, Q_org, V_org, coefficients, std_abs, \
           std_V_real, std_V_imag


if __name__ == '__main__':
    data = load_full_loadflow('cigre4')
    print('Done!')