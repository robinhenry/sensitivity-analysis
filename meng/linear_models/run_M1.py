import numpy as np
import matplotlib.pyplot as plt

from meng.data_loader import load_true_data, simulate_noisy_meas, load_coefficients
from meng.linear_models import utils
from meng.linear_models.core import linear_model
from meng.my_io import ExperimentLogger
from meng import noise, constants, crlb
from meng.plotting import simple as plotting, labels as plotting_labels
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run(dataset, sensor_class, coeff_type, N_exp, use_sigma, tau, freq, k_nn,
        which_i, k_pcr, qr, folder, pre_filtering, epochs):
    """
    Run `N_exp` sensitivity coefficient estimation experiments using Model 1.

    Model 1 is the linear model of the form: dx = du S + w.

    Parameters
    ----------
    dataset : str
        The unique dataset ID (see `meng.constants`).
    sensor_class : float
        The PMU measurement sensor class (see `meng.constants`).
    coeff_type : {'magn', 'real-imag'}
        Which sensitivity coefficients to estimate.
    N_exp : int
        The number of experiments to run.
    use_sigma : bool
        Whether or not to use the correlation matrix :math:`\Sigma`. If False, it
        is set to the identity matrix I.
    tau : int
        The time window size.
    freq : int
        How often (in timesteps) to estimate the coefficients).
    k_nn : int
        The number of past timesteps given as input to the neural network.
    which_i : list of int or 'all'
        Which coefficients to estimate (eg for voltage coefficients, at which
        buses). Note: a value of 1 corresponds to the 1st bus after the slack.
    k_pcr : int or None
        How many principal components to keep in PCR; set to None to skip the
        PCR step.
    qr : bool
        Set to True to solve the LS problem using Q-R decomposition.
    folder : str
        The name of the experiment folder in which the results are stored.
    pre_filtering : bool
        Whether to do pre-filtering or not.
    epochs : int
        The number of epochs during whith to train the neural network.
    """

    is_real_imag = coeff_type == 'real_imag'

    # Create the folder in which the experiment results will be stored.
    exp_logger = ExperimentLogger(folder, locals())

    # Load true load flow.
    V_org, I_org, P_org, Q_org, current_idx = load_true_data(dataset)

    # Select the noise std that corresponds to the type of coefficient(s).
    std_abs = constants.SENSOR_STD_ABS[sensor_class]
    std_arg = constants.SENSOR_STD_ANG[sensor_class]
    std_real, std_imag = noise.project(np.abs(V_org), std_abs, std_arg)
    std_a = std_real[1:] if is_real_imag else std_abs
    std_b = std_imag[1:] if is_real_imag else None

    # Pre-estimation pre-filtering step (doing nothing atm).
    pf_mask = np.ones(V_org.shape[1], dtype=np.bool)

    # Extract the list of which coefficients to estimate.
    which_i = utils.unpack_which_i(which_i, V_org.shape[0])

    # Load the true sensitivity coefficients of interest.
    coefficients = load_coefficients(dataset)
    S_a_true, S_b_true = utils.coefficients_setup(coefficients, coeff_type, which_i)
    S_a_true = {k: v[:, pf_mask] for k, v in S_a_true.items()}
    if S_b_true is not None:
        S_b_true = {k: v[:, pf_mask] for k, v in S_b_true.items()}
    del coefficients

    # Transform voltage phasor measurements into either
    # (|x|, Re{x}, or Im{x}) based on `coeff_type`.
    X_org_a, X_org_b = utils.dependent_vars_setup(V_org, coeff_type)

    # Remove slack bus measurements and create delta matrices.
    dPQ_org = np.vstack((np.diff(P_org[1:], axis=1),
                         np.diff(Q_org[1:], axis=1)))[:, pf_mask[1:]]
    dX_org_a = np.diff(X_org_a[1:], axis=1)[:, pf_mask[1:]]
    if is_real_imag:
        dX_org_b = np.diff(X_org_b[1:], axis=1)[:, pf_mask[1:]]

    # Run `N_exp` experiments (noise realizations).
    S_a_all, S_b_all, dV_est_all, dV_true_coeff_all =  {}, {}, {}, {}
    S_a_nn_all, dV_nn_all, ts_nn = {}, {}, []
    ts, cns_a_all, cns_b_all = [], [], []
    for n in range(N_exp):

        # Add noise to load flow data.
        V_meas, _, P_meas, Q_meas, _, _, _, _, _, _ = \
            simulate_noisy_meas(sensor_class, V_org, I_org, current_idx)

        # Transform voltage phasor measurements into either
        # (|x|, Re{x}, or Im{x}) based on `coeff_type`.
        X_meas_a, X_meas_b = utils.dependent_vars_setup(V_meas, coeff_type)

        # Remove slack bus measurements and create delta matrices.
        dPQ_meas = np.vstack((np.diff(P_meas[1:], axis=1),
                              np.diff(Q_meas[1:], axis=1)))[:, pf_mask[1:]]
        dX_meas_a = np.diff(X_meas_a[1:], axis=1)[:, pf_mask[1:]]
        if is_real_imag:
            dX_meas_b = np.diff(X_meas_b[1:], axis=1)[:, pf_mask[1:]]
        else:
            dX_meas_b = None

        # Pre-filtering step: check which measurements are valid.
        if pre_filtering:
            valid_timesteps = np.any(dPQ_meas > std_abs, axis=0)
        else:
            valid_timesteps = np.ones(dPQ_meas.shape[1])

        # Estimate the coefficients using linear nn_models.
        S_a, ts, cns_a = linear_model(dX_meas_a, dPQ_meas, use_sigma, tau,
                                        freq, which_i, k_pcr, qr, valid_timesteps)
        if is_real_imag:
            S_b, _, cns_b = linear_model(dX_meas_b, dPQ_meas, use_sigma, tau,
                                            freq, which_i, k_pcr, qr, valid_timesteps)
        else:
            S_b, cns_b = None, None

        # Construct estimated |X| (using estimated coefficients).
        dPQ_meas = dPQ_meas[:, ts]
        dV_est = _estimate_dVmagn(dPQ_meas, S_a, S_b, is_real_imag)

        # Construct estimated |X| (using true coefficients).
        S_a_true_ts = {k: v[:, ts] for k, v in S_a_true.items()}
        S_b_true_ts = {k: v[:, ts] for k, v in S_b_true.items()} if S_b is not None else None
        dV_true_coeff = _estimate_dVmagn(dPQ_meas, S_a_true_ts, S_b_true_ts,
                                         is_real_imag)

        # Store experiment results.
        _add_results_to_dict(S_a, S_a_all)
        # _add_results_to_dict(S_a_nn, S_a_nn_all)
        if is_real_imag:
            _add_results_to_dict(S_b, S_b_all)
        _add_results_to_dict(dV_est, dV_est_all)
        _add_results_to_dict(dV_true_coeff, dV_true_coeff_all)
        # _add_results_to_dict(dV_nn, dV_nn_all)
        cns_a_all.append(cns_a)
        if is_real_imag:
            cns_b_all.append(cns_b)

    # Compute the mean of the estimated coefficients and predicted dx.
    compute_dict_mean = lambda x: {k: np.mean(v, axis=0) for k, v in x.items()}
    S_a_mean = compute_dict_mean(S_a_all)
    S_a_nn_mean = compute_dict_mean(S_a_nn_all)
    S_b_mean = compute_dict_mean(S_b_all) if is_real_imag else None
    dV_mean = compute_dict_mean(dV_est_all)
    dV_true_coeff_mean = compute_dict_mean(dV_true_coeff_all)
    dV_nn_mean = compute_dict_mean(dV_nn_all)

    # Compute the std of the estimated coefficients and predicted dx.
    # compute_dict_std = lambda x: {k: np.std(v, axis=0) for k, v in x.items()}

    def _compute_dict_std(d):
        answer = {}
        for k, v in d.items():
            answer[k] = np.std(v, axis=0)

        return answer

    S_a_std = _compute_dict_std(S_a_all)
    S_a_nn_std = _compute_dict_std(S_a_nn_all)
    S_b_std = _compute_dict_std(S_b_all) if is_real_imag else None
    dV_std = _compute_dict_std(dV_est_all)
    dV_true_coeff_std = _compute_dict_std(dV_true_coeff_all)
    dV_nn_std = _compute_dict_std(dV_nn_all)

    # Compute the true voltage magnitude deviations (from the load flow).
    dV_load_flow = np.diff(np.abs(V_org[1:]), axis=1)
    dV_load_flow = {i: dV_load_flow[i-1, ts] for i in which_i}

    # Compute the mean of the condition numbers.
    cns_a_mean = np.mean(cns_a_all, axis=0)
    cns_b_mean = np.mean(cns_b_all, axis=0) if cns_b_all else []

    # Compute Cramer-Rao lower bound on the coefficient estimations.
    crlbs_a, crlbs_b, crlb_cns_a, crlb_cns_b = [], [], [], []
    for t in ts:

        # Compute the average std over the time window for real-imag coefficients.
        if not is_real_imag:
            std = std_a
        else:
            std = np.mean(std_a[:, t-tau: t], axis=1)
            std = np.hstack((std, std))

        H = dPQ_org[:, pf_mask[1:]][:, t-tau: t].T
        lb, cn = crlb.compute_crlb(H, std, use_sigma)
        crlbs_a.append(lb)
        crlb_cns_a.append(cn)

        if is_real_imag:
            std = np.mean(std_b[:, t-tau: t], axis=1)
            std = np.hstack((std, std))
            lb, cn = crlb.compute_crlb(H, std, use_sigma)
            crlbs_b.append(lb)
            crlb_cns_b.append(cn)

    crlbs_a = np.vstack(crlbs_a).T
    if is_real_imag:
        crlbs_b = np.vstack(crlbs_b).T

    # Keep the true coefficients for the timesteps of interest only.
    S_a_true = {k: v[:, ts] for k, v in S_a_true.items()}
    S_b_true = {k: v[:, ts] for k, v in S_b_true.items()} if is_real_imag else {}

    # Store numerical results to files.
    data = {
        'S_a_mean': S_a_mean,
        'S_a_std': S_a_std,
        'S_b_mean': S_b_mean,
        'S_b_std': S_b_std,
        'dV_mean': dV_mean,
        'dV_std': dV_std,
        'S_a_true': S_a_true,
        'S_b_true': S_b_true,
        'dV_load_flow': dV_load_flow,
        'dV_true_coeff_mean': dV_true_coeff_mean,
        'dV_true_coeff_std': dV_true_coeff_std,
        'ts': ts,
        'cns_a': cns_a_mean,
        'cns_b': cns_b_mean,
        'crlb_a': crlbs_a,
        'crlb_b': crlbs_b,
        'S_a_nn_mean': S_a_nn_mean,
        'S_a_nn_std': S_a_nn_std,
        'dV_nn_mean': dV_nn_mean,
        'dV_nn_std': dV_nn_std,
        'ts_nn': ts_nn
    }
    exp_logger.save_data(data, 'results')


    #######################################
    ############## PLOTTING ###############
    #######################################
    xlabel = 'Time (s)'

    # Select timesteps estimated by neural network to plot.
    # nn_mask = np.array(ts) - ts_nn[0]
    # nn_mask = nn_mask[nn_mask >= 0]
    # ts_nn_plot = ts_nn_plot = ts_nn[nn_mask]

    # Estimated coefficients (for |.| or Re{.}).
    fig, axs = plt.subplots(len(which_i)+1, 1, figsize=(10, 2.5*len(which_i)),
                            sharex=True)

    for ax, x_i in zip(axs[:-1], which_i):
        y_true = (S_a_true[x_i][x_i-1], np.zeros(len(ts)))
        y_est = (S_a_mean[x_i][x_i-1], S_a_std[x_i][x_i-1])
        # y_nn = (S_a_nn_mean[x_i][x_i-1][nn_mask], S_a_nn_std[x_i][x_i-1][nn_mask])

        if not is_real_imag:
            labels = [plotting_labels.magn_coeff(x_i, x_i, 'P', 'true'),
                      plotting_labels.magn_coeff(x_i, x_i, 'P', 'M1')]
                      # plotting_labels.magn_coeff(x_i, x_i, 'P', 'NN')]
        else:
            labels = [plotting_labels.real_coeff(x_i, x_i, 'P', 'true'),
                      plotting_labels.real_coeff(x_i, x_i, 'P', 'M1')]
                      # plotting_labels.real_coeff(x_i, x_i, 'P', 'NN'),]
        ylabel = 'Sens. Coeff.'

        plotting.shaded_plot([ts, ts], [y_true, y_est],
                             ylabel=ylabel, ax=ax, labels=labels)

    ax = axs[-1]
    ax.plot(ts, cns_a_mean, label='Condition number')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Condition number')

    exp_logger.save_figure(fig, 'coefficients_a')

    # Estimated coefficients (for Im{.}).
    if is_real_imag:
        fig, axs = plt.subplots(len(which_i) + 1, 1,
                                figsize=(10, 2.5 * len(which_i)), sharex=True)

        for ax, x_i in zip(axs[:-1], which_i):
            y_true = (S_b_true[x_i][x_i - 1], np.zeros(len(ts)))
            y_est = (S_b_mean[x_i][x_i - 1], S_b_std[x_i][x_i - 1])

            labels = [plotting_labels.imag_coeff(x_i, x_i, 'P', True),
                      plotting_labels.imag_coeff(x_i, x_i, 'P', False),]
            ylabel = 'Sens. Coeff.'

            plotting.shaded_plot(ts, [y_true, y_est], ylabel=ylabel,
                                           ax=ax, labels=labels)

        ax = axs[-1]
        ax.plot(ts, cns_b_mean, label='Condition number')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Condition number')

        exp_logger.save_figure(fig, 'coefficients_b')

    # Estimated d|X| (using (a) load flow, (b) true coefficients,
    # (c) estimated coefficients).
    fig, axs = plt.subplots(len(which_i)+1, 1, figsize=(10, 2.5*len(which_i)),
                            sharex=True)

    for ax, x_i in zip(axs[:-1], which_i):
        y_lf = (dV_load_flow[x_i], np.zeros(len(ts)))
        y_true_coeff = (dV_true_coeff_mean[x_i], dV_true_coeff_std[x_i])
        y_est = (dV_mean[x_i], dV_std[x_i])
        # y_nn = (dV_nn_mean[x_i][nn_mask], dV_nn_std[x_i][nn_mask])

        labels = ['Load flow', 'True SCs', 'M1', 'NN']
        ylabel = r'$\Delta|V_{%d}|$' % x_i

        plotting.shaded_plot([ts, ts, ts],
                             [y_lf, y_true_coeff, y_est],
                             labels=labels, ylabel=ylabel, ax=ax)

    ax = axs[-1]
    ax.plot(ts, cns_a_mean, label='Condition number')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Condition number')

    exp_logger.save_figure(fig, 'dV')

    # Cramer-Rao lower bound and estimation variance (for |.| or Re{.}).
    fig, axs = plt.subplots(len(which_i)+1, 1, figsize=(10, 2.5*len(which_i)),
                            sharex=True)

    for ax, x_i in zip(axs[:-1], which_i):
        lb = crlbs_a[x_i-1]
        variance = S_a_std[x_i][x_i-1] ** 2

        labels = ['CRLB', 'Variance']
        ylabel = 'Est. variance'

        plotting.single_plot(ts, [lb, variance], labels=labels, ax=ax,
                             ylabel=ylabel)

    ax = axs[-1]
    ax.plot(ts, crlb_cns_a, label='CRLB condition number')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CRLB condition number')

    exp_logger.save_figure(fig, 'crlb_a')

    # Cramer-Rao lower bound and estimation variance (for Im{.}).
    if is_real_imag:
        fig, axs = plt.subplots(len(which_i) + 1, 1,
                                figsize=(10, 2.5 * len(which_i)),
                                sharex=True)

        for ax, x_i in zip(axs[:-1], which_i):
            lb = crlbs_b[x_i - 1]
            variance = S_b_std[x_i][x_i - 1] ** 2

            labels = ['CRLB', 'Variance']
            ylabel = 'Est. variance'

            plotting.single_plot(ts, [lb, variance], labels=labels, ax=ax,
                                 ylabel=ylabel)

        ax = axs[-1]
        ax.plot(ts, crlb_cns_b, label='CRLB condition number')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('CRLB condition number')

        exp_logger.save_figure(fig, 'crlb_b')

    plt.show()

    return


def _add_results_to_dict(run_result, all_results):
    """
    From a dict of values, append those values to lists in another dict.

    Parameters
    ----------
    run_result : dict
        The results of the current experiment.
    all_results : dict
        The dict that gathers the results from all experiments.
    """
    for k, v in run_result.items():
        if k in list(all_results.keys()):
            all_results[k].append(v)
        else:
            all_results[k] = [v]


def _estimate_dVmagn(dPQ_meas, S_a, S_b, is_real_imag):
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


if __name__ == '__main__':

    tau = 1000
    config = {
        'dataset': 'cigre13',
        'sensor_class': 0.5,
        'coeff_type': 'magn',
        'N_exp': 1,
        'use_sigma': True,
        'tau': tau,
        'freq': 120,
        'k_nn': 100,
        'epochs': 1,
        'which_i': [11],
        'k_pcr': None,
        'qr': False,
        'folder': 'model1',
        'pre_filtering': False,
    }

    run(**config)
    plt.show()

    print('Done!')
