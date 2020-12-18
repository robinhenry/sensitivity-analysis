import numpy as np
import matplotlib.pyplot as plt

from meng.data_loader import load_true_data, simulate_noisy_meas, load_coefficients
from meng.linear_models import utils
from meng.linear_models.core import linear_model
from meng.prefiltering import prefilter
from meng.my_io import ExperimentLogger
from meng import noise, constants, crlb
from meng.plotting import simple as plotting, labels as plotting_labels


def run(dataset, sensor_class, coeff_type, N_exp, tau, freq, which_i,
        k_pcr, qr, folder, pre_filtering):
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
    tau : int
        The time window size.
    freq : int
        How often (in timesteps) to estimate the coefficients).
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

    # Pre-filtering step.
    if pre_filtering:
        pf_mask = prefilter(np.abs(V_org), std_abs)
    else:
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

    # Remove slack bus measurements.
    PQ_org = np.vstack((P_org[1:], Q_org[1:]))[:, pf_mask]
    X_org_a = X_org_a[1:, pf_mask]
    if is_real_imag:
        X_org_b = X_org_b[1:, pf_mask]

    # Extract the true bias term using the true coefficients and the true power
    # injections (from the load flow).
    V0_a_org, V0_b_org = {}, {}
    for x_i, s_a in S_a_true.items():
        v0 = X_org_a[x_i-1] - np.sum(s_a * PQ_org, axis=0)
        V0_a_org[x_i] = v0
        if is_real_imag:
            s_b = S_b_true[x_i]
            v0 = X_org_b[x_i-1] - np.sum(s_b * PQ_org, axis=0)

    # Run `N_exp` experiments (noise realizations).
    S_a_all, S_b_all, V_est_all, V_true_coeff_all, V0_a_all, V0_b_all =  \
        {}, {}, {}, {}, {}, {}
    ts, cns_a_all, cns_b_all = [], [], []
    for n in range(N_exp):

        # Add noise to load flow data.
        V_meas, _, P_meas, Q_meas, _, _, _, _, _, _ = \
            simulate_noisy_meas(sensor_class, V_org, I_org, current_idx)

        # Transform voltage phasor measurements into either
        # (|x|, Re{x}, or Im{x}) based on `coeff_type`.
        X_meas_a, X_meas_b = utils.dependent_vars_setup(V_meas, coeff_type)

        # Remove slack bus measurements and add bias row (last one).
        PQ_meas = np.vstack((P_meas[1:], Q_meas[1:]))[:, pf_mask]
        PQ_meas = np.vstack((PQ_meas, np.ones(PQ_meas.shape[1])))
        X_meas_a = X_meas_a[1:, pf_mask]
        if is_real_imag:
            X_meas_b = X_meas_b[1:, pf_mask]
        else:
            dX_meas_b = None

        # Estimate the coefficients.
        use_sigma = False
        S_a, ts, cns_a = linear_model(X_meas_a, PQ_meas, use_sigma, tau,
                                        freq, which_i, k_pcr, qr)
        if is_real_imag:
            S_b, _, cns_b = linear_model(X_meas_b, PQ_meas, use_sigma, tau,
                                            freq, which_i, k_pcr, qr)
        else:
            S_b, cns_b = None, None

        # Extract the bias terms V0 from the estimated coeefficients.
        V0_a = {x_i: v[-1] for x_i, v in S_a.items()}
        V0_b = {x_i: v[-1] for x_i, v in S_b.items()} if is_real_imag else None

        S_a = {x_i: v[:-1] for x_i, v in S_a.items()}
        S_b = {x_i: v[:-1] for x_i, v in S_b.items()} if is_real_imag else None

        # Construct estimated |X| (using estimated coefficients).
        PQ_meas = PQ_meas[:-1, ts]
        V_est = _estimate_Vmagn(PQ_meas, V0_a, V0_b, S_a, S_b, is_real_imag)

        # Construct estimated |X| (using true coefficients).
        S_a_true_ts = {k: v[:, ts] for k, v in S_a_true.items()}
        S_b_true_ts = {k: v[:, ts] for k, v in S_b_true.items()} if is_real_imag else None

        V0_a_org_ts = {k: v[ts] for k, v in V0_a_org.items()}
        V0_b_org_ts = {k: v[ts] for k, v in V0_b_org.items()} if is_real_imag else None
        V_true_coeff = _estimate_Vmagn(PQ_meas, V0_a_org_ts, V0_b_org_ts,
                                       S_a_true_ts, S_b_true_ts, is_real_imag)

        # Store experiment results.
        _add_results_to_dict(S_a, S_a_all)
        if is_real_imag:
            _add_results_to_dict(S_b, S_b_all)
        _add_results_to_dict(V_est, V_est_all)
        _add_results_to_dict(V_true_coeff, V_true_coeff_all)
        cns_a_all.append(cns_a)
        if is_real_imag:
            cns_b_all.append(cns_b)

    # Compute the mean of the estimated coefficients and predicted dx.
    compute_dict_mean = lambda x: {k: np.mean(v, axis=0) for k, v in x.items()}
    S_a_mean = compute_dict_mean(S_a_all)
    S_b_mean = compute_dict_mean(S_b_all) if is_real_imag else None
    V_mean = compute_dict_mean(V_est_all)
    V_true_coeff_mean = compute_dict_mean(V_true_coeff_all)
    V0_a_mean = compute_dict_mean(V0_a_all)

    # Compute the std of the estimated coefficients and predicted dx.
    compute_dict_std = lambda x: {k: np.std(v, axis=0) for k, v in x.items()}
    S_a_std = compute_dict_std(S_a_all)
    S_b_std = compute_dict_std(S_b_all) if is_real_imag else None
    V_std = compute_dict_std(V_est_all)
    V_true_coeff_std = compute_dict_std(V_true_coeff_all)
    V0_b_std = compute_dict_std(V0_b_all)

    # Compute the true voltage magnitude deviations (from the load flow).
    V_load_flow = np.abs(V_org[1:])
    V_load_flow = {i: V_load_flow[i-1, ts] for i in which_i}

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

        H = PQ_org[:, pf_mask][:, t-tau: t].T
        lb, cn = crlb.compute_crlb(H, std)
        crlbs_a.append(lb)
        crlb_cns_a.append(cn)

        if is_real_imag:
            std = np.mean(std_b[:, t-tau: t], axis=1)
            std = np.hstack((std, std))
            lb, cn = crlb.compute_crlb(H, std)
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
        'V_mean': V_mean,
        'V_std': V_std,
        'S_a_true': S_a_true,
        'S_b_true': S_b_true,
        'V_load_flow': V_load_flow,
        'V_true_coeff_mean': V_true_coeff_mean,
        'V_true_coeff_std': V_true_coeff_std,
        'ts': ts,
        'cns_a': cns_a_mean,
        'cns_b': cns_b_mean,
        'crlb_a': crlbs_a,
        'crlb_b': crlbs_b
    }
    exp_logger.save_data(data, 'results')


    #######################################
    ############## PLOTTING ###############
    #######################################
    xlabel = 'Time (s)'

    # Estimated coefficients (for |.| or Re{.}).
    fig, axs = plt.subplots(len(which_i)+1, 1, figsize=(10, 2.5*len(which_i)),
                            sharex=True)

    for ax, x_i in zip(axs[:-1], which_i):
        y_true = (S_a_true[x_i][x_i-1], np.zeros(len(ts)))
        y_est = (S_a_mean[x_i][x_i-1], S_a_std[x_i][x_i-1])

        if not is_real_imag:
            labels = [plotting_labels.magn_coeff(x_i, x_i, 'P', True),
                      plotting_labels.magn_coeff(x_i, x_i, 'P', False),]
        else:
            labels = [plotting_labels.real_coeff(x_i, x_i, 'P', True),
                      plotting_labels.real_coeff(x_i, x_i, 'P', False),]
        ylabel = 'Sens. Coeff.'

        plotting.shaded_plot(ts, [y_true, y_est], ylabel=ylabel,
                             ax=ax, labels=labels)

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
                      plotting_labels.imag_coeff(x_i, x_i, 'P', False), ]
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
        y_lf = (V_load_flow[x_i], np.zeros(len(ts)))
        y_true_coeff = (V_true_coeff_mean[x_i], V_true_coeff_std[x_i])
        y_est = (V_mean[x_i], V_std[x_i])

        labels = ['Load flow', 'Using true SCs', 'Using est. SCs']
        ylabel = r'$|V_{%d}|$' % x_i

        plotting.shaded_plot(ts, [y_lf, y_true_coeff, y_est], labels=labels,
                             ylabel=ylabel, ax=ax)

    ax = axs[-1]
    ax.plot(ts, cns_a_mean, label='Condition number')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Condition number')

    exp_logger.save_figure(fig, 'X')

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


def _estimate_Vmagn(PQ_meas, V0_a, V0_b, S_a, S_b, is_real_imag):
    """
    Estimate |X| using {P, Q}, the sensitivity coefficients, and the bias `V0`.

    Parameters
    ----------
    PQ_meas : (2N, ts) array_like
        The power injections at `ts` different timesteps.
    V0_a : dict of {int : (ts,) array_like}
        The bias terms in the linear model of either |.| or Re{.} coefficients.
    V0_b : dict of {int : (ts,) array_like}
        The bias terms in the linear model of Im{.} coefficients; or None.
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
        The estimated |X| at the different `ts` timesteps, for all buses in the
        input dicts.
    """
    V_est = {}

    if not is_real_imag:
        for x_i, Si_a in S_a.items():
            V_est[x_i] = np.sum(Si_a * PQ_meas, axis=0) + V0_a[x_i]

    else:
        for (x_i, Si_a), (_, Si_b) in zip(S_a.items(), S_b.items()):
            Vi_real = np.sum(Si_a * PQ_meas, axis=0) + V0_a[x_i]
            Vi_imag = np.sum(Si_b * PQ_meas, axis=0) + V0_b[x_i]
            V_est[x_i] = np.sqrt(Vi_real ** 2 + Vi_imag ** 2)

    return V_est


if __name__ == '__main__':

    tau = 1000
    config = {
        'dataset': 'cigre4',
        'sensor_class': 0.,
        'coeff_type': 'magn',
        'N_exp': 1,
        'tau': tau,
        'freq': tau,
        'which_i': [1, 2, 3],
        'k_pcr': None,
        'qr': False,
        'folder': 'model2',
        'pre_filtering': False,
    }

    run(**config)
    plt.show()

    print('Done!')
