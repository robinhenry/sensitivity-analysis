import numpy as np
import matplotlib.pyplot as plt
import os

from meng.data_loader import load_true_data, simulate_noisy_meas, load_coefficients
from meng.neural_nets.dataloaders import build_training_dataloader, build_testing_dataloader
from meng.neural_nets import core as nn
from meng.neural_nets.models import FeedForward
from meng.my_io import ExperimentLogger
from meng import metrics, crlb
from meng.linear_models.core import linear_model
from meng.methods import utils
from meng import lstm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

####################### PARAMETERS #######################

# General parameters.
dataset = 'cigre13'
sensor_class = 0.5
x_i = 11
folder = 'all_methods'
N_exp = 1

# Linear nn_models.
tau = 60 * 10
freq = 60 * 2
k_pcr = None
qr = False

# Feedforward neural network parameters.
k_nn = 100
epochs = 15
hidden_shapes = [64, 64]

# LSTM parameters.
hidden_layer_size = 100
tw_lstm = k_nn
epochs_lstm = 10
batch_size = 64
T_train = 6 * 60 * 60
T_test = 6 * 60 * 60

# Decide which experiments get ran.
# 0: magn + M1   3: magn + M2
# 1: real + M1   4: real + M2
# 2: imag + M1   5: imag + M2
# Caution: real/imag should always go together.
methods = {'NN': [3, 4, 5],
           'Linear': [0, 1, 2],
           'LSTM': [0, 1, 2]}
method_tags = methods.keys()

# Plot parameters.
fill_alpha = 0.2
x_plot = np.arange(T_train + tau, T_train + T_test, freq)

# Create the folder in which the experiment results will be stored.
exp_logger = ExperimentLogger(folder, locals())


####################### LOAD TRUE DATA #######################

# Load the true sensitivity coefficients of interest.
coefficients = load_coefficients(dataset)
for name, coeffs in coefficients.items():
    coefficients[name] = coeffs[x_i - 1, x_i - 1]

# Load true load flow.
V_org, I_org, P_org, Q_org, current_idx = load_true_data(dataset)

# Remove slack measurements and create measurement matrices used in Model 2.
V_org_magn = np.abs(V_org[1:])
V_org_re = np.real(V_org[1:])
V_org_im = np.imag(V_org[1:])
PQ_org = np.vstack((P_org[1:], Q_org[1:]))
N, T = V_org_magn.shape
PQ_org_bias = np.vstack((PQ_org, np.ones(T)))


# Create measurement delta matrices used in Model 1.
dV_org_magn = np.diff(V_org_magn, axis=1)
dV_org_re = np.diff(V_org_re, axis=1)
dV_org_im = np.diff(V_org_im, axis=1)
dPQ_org = np.diff(PQ_org, axis=1)

####################### RUN EXPERIMENTS #######################

# Initialize empty dicts and arrays to store results.
def _init_dicts(n):
    dicts = []
    for _ in range(n):
        d = {t: [] for t in method_tags}
        dicts.append(d)
    return dicts

Kp_est_all, Kq_est_all, dV_est_all = _init_dicts(3)
ts_all = {}
nn_models, lstms = [], []

std_abs, std_re, std_im = None, None, None
for iter in range(N_exp):

    ################### GENERATE NOISY DATA ###################

    # Add noise to load flow data.
    V_meas, _, P_meas, Q_meas, std_abs, std_ang, std_re, std_im, _, _ = \
        simulate_noisy_meas(sensor_class, V_org, I_org, current_idx)

    # Remove slack measurements and create measurement matrices used in Model 2.
    V_meas_magn = np.abs(V_meas[1:])
    V_meas_re = np.real(V_meas[1:])
    V_meas_im = np.imag(V_meas[1:])
    PQ_meas = np.vstack((P_meas[1:], Q_meas[1:]))
    PQ_meas_bias = np.vstack((PQ_meas, np.ones(T)))

    # Create measurement delta matrices used in Model 1.
    dV_meas_magn = np.diff(V_meas_magn, axis=1)
    dV_meas_re = np.diff(V_meas_re, axis=1)
    dV_meas_im = np.diff(V_meas_im, axis=1)
    dPQ_meas = np.diff(PQ_meas, axis=1)


    ################### SPLIT TRAINING AND TESTING DATA ###################
    test = slice(T_train, T_train + T_test)

    # Delta matrices train.
    dV_meas_magn_tr = dV_meas_magn[:, :T_train]
    dV_meas_re_tr   = dV_meas_re[:, :T_train]
    dV_meas_im_tr   = dV_meas_im[:, :T_train]
    dPQ_meas_tr     = dPQ_meas[:, :T_train]

    # Delta matrices test.
    dV_meas_magn_test = dV_meas_magn[:, test]
    dV_meas_re_test   = dV_meas_re[:, test]
    dV_meas_im_test   = dV_meas_im[:, test]
    dPQ_meas_test     = dPQ_meas[:, test]

    # Non-delta matrices train.
    V_meas_magn_tr  = V_meas_magn[:, :T_train]
    V_meas_re_tr    = V_meas_re[:, :T_train]
    V_meas_im_tr    = V_meas_im[:, :T_train]
    PQ_meas_tr      = PQ_meas[:, :T_train]
    PQ_meas_bias_tr = PQ_meas_bias[:, :T_train]

    V_meas_magn_test  = V_meas_magn[:, test]
    V_meas_re_test    = V_meas_re[:, test]
    V_meas_im_test    = V_meas_im[:, test]
    PQ_meas_test      = PQ_meas[:, test]
    PQ_meas_bias_test = PQ_meas_bias[:, test]


    ################### TRAIN FEEDFORWARD NEURAL NETWORKS ###################
    # Train only during the first run.
    if iter == 0:

        # Initialize neural nets.
        in_shape = 3 * N * k_nn
        sc_matrix_shape = 2 * N
        model_dv_magn = FeedForward(in_shape, hidden_shapes, sc_matrix_shape, k_nn)
        model_dv_re = FeedForward(in_shape, hidden_shapes, sc_matrix_shape, k_nn)
        model_dv_im = FeedForward(in_shape, hidden_shapes, sc_matrix_shape, k_nn)

        in_shape = (3 * N + 1) * k_nn
        sc_matrix_shape = 2 * N + 1
        model_v_magn = FeedForward(in_shape, hidden_shapes, sc_matrix_shape, k_nn)
        model_v_re = FeedForward(in_shape, hidden_shapes, sc_matrix_shape, k_nn)
        model_v_im = FeedForward(in_shape, hidden_shapes, sc_matrix_shape, k_nn)

        nn_models = [model_dv_magn, model_dv_re, model_dv_im,
                     model_v_magn, model_v_re, model_v_im]

        # Define training data sets.
        train_dv_magn = build_training_dataloader(dV_meas_magn_tr, dPQ_meas_tr, x_i, k_nn)
        train_dv_re   = build_training_dataloader(dV_meas_re_tr, dPQ_meas_tr, x_i, k_nn)
        train_dv_im   = build_training_dataloader(dV_meas_im_tr, dPQ_meas_tr, x_i, k_nn)

        train_v_magn  = build_training_dataloader(V_meas_magn_tr, PQ_meas_bias_tr, x_i, k_nn)
        train_v_re    = build_training_dataloader(V_meas_re_tr, PQ_meas_bias_tr, x_i, k_nn)
        train_v_im    = build_training_dataloader(V_meas_im_tr, PQ_meas_bias_tr, x_i, k_nn)

        train_datasets = [train_dv_magn, train_dv_re, train_dv_im,
                          train_v_magn, train_v_re, train_v_im]

        # Train neural networks.
        for idx, (m, data) in enumerate(zip(nn_models, train_datasets)):
            if idx in methods['NN']:
                nn.train(m, data, lr=1e-3, epochs=epochs, l2=0., plot=False)


    ################### TRAIN LSTM ###################
    if iter == 0:

        # Initialize LSTMs.
        in_shape = 3 * N
        out_shape = 2 * N
        lstm_m1 = [lstm.LSTM(in_shape, hidden_layer_size, out_shape, batch_size) for _ in range(3)]

        in_shape = 3 * N + 1
        out_shape = 2 * N + 1
        lstm_m2 = [lstm.LSTM(in_shape, hidden_layer_size, out_shape, batch_size) for _ in range(3)]

        lstms = lstm_m1 + lstm_m2

        # Define training data sets.
        train_dv_magn = lstm.build_dataloader(dV_meas_magn_tr, dPQ_meas_tr, x_i, tw_lstm, batch_size)
        train_dv_re   = lstm.build_dataloader(dV_meas_re_tr, dPQ_meas_tr, x_i, tw_lstm, batch_size)
        train_dv_im   = lstm.build_dataloader(dV_meas_im_tr, dPQ_meas_tr, x_i, tw_lstm, batch_size)

        train_v_magn  = lstm.build_dataloader(V_meas_magn_tr, PQ_meas_bias_tr, x_i, tw_lstm, batch_size)
        train_v_re    = lstm.build_dataloader(V_meas_re_tr, PQ_meas_bias_tr, x_i, tw_lstm, batch_size)
        train_v_im    = lstm.build_dataloader(V_meas_im_tr, PQ_meas_bias_tr, x_i, tw_lstm, batch_size)

        train_datasets = [train_dv_magn, train_dv_re, train_dv_im,
                          train_v_magn, train_v_re, train_v_im]

        # Train LSTMs.
        for idx, (m, data) in enumerate(zip(lstms, train_datasets)):
            if idx in methods['LSTM']:
                lstm.train(m, data, epochs_lstm, lr=1e-3, batch_size=batch_size)


    ################### TEST THE FEEDFORWARD NEURAL NETS ###################

    # Timesteps corresponding to the estimations.
    ts = np.arange(T_train + k_nn, T_train + T_test - 1)

    # Define testing data sets.
    test_dv_magn = build_testing_dataloader(dV_meas_magn_test, dPQ_meas_test, x_i, k_nn)
    test_dv_re   = build_testing_dataloader(dV_meas_re_test, dPQ_meas_test, x_i, k_nn)
    test_dv_im   = build_testing_dataloader(dV_meas_im_test, dPQ_meas_test, x_i, k_nn)

    test_v_magn  = build_testing_dataloader(V_meas_magn_test, PQ_meas_bias_test, x_i, k_nn)
    test_v_re    = build_testing_dataloader(V_meas_re_test, PQ_meas_bias_test, x_i, k_nn)
    test_v_im    = build_testing_dataloader(V_meas_im_test, PQ_meas_bias_test, x_i, k_nn)

    test_datasets = [test_dv_magn, test_dv_re, test_dv_im,
                     test_v_magn, test_v_re, test_v_im]

    # Predict coefficients and dependent variables.
    Kp_est = np.zeros((len(nn_models), T))
    Kq_est = np.zeros_like(Kp_est)
    dV_est = np.zeros_like(Kp_est)

    for idx, (model, data) in enumerate(zip(nn_models, test_datasets)):
        if idx in methods['NN']:

            S, dV, _ = model.predict(data)
            Kp_est[idx, ts] = S[x_i - 1, :len(ts)]
            Kq_est[idx, ts] = S[x_i + N - 1, :len(ts)]
            dV_est[idx, ts] = dV[:len(ts)]

    Kp_est_all['NN'].append(Kp_est)
    Kq_est_all['NN'].append(Kq_est)
    dV_est_all['NN'].append(dV_est)
    ts_all['NN'] = ts


    ################### TEST THE LSTM ###################

    # Timesteps corresponding to the estimations.
    ts = np.arange(T_train + tw_lstm, T_train + T_test - 1)

    # Build testing data sets.
    test_dv_magn = lstm.build_dataloader(dV_meas_magn_test, dPQ_meas_test, x_i, tw_lstm, batch_size)
    test_dv_re = lstm.build_dataloader(dV_meas_re_test, dPQ_meas_test, x_i, tw_lstm, batch_size)
    test_dv_im = lstm.build_dataloader(dV_meas_im_test, dPQ_meas_test, x_i, tw_lstm, batch_size)

    test_v_magn = lstm.build_dataloader(V_meas_magn_test, PQ_meas_bias_test, x_i, tw_lstm, batch_size)
    test_v_re = lstm.build_dataloader(V_meas_re_test, PQ_meas_bias_test, x_i, tw_lstm, batch_size)
    test_v_im = lstm.build_dataloader(V_meas_im_test, PQ_meas_bias_test, x_i, tw_lstm, batch_size)

    test_datasets = [test_dv_magn, test_dv_re, test_dv_im,
                     test_v_magn, test_v_re, test_v_im]

    # Predict coefficients and dependent variables.
    Kp_est = np.zeros((len(lstms), T))
    Kq_est = np.zeros_like(Kp_est)
    dV_est = np.zeros_like(Kp_est)

    for idx, (model, data) in enumerate(zip(lstms, test_datasets)):
        if idx in methods['LSTM']:
            S, dV, _ = lstm.predict(model, data, batch_size)
            Kp_est[idx, ts] = S[x_i - 1, :len(ts)]
            Kq_est[idx, ts] = S[x_i + N - 1, :len(ts)]
            dV_est[idx, ts] = dV[:len(ts)]

    Kp_est_all['LSTM'].append(Kp_est)
    Kq_est_all['LSTM'].append(Kq_est)
    dV_est_all['LSTM'].append(dV_est)
    ts_all['LSTM'] = ts


    ################### LINEAR MODEL 1 ###################
    Kp_est = np.zeros((len(nn_models), T))
    Kq_est = np.zeros_like(Kp_est)
    dV_est = np.zeros((len(nn_models), T))

    valid_timesteps = np.ones(T-1).astype(np.bool)
    use_sigma = True
    ts_lm = np.arange(0, T_test)
    which_i = np.array([x_i])

    linear_methods = methods['Linear']
    if 0 in linear_methods:
        S_m1_magn, ts_lm, _ = linear_model(dV_meas_magn_test, dPQ_meas_test, use_sigma, tau, freq,
                                           which_i, k_pcr, qr, valid_timesteps)
        ts_lm += T_train
        Kp_est[0, ts_lm] = S_m1_magn[x_i][x_i - 1]
        Kq_est[0, ts_lm] = S_m1_magn[x_i][x_i - 1 + N]
        dV_est[0, ts_lm] = utils.lm_estimate_dVmagn(dPQ_meas[:, ts_lm], S_m1_magn, None, False)[x_i]

    if 1 in linear_methods:
        S_m1_re, ts_lm, _ = linear_model(dV_meas_re_test, dPQ_meas_test, use_sigma, tau, freq,
                                         which_i, k_pcr, qr, valid_timesteps)
        ts_lm += T_train
        Kp_est[1, ts_lm] = S_m1_re[x_i][x_i - 1]
        Kq_est[1, ts_lm] = S_m1_re[x_i][x_i - 1 + N]
        dV_est[1, ts_lm] = utils.lm_estimate_dVmagn(dPQ_meas[:, ts_lm], S_m1_re, None, False)[x_i]

    if 2 in linear_methods:
        S_m1_im, ts_lm, _ = linear_model(dV_meas_im_test, dPQ_meas_test, use_sigma, tau, freq,
                                         which_i, k_pcr, qr, valid_timesteps)
        ts_lm += T_train
        Kp_est[2, ts_lm] = S_m1_im[x_i][x_i - 1]
        Kq_est[2, ts_lm] = S_m1_im[x_i][x_i - 1 + N]
        dV_est[2, ts_lm] = utils.lm_estimate_dVmagn(dPQ_meas[:, ts_lm], S_m1_im, None, False)[x_i]


    ################### LINEAR MODEL 2 (BIAS) ###################
    valid_timesteps = np.ones(T).astype(np.bool)
    use_sigma = False
    which_i = np.array([x_i])

    if 3 in linear_methods:
        S_m2_magn, ts_lm, _ = linear_model(V_meas_magn_test, PQ_meas_bias_test, use_sigma, tau, freq,
                                           which_i, k_pcr, qr, valid_timesteps)
        ts_lm += T_train
        Kp_est[3, ts_lm] = S_m2_magn[x_i][x_i - 1]
        Kq_est[3, ts_lm] = S_m2_magn[x_i][x_i - 1 + N]
        dV_est[3, ts_lm] = utils.lm_estimate_dVmagn(PQ_meas_bias[:, ts_lm], S_m2_magn, None, False)[x_i]

    if 4 in linear_methods:
        S_m2_re, ts_lm, _ = linear_model(V_meas_re_test, PQ_meas_bias_test, use_sigma, tau, freq,
                                         which_i, k_pcr, qr, valid_timesteps)
        ts_lm += T_train
        Kp_est[4, ts_lm] = S_m2_re[x_i][x_i - 1]
        Kq_est[4, ts_lm] = S_m2_re[x_i][x_i - 1 + N]
        dV_est[4, ts_lm] = utils.lm_estimate_dVmagn(PQ_meas_bias[:, ts_lm], S_m2_re, None, False)[x_i]

    if 5 in linear_methods:
        S_m2_im, ts_lm, _ = linear_model(V_meas_im_test, PQ_meas_bias_test, use_sigma, tau, freq,
                                         which_i, k_pcr, qr, valid_timesteps)
        ts_lm += T_train
        Kp_est[5, ts_lm] = S_m2_im[x_i][x_i - 1]
        Kq_est[5, ts_lm] = S_m2_im[x_i][x_i - 1 + N]
        dV_est[5, ts_lm] = utils.lm_estimate_dVmagn(PQ_meas_bias[:, ts_lm], S_m2_im, None, False)[x_i]

    ################### LINEAR MODEL 2 (NO BIAS) ###################

    ################### STORE LINEAR MODEL ESTIMATIONS ###################
    Kp_est_all['Linear'].append(Kp_est)
    Kq_est_all['Linear'].append(Kq_est)
    ts_all['Linear'] = ts_lm
    dV_est_all['Linear'].append(dV_est)


####################### AVERAGE RESULTS OVER EXPERIMENTS #######################

# Compute mean and std over all experiments.
def _process_results(results):
    mean, std = {}, {}
    for name, c in results.items():
        mean[name] = np.mean(c, axis=0)
        std[name] = np.std(c, axis=0)
        results[name] = np.stack(c, axis=0)

    return mean, std

Kp_est_mean, Kp_est_std = _process_results(Kp_est_all)
Kq_est_mean, Kq_est_std = _process_results(Kq_est_all)


####################### RECONSTRUCT d|V| or |V| #######################

# Reconstruct estimated d|X| (Model 1) or |X| (Model 2).
dV_est_mean, dV_est_std = {}, {}
V_est_mean, V_est_std = {}, {}

for name, v in dV_est_all.items():

    dV_mean = np.zeros((2, T-1))
    dV_std = np.zeros_like(dV_mean)
    V_mean = np.zeros((2, T))
    V_std = np.zeros_like(V_mean)

    # Stack results from different experiments.
    dV_est_all[name] = np.stack(v, axis=0)

    # Model 1 d|X|.
    dV_mean[0] = np.mean(dV_est_all[name][:, 0], axis=0)[:-1]
    dV_std[0] = np.std(dV_est_all[name][:, 0], axis=0)[:-1]

    # Model 1 dRe{X}, dIm{X}.
    dV = np.sqrt(dV_est_all[name][:, 1] ** 2 + dV_est_all[name][:, 2] ** 2)
    dV_mean[1] = np.mean(dV, axis=0)[:-1]
    dV_std[1] = np.std(dV, axis=0)[:-1]

    # Model 2 |X|.
    V_mean[0] = np.mean(dV_est_all[name][:, 3], axis=0)
    V_std[0] = np.std(dV_est_all[name][:, 3], axis=0)

    # Model 2 Re{X}, Im{X}.
    V = np.sqrt(dV_est_all[name][:, 4] ** 2 + dV_est_all[name][:, 5] ** 2)
    V_mean[1] = np.mean(V, axis=0)
    V_std[1] = np.std(V, axis=0)

    dV_est_mean[name] = dV_mean
    dV_est_std[name] = dV_std
    V_est_mean[name] = V_mean
    V_est_std[name] = V_std


####################### NORMALIZED ERRORS #######################

Kp_errors, Kq_errors, dV_errors, V_errors = {}, {}, {}, {}

# K_P coefficient error.
for name, c in Kp_est_mean.items():
    errors = np.zeros_like(c)

    # |X|/P
    y_true = coefficients['vmagn_p']
    errors[0] = metrics.normalized_error(y_true, c[0])
    errors[3] = metrics.normalized_error(y_true, c[3])

    # Re{X}/P
    y_true = coefficients['vreal_p']
    errors[1] = metrics.normalized_error(y_true, c[1])
    errors[4] = metrics.normalized_error(y_true, c[4])

    # Im{X}/P
    y_true = coefficients['vimag_p']
    errors[2] = metrics.normalized_error(y_true, c[2])
    errors[5] = metrics.normalized_error(y_true, c[5])

    Kp_errors[name] = errors

# K_Q coefficient error.
for name, c in Kq_est_mean.items():
    errors = np.zeros_like(c)

    # |X|/Q
    y_true = coefficients['vmagn_q']
    errors[0] = metrics.normalized_error(y_true, c[0])
    errors[3] = metrics.normalized_error(y_true, c[3])

    # Re{X}/Q
    y_true = coefficients['vreal_q']
    errors[1] = metrics.normalized_error(y_true, c[1])
    errors[4] = metrics.normalized_error(y_true, c[4])

    # Im{X}/Q
    y_true = coefficients['vimag_q']
    errors[2] = metrics.normalized_error(y_true, c[2])
    errors[5] = metrics.normalized_error(y_true, c[5])

    Kq_errors[name] = errors

# d|X| error (Model 1)
for name, dv in dV_est_mean.items():
    errors = np.zeros_like(dv)

    y_true = np.abs(dV_org_magn[x_i - 1])
    errors[0] = metrics.normalized_error(y_true, np.abs(dv[0]))
    errors[1] = metrics.normalized_error(y_true, np.abs(dv[1]))

    dV_errors[name] = errors

# |X| error (Model 2)
for name, v in V_est_mean.items():
    errors = np.zeros_like(v)

    y_true = V_org_magn[x_i - 1]
    errors[0] = metrics.normalized_error(y_true, v[0])
    errors[1] = metrics.normalized_error(y_true, v[1])

    V_errors[name] = errors


####################### CRAMER-RAO LOWER BOUND #######################
ts = ts_all['Linear']

### Lower bounds for Model 1.
crlbs_P = np.zeros((6, T))
crlbs_Q = np.zeros_like(crlbs_P)

for t in ts:

    # Model 1.
    H = dPQ_org[:, t-tau: t].T
    use_sigma = True
    lb_magn, _ = crlb.compute_crlb(H, std_abs, use_sigma)
    lb_re, _ = crlb.compute_crlb(H, np.mean(std_re), use_sigma)
    lb_im, _ = crlb.compute_crlb(H, np.mean(std_im), use_sigma)

    crlbs_P[0, t] = lb_magn[x_i - 1]
    crlbs_P[1, t] = lb_re[x_i - 1]
    crlbs_P[2, t] = lb_im[x_i - 1]

    crlbs_Q[0, t] = lb_magn[x_i - 1 + N]
    crlbs_Q[1, t] = lb_re[x_i - 1 + N]
    crlbs_Q[2, t] = lb_im[x_i - 1 + N]

    # Model 2.
    H = PQ_org_bias[:, t-tau: t].T
    use_sigma = False

    lb_magn, _ = crlb.compute_crlb(H, std_abs, use_sigma)
    lb_re, _ = crlb.compute_crlb(H, np.mean(std_re), use_sigma)
    lb_im, _ = crlb.compute_crlb(H, np.mean(std_im), use_sigma)

    crlbs_P[3, t] = lb_magn[x_i - 1]
    crlbs_P[4, t] = lb_re[x_i - 1]
    crlbs_P[5, t] = lb_im[x_i - 1]

    crlbs_Q[3, t] = lb_magn[x_i - 1 + N]
    crlbs_Q[4, t] = lb_re[x_i - 1 + N]
    crlbs_Q[5, t] = lb_im[x_i - 1 + N]


####################### PLOTTING SETUP #######################

# Which timesteps to plot and the corresponding indices.
x = x_plot

# Which timesteps to consider in the boxplots.
x_boxplot = x_plot


####################### PLOT ESTIMATED K_P #######################
n_plots = 3
fig_P, axs_P = plt.subplots(n_plots, 1, figsize=(10, 2.5*n_plots), sharex=True)

### True values.
axs_P[0].plot(x, coefficients['vmagn_p'][x], label='True')
axs_P[1].plot(x, coefficients['vreal_p'][x], label='True')
axs_P[2].plot(x, coefficients['vimag_p'][x], label='True')

### Estimated values.
for name, which_methods in methods.items():

    # |X| Model 1.
    if 0 in which_methods:
        mu = Kp_est_mean[name][0][x]
        std = Kp_est_std[name][0][x]
        label = name + ' (M1)'
        axs_P[0].plot(x, mu, label=label)
        axs_P[0].fill_between(x, mu-std, mu+std, alpha=fill_alpha)

    # Re{X} Model 1.
    if 1 in which_methods:
        mu = Kp_est_mean[name][1][x]
        std = Kp_est_std[name][1][x]
        label = name + ' (M1)'
        axs_P[1].plot(x, mu, label=label)
        axs_P[1].fill_between(x, mu-std, mu+std, alpha=fill_alpha)

    # Im{X} Model 1.
    if 2 in which_methods:
        mu = Kp_est_mean[name][2][x]
        std = Kp_est_std[name][2][x]
        label = name + ' (M1)'
        axs_P[2].plot(x, mu, label=label)
        axs_P[2].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

    # |X| Model 2 with bias.
    if 3 in which_methods:
        mu = Kp_est_mean[name][3][x]
        std = Kp_est_std[name][3][x]
        label = name + ' (M2 + bias)'
        axs_P[0].plot(x, mu, label=label)
        axs_P[0].fill_between(x, mu-std, mu+std, alpha=fill_alpha)

    # Re{X} Model 2 with bias.
    if 4 in which_methods:
        mu = Kp_est_mean[name][4][x]
        std = Kp_est_std[name][4][x]
        label = name + ' (M2 + bias)'
        axs_P[1].plot(x, mu, label=label)
        axs_P[1].fill_between(x, mu-std, mu+std, alpha=fill_alpha)

    # Im{X} Model 2 with bias.
    if 5 in which_methods:
        mu = Kp_est_mean[name][5][x]
        std = Kp_est_std[name][5][x]
        label = name + ' (M2 + bias)'
        axs_P[2].plot(x, mu, label=label)
        axs_P[2].fill_between(x, mu-std, mu+std, alpha=fill_alpha)

### Add labels.
axs_P[0].set_ylabel(r'$\partial |V_{%d}| / \partial P_{%d}$' % (x_i, x_i))
axs_P[1].set_ylabel(r'$\partial Re\{V_{%d}\} / \partial P_{%d}$' % (x_i, x_i))
axs_P[2].set_ylabel(r'$\partial Im\{V_{%d}\} / \partial P_{%d}$' % (x_i, x_i))
axs_P[-1].set_xlabel('Time step (s)')
axs_P[1].legend(loc='upper right')

exp_logger.save_figure(fig_P, f'KP_{dataset}_{sensor_class}.pdf')
plt.show()


####################### PLOT ESTIMATED K_Q #######################
n_plots = 3
fig_Q, axs_Q = plt.subplots(n_plots, 1, figsize=(10, 2.5*n_plots), sharex=True)

### True values.
axs_Q[0].plot(x, coefficients['vmagn_q'][x], label='True')
axs_Q[1].plot(x, coefficients['vreal_q'][x], label='True')
axs_Q[2].plot(x, coefficients['vimag_q'][x], label='True')

### Estimated values.
for name, which_methods in methods.items():


    # |X| Model 1.
    if 0 in which_methods:
        mu = Kq_est_mean[name][0][x]
        std = Kq_est_std[name][0][x]
        label = name + ' (M1)'
        axs_Q[0].plot(x, mu, label=label)
        axs_Q[0].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

    # Re{X} Model 1.
    if 1 in which_methods:
        mu = Kq_est_mean[name][1][x]
        std = Kq_est_std[name][1][x]
        label = name + ' (M1)'
        axs_Q[1].plot(x, mu, label=label)
        axs_Q[1].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

    # Im{X} Model 1.
    if 2 in which_methods:
        mu = Kq_est_mean[name][2][x]
        std = Kq_est_std[name][2][x]
        label = name + ' (M1)'
        axs_Q[2].plot(x, mu, label=label)
        axs_Q[2].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

    # |X| Model 2 with bias.
    if 3 in which_methods:
        mu = Kq_est_mean[name][3][x]
        std = Kq_est_std[name][3][x]
        label = name + ' (M2 + bias)'
        axs_Q[0].plot(x, mu, label=label)
        axs_Q[0].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

    # Re{X} Model 2 with bias.
    if 4 in which_methods:
        mu = Kq_est_mean[name][4][x]
        std = Kq_est_std[name][4][x]
        label = name + ' (M2 + bias)'
        axs_Q[1].plot(x, mu, label=label)
        axs_Q[1].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

    # Im{X} Model 2 with bias.
    if 5 in which_methods:
        mu = Kq_est_mean[name][5][x]
        std = Kq_est_std[name][5][x]
        label = name + ' (M2 + bias)'
        axs_Q[2].plot(x, mu, label=label)
        axs_Q[2].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

### Add labels.
axs_Q[0].set_ylabel(r'$\partial |V_{%d}| / \partial Q_{%d}$' % (x_i, x_i))
axs_Q[1].set_ylabel(r'$\partial Re\{V_{%d}\} / \partial Q_{%d}$' % (x_i, x_i))
axs_Q[2].set_ylabel(r'$\partial Im\{V_{%d}\} / \partial Q_{%d}$' % (x_i, x_i))
axs_Q[-1].set_xlabel('Time step (s)')
axs_Q[1].legend(loc='upper right')

exp_logger.save_figure(fig_Q, f'KQ_{dataset}_{sensor_class}.pdf')
plt.show()


####################### PLOT ESTIMATED dV AND X #######################
n_plots = 2
fig_V, ax_V = plt.subplots(n_plots, 1, figsize=(10, 2.5*n_plots), sharex=True)

# True values.
ax_V[0].plot(x, np.abs(dV_org_magn[x_i-1][x]), label='True')
ax_V[1].plot(x, V_org_magn[x_i-1][x], label='True')

# Estimated d|X| values.
for name, which_methods in methods.items():

    # Model 1 d|X|.
    if 0 in which_methods:
        mu = np.abs(dV_est_mean[name][0][x])
        std = dV_est_std[name][0][x]
        label = name + ' (|X|)'
        ax_V[0].plot(x, mu, label=label)
        ax_V[0].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

    # Model 1 dRe{X}, dIm{X}.
    if 1 in which_methods:
        mu = np.abs(dV_est_mean[name][1][x])
        std = dV_est_std[name][1][x]
        label = name + ' (Re/Im)'
        ax_V[0].plot(x, mu, label=label)
        ax_V[0].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

# Estimated |X| values.
for name, which_methods in methods.items():

    # Model 2 |X|.
    if 3 in which_methods:
        mu = V_est_mean[name][0][x]
        std = V_est_std[name][0][x]
        label = name + ' (|X|)'
        ax_V[1].plot(x, mu, label=label)
        ax_V[1].fill_between(x, mu + std, mu - std, alpha=fill_alpha)

    # Model 2 Re{X}, Im{X}.
    if 4 in which_methods:
        mu = V_est_mean[name][1][x]
        std = V_est_std[name][1][x]
        label = name + ' (|X|)'
        ax_V[1].plot(x, mu, label=label)
        ax_V[1].fill_between(x, mu - std, mu + std, alpha=fill_alpha)

ax_V[0].set_ylabel(r'$\Delta |V_{%d}|$' % x_i)
ax_V[1].set_ylabel(r'$|V_{%d}|$' % x_i)
ax_V[0].legend(loc='upper right')
ax_V[-1].set_xlabel('Time step (s)')

exp_logger.save_figure(fig_V, f'V_{dataset}_{sensor_class}.pdf')
plt.show()


####################### PLOT CRLB W.R.T. P #######################
fig_crlb_P, axs_crlb_P = plt.subplots(3, 2, figsize=(12, 6), sharex=True)

# Plot CRLBs.
axs_crlb_P[0, 0].plot(x_plot, crlbs_P[0][x_plot], label='CRLB')
axs_crlb_P[1, 0].plot(x_plot, crlbs_P[1][x_plot], label='CRLB')
axs_crlb_P[2, 0].plot(x_plot, crlbs_P[2][x_plot], label='CRLB')
axs_crlb_P[0, 1].plot(x_plot, crlbs_P[3][x_plot], label='CRLB')
axs_crlb_P[1, 1].plot(x_plot, crlbs_P[4][x_plot], label= 'CRLB')
axs_crlb_P[2, 1].plot(x_plot, crlbs_P[5][x_plot], label='CRLB')

for name, std in Kp_est_std.items():
    for i in range(6):
        if i in methods[name]:
            ax = axs_crlb_P[i % 3, i // 3]
            ax.plot(x_plot, std[i][x_plot] ** 2, label=name)

# Labels.
axs_crlb_P[0, 0].legend(loc='upper right')

axs_crlb_P[0, 0].set_ylabel(r'Var of $\partial |V_{%d}| / \partial P_{%d}$ (M1)' % (x_i, x_i))
axs_crlb_P[1, 0].set_ylabel(r'Var of $\partial Re\{V_{%d}\} / \partial P_{%d}$ (M1)' % (x_i, x_i))
axs_crlb_P[2, 0].set_ylabel(r'Var of $\partial Im\{V_{%d}\} / \partial P_{%d}$ (M1)' % (x_i, x_i))
axs_crlb_P[0, 1].set_ylabel(r'Var of $\partial |V_{%d}| / \partial P_{%d}$ (M2)' % (x_i, x_i))
axs_crlb_P[1, 1].set_ylabel(r'Var of $\partial Re\{V_{%d}\} / \partial P_{%d}$ (M2)' % (x_i, x_i))
axs_crlb_P[2, 1].set_ylabel(r'Var of $\partial Im\{V_{%d}\} / \partial P_{%d}$ (M2)' % (x_i, x_i))

axs_crlb_P[-1, 0].set_xlabel('Time step (s)')
axs_crlb_P[-1, 1].set_xlabel('Time step (s)')

exp_logger.save_figure(fig_crlb_P, f'crlb_P_{dataset}_{sensor_class}.pdf')
plt.show()


####################### PLOT K_P ERRORS #######################
n_plots = 3
fig_P_e, ax_P_e = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))

x_idx = 0
for name, errors in Kp_errors.items():

    # Model 1 (d|X|, dRe{X}, dIm{X}) / P
    for i in range(3):
        if i in methods[name]:
            x_idx += 1
            e = 100 * errors[i, x_boxplot].T
            label = [name + ' (M1)']
            ax_P_e[i].boxplot(e, positions=[x_idx], labels=label)

    # Model 2 (|X|, Re{X}, Im{X}) / P
    for i in range(3, 6):
        if i in methods[name]:
            x_idx += 1
            e = 100 * errors[i, x_boxplot].T
            label = [name + ' (M2)']
            ax_P_e[i-3].boxplot(e, positions=[x_idx], labels=label)

# Add labels.
ax_P_e[0].set_ylabel(r'Normalized error on $\partial |V_{%d}| / \partial P_{%d}$ [%%]' % (x_i, x_i))
ax_P_e[1].set_ylabel(r'Normalized error on $\partial Re\{V_{%d}\} / \partial P_{%d}$ [%%]' % (x_i, x_i))
ax_P_e[2].set_ylabel(r'Normalized error on $\partial Im\{V_{%d}\} / \partial P_{%d}$ [%%]' % (x_i, x_i))

# Rotate xticks.
for ax in ax_P_e:
    ax.xaxis.set_tick_params(rotation=45)

exp_logger.save_figure(fig_P_e, f'KP_error_{dataset}_{sensor_class}.pdf')
plt.show()


####################### PLOT K_Q ERRORS #######################
n_plots = 3
fig_Q_e, ax_Q_e = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))

x_idx = 0
for name, errors in Kq_errors.items():

    # Model 1 (d|X|, dRe{X}, dIm{X}) / Q
    for i in range(3):
        if i in methods[name]:
            x_idx += 1
            e = 100 * errors[i, x_boxplot].T
            label = [name + ' (M1)']
            ax_Q_e[i].boxplot(e, positions=[x_idx], labels=label)

    # Model 2 (|X|, Re{X}, Im{X}) / Q
    for i in range(3, 6):
        if i in methods[name]:
            x_idx += 1
            e = 100 * errors[i, x_boxplot].T
            label = [name + ' (M2)']
            ax_Q_e[i - 3].boxplot(e, positions=[x_idx], labels=label)

ax_Q_e[0].set_ylabel(r'Normalized error on $\partial |V_{%d}| / \partial Q_{%d}$ [%%]' % (x_i, x_i))
ax_Q_e[1].set_ylabel(r'Normalized error on $\partial Re\{V_{%d}\} / \partial Q_{%d}$ [%%]' % (x_i, x_i))
ax_Q_e[2].set_ylabel(r'Normalized error on $\partial Im\{V_{%d}\} / \partial Q_{%d}$ [%%]' % (x_i, x_i))

# Rotate xticks.
for ax in ax_Q_e:
    ax.xaxis.set_tick_params(rotation=45)

exp_logger.save_figure(fig_Q_e, f'KQ_error_{dataset}_{sensor_class}.pdf')
plt.show()


####################### PLOT dV AND V ERRORS #######################
n_plots = 2
fig_V_e, ax_V_e = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))

# d|X|
x_idx = 0
for name, errors in dV_errors.items():

    # d|X| using magnitude
    if 0 in methods[name]:
        x_idx += 1
        e = 100 * errors[0, x_boxplot]
        label = [name + ' (magn)']
        ax_V_e[0].boxplot(e, positions=[x_idx], labels=label, showfliers=False)

    # d|X| using real/imag
    if 1 in methods[name]:
        x_idx += 1
        e = 100 * errors[1, x_boxplot]
        label = [name + ' (re/im)']
        ax_V_e[0].boxplot(e, positions=[x_idx], labels=label, showfliers=False)

# |X|
x_idx = 0
for name, errors in V_errors.items():

    if 3 in methods[name]:
        x_idx += 1
        e = 100 * errors[0, x_boxplot]
        label = [name + ' (magn)']
        ax_V_e[1].boxplot(e, positions=[x_idx], labels=label)

    if 4 in methods[name]:
        x_idx += 1
        e = 100 * errors[1, x_boxplot]
        label = [name + ' (re/im)']
        ax_V_e[1].boxplot(e, positions=[x_idx], labels=label)

ax_V_e[0].set_ylabel(r'Normalized error on $\Delta|V_{%d}|$ [%%]' % x_i)
ax_V_e[1].set_ylabel(r'Normalized error on $|V_{%d}|$ [%%]' % x_i)

# Rotate xticks.
for ax in ax_V_e:
    ax.xaxis.set_tick_params(rotation=45)

exp_logger.save_figure(fig_V_e, f'V_error_{dataset}_{sensor_class}.pdf')
plt.show()

print('Done!')

if __name__ == '__main__':
    pass
