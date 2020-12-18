"""
This script runs a series of experiments (random noise realisations) and
estimate sensitivity coefficients for a number of models considered. Those models
are:
    - Linear Model: M1 (dy = H dx + w), M2 (y = H x + w).
    - Feedforward neural networks using M1 and M2.
    - LSTMs using M1 and M2.

Each of these models is used to estimate 3 sorts of coefficients:
    - |V| coefficients
    - Re{V} coefficients
    - Im{V} coefficients

This thus results in estimations of 18 types of sensitivity coefficients, which
are also used to estimate dependent variable measurements.

The estimations are saved in a .mat file of the form:
    `est_<dataset>_<sensor_class>_x<node>_Nexp<N_exp>.mat`
which can then be loaded to visualize the results (see `compare.py`).
"""

import numpy as np
import os
import time
import torch

from meng.data_loader import load_true_data, simulate_noisy_meas, load_coefficients

from meng.methods.utils import estimations_filename
from meng.neural_nets.dataloaders import build_testing_dataloader
from meng.my_io import EstimationLogger, NeuralNetLogger
from meng.methods import nn_utils, utils
from meng import lstm, crlb
from meng.linear_models.core import linear_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'

####################### PARAMETERS #######################

# General parameters.
dataset = 'cigre13'
sensor_classes = [0.5, 1., 0.2]
x_is = [11, 10, 5]
train_seed = 1
seeds = np.arange(0, 10)

# Training (12h), validation (6h), testing (6h) splits.
ts_test = np.arange(18 * 3600, 24 * 3600 - 1)
T_test = len(ts_test)

# Linear nn_models.
tau = 1000
freq = 60 * 2
k_pcr = None
qr = False

# Feedforward neural network parameters.
k_nn = 100

# LSTM parameters.
batch_size = 64

# Create the folder in which the experiment results will be stored.
nn_logger = NeuralNetLogger()
est_logger = EstimationLogger()


####################### LOAD TRUE DATA #######################

# Load the true sensitivity coefficients of interest.
_shape = (len(x_is), 3, T_test)
Kp_true = np.zeros(_shape)
Kq_true = np.zeros(_shape)

coefficients = load_coefficients(dataset)
for x_idx, x_i in enumerate(x_is):
    Kp_true[x_idx, 0] = coefficients['vmagn_p'][x_i - 1, x_i - 1, ts_test]
    Kq_true[x_idx, 0] = coefficients['vmagn_q'][x_i - 1, x_i - 1, ts_test]

    Kp_true[x_idx, 1] = coefficients['vreal_p'][x_i - 1, x_i - 1, ts_test]
    Kq_true[x_idx, 1] = coefficients['vreal_q'][x_i - 1, x_i - 1, ts_test]

    Kp_true[x_idx, 2] = coefficients['vimag_p'][x_i - 1, x_i - 1, ts_test]
    Kq_true[x_idx, 2] = coefficients['vimag_q'][x_i - 1, x_i - 1, ts_test]

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


################### ITERATE OVER SENSOR CLASSES ###################
N_exp = len(seeds)
n_runs = N_exp * len(sensor_classes)
n_run = 0
for sensor_class in sensor_classes:

    start_time = time.time()

    # Store all estimations in a single array.
    #   axis 0: different noise realisations.
    #   axis 1: 6 Linear, 6 NNs, 6 LSTMs.
    #   axis 2: all test timesteps.
    _shape = (N_exp, 18, T_test)
    Kp_all = [np.zeros(_shape) for _ in range(len(x_is))]
    Kq_all = [np.zeros(_shape) for _ in range(len(x_is))]
    X_all = [np.zeros(_shape) for _ in range(len(x_is))]

    _shape = (N_exp, 6, T_test)
    crlbs_P = [np.zeros(_shape) for _ in range(len(x_is))]
    crlbs_Q = [np.zeros(_shape) for _ in range(len(x_is))]

    ts_linear = None

    ################### RUN N_EXP EXPERIMENTS ###################
    for n_exp, seed in enumerate(seeds):

        n_run += 1
        print(f'Run {n_run}/{n_runs}...')

        ####################### SET RANDOM SEED #######################
        torch.manual_seed(seed)
        np.random.seed(seed)


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
        std_re = std_re[1:]
        std_im = std_im[1:]

        # Create measurement delta matrices used in Model 1.
        dV_meas_magn = np.diff(V_meas_magn, axis=1)
        dV_meas_re = np.diff(V_meas_re, axis=1)
        dV_meas_im = np.diff(V_meas_im, axis=1)
        dPQ_meas = np.diff(PQ_meas, axis=1)


        ################### SPLIT TESTING DATA ###################

        # Delta matrices test.
        dV_meas_magn_test = dV_meas_magn[:, ts_test]
        dV_meas_re_test   = dV_meas_re[:, ts_test]
        dV_meas_im_test   = dV_meas_im[:, ts_test]
        dPQ_meas_test     = dPQ_meas[:, ts_test]

        # Non-delta matrices test.
        V_meas_magn_test  = V_meas_magn[:, ts_test]
        V_meas_re_test    = V_meas_re[:, ts_test]
        V_meas_im_test    = V_meas_im[:, ts_test]
        PQ_meas_test      = PQ_meas[:, ts_test]
        PQ_meas_bias_test = PQ_meas_bias[:, ts_test]

        # Matrices used in the computation of CRLB.
        dPQ_org_test     = dPQ_org[:, ts_test]
        PQ_org_test      = PQ_org[:, ts_test]
        PQ_org_bias_test = PQ_org_bias[:, ts_test]
        std_re_test      = std_re[:, ts_test]
        std_im_test      = std_im[:, ts_test]

        model_types = ['M1magn', 'M1real', 'M1imag', 'M2magn', 'M2real', 'M2imag']

        # Define matrices to be used for each type of model.
        X_matrices = [dV_meas_magn_test, dV_meas_re_test, dV_meas_im_test,
                      V_meas_magn_test, V_meas_re_test, V_meas_im_test]
        PQ_matrices = [dPQ_meas_test] * 3 + [PQ_meas_bias_test] * 3


        ################### USE LEAST SQUARES FOR INFERENCE ###################
        print('Least squares...')

        offset = 0
        which_i = np.array(x_is)
        valid_timesteps = np.ones(T_test - 1).astype(np.bool)

        if sensor_class != 0.:
            use_sigma = [True] * 3 + [False] * 3
        else:
            use_sigma = [True] * 6

        for i in range(6):

            # Estimate coefficients.
            S, ts, _ = linear_model(X_matrices[i], PQ_matrices[i], use_sigma[i],
                                    tau, freq, which_i, k_pcr, qr, valid_timesteps)
            dV = utils.lm_estimate_dVmagn(PQ_matrices[i][:, ts], S, None, False)

            # Save predicted data.
            for x_idx, x_i in enumerate(x_is):
                Kp_all[x_idx][n_exp, offset + i, ts] = S[x_i][x_i - 1]
                Kq_all[x_idx][n_exp, offset + i, ts] = S[x_i][x_i - 1 + N]
                X_all[x_idx][n_exp, offset + i, ts] = dV[x_i]

            ts_linear = ts

        ################### USED THE TRAINED NN FOR INFERENCE ###################
        print('Neural nets...')

        offset = 6
        for x_idx, x_i in enumerate(x_is):

            for i in range(6):

                # Create test dataset.
                data = build_testing_dataloader(X_matrices[i], PQ_matrices[i], x_i, k_nn)

                # Load trained model.
                filename = nn_utils.checkpoint_filename(dataset, sensor_class, 'NN', model_types[i], x_i, train_seed)
                model = nn_logger.load_model(filename)

                # Predict coefficients on the test set.
                S, y_pred, _ = model.predict(data)

                # Save predicted data.
                ts = np.arange(k_nn, T_test - 1)
                Kp_all[x_idx][n_exp, offset + i, ts] = S[x_i - 1, :len(ts)]
                Kq_all[x_idx][n_exp, offset + i, ts] = S[x_i - 1 + N, :len(ts)]
                X_all[x_idx][n_exp, offset + i, ts] = y_pred[:len(ts)]


        ################### USED THE TRAINED LSTM FOR INFERENCE ####################
        print('LSTMs...')

        offset = 12
        for x_idx, x_i in enumerate(x_is):

            for i in range(6):
                # Create test dataset.
                data = lstm.build_dataloader(X_matrices[i], PQ_matrices[i], x_i, k_nn, batch_size)

                # Load trained model.
                filename = nn_utils.checkpoint_filename(dataset, sensor_class, 'LSTM', model_types[i], x_i, train_seed)
                model = nn_logger.load_model(filename)

                # Predict coefficients on the test set.
                S, y_pred, _ = lstm.predict(model, data, batch_size)

                # Save predicted data.
                ts = np.arange(k_nn, T_test - 1)
                Kp_all[x_idx][n_exp, offset + i, ts] = S[x_i - 1, :len(ts)]
                Kq_all[x_idx][n_exp, offset + i, ts] = S[x_i - 1 + N, :len(ts)]
                X_all[x_idx][n_exp, offset + i, ts] = y_pred[:len(ts)]

        ####################### CRAMER-RAO LOWER BOUND #######################
        print('CRLB...')

        for x_idx, x_i in enumerate(x_is):

            for t in ts_linear:

                # Model 1.
                H = dPQ_org_test[:, t - tau: t].T
                use_sigma = True
                lb_magn, _ = crlb.compute_crlb(H, std_abs, use_sigma)
                lb_re, _ = crlb.compute_crlb(H, np.mean(std_re[x_i - 1, t-tau: t]), use_sigma)
                lb_im, _ = crlb.compute_crlb(H, np.mean(std_im[x_i - 1, t-tau: t]), use_sigma)

                crlbs_P[x_idx][n_exp, 0, t] = lb_magn[x_i - 1]
                crlbs_P[x_idx][n_exp, 1, t] = lb_re[x_i - 1]
                crlbs_P[x_idx][n_exp, 2, t] = lb_im[x_i - 1]

                crlbs_Q[x_idx][n_exp, 0, t] = lb_magn[x_i - 1 + N]
                crlbs_Q[x_idx][n_exp, 1, t] = lb_re[x_i - 1 + N]
                crlbs_Q[x_idx][n_exp, 2, t] = lb_im[x_i - 1 + N]

                # Model 2.
                H = PQ_org_bias_test[:, t - tau: t].T
                use_sigma = False

                lb_magn, _ = crlb.compute_crlb(H, std_abs, use_sigma)
                lb_re, _ = crlb.compute_crlb(H, np.mean(std_re[x_i - 1, t-tau: t]), use_sigma)
                lb_im, _ = crlb.compute_crlb(H, np.mean(std_im[x_i - 1, t-tau: t]), use_sigma)

                crlbs_P[x_idx][n_exp, 3, t] = lb_magn[x_i - 1]
                crlbs_P[x_idx][n_exp, 4, t] = lb_re[x_i - 1]
                crlbs_P[x_idx][n_exp, 5, t] = lb_im[x_i - 1]

                crlbs_Q[x_idx][n_exp, 3, t] = lb_magn[x_i - 1 + N]
                crlbs_Q[x_idx][n_exp, 4, t] = lb_re[x_i - 1 + N]
                crlbs_Q[x_idx][n_exp, 5, t] = lb_im[x_i - 1 + N]


    ####################### RECONSTRUCT d|V| or |V| #######################

    X_reconstructed = [np.zeros((N_exp, 12, T_test)) for _ in range(len(x_is))]
    X_true = [np.zeros((2, T_test)) for _ in range(len(x_is))]

    for x_idx, x_i in enumerate(x_is):

        for i in range(6):

            # No need to modify estimations using |V| coefficients.
            X_reconstructed[x_idx][:, 2 * i] = X_all[x_idx][:, 3 * i]

            # Combine real and imag estimations.
            x_real = X_all[x_idx][:, 3 * i + 1]
            x_imag = X_all[x_idx][:, 3 * i + 2]
            X_reconstructed[x_idx][:, 2 * i + 1] = np.sqrt(x_real ** 2 + x_imag ** 2)

        # True values.
        X_true[x_idx][0] = dV_org_magn[x_i - 1, ts_test]
        X_true[x_idx][1] = V_org_magn[x_i - 1, ts_test]


    ################### SAVE ESTIMATIONS ####################

    # Which timesteps to select within the test timesteps.
    ts_final = ts_linear

    # Which actual timesteps that corresponds to.
    ts = ts_test[ts_final]

    for x_idx, x_i in enumerate(x_is):

        # Save data in a dictionary.
        data = {'Kp_est': Kp_all[x_idx][:, :, ts_final],
                'Kq_est': Kq_all[x_idx][:, :, ts_final],
                'X_est': X_reconstructed[x_idx][:, :, ts_final],
                'Kp_true': Kp_true[x_idx][:, ts_final],
                'Kq_true': Kq_true[x_idx][:, ts_final],
                'X_true': X_true[x_idx][:, ts_final],
                'CRLB_P': crlbs_P[x_idx][:, :, ts_final],
                'CRLB_Q': crlbs_Q[x_idx][:, :, ts_final],
                'ts': ts}

        # Save data.
        f = estimations_filename(dataset, sensor_class, x_i, N_exp)
        est_logger.save_estimation(data, f)

    print(f'Elapsed time (sc = {sensor_class}): {(time.time() - start_time) / 60} min.')


print('Done!')

if __name__ == '__main__':
    pass
