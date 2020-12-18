"""
This script compares the performance of the 3 key methods:
    - least squares (using Model 2)
    - feedforward neural network
    - LSTM neural network

Each method is trained on the first 12h, validated on the next 6h, and then they
are compared on the last 6h. This is done for a single noise realization.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from meng.data_loader import load_true_data, simulate_noisy_meas, load_coefficients
from meng import neural_nets as nn, lstm, linear_models as linear
import meng.final_comparison as fc
from meng.metrics import normalized_error as norm_e
from meng.my_io import ComparisonLogger

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run(x_i, sensor_class, seed):

    ####################### PARAMETERS #######################

    # General parameters.
    dataset = 'cigre13'
    sc_train = sensor_class

    # Training (12h), validation (6h), testing (6h) splits.
    ts_train = np.arange(0, 12 * 3600)
    ts_val   = np.arange(12 * 3600, 18 * 3600)
    ts_test  = np.arange(18 * 3600, 24 * 3600 - 1)
    T_train, T_val, T_test = len(ts_train), len(ts_val), len(ts_test)

    # Feedforward neural network parameters.
    k_nn = 10
    nn_epoch_max = 10
    hidden_shape = [128, 64]

    # LSTM parameters.
    k_lstm = 50
    hidden_layer_size = 64
    lstm_epoch_max = 10
    batch_size = 64

    # Least squares model.
    tau = 1000
    freq = 50


    ####################### SET RANDOM SEED #######################
    torch.manual_seed(seed)
    np.random.seed(seed)


    ####################### LOAD TRUE DATA #######################

    # Load true load flow.
    V_org, I_org, P_org, Q_org, current_idx = load_true_data(dataset)

    # Remove slack measurements and create measurement matrices used in Model 2.
    V_org_magn = np.abs(V_org[1:])
    N, T = V_org_magn.shape

    # Create measurement delta matrix to be used as target.
    dV_org_magn = np.diff(V_org_magn, axis=1)


    ################### GENERATE NOISY DATA FOR TRAINING ###################

    # Add noise to load flow data.
    V_meas, _, P_meas, Q_meas, std_abs, std_ang, std_re, std_im, _, _ = \
        simulate_noisy_meas(sc_train, V_org, I_org, current_idx)

    # Remove slack measurements and create measurement matrices used in Model 2.
    V_meas = V_meas[1:]
    PQ_meas = np.vstack((P_meas[1:], Q_meas[1:]))


    ################### SELECT TYPE OF COEFFICIENT ###################

    # Select type of dependent variable.
    V_meas = np.abs(V_meas)

    # Load true coefficients.
    coefficients = load_coefficients(dataset)
    Kp_true = coefficients['vmagn_p'][x_i - 1, x_i - 1]
    Kq_true = coefficients['vmagn_q'][x_i - 1, x_i - 1]


    ################### SPLIT TRAINING, VALIDATION AND TESTING DATA ###################

    # Train matrices
    V_meas_tr  = V_meas[:, ts_train]
    PQ_meas_tr = PQ_meas[:, ts_train]
    X_train = np.vstack((V_meas_tr, PQ_meas_tr))

    # Validation matrices.
    V_meas_val  = V_meas[:, ts_val]
    PQ_meas_val = PQ_meas[:, ts_val]
    X_val = np.vstack((V_meas_val, PQ_meas_val))


    ################### PRE-PROCESS DATA ###################

    # Normalize training input data.
    norm_scaler = MinMaxScaler()
    X_train = norm_scaler.fit_transform(X_train.T).T
    X_val = norm_scaler.transform(X_val.T).T


    ################### FEEDFORWARD NEURAL NET ###################
    print('Training feedforward neural net...')

    in_shape = 3 * N * k_nn
    out_shape = 2 * N

    # Build training, validation, and test sets.
    train_data = nn.build_training_dataloader(X_train, PQ_meas_tr, V_meas_tr, x_i, k_nn)
    val_data = nn.build_training_dataloader(X_val, PQ_meas_val, V_meas_val, x_i, k_nn)

    # Initialize and train the models.
    nn_model = nn.FeedForward(in_shape, hidden_shape, out_shape, k_nn)
    nn_model, _ = fc.nn.train(nn_model, train_data, val_data, epochs=nn_epoch_max)


    ################### LSTM NEURAL NET ###################
    print('\nTraining LSTMs...')

    in_shape = 3 * N
    out_shape = 2 * N

    # Build training and validation sets.
    train_data = lstm.build_dataloader(X_train, PQ_meas_tr, V_meas_tr, x_i, k_lstm, batch_size)
    val_data   = lstm.build_dataloader(X_val, PQ_meas_val, V_meas_val, x_i, k_lstm, batch_size)

    # Initialize and train the models.
    lstm_model = lstm.LSTM(in_shape, hidden_layer_size, out_shape, batch_size)
    lstm_model, _ = fc.lstm.train(lstm_model, train_data, val_data, lr=1e-3, epochs=lstm_epoch_max, l2=0.)


    for sc_test in [0., 0.2, 0.5, 1.0]:

        folder = f'cross_trained_{dataset}_train{sc_train}_test{sc_test}'
        logger = ComparisonLogger(folder)


        ################### GENERATE NOISY DATA FOR TESTING ###################

        # Add noise to load flow data.
        V_meas, _, P_meas, Q_meas, std_abs, std_ang, std_re, std_im, _, _ = \
            simulate_noisy_meas(sc_test, V_org, I_org, current_idx)

        # Remove slack measurements and create measurement matrices used in Model 2.
        V_meas = V_meas[1:]
        PQ_meas = np.vstack((P_meas[1:], Q_meas[1:]))
        PQ_meas_bias = np.vstack((PQ_meas, np.ones(T)))


        ################### SELECT TYPE OF COEFFICIENT ###################

        # Select type of dependent variable.
        V_meas = np.abs(V_meas)
        dPQ_meas = np.diff(PQ_meas, axis=1)


        ################### SPLIT TESTING DATA ###################

        # Testing matrices.
        V_meas_test  = V_meas[:, ts_test]
        PQ_meas_test = PQ_meas[:, ts_test]
        X_test = np.vstack((V_meas_test, PQ_meas_test))
        PQ_meas_bias_test = PQ_meas_bias[:, ts_test]
        dPQ_meas_test = dPQ_meas[:, ts_test]


        ################### PRE-PROCESS DATA ###################

        # Normalize training input data.
        X_test = norm_scaler.transform(X_test.T).T


        ################### INFERENCE WITH PRE-TRAINED MODELS ###################

        # Feedforward model.
        test_data = nn.build_testing_dataloader(X_test, PQ_meas_test, V_meas_test, x_i, k_nn)
        S_nn, y_pred_nn, _ = nn_model.predict(test_data)
        ts_nn = np.arange(k_nn-1, T_test-1)

        # LSTM model.
        test_data  = lstm.build_dataloader(X_test, PQ_meas_test, V_meas_test, x_i, k_lstm, batch_size)
        S_lstm, y_pred_lstm, _ = lstm.predict(lstm_model, test_data, batch_size)
        ts_lstm = np.arange(k_nn-1, T_test-1)


        ################### LEAST SQUARES MODEL ###################
        print('\tLeast squares estimation...')

        which_i = np.array([x_i])
        valid_timesteps = np.ones(T_test - 1).astype(np.bool)
        use_sigma = False
        k_pcr = None
        qr = False

        S_ls, ts_ls, _ = linear.linear_model(V_meas_test, PQ_meas_bias_test, use_sigma,
                                             tau, freq, which_i, k_pcr, qr, valid_timesteps)

        # Remove bias terms.
        S_ls = {a: b[:-1] for a, b in S_ls.items()}

        y_pred_ls = fc.linear.lm_estimate_dVmagn(dPQ_meas_test[:, ts_ls], S_ls, None, False)
        S_ls, y_pred_ls = S_ls[x_i], y_pred_ls[x_i]

        ################### VISUALIZE RESULTS ON TEST SET ###################

        ts_all = ts_test[ts_ls]
        ts_all_hour = ts_all / 3600
        x_nn = ts_ls - k_nn + 1
        x_lstm = ts_ls - k_lstm + 1

        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(3, 4, hspace=0.05)

        # Plot Kp coefficients.
        ax = fig.add_subplot(gs[0, :-1])
        ax.plot(ts_all_hour, Kp_true[ts_all], label='True')
        ax.plot(ts_all_hour, S_ls[x_i - 1], label='LS')
        ax.plot(ts_all_hour, S_nn[x_i - 1, x_nn], label='NN')
        ax.plot(ts_all_hour, S_lstm[x_i - 1, x_lstm], label='LSTM')
        ax.set_ylabel(r'$\partial |V_{%d}|/\partial P_{%d}$' % (x_i, x_i))
        ax.set_xticks([])

        # Plot Kq coefficients.
        ax = fig.add_subplot(gs[1, :-1])
        ax.plot(ts_all_hour, Kq_true[ts_all], label='True')
        ax.plot(ts_all_hour, S_ls[x_i - 1 + N], label='LS')
        ax.plot(ts_all_hour, S_nn[x_i - 1 + N, x_nn], label='NN')
        ax.plot(ts_all_hour, S_lstm[x_i - 1 + N, x_lstm], label='LSTM')
        ax.legend(loc='upper right')
        ax.set_ylabel(r'$\partial |V_{%d}|/\partial Q_{%d}$' % (x_i, x_i))
        ax.set_xticks([])

        # Plot dV.
        ax = fig.add_subplot(gs[2, :-1])
        ax.plot(ts_all_hour[::2], dV_org_magn[x_i - 1, ts_all[::2]], label='True')
        ax.plot(ts_all_hour[::2], y_pred_ls[::2], label='LS')
        ax.plot(ts_all_hour[::2], y_pred_nn[x_nn[::2]], label='NN')
        ax.plot(ts_all_hour[::2], y_pred_lstm[x_lstm[::2]], label='LSTM')
        ax.set_ylabel(r'$\Delta |V_{%d}|$' % (x_i))
        ax.set_xlabel('Time (h)')

        # Plot Kp errors.
        ax = fig.add_subplot(gs[0, -1])
        e_ls   = 100 * norm_e(Kp_true[ts_all], S_ls[x_i - 1])
        e_nn   = 100 * norm_e(Kp_true[ts_all], S_nn[x_i - 1, x_nn])
        e_lstm = 100 * norm_e(Kp_true[ts_all], S_lstm[x_i - 1, x_lstm])
        ax.boxplot([e_ls, e_nn, e_lstm], labels=['LS', 'NN', 'LSTM'])
        ax.set_xticks([])

        # Plot Kq errors.
        ax = fig.add_subplot(gs[1, -1])
        e_ls   = 100 * norm_e(Kq_true[ts_all], S_ls[x_i - 1 + N])
        e_nn   = 100 * norm_e(Kq_true[ts_all], S_nn[x_i - 1 + N, x_nn])
        e_lstm = 100 * norm_e(Kq_true[ts_all], S_lstm[x_i - 1 + N, x_lstm])
        ax.boxplot([e_ls, e_nn, e_lstm], labels=['LS', 'NN', 'LSTM'])
        ax.set_ylabel('Normalized error [%]')
        ax.set_xticks([])

        # Plot d|V| errors.
        ax = fig.add_subplot(gs[2, -1])
        e_ls = 100 * norm_e(dV_org_magn[x_i - 1, ts_all], y_pred_ls)
        e_nn = 100 * norm_e(dV_org_magn[x_i - 1, ts_all], y_pred_nn[x_nn])
        e_lstm = 100 * norm_e(dV_org_magn[x_i - 1, ts_all], y_pred_lstm[x_lstm])
        ax.boxplot([e_ls, e_nn, e_lstm], labels=['LS', 'NN', 'LSTM'], showfliers=False)

        gs.tight_layout(fig)
        plt.show()

        logger.save_fig(fig, f'x_{x_i}_s{seed}.png')

        print('Done!')

if __name__ == '__main__':

    args = fc.utils.parse_args().__dict__
    run(**args)

    pass