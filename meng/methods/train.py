"""
This script to train feedforward neural nets and LSTMs to estimate coefficients
according to 6 models:
    - M1 (dy = H dx + w) using |V| coefficients
    - M1 (dy = H dx + w) using Re{V} coefficients
    - M1 (dy = H dx + w) using Im{V} coefficients
    - M2 (y = H x + w) using |V| coefficients
    - M2 (y = H x + w) using Re{V} coefficients
    - M2 (y = H x + w) using Im{V} coefficients

The first 12 hours are used for training and the next 6h for validation.

The best (validation) model for each of the 6 model type is saved for future
inference as a Pytorch named as:
    `trained_<NN-type>_<dataset>_<sensor_class>_<model-type>_x{node}_rs<seed>.pt`.
"""

import numpy as np
import time
import os
import torch

from meng.data_loader import load_true_data, simulate_noisy_meas
from meng.neural_nets.dataloaders import build_training_dataloader
from meng.neural_nets.models import FeedForward
from meng.my_io import NeuralNetLogger
from meng.methods import nn_utils, lstm_utils
from meng import lstm

os.environ['KMP_DUPLICATE_LIB_OK']='True'


####################### PARAMETERS #######################

# General parameters.
dataset = 'cigre13'
sensor_classes = [0.] #[0.5, 1., 0.2]
x_is = [11, 10, 5]
seed = 1

# Training (12h), validation (6h), testing (6h) splits.
ts_train = np.arange(0, 12 * 3600)
ts_val   = np.arange(12 * 3600, 18 * 3600)

# Feedforward neural network parameters.
k_nns = [100]
nn_epoch_max = 5
hidden_shapes = [[128, 64]]

# LSTM parameters.
hidden_layer_sizes = [64]
lstm_epoch_max = [3, 10]  # [M1, M2]
batch_size = 64

# Create the folder in which the experiment results will be stored.
nn_logger = NeuralNetLogger()

N_runs = len(sensor_classes) * len(x_is)


####################### SET RANDOM SEED #######################
torch.manual_seed(seed)
np.random.seed(seed)


####################### LOAD TRUE DATA #######################

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


################### GENERATE NOISY DATA ###################
for sensor_class in sensor_classes:
    n_run = 0

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


    ################### SPLIT TRAINING, VALIDATION AND TESTING DATA ###################

    # Delta matrices train.
    dV_meas_magn_tr = dV_meas_magn[:, ts_train]
    dV_meas_re_tr   = dV_meas_re[:, ts_train]
    dV_meas_im_tr   = dV_meas_im[:, ts_train]
    dPQ_meas_tr     = dPQ_meas[:, ts_train]

    # Delta matrices val.
    dV_meas_magn_val = dV_meas_magn[:, ts_val]
    dV_meas_re_val   = dV_meas_re[:, ts_val]
    dV_meas_im_val   = dV_meas_im[:, ts_val]
    dPQ_meas_val     = dPQ_meas[:, ts_val]

    # Non-delta matrices train.
    V_meas_magn_tr  = V_meas_magn[:, ts_train]
    V_meas_re_tr    = V_meas_re[:, ts_train]
    V_meas_im_tr    = V_meas_im[:, ts_train]
    PQ_meas_tr      = PQ_meas[:, ts_train]
    PQ_meas_bias_tr = PQ_meas_bias[:, ts_train]

    # Non-delta matrices val.
    V_meas_magn_val  = V_meas_magn[:, ts_val]
    V_meas_re_val    = V_meas_re[:, ts_val]
    V_meas_im_val    = V_meas_im[:, ts_val]
    PQ_meas_val      = PQ_meas[:, ts_val]
    PQ_meas_bias_val = PQ_meas_bias[:, ts_val]

    model_types = ['M1magn', 'M1real', 'M1imag', 'M2magn', 'M2real', 'M2imag']


    ################### TRAIN THE MODELS ###################
    for x_i in x_is:
        n_run += 1
        print(f'\n### Starting run {n_run}/{N_runs}... ###\n')

        ################### FEEDFORWARD NETS ###################

        best_models = [None] * 6
        best_val_loss = [None] * 6

        nn_run = 0
        nn_runs = len(k_nns) * len(hidden_shapes) * 6
        for k_nn in k_nns:
            for h_shapes in hidden_shapes:

                # Define input/output shapes for each type of neural net.
                in_shapes = [3 * N * k_nn] * 3 + [(3 * N + 1) * k_nn] * 3
                out_shapes = [2 * N] * 3 + [2 * N + 1] * 3

                # Define matrices to be used in training for each type of neural net.
                X_matrices_tr = [dV_meas_magn_tr, dV_meas_re_tr, dV_meas_im_tr,
                                 V_meas_magn_tr, V_meas_re_tr, V_meas_im_tr]
                PQ_matrices_tr = [dPQ_meas_tr] * 3 + [PQ_meas_bias_tr] * 3

                X_matrices_val = [dV_meas_magn_val, dV_meas_re_val, dV_meas_im_val,
                                  V_meas_magn_val, V_meas_re_val, V_meas_im_val]
                PQ_matrices_val = [dPQ_meas_val] * 3 + [PQ_meas_bias_val] * 3

                for idx in range(6):

                    nn_run += 1
                    print(f'\nnn run {nn_run}/{nn_runs}...')
                    start_time = time.time()

                    # Extract the correct parameters for this type of neural net.
                    in_shape = in_shapes[idx]
                    out_shape = out_shapes[idx]
                    X_tr = X_matrices_tr[idx]
                    PQ_tr = PQ_matrices_tr[idx]
                    X_val = X_matrices_val[idx]
                    PQ_val = PQ_matrices_val[idx]

                    # Build training and validation sets.
                    train_data = build_training_dataloader(X_tr, PQ_tr, x_i, k_nn)
                    val_data = build_training_dataloader(X_val, PQ_val, x_i, k_nn)

                    # Initialize and train the neural net.
                    model = FeedForward(in_shape, h_shapes, out_shape, k_nn)
                    model, val_loss = nn_utils.train(model, train_data, val_data, lr=1e-3, epochs=nn_epoch_max, l2=0.)

                    # Keep the model if it's the best seen so far.
                    if best_val_loss[idx] is None or val_loss < best_val_loss[idx]:
                        best_val_loss[idx] = val_loss
                        best_models[idx] = model

                    print(f' time: {(time.time() - start_time) / 60:.2} min.')

        # Save the 6 neural nets.
        for idx, model in enumerate(best_models):
            filename = nn_utils.checkpoint_filename(dataset, sensor_class, 'NN',
                                                    model_types[idx], x_i, seed)
            nn_logger.save_model(model, filename)


        ################### LSTM NETS ###################

        best_models = [None] * 6
        best_val_loss = [None] * 6

        lstm_run = 0
        lstm_runs = len(k_nns) * len(hidden_layer_sizes) * 6
        for k_nn in k_nns:
            for hidden_shape in hidden_layer_sizes:

                # Define input/output shapes for each type of neural net.
                in_shapes = [3 * N] * 3 + [(3 * N + 1)] * 3
                out_shapes = [2 * N] * 3 + [2 * N + 1] * 3

                # Define matrices to be used in training for each type of neural net.
                X_matrices_tr = [dV_meas_magn_tr, dV_meas_re_tr, dV_meas_im_tr,
                                 V_meas_magn_tr, V_meas_re_tr, V_meas_im_tr]
                PQ_matrices_tr = [dPQ_meas_tr] * 3 + [PQ_meas_bias_tr] * 3

                X_matrices_val = [dV_meas_magn_val, dV_meas_re_val, dV_meas_im_val,
                                  V_meas_magn_val, V_meas_re_val, V_meas_im_val]
                PQ_matrices_val = [dPQ_meas_val] * 3 + [PQ_meas_bias_val] * 3

                for idx in range(6):

                    lstm_run += 1
                    print(f'\nlstm run {lstm_run}/{lstm_runs}...')
                    start_time = time.time()

                    # Extract the correct parameters for this type of neural net.
                    in_shape = in_shapes[idx]
                    out_shape = out_shapes[idx]
                    X_tr = X_matrices_tr[idx]
                    PQ_tr = PQ_matrices_tr[idx]
                    X_val = X_matrices_val[idx]
                    PQ_val = PQ_matrices_val[idx]

                    # Build training and validation sets.
                    train_data = lstm.build_dataloader(X_tr, PQ_tr, x_i, k_nn, batch_size)
                    val_data = lstm.build_dataloader(X_val, PQ_val, x_i, k_nn, batch_size)

                    # Initialize and train the neural net.
                    model = lstm.LSTM(in_shape, hidden_shape, out_shape, batch_size)
                    epoch_max = lstm_epoch_max[idx // 3]
                    model, val_loss = lstm_utils.train(model, train_data, val_data, lr=1e-3, epochs=epoch_max, l2=0.)

                    # Keep the model if it's the best seen so far.
                    if best_val_loss[idx] is None or val_loss < best_val_loss[idx]:
                        best_val_loss[idx] = val_loss
                        best_models[idx] = model

                    print(f' time: {(time.time() - start_time) / 60:.2} min.')

        # Save the 6 neural nets.
        for idx, model in enumerate(best_models):
            filename = nn_utils.checkpoint_filename(dataset, sensor_class, 'LSTM',
                                                    model_types[idx], x_i, seed)
            nn_logger.save_model(model, filename)


print('Done!')

if __name__ == '__main__':
    pass
