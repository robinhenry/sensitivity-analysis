"""
This script was used to test a range of different hyper-parameters for the
neural network method to select the ones that performed best.
"""

import numpy as np
import os

from meng.data_loader import load_voltage_exp_data
from meng.neural_nets.dataloaders import build_training_dataloader, build_testing_dataloader
from meng.neural_nets.models import FeedForward
from meng.neural_nets.utils import train
from meng.my_io import ExperimentLogger

os.environ['KMP_DUPLICATE_LIB_OK']='True'

####################### PARAMETERS #######################

# General parameters.
dataset_train = 'cigre13'
dataset_val = 'cigre13-v1'
sensor_class = 0.5
x_i = 11
folder = 'nn_param_search'
N_exp = 20

# Feedforward neural network parameters.
k_nns = [10, 100, 300]
epochs = 25
hidden_shapes = [64, 128]

# Create the folder in which the experiment results will be stored.
exp_logger = ExperimentLogger(folder, locals())

####################### LOAD TRAINING DATA #######################
P_meas_tr, Q_meas_tr, V_meas_tr, P_org, Q_org, V_org, coefficients, std_abs, \
std_V_real, std_V_imag = load_voltage_exp_data(dataset_train, sensor_class)
N, T = P_meas_tr.shape

# Remove slack measurements and create measurement matrices used in Model 2.
V_meas_magn_tr = np.abs(V_meas_tr)
V_meas_re_tr = np.real(V_meas_tr)
V_meas_im_tr = np.imag(V_meas_tr)
PQ_meas_tr = np.vstack((P_meas_tr, Q_meas_tr))
PQ_meas_bias_tr = np.vstack((PQ_meas_tr, np.ones(T)))

# Create measurement delta matrices used in Model 1.
dV_meas_magn_tr = np.diff(V_meas_magn_tr, axis=1)
dV_meas_re_tr = np.diff(V_meas_re_tr, axis=1)
dV_meas_im_tr = np.diff(V_meas_im_tr, axis=1)
dPQ_meas_tr = np.diff(PQ_meas_tr, axis=1)

####################### LOAD VALIDATION DATA #######################
P_meas_val, Q_meas_val, V_meas_val, _, _, _, _, _, _, _ = \
    load_voltage_exp_data(dataset_val, sensor_class)

# Remove slack measurements and create measurement matrices used in Model 2.
V_meas_magn_val = np.abs(V_meas_val)
V_meas_re_val = np.real(V_meas_val)
V_meas_im_val = np.imag(V_meas_val)
PQ_meas_val = np.vstack((P_meas_val, Q_meas_val))
PQ_meas_bias_val = np.vstack((PQ_meas_val, np.ones(T)))

# Create measurement delta matrices used in Model 1.
dV_meas_magn_val = np.diff(V_meas_magn_val, axis=1)
dV_meas_re_val = np.diff(V_meas_re_val, axis=1)
dV_meas_im_val = np.diff(V_meas_im_val, axis=1)
dPQ_meas_val = np.diff(PQ_meas_val, axis=1)

####################### TRAIN NEURAL NETS #######################

best_losses = [None] * 6
best_epochs = [0] * 6
best_ks = [0] * 6
best_h = [0] * 6

for k_nn in k_nns:
    for h in hidden_shapes:

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

        models = [model_dv_magn, model_dv_re, model_dv_im,
                  model_v_magn, model_v_re, model_v_im]

        # Define training data sets.
        train_dv_magn = build_training_dataloader(dV_meas_magn_tr, dPQ_meas_tr, x_i, k_nn)
        train_dv_re = build_training_dataloader(dV_meas_re_tr, dPQ_meas_tr, x_i, k_nn)
        train_dv_im = build_training_dataloader(dV_meas_im_tr, dPQ_meas_tr, x_i, k_nn)

        train_v_magn = build_training_dataloader(V_meas_magn_tr, PQ_meas_bias_tr, x_i, k_nn)
        train_v_re = build_training_dataloader(V_meas_re_tr, PQ_meas_bias_tr, x_i, k_nn)
        train_v_im = build_training_dataloader(V_meas_im_tr, PQ_meas_bias_tr, x_i, k_nn)

        train_datasets = [train_dv_magn, train_dv_re, train_dv_im,
                          train_v_magn, train_v_re, train_v_im]

        # Define validation data sets.
        val_dv_magn = build_testing_dataloader(dV_meas_magn_val, dPQ_meas_val, x_i, k_nn)
        val_dv_re = build_testing_dataloader(dV_meas_re_val, dPQ_meas_val, x_i, k_nn)
        val_dv_im = build_testing_dataloader(dV_meas_im_val, dPQ_meas_val, x_i, k_nn)

        val_v_magn = build_testing_dataloader(V_meas_magn_val, PQ_meas_bias_val, x_i, k_nn)
        val_v_re = build_testing_dataloader(V_meas_re_val, PQ_meas_bias_val, x_i, k_nn)
        val_v_im = build_testing_dataloader(V_meas_im_val, PQ_meas_bias_val, x_i, k_nn)

        val_datasets = [val_dv_magn, val_dv_re, val_dv_im,
                        val_v_magn, val_v_re, val_v_im]

        for idx, (model, train_data, val_data) in enumerate(zip(models, train_datasets, val_datasets)):

            # Train neural network.
            best_loss, best_model, best_epoch = \
                train(model, train_data, val_data, lr=1e-3, epoch_max=epochs, l2=0.)

            if best_losses[idx] is None or best_loss < best_losses[idx]:
                best_losses[idx] = best_loss
                best_epochs[idx] = best_epoch
                best_ks[idx] = k_nn
                best_h[idx] = h

print('Best val losses: ', best_losses)
print('Best epochs: ', best_epochs)
print('Best ks: ', best_ks)
print('Best # hidden units: ', best_h)


if __name__ == '__main__':
    pass