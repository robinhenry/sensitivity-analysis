"""
This script tests a neural network that estimates both real and imaginary
coefficients at the same time;
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from meng.data_loader import load_true_data, simulate_noisy_meas, load_coefficients
from meng.nn_experimenting import core_1, real_imag_model
from meng.methods import nn_utils
from meng.metrics import normalized_error

os.environ['KMP_DUPLICATE_LIB_OK']='True'


####################### PARAMETERS #######################

# General parameters.
dataset = 'cigre13'
sensor_class = 0.5
x_i = 11
seed = 1

# Training (12h), validation (6h), testing (6h) splits.
ts_train = np.arange(0, 12 * 3600)
ts_val   = np.arange(12 * 3600, 18 * 3600)

# Feedforward neural network parameters.
k_nn = 100
nn_epoch_max = 10
hidden_shape = [128, 64]

# LSTM parameters.
hidden_layer_size = 64
lstm_epoch_max = 10
batch_size = 64

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

# Add noise to load flow data.
V_meas, _, P_meas, Q_meas, std_abs, std_ang, std_re, std_im, _, _ = \
    simulate_noisy_meas(sensor_class, V_org, I_org, current_idx)

# Remove slack measurements and create measurement matrices used in Model 2.
V_meas = V_meas[1:]
PQ_meas = np.vstack((P_meas[1:], Q_meas[1:]))


################### SELECT TYPE OF COEFFICIENT ###################

V_re_im = np.vstack((V_meas.real, V_meas.imag))
V_magn = np.abs(V_meas)

# Load true coefficients.
coefficients = load_coefficients(dataset)

Kp_re_true = coefficients['vreal_p'][x_i - 1, x_i - 1]
Kq_re_true = coefficients['vreal_q'][x_i - 1, x_i - 1]
Kp_im_true = coefficients['vimag_p'][x_i - 1, x_i - 1]
Kq_im_true = coefficients['vimag_q'][x_i - 1, x_i - 1]

################### SPLIT TRAINING, VALIDATION AND TESTING DATA ###################

# Train matrices
V_tr  = V_re_im[:, ts_train]
PQ_meas_tr = PQ_meas[:, ts_train]
X_train = np.vstack((V_tr, PQ_meas_tr))
y_train = V_magn[:, ts_train]

# Validation matrices.
V_val  = V_re_im[:, ts_val]
PQ_meas_val = PQ_meas[:, ts_val]
X_val = np.vstack((V_val, PQ_meas_val))
y_val = V_magn[:, ts_train]


################### PRE-PROCESS TRAINING AND VALIDATION DATA ###################

# Normalize training input data.
# norm_scaler = MinMaxScaler()
# X_train = norm_scaler.fit_transform(X_train.T).T
# X_val = norm_scaler.transform(X_val.T).T

# Standardize training input data.
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.T).T
# X_val = scaler.transform(X_val.T).T


################### TRAIN THE FEEDFORWARD MODELS ###################

in_shape = 4 * N * k_nn
out_shape = 4 * N

# Build training and validation sets.
train_data = core_1.build_training_dataloader(X_train, PQ_meas_tr, y_train, x_i, k_nn)
val_data = core_1.build_testing_dataloader(X_val, PQ_meas_val, y_val, x_i, k_nn)

# Initialize and train the models.
model = real_imag_model.RealImagForward(in_shape, hidden_shape, out_shape, k_nn)
mode, val_loss = nn_utils.train(model, train_data, val_data, epochs=nn_epoch_max)
ts = ts_val[k_nn-1:-1]


################### VISUALIZE RESULTS ON VALIDATION SET ###################

# Run inference.
S, y_pred, y_true = model.predict(val_data)
S_re = S[: 2 * N]
S_im = S[2 * N:]

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(5, 4)
axs = []

# Plot Kp real coefficients.
ax = fig.add_subplot(gs[0, :-1])
ax.plot(ts, Kp_re_true[ts], label='True')
ax.plot(ts, S_re[x_i - 1], label='Estimated')

# Plot Kq real coefficients.
ax = fig.add_subplot(gs[1, :-1])
ax.plot(ts, Kq_re_true[ts], label='True')
ax.plot(ts, S_re[x_i - 1 + N], label='Estimated')

# Plot Kp imag coefficients.
ax = fig.add_subplot(gs[0, :-1])
ax.plot(ts, Kp_im_true[ts], label='True')
ax.plot(ts, S_im[x_i - 1], label='Estimated')

# Plot Kq imag coefficients.
ax = fig.add_subplot(gs[1, :-1])
ax.plot(ts, Kq_im_true[ts], label='True')
ax.plot(ts, S_im[x_i - 1 + N], label='Estimated')

# Plot dV.
ax = fig.add_subplot(gs[2, :-1])
x = slice(0, 100)
ax.plot(ts[x], dV_org_magn[x_i - 1, ts[x]], label='True')
ax.plot(ts[x], y_pred[x], label='Estimated')

# Plot Kp real errors.
ax = fig.add_subplot(gs[0, -1])
e = 100 *  normalized_error(Kp_re_true[ts], S_re[x_i - 1])
ax.boxplot(e)

# Plot Kq real errors.
ax = fig.add_subplot(gs[1, -1])
e = 100 *  normalized_error(Kq_re_true[ts], S_re[x_i - 1 + N])
ax.boxplot(e)

# Plot Kp imag errors.
ax = fig.add_subplot(gs[0, -1])
e = 100 *  normalized_error(Kp_im_true[ts], S_im[x_i - 1])
ax.boxplot(e)

# Plot Kq imag errors.
ax = fig.add_subplot(gs[1, -1])
e = 100 *  normalized_error(Kq_im_true[ts], S_im[x_i - 1 + N])
ax.boxplot(e)

# Plot d|V| errors.
ax = fig.add_subplot(gs[2, -1])
e = 100 *  normalized_error(dV_org_magn[x_i - 1, ts], y_pred)
ax.boxplot(e, showfliers=False)


plt.show()

print('Done!')

if __name__ == '__main__':
    pass
