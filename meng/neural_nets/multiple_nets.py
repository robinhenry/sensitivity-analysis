"""
This script investigates the variance found in the neural network estimates by
training several neural networks on the same data.
"""
import numpy as np
import os
import matplotlib.pyplot as plt

from meng.data_loader import load_voltage_exp_data
from meng.neural_nets.dataloaders import build_training_dataloader, build_testing_dataloader
from meng.neural_nets.models import FeedForward
from meng.neural_nets.core import train
from meng.my_io import ExperimentLogger

os.environ['KMP_DUPLICATE_LIB_OK']='True'

####################### PARAMETERS #######################

# General parameters.
dataset_train = 'cigre13'
dataset_val = 'cigre13-v1'
sensor_class = 0.5
x_i = 11
folder = 'multiple_feedforward'
N_nets = 20
T_train = 60 * 60 * 12

# Feedforward neural network parameters.
k_nn = 100
epochs = 40
hidden_shapes = [64, 64]

# Create the folder in which the experiment results will be stored.
exp_logger = ExperimentLogger(folder, locals())


####################### LOAD TRAINING DATA #######################
P_meas, Q_meas, V_meas, P_org, Q_org, V_org, coefficients, std_abs, \
std_V_real, std_V_imag = load_voltage_exp_data(dataset_train, sensor_class)
N, T = P_meas.shape

# True data.
dV_org = np.diff(np.abs(V_org), axis=1)
V_org_magn = np.abs(V_org)

# Remove slack measurements and create measurement matrices used in Model 2.
V_meas_magn = np.abs(V_meas)
PQ_meas = np.vstack((P_meas, Q_meas))
PQ_meas_bias = np.vstack((PQ_meas, np.ones(T)))

# Create measurement delta matrices used in Model 1.
dV_meas_magn = np.diff(V_meas_magn, axis=1)
dPQ_meas = np.diff(PQ_meas, axis=1)

# Split training and testing data.
dV_train, dPQ_train = dV_meas_magn[:, :T_train], dPQ_meas[:, :T_train]
dV_test, dPQ_test = dV_meas_magn[:, T_train:], dPQ_meas[:, T_train:]
T_test = T - T_train

V_train, PQ_train = V_meas_magn[:, :T_train], PQ_meas_bias[:, :T_train]
V_test, PQ_test = V_meas_magn[:, T_train:], PQ_meas_bias[:, T_train:]

####################### TRAIN NEURAL NETWORKS #######################

# Define training and testing data sets.
train_dv = build_training_dataloader(dV_train, dPQ_train, x_i, k_nn)
test_dv = build_testing_dataloader(dV_test, dPQ_test, x_i, k_nn)

train_v = build_training_dataloader(V_train, PQ_train, x_i, k_nn)
test_v = build_testing_dataloader(V_test, PQ_test, x_i, k_nn)

# models_dv = []
models_v = []

for iter in range(N_nets):
    print(f'\nTraining model {iter}...')

    # # Initialize neural nets.
    # in_shape = 3 * N * k_nn
    # sc_matrix_shape = 2 * N
    # model_dv = FeedForward(in_shape, hidden_shapes, sc_matrix_shape, k_nn)
    #
    # # Train dV neural net.
    # _ = train(model_dv, train_dv, lr=1e-3, epochs=epochs, l2=0., plot=True)
    # plt.show()
    # models_dv.append(model_dv)

    # Initialize neural net.
    in_shape = (3 * N + 1) * k_nn
    sc_matrix_shape = 2 * N + 1
    model_v = FeedForward(in_shape, hidden_shapes, sc_matrix_shape, k_nn)

    # Train X neural net.
    _ = train(model_v, train_v, lr=1e-3, epochs=epochs, l2=0., plot=True)
    plt.show()
    models_v.append(model_v)

####################### TEST NEURAL NETWORKS #######################
Kps = []
Kqs = []
y_preds = []
y_trues = []

models = models_v
test_data = test_v
for idx, model in enumerate(models):

    # Predict using neural net.
    S, y_pred, y_true = model.predict(test_data)

    # Store predicted values.
    Kps.append(S[x_i - 1])
    Kqs.append(S[x_i - 1 + N])
    y_preds.append(y_pred)
    y_trues.append(y_true)

# Stack estimations from all neural nets.
Kps = np.vstack(Kps)
Kqs = np.vstack(Kqs)
y_preds = np.vstack(y_preds)
y_trues = np.vstack(y_trues)

####################### PLOT RESULTS #######################
fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

x_plot_true = np.arange(k_nn+T_train, T, 250)
x_plot_nn = np.arange(0, T_test - k_nn, 250)

# True values to plot.
Kp_true = coefficients['vmagn_p'][x_i - 1, x_i - 1]
Kq_true = coefficients['vmagn_q'][x_i - 1, x_i - 1]
dv_true = dV_org[x_i - 1]
v_true = V_org_magn[x_i - 1]

# Plot true values.
ax1.plot(x_plot_true, Kp_true[x_plot_true], label='True Kp')
ax2.plot(x_plot_true, Kq_true[x_plot_true], label='True Kq')
# ax3.plot(x_plot_true, dv_true[x_plot_true], label='True dV')
ax3.plot(x_plot_true, v_true[x_plot_true], label='True dV')

# Plot estimations.
mu, std = np.mean(Kps, axis=0), np.std(Kps, axis=0)
ax1.plot(x_plot_true, mu[x_plot_nn], label=f'NN')
ax1.fill_between(x_plot_true, (mu + std)[x_plot_nn], (mu - std)[x_plot_nn], facecolor='orange', alpha=0.4)

mu, std = np.mean(Kqs, axis=0), np.std(Kqs, axis=0)
ax2.plot(x_plot_true, mu[x_plot_nn], label=f'NN')
ax2.fill_between(x_plot_true, (mu - std)[x_plot_nn], (mu + std)[x_plot_nn], facecolor='orange', alpha=0.4)

mu, std = np.mean(y_preds, axis=0), np.std(y_preds, axis=0)
ax3.plot(x_plot_true, mu[x_plot_nn], label=f'NN')
ax3.fill_between(x_plot_true, (mu - std)[x_plot_nn], (mu + std)[x_plot_nn], facecolor='orange', alpha=0.4)

# Add legends.
for ax in [ax1, ax2, ax3]:
    ax.legend(loc='lower right')

# Add y-labels.
ax1.set_ylabel(r'$\partial |V_{%d}| / \partial P_{%d}$' % (x_i, x_i))
ax2.set_ylabel(r'$\partial |V_{%d}| / \partial Q_{%d}$' % (x_i, x_i))
# ax3.set_ylabel(r'$\Delta |V_{%d}|$' % (x_i))
ax3.set_ylabel(r'$|V_{%d}|$' % (x_i))

ax3.set_xlabel('Time step (s)')

exp_logger.save_figure(fig, f'multi_nets_{dataset_train}_{sensor_class}.pdf')
plt.show()

print('Done!')

if __name__ == '__main__':
    pass
