import os
import numpy as np
import matplotlib.pyplot as plt

from meng.data_loader import load_voltage_exp_data
import meng.lstm.core as core
from meng.lstm.models import LSTM
from meng.lstm.dataloader import build_dataloader
from meng.my_io import ExperimentLogger

os.environ['KMP_DUPLICATE_LIB_OK']='True'


####################### PARAMETERS #######################

# General parameters.
dataset_train = 'cigre13'
sensor_class = 0.5
folder = 'lstm'

# LSTM parameters.
hidden_layer_size = 100
time_window = 100
epochs = 3
batch_size = 64
T_train = 12 * 60 * 60
T_test = 12 * 60 * 60

# Plot parameters.
y_lims = [-0.5, 0.5]

# Create the folder in which the experiment results will be stored.
exp_logger = ExperimentLogger(folder, locals())

for x_i in range(12):

    ####################### LOAD TRAINING DATA #######################
    P_meas_tr, Q_meas_tr, V_meas_tr, P_org, Q_org, V_org, coefficients, std_abs, \
    std_V_real, std_V_imag = load_voltage_exp_data(dataset_train, sensor_class)
    N, T = P_meas_tr.shape

    # Remove slack measurements and create measurement matrices used in Model 2.
    V_meas_magn_tr = np.abs(V_meas_tr)
    PQ_meas_tr = np.vstack((P_meas_tr, Q_meas_tr))

    # Create measurement delta matrices used in Model 1.
    dV_org = np.diff(np.abs(V_org[1:]), axis=1)
    dV_meas_magn_tr = np.diff(V_meas_magn_tr, axis=1)
    dPQ_meas_tr = np.diff(PQ_meas_tr, axis=1)

    # Create sequences to be used during training.
    dv = dV_meas_magn_tr[:, :T_train]
    dpq = dPQ_meas_tr[:, :T_train]
    train_dataloader = build_dataloader(dv, dpq, x_i, time_window, batch_size)


    ####################### LOAD TESTING DATA #######################

    # Create sequences to be used during testing.
    dv = dV_meas_magn_tr[:, T_train: T_train + T_test]
    dpq = dPQ_meas_tr[:, T_train: T_train + T_test]
    test_dataloader = build_dataloader(dv, dpq, x_i, time_window, batch_size)

    # Load true coefficients.
    Kp_true = coefficients['vmagn_p'][x_i - 1, x_i - 1][T_train + time_window: T_train + T_test]
    Kq_true = coefficients['vmagn_q'][x_i - 1, x_i - 1][T_train + time_window: T_train + T_test]
    dV_true = dV_org[x_i - 1, T_train + time_window: T_train + T_test]

    ####################### TRAIN LSTM #######################

    # Initialize LSTM.
    model = LSTM(3 * N, hidden_layer_size, 2 * N, batch_size)

    # Train LSTM.
    core.train(model, train_dataloader, epochs, lr=1e-3, batch_size=batch_size)


    ####################### TEST LSTM #######################
    coeff_predicted, v_predicted, v_target = core.predict(model, test_dataloader)


    ####################### PLOT RESULTS #######################

    # Plot true vs. estimated values.
    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    x_plot = np.arange(0, T_test, int(T_test / 200))

    ax1.plot(Kp_true, label='True')
    ax1.plot(coeff_predicted[x_i - 1], label='LSTM')
    ax1.set_ylim(y_lims)
    ax1.legend()
    ax1.set_ylabel(r'$\partial |V_{%d}| / \partial P_{%d}$' % (x_i, x_i))

    ax2.plot(Kq_true, label='True Kq')
    ax2.plot(coeff_predicted[x_i - 1 + N], label='LSTM')
    ax2.set_ylim(y_lims)
    ax2.legend()
    ax2.set_ylabel(r'$\partial |V_{%d}| / \partial Q_{%d}$' % (x_i, x_i))

    ax3.plot(x_plot, dV_true[x_plot], label='True')
    # ax3.plot(x_plot, v_target[x_plot], label='Noisy')
    ax3.plot(x_plot, v_predicted[x_plot], label='LSTM')
    ax3.legend()
    ax3.set_ylabel(r'$\Delta |V_{%d}|$' % (x_i))
    ax3.set_xlabel('Time step (s)')

    exp_logger.save_figure(fig, f'estimations_x{x_i}.pdf')

    plt.show()

    print('Done!')


if __name__ == '__main__':
    pass
