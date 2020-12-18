import numpy as np
import matplotlib.pyplot as plt
import os

from meng.data_loader import load_true_data, simulate_noisy_meas, load_coefficients
from meng import constants
from meng.linear_models import utils
from meng.neural_nets.dataloaders import build_training_dataloader, build_testing_dataloader
from meng.neural_nets import core as nn
from meng.neural_nets.models import FeedForward
from meng.my_io import ExperimentLogger

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run(dataset, sensor_class, which_i, coeff_type, k_nn, epochs, folder,
        T_train, freq):

    # Create the folder in which the experiment results will be stored.
    exp_logger = ExperimentLogger(folder, locals())

    # Load true load flow.
    V_org, I_org, P_org, Q_org, current_idx = load_true_data(dataset)
    T = V_org.shape[1]

    # Extract the list of which coefficients to estimate.
    which_i = utils.unpack_which_i(which_i, V_org.shape[0])

    # Select the noise std that corresponds to the type of coefficient(s).
    std_abs = constants.SENSOR_STD_ABS[sensor_class]
    std_arg = constants.SENSOR_STD_ANG[sensor_class]

    # Transform voltage phasor measurements into either
    # (|x|, Re{x}, or Im{x}) based on `coeff_type`.
    X_org_a, X_org_b = utils.dependent_vars_setup(V_org, coeff_type)

    # Remove slack bus measurements and create delta matrices.
    dPQ_org = np.vstack((np.diff(P_org[1:], axis=1),
                         np.diff(Q_org[1:], axis=1)))
    dX_org_a = np.diff(X_org_a[1:], axis=1)

    # Add noise to load flow data.
    V_meas, _, P_meas, Q_meas, _, _, _, _, _, _ = \
        simulate_noisy_meas(sensor_class, V_org, I_org, current_idx)

    # Transform voltage phasor measurements into either
    # (|x|, Re{x}, or Im{x}) based on `coeff_type`.
    X_meas_a, X_meas_b = utils.dependent_vars_setup(V_meas, coeff_type)

    # Remove slack bus measurements and create delta matrices.
    dPQ_meas = np.vstack((np.diff(P_meas[1:], axis=1),
                          np.diff(Q_meas[1:], axis=1)))
    dX_meas_a = np.diff(X_meas_a[1:], axis=1)

    # Load the true sensitivity coefficients of interest.
    coefficients = load_coefficients(dataset)
    S_a_true, S_b_true = utils.coefficients_setup(coefficients, coeff_type, which_i)
    del coefficients

    # Compute the true voltage magnitude deviations (from the load flow).
    dV_load_flow = np.diff(np.abs(V_org[1:]), axis=1)

    # T_plot = 200
    t_nn = np.arange(T_train + k_nn, T)  # which timesteps the NN estimates.
    t_plot = np.arange(T_train + k_nn, T, freq)   # which timesteps should be plotted.
    t_nn_plot = np.arange(0, T - T_train - k_nn, freq)  # the idx of the nn estimations that should be plotted

    X_train, X_test = dX_meas_a[:, :T_train], dX_meas_a[:, T_train:]
    PQ_train, PQ_test = dPQ_meas[:, :T_train], dPQ_meas[:, T_train:]

    fig_1, axs_1 = plt.subplots(len(which_i), 1, sharex=True, figsize=(10, 2.5*len(which_i)))
    fig_2, axs_2 = plt.subplots(len(which_i), 1, sharex=True, figsize=(10, 2.5*len(which_i)))

    for idx, x_i in enumerate(which_i):

        training_dataset = build_training_dataloader(X_train, PQ_train, x_i, k_nn)
        in_shape = (dX_meas_a.shape[0] + dPQ_meas.shape[0]) * k_nn
        hidden_shapes = [128, 128]
        sc_matrix_shape = (dPQ_meas.shape[0])

        model = FeedForward(in_shape, hidden_shapes, sc_matrix_shape, k_nn)
        train_loss = nn.train(model, training_dataset, lr=1e-3, epochs=epochs, l2=0.)

        # Use the neural net to estimate the coefficients for the last 12 hours.
        testing_dataset = build_testing_dataloader(X_test, PQ_test, x_i, k_nn)
        S_a_nn, dV_nn, dV_nn_true = model.predict(testing_dataset)

        # Plot estimated coefficient.
        ax = axs_1[idx]
        ax.plot(t_plot, S_a_true[x_i][x_i - 1][t_plot], label='True')
        ax.plot(t_plot, S_a_nn[x_i - 1][t_nn_plot], label='Neural net')
        ax.legend(loc='upper right')

        # Plot predicted d|X|.
        ax = axs_2[idx]
        ax.plot(t_plot, dV_load_flow[x_i-1][t_plot], label='True')
        ax.plot(t_plot, dV_nn[t_nn_plot], label='Neural net')
        ax.legend(loc='upper right')

    exp_logger.save_figure(fig_1, 'coefficients')
    exp_logger.save_figure(fig_2, 'dV')

    plt.show()

    print('Done!')

if __name__ == '__main__':
    config = {
        'dataset': 'cigre13',
        'sensor_class': 0.5,
        'which_i': [9, 10, 11],
        'coeff_type': 'magn',
        'k_nn': 100,
        'epochs': 25,
        'folder': 'nn_single_coeff',
        'freq': 500,
        'T_train': 43_000,
    }

    run(**config)
