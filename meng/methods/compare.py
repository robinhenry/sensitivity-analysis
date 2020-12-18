import matplotlib.pyplot as plt
import numpy as np

from meng.my_io import EstimationLogger, ComparisonLogger
from meng.methods import utils
from meng.metrics import normalized_error


####################### PARAMETERS #######################

# General parameters.
dataset = 'cigre13'
sensor_classes = [0.] #[0.5, 1., 0.2]
x_is = [11, 10, 5]
N_exp = 1

# Plot parameters.
alpha = 0.3
colors = ['tab:%s' % c for c in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']]

coeff_labels = ['Linear M1', 'Linear M2', 'NN M1', 'NN M2', 'LSTM M1', 'LSTM M2']
X_labels = ['Linear magn', 'Linear Re/Im', 'NN magn', 'NN Re/Im', 'LSTM magn', 'LSTM Re/Im']
coeff_filename_tags = ['magn', 'real', 'imag']
X_filename_tags = ['dV', 'V']

# Timesteps that correspond to the estimations.
# Last 6 hours, with time windows of 1000 and a frequency of 2 min.
ts = np.arange(18 * 3600 + 1000, 24 * 3600 - 1, 2 * 60)
ts_hour = ts / 3600


####################### PLOTTING FUNCTIONS #######################

def make_master_Kx_plot(data_all, y_true_label, y_est_label, ylabel, y_true_idx, methods_idx):

    # Create an empty figure.
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 5, hspace=0.05)
    axs, axs_box = [], []

    # Iterate over nodes and plot each one on a different row.
    for x_idx, x_i in enumerate(x_is):

        # Create a new subplot for the estimated coefficient.
        ax = fig.add_subplot(gs[x_idx, :-1])

        data = data_all[x_idx]
        true = data[y_true_label][y_true_idx]
        ax.plot(ts_hour, true, label='True', color=colors[0])

        # Create a new subplot for the estimation errors.
        ax_box = fig.add_subplot(gs[x_idx, -1])

        for idx, i in enumerate(methods_idx):

            col = colors[idx + 1]

            # Extract estimated coefficient and compute mean and std.
            c = data[y_est_label][:, i]
            mu, std = np.mean(c, axis=0), np.std(c, axis=0)

            # Plot estimations.
            ax.plot(ts_hour, mu, label=coeff_labels[i // 3], color=col)
            ax.fill_between(ts_hour, mu - std, mu + std, alpha=alpha, facecolor=col)

            # Plot error boxplot.
            e = 100 * normalized_error(true, c).flatten()
            ax_box.boxplot(e, positions=[idx], labels=[coeff_labels[i // 3]],
                           showfliers=False)

        ax.set_ylabel(ylabel % (x_i, x_i))

        axs.append(ax)
        axs_box.append(ax_box)

    axs[1].legend(loc='lower right')
    for ax in axs[:-1] + axs_box[:-1]:
        ax.set_xticks([])
    axs[-1].set_xlabel('Time (hour)')

    axs_box[1].set_ylabel('Normalized error [%%]')
    axs_box[-1].xaxis.set_tick_params(rotation=45)

    gs.tight_layout(fig)

    return fig, axs, axs_box


def make_master_X_plot(data_all, y_true_label, y_est_label, ylabel, y_true_idx, methods_idx):

    # Create an empty figure.
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 5, hspace=0.05)
    axs, axs_box = [], []

    # Iterate over nodes and plot each one on a different row.
    for x_idx, x_i in enumerate(x_is):

        # Create a new subplot for the time series.
        ax = fig.add_subplot(gs[x_idx, :-1])

        data = data_all[x_idx]
        true = np.abs(data[y_true_label][y_true_idx])
        ax.plot(ts_hour, true, label='True', color=colors[0])

        # Create a new subplot for the estimation errors.
        ax_box = fig.add_subplot(gs[x_idx, -1])

        for idx, i in enumerate(methods_idx):

            col = colors[idx + 1]

            # Extract estimated values and compute mean and std.
            c = np.abs(data[y_est_label][:, i])
            mu, std = np.mean(c, axis=0), np.std(c, axis=0)

            # Plot estimations.
            ax.plot(ts_hour, mu, label=X_labels[idx], color=col)
            ax.fill_between(ts_hour, mu - std, mu + std, alpha=alpha, facecolor=col)

            # Plot error boxplot.
            e = 100 * normalized_error(true, c).flatten()
            ax_box.boxplot(e, positions=[idx], labels=[X_labels[idx]],
                           showfliers=False)

        ax.set_ylabel(ylabel % (x_i))

        axs.append(ax)
        axs_box.append(ax_box)

    axs[1].legend(loc='lower right')
    for ax in axs[:-1] + axs_box[:-1]:
        ax.set_xticks([])
    axs[-1].set_xlabel('Time (hour)')

    axs_box[1].set_ylabel('Normalized error [%%]')
    axs_box[-1].xaxis.set_tick_params(rotation=45)

    gs.tight_layout(fig)

    return fig, axs, axs_box


################### PLOT RESULTS ####################

for sensor_class in sensor_classes:

    folder = f'{dataset}_{sensor_class}'
    logger = ComparisonLogger(folder)


    ################### LOAD SAVED ESTIMATIONS ####################

    data_all = []
    for x_i in x_is:

        # Load saved estimations.
        f = utils.estimations_filename(dataset, sensor_class, x_i, N_exp)
        est_logger = EstimationLogger()
        data = est_logger.load_estimation(f)

        data_all.append(data)


    ################### PLOT KP COEFFICIENTS ####################

    ylabels = [r'$\partial |V_{%d}|/\partial P_{%d}$',
               r'$\partial Re\{V_{%d}\}/\partial P_{%d}$',
               r'$\partial Im\{V_{%d}\}/\partial P_{%d}$']

    # KP for |V|, Re{V} and Im{V}
    y_true_label = 'Kp_true'
    y_est_label = 'Kp_est'
    idx = np.array([0, 6, 9, 12, 15])

    for i in range(3):
        ylabel = ylabels[i]
        methods_idx = idx + i
        fig, axs, axs_box = make_master_Kx_plot(data_all, y_true_label, y_est_label, ylabel, i, methods_idx)
        logger.save_fig(fig, f'KP_{coeff_filename_tags[i]}.png')


    ################### PLOT KQ COEFFICIENTS ####################

    ylabels = [r'$\partial |V_{%d}|/\partial Q_{%d}$',
               r'$\partial Re\{V_{%d}\}/\partial Q_{%d}$',
               r'$\partial Im\{V_{%d}\}/\partial Q_{%d}$']

    # KQ for |V|, Re{V} and Im{V}
    y_true_label = 'Kq_true'
    y_est_label = 'Kq_est'
    idx = np.array([0, 6, 9, 12, 15])

    for i in range(3):
        ylabel = ylabels[i]
        methods_idx = idx + i
        fig, axs, axs_box = make_master_Kx_plot(data_all, y_true_label, y_est_label, ylabel, i, methods_idx)
        logger.save_fig(fig, f'KQ_{coeff_filename_tags[i]}.png')


    ################### PLOT d|V| and |V| ESTIMATIONS ####################

    ylabels = [r'$\Delta |V_{%d}|$',
               r'$|V_{%d}|$']

    y_true_label = 'X_true'
    y_est_label = 'X_est'
    idx = [[0, 1, 4, 5, 8, 9], [2, 3, 6, 7, 10, 11]]

    for i in range(2):
        y_true_idx = i
        methods_idx = idx[i]
        ylabel = ylabels[i]
        fig, ax, axs_box = make_master_X_plot(data_all, y_true_label, y_est_label, ylabel, y_true_idx, methods_idx)
        logger.save_fig(fig, f'{X_filename_tags[i]}.png')


    ################### PLOT CRLB ####################
    def make_crlb_master_plot(data_all, crlb_label, y_est_label, ylabel, y_true_idx, methods_idx):

        # Create an empty figure.
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(3, 5, hspace=0.05)
        axs, axs_box = [], []

        # Iterate over nodes and plot each one on a different row.
        for x_idx, x_i in enumerate(x_is):

            # Create a new subplot for the estimated coefficient.
            ax = fig.add_subplot(gs[x_idx, :-1])

            data = data_all[x_idx]

            crlb = data[crlb_label][:, y_true_idx]
            mu, std = np.mean(crlb, axis=0), np.std(crlb, axis=0)

            ax.plot(ts_hour, mu, label='CRLB', color=colors[0])
            ax.fill_between(ts_hour, mu + std, mu - std, alpha=alpha, facecolor=colors[0])

            # Create a new subplot for the estimation errors.
            ax_box = fig.add_subplot(gs[x_idx, -1])

            for idx, i in enumerate(methods_idx):
                col = colors[idx + 1]

                # Extract estimated coefficient and compute mean and std.
                c = data[y_est_label][:, i]
                var = np.var(c, axis=0)

                # Plot variance of estimations.
                ax.plot(ts_hour, var, label=coeff_labels[i // 3], color=col)

                # # Plot error boxplot.
                # e = 100 * normalized_error(true, c).flatten()
                # ax_box.boxplot(e, positions=[idx], labels=[coeff_labels[i // 3]],
                #                showfliers=False)

            ax.set_ylabel(ylabel % (x_i, x_i))

            axs.append(ax)
            axs_box.append(ax_box)

        axs[1].legend(loc='lower right')
        for ax in axs[:-1] + axs_box[:-1]:
            ax.set_xticks([])
        axs[-1].set_xlabel('Time (hour)')

        axs_box[1].set_ylabel('Normalized error [%%]')
        axs_box[-1].xaxis.set_tick_params(rotation=45)

        gs.tight_layout(fig)

        return fig, axs, axs_box

    # Variance of KP coefficients.
    crlb_label = 'CRLB_P'
    y_est_label = 'Kp_est'
    ylabels = [r'Var of $\partial |V_{%d}|/\partial P_{%d}$',
               r'Var of $\partial Re\{V_{%d}\}/\partial P_{%d}$',
               r'Var of $\partial Im\{V_{%d}\}/\partial P_{%d}$',
               r'Var of $\partial |V_{%d}|/\partial P_{%d}$',
               r'Var of $\partial Re\{V_{%d}\}/\partial P_{%d}$',
               r'Var of $\partial Im\{V_{%d}\}/\partial P_{%d}$']

    idx = np.array([0, 6, 12])

    for i in range(6):
        ylabel = ylabels[i]
        y_true_idx = i
        methods_idx = idx + i
        fig, axs, axs_box = make_crlb_master_plot(data_all, crlb_label, y_est_label, ylabel,
                                                  y_true_idx, methods_idx)

    # Variance of KQ coefficients.
    crlb_label = 'CRLB_Q'
    y_est_label = 'Kq_est'
    ylabels = [r'Var of $\partial |V_{%d}|/\partial Q_{%d}$',
               r'Var of $\partial Re\{V_{%d}\}/\partial Q_{%d}$',
               r'Var of $\partial Im\{V_{%d}\}/\partial Q_{%d}$',
               r'Var of $\partial |V_{%d}|/\partial Q_{%d}$',
               r'Var of $\partial Re\{V_{%d}\}/\partial Q_{%d}$',
               r'Var of $\partial Im\{V_{%d}\}/\partial Q_{%d}$']

    idx = np.array([0, 6, 12])

    for i in range(6):
        ylabel = ylabels[i]
        y_true_idx = i
        methods_idx = idx + i
        fig, axs, axs_box = make_crlb_master_plot(data_all, crlb_label, y_est_label, ylabel,
                                                  y_true_idx, methods_idx)


    plt.show()

if __name__ == '__main__':
    pass