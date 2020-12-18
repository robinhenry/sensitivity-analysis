import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot

from meng.data_loader import load_true_data, simulate_noisy_meas
from meng import my_io


def run(N_exp, dataset, sensor_class):

    # Load true load flow.
    V_org, I_org, P_org, Q_org, current_idx = load_true_data(dataset)

    V_real_noise_all = []
    V_imag_noise_all = []

    for n in range(N_exp):

        # Add noise to load flow data.
        V_meas, _, P_meas, Q_meas, std_abs, std_ang, std_real, std_imag, _, _ = \
            simulate_noisy_meas(sensor_class, V_org, I_org, current_idx)

        # Compute noise on real part of voltage.
        # V_real_noise_all.append((V_org.real - V_meas.real) / std_real)
        # V_imag_noise_all.append((V_org.imag - V_meas.imag) / std_imag)
        V_real_noise_all.append((V_org.real - V_meas.real))
        V_imag_noise_all.append((V_org.imag - V_meas.imag))

    V_real_noise_all = np.array(V_real_noise_all).flatten()
    V_imag_noise_all = np.array(V_imag_noise_all).flatten()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))

    probplot(V_real_noise_all, plot=ax1)
    ax1.set_ylabel('Quantiles of real part of $X$ error')
    ax1.set_xlabel('Standard normal quantiles')
    ax1.set_title('')

    probplot(V_imag_noise_all, plot=ax2)
    ax2.set_ylabel('Quantiles of imaginary part of $X$ error')
    ax2.set_xlabel('Standard normal quantiles')
    ax2.set_title('')

    # Change size of scatter points.
    ax1.get_lines()[0].set_markersize(5.0)
    ax2.get_lines()[0].set_markersize(5.0)

    my_io.save_figure(f'figures/qq_{sensor_class}_{dataset}.png', fig)
    plt.show()

    print('Done!')

if __name__ == '__main__':

    config = {
        'N_exp': 1,
        'dataset': 'cigre13',
        'sensor_class': 1.
    }

    run(**config)
