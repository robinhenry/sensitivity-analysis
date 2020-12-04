import numpy as np


def project(X, std_abs, std_arg):
    """
    Project measurement noise from polar coordinates to rectangular ones.

    Parameters
    ----------
    X : (N, T) array_like
        The complex variables (true values).
    std_abs : float
        The standard deviation of noise w.r.t. absolute value.
    std_arg : float
        The standard deviation of noise w.r.t. argument.

    Returns
    -------
    std_real : float
        The standard deviation of noise w.r.t. real part.
    std_imag : float
        The standard deviation of noise w.r.t. imaginary part.
    """
    X_abs = np.abs(X)
    X_arg = np.angle(X)

    var_abs = std_abs ** 2
    var_arg = std_arg ** 2

    var_real = X_abs**2 * np.exp(-2 * var_arg) * ((np.cos(X_arg)**2 * (np.cosh(2 * var_arg) - np.cosh(var_arg))) + np.sin(X_arg)**2 *  (np.sinh(2*var_arg) - np.sinh(var_arg))) \
               + var_abs * np.exp(-2 * var_arg) * (np.cos(X_arg)**2 * (2 * np.cosh(2 * var_arg) - np.cosh(var_arg)) + np.sin(X_arg)**2 * (2 * np.sinh(2 * var_arg) - np.sinh(var_arg)))

    var_imag = X_abs**2 * np.exp(-2 * var_arg) * ((np.sin(X_arg)**2 * (np.cosh(2 * var_arg) - np.cosh(var_arg))) + np.cos(X_arg)**2 * (np.sinh(2 * var_arg) - np.sinh(var_arg))) \
               + var_abs * np.exp(-2 * var_arg) * (np.sin(X_arg)**2 * (2 * np.cosh(2 * var_arg) - np.cosh(var_arg)) + np.cos(X_arg)**2 * (2 * np.sinh(2 * var_arg) - np.sinh(var_arg)))

    std_real = np.sqrt(var_real)
    std_imag = np.sqrt(var_imag)

    return std_real, std_imag


def add_noise(X, std_abs, std_ang):
    """
    Add white Gaussian noise to |X| and the angle of X.

    Parameters
    ----------
    X : (N, T) array_like
        The true complex phasor measurements.
    std_abs : float
        The std of the white Gaussian noise to add to |X|.
    std_ang : float
        The std of the white Gaussian noise to add to the angle of X.

    Returns
    -------
    (N, T) array_like
        The noisy matrix X.
    """
    N, T = X.shape

    # Sample Gaussian i.i.d. errors.
    e_abs = np.random.multivariate_normal(mean=np.zeros(N),
                                          cov=np.diag(std_abs ** 2 * np.ones(N)),
                                          size=T)
    e_ang = np.random.multivariate_normal(mean=np.zeros(N),
                                          cov=np.diag(std_ang ** 2 * np.ones(N)),
                                          size=T)

    X_noisy = np.abs(X) * (1 + e_abs.T) * np.exp(1j * (np.angle(X) + e_ang.T))

    return X_noisy