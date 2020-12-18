import numpy as np
from tqdm import tqdm
from warnings import warn


def linear_model(y, X, use_sigma, tau, freq, which_i, k_pcr, qr,
                 valid_timesteps):
    """
    Solve the least squares problem :math:`y = X \theta + e`.

    Parameters
    ----------
    y : (M, T) array_like
        The dependent variable measurements.
    X : (N, T) array_like
        The power injection measurements.
    use_sigma : bool
        Set to False to use :math:`\Sigma = I`.
    tau : int
        The size of the time windows.
    freq : int
        How often to estimate coefficients.
    which_i : (x_num,) array_like
        Which x_i coefficients to estimate
    k_pcr : int or None
        If None, PCR step is skipped. Otherwise, `k` principal components are
        kept.
    qr : bool
        Set to True to solve the LS problem through QR-decomposition.
    valid_timesteps : (T,) array_like
        A boolean mask that indicates which measurements are valid.

    Returns
    -------
    S : dict of {int : (N, T') array_like}
        The matrices of sensitivity coefficients for each x_i, where T' = T/freq.
    ts_linear : array_like
        At which time steps the coefficients where computed.
    cond_nums : array_like
        The condition number for estimation.
    """

    # Warnings.
    if qr:
        warn('Solving using QR-decomposition ignores \Sigma.')

    M, T = y.shape
    N = X.shape[0]
    do_pcr = k_pcr is not None

    # The time steps at which to compute coefficients.
    ts = np.arange(max(tau, freq), T, freq)

    # Empty arrays to store coefficients.
    S = np.zeros((which_i.size, N, ts.size))
    cond_nums = []

    for t_idx, t in enumerate(tqdm(ts)):

        # Select the last `tau` timesteps before time t that are valid.
        mask = np.argwhere(valid_timesteps[:t])[max(0, -tau):, 0]
        mask = mask[max(0, mask.size - tau):]

        # Select time window.
        # The -1 comes from the fact that `which_i` variable elements start at 1.
        y_window = y[which_i-1][:, mask].T  # (tau, x_num)
        X_window = X[:, mask].T        # (tau, N)
        m = X_window.shape[0]

        # Select power injection nodes that have non-zero injections.
        mask = np.where(np.mean(np.abs(X_window), axis=0) > 1e-2)

        # Construct correlation matrix.
        if use_sigma:
            temp = - 0.5 * np.ones(m - 1)
            sigma = np.identity(m) + np.diag(temp, k=1) + np.diag(temp, k=-1)
        else:
            sigma = np.identity(m)

        # Do the Principal Component Regression step.
        if do_pcr:
            X_window, V_k = pcr(X_window, k_pcr)

        # Solve the least squares problem.
        beta, cond_num = [], []
        for j in range(len(which_i)):
            if not qr:
                b, cn = _solve_ls(X_window, y_window[:, j], sigma)  # (2N,)
            else:
                b, cn = _solve_ls_qr(X_window, y_window[:, j])      # (2N,)
            beta.append(b)
            cond_num.append(cn)
        beta = np.vstack(beta)        # (M, 2N)
        cond_num = np.mean(cond_num)

        # Transform solution back to the original domain if PCR was done.
        if do_pcr:
            beta = beta.dot(V_k.T)

        # Store results.
        S[:, :, t_idx] = beta
        cond_nums.append(cond_num)

    # Construct the dictionary to return.
    S_dict = {i: S[idx] for idx, i in enumerate(which_i)}
    cond_nums = np.array(cond_nums)

    return S_dict, ts, cond_nums


def _solve_ls(X, y, sigma):
    """Solve the least squares problem :math:`y = X \beta + e`."""

    sigma_inv = np.linalg.inv(sigma)
    beta = X.T.dot(sigma_inv).dot(X)
    if np.linalg.matrix_rank(beta) != beta.shape[1]:
        print('Singular matrix')
    cond_num = np.linalg.cond(beta)
    beta = np.linalg.inv(beta).dot(X.T).dot(sigma_inv).dot(y)  # (2N, M)

    return beta, cond_num

def _solve_ls_qr(X, y):
    """Solve the least squares problem through QR decomposition."""

    q, r = np.linalg.qr(X)
    p = np.dot(q.T, y)
    cond_num = np.linalg.cond(r)
    beta = np.dot(np.linalg.inv(r), p)

    return beta, cond_num

def pcr(X, k):
    """Compute the singular value decomposition of X and keep k components. """

    X = X - np.mean(X, axis=0)
    _, _, V = np.linalg.svd(X, full_matrices=False)
    V_k = V.T[:, :k]
    X_k = X.dot(V_k)

    return X_k, V_k