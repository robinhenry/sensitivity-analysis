import numpy as np


def compute_crlb(H, noise_std, use_sigma):
    m = H.shape[0]

    if use_sigma:
        temp = - 0.5 * np.ones(m - 1)
        sigma = np.identity(m) + np.diag(temp, k=1) + np.diag(temp, k=-1)
    else:
        sigma = np.eye(m)

    H_squared = H.T.dot(np.linalg.inv(sigma)).dot(H)
    cn = np.linalg.cond(H_squared)

    if np.linalg.matrix_rank(H_squared) == H_squared.shape[0]:
        bound = noise_std **2 * np.linalg.inv(H_squared)
        bounds = np.diag(bound)
    else:
        bounds = np.nan * np.ones(H_squared.shape[0])

    return bounds, cn
