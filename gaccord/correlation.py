import numpy as np


def compute_partial_correlation(theta):
    # D: diagonal matrix with diagonal elements of Theta
    D = np.diag(np.diag(theta))

    # Compute D^(-1/2)
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))

    # Compute P = -D^(-1/2) * Theta * D^(-1/2)
    return -D_inv_sqrt @ theta @ D_inv_sqrt


def compute_simple_correlation(theta):
    # Compute Sigma = Theta^(-1)
    sigma = np.linalg.inv(theta)

    # D: diagonal matrix with diagonal elements of Sigma
    D = np.diag(np.diag(sigma))

    # Compute D^(-1/2)
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))

    # Compute R = D^(-1/2) * Sigma * D^(-1/2)
    return D_inv_sqrt @ sigma @ D_inv_sqrt
