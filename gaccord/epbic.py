import numpy as np


def count_nonzero_entries(matrix):
    """Count the number of nonzero entries in a matrix."""
    return np.count_nonzero(matrix)


def compute_g(Xn, S):
    """Compute g(Ω^) = -logdet(diag(Xn)) + (1/2) * tr(Xn^T Xn S)."""
    return -np.sum(np.log(np.diag(Xn))) + 0.5 * np.trace(Xn.T @ Xn @ S)


def compute_epBIC(Xn, S, gamma=0.5):
    """Compute the epBIC score for a given Xn."""
    (n, p) = S.shape

    g_hat = compute_g(Xn, S)
    nonzero_count = count_nonzero_entries(Xn)

    return (
        (2 * n * g_hat)
        + (nonzero_count * np.log(n))
        + (4 * gamma * nonzero_count * np.log(p))
    )
