import _gaccord as _accord
import numpy as np
import time
from datetime import datetime
import psutil
import os
from gaccord.epbic import compute_epBIC


def check_symmetry(a, rtol=1e-5, atol=1e-8):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def get_lambda_matrix(lam1, S):
    if isinstance(lam1, float) | isinstance(lam1, int):
        lam_mat = np.full_like(S, lam1, order="F", dtype="float64")
    elif isinstance(lam1, np.ndarray):
        lam_mat = lam1
    return lam_mat


def accord(
    S,
    Omega_star,
    X_init=None,
    lam1=0.1,
    lam2=0.0,
    split="fbs",
    stepsize_multiplier=1.0,
    constant_stepsize=0.5,
    backtracking=True,
    epstol=1e-5,
    maxitr=100,
    penalize_diag=True,
    logging_interval=0,
):
    """
    Modified ACCORD algorithm for convergence analysis

    Parameters
    ----------
    S : ndarray of shape (n_features, n_features)
        Sample covariance matrix
    Omega_star : ndarray of shape (n_features, n_features)
        Proxy of converged Omega
    X_init : ndarray of shape (n_features, n_features)
        Initial omega value
    lam1 : float
        The l1-regularization parameter
    lam2 : float
        The l2-regularization parameter
    split : {'fbs', 'ista'}, default='fbs'
        The type of split
    stepsize_multiplier : int or float
        Multiplier for stepsize
    backtracking : bool, default=True
        Whether ot nor to perform backtracking with lower bound
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements
    logging_interval : int, default=0
        Logging interval of iterations (zero means no logging)

    Returns
    -------
    Omega : ndarray of shape (n_features, n_features)
        Estimated Omega
    hist : ndarray of shape (n_iters, 5)
        The list of values of (inner_iter_count, objective, successive_norm, omega_star_norm, iter_time) at each iteration until convergence
        inner_iter_count is included only when backtracking=True
    """
    assert type(S) == np.ndarray and S.dtype == "float64"

    if X_init is None:
        X_init = np.eye(S.shape[0])
    assert X_init.shape[0] == X_init.shape[1] == S.shape[0]

    lam_mat = get_lambda_matrix(lam1, S)

    assert type(lam_mat) == np.ndarray and lam_mat.dtype == "float64"

    if not penalize_diag:
        np.fill_diagonal(lam_mat, 0)

    hist_inner_itr_count = np.full((maxitr, 1), -1, order="F", dtype="int32")
    hist_hn = np.full((maxitr, 1), -1, order="F", dtype="float64")
    hist_successive_norm = np.full((maxitr, 1), -1, order="F", dtype="float64")
    hist_norm = np.full((maxitr, 1), -1, order="F", dtype="float64")
    hist_iter_time = np.full((maxitr, 1), -1, order="F", dtype="float64")

    tau = (stepsize_multiplier * 1) / np.linalg.svd(S)[1][0]

    if split == "fbs":
        if backtracking:
            Omega = _accord.accord_fbs_backtracking(
                S,
                X_init,
                Omega_star,
                lam_mat,
                lam2,
                epstol,
                maxitr,
                tau,
                penalize_diag,
                hist_inner_itr_count,
                hist_hn,
                hist_successive_norm,
                hist_norm,
                hist_iter_time,
                logging_interval,
            )
            hist = np.hstack(
                [
                    hist_inner_itr_count,
                    hist_hn,
                    hist_successive_norm,
                    hist_norm,
                    hist_iter_time,
                ]
            )
        else:
            Omega = _accord.accord_fbs_constant(
                S,
                X_init,
                Omega_star,
                lam_mat,
                lam2,
                epstol,
                maxitr,
                tau,
                penalize_diag,
                hist_hn,
                hist_successive_norm,
                hist_norm,
                hist_iter_time,
                logging_interval,
            )
            hist = np.hstack([hist_hn, hist_successive_norm, hist_norm, hist_iter_time])
    elif split == "ista":
        if backtracking:
            Omega = _accord.accord_ista_backtracking(
                S,
                X_init,
                Omega_star,
                lam_mat,
                lam2,
                epstol,
                maxitr,
                hist_inner_itr_count,
                hist_hn,
                hist_successive_norm,
                hist_norm,
                hist_iter_time,
                logging_interval,
            )
            hist = np.hstack(
                [
                    hist_inner_itr_count,
                    hist_hn,
                    hist_successive_norm,
                    hist_norm,
                    hist_iter_time,
                ]
            )
        else:
            Omega = _accord.accord_ista_constant(
                S,
                X_init,
                Omega_star,
                lam_mat,
                lam2,
                epstol,
                maxitr,
                constant_stepsize,
                hist_hn,
                hist_successive_norm,
                hist_norm,
                hist_iter_time,
                logging_interval,
            )
            hist = np.hstack([hist_hn, hist_successive_norm, hist_norm, hist_iter_time])

    hist = hist[np.where(hist[:, 0] != -1)]

    return Omega, hist


class GraphicalAccord:
    """
    Modified ACCORD algorithm for convergence analysis

    Parameters
    ----------
    Omega_star : ndarray of shape (n_features, n_features)
        Proxy of converged Omega
    lam1_values : array of float
        The l1-regularization parameter
    gamma: float
        The constant for epBIC calculation
    lam2_values : array of float
        The l2-regularization parameter
    split : {'fbs', 'ista'}, default='fbs'
        The type of split
    stepsize_multiplier : int or float
        Multiplier for stepsize
    backtracking : bool, default=True
        Whether ot nor to perform backtracking with lower bound
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements
    logging_interval : int, default=0
        Logging interval of iterations (zero means no logging)

    Attributes
    ----------
    omega_ : ndarray of shape (n_features, n_features)
        Estimated Omega
    hist_ : ndarray of shape (n_iters, 5)
        The list of values of (inner_iter_count, objective, successive_norm, omega_star_norm, iter_time) at each iteration until convergence
        inner_iter_count is included only when backtracking=True
    """

    def __init__(
        self,
        Omega_star=None,
        lam1_values=[0.1],
        gamma=0.5,
        lam2_values=[0.0],
        split="fbs",
        stepsize_multiplier=1.0,
        constant_stepsize=0.5,
        backtracking=True,
        epstol=1e-5,
        maxitr=100,
        penalize_diag=True,
        logging_interval=0,
    ):
        self.Omega_star = Omega_star
        self.lam1_values = lam1_values
        self.gamma = gamma
        self.lam2_values = lam2_values
        self.split = split
        self.stepsize_multiplier = stepsize_multiplier
        self.constant_stepsize = constant_stepsize
        self.backtracking = backtracking
        self.epstol = epstol
        self.maxitr = maxitr
        self.penalize_diag = penalize_diag
        self.logging_interval = logging_interval
        self.epbic_values = []

    def fit(self, X, y=None, initial=None):
        """
        Fit ACCORD

        Parameters
        ----------
        X       : ndarray, shape (n_samples, p_features)
                Data from which to compute the inverse covariance matrix
        initial : ndarray, shape (p_features, p_features)
                Warm up data from which to be initial point
        y       : (ignored)
        """
        assert initial is None or initial.shape[0] == initial.shape[1] == X.shape[1]

        S = np.matmul(X.T, X) / X.shape[0]

        # Get initial process memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

        start_time = time.time()

        best_lam1 = None
        best_epBIC = float("inf")

        for lam1 in self.lam1_values:
            for lam2 in self.lam2_values:
                print(
                    f'[ACCORD]{datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")} ACCORD started with lam1: {lam1}, lam2: {lam2}'
                )
                omega, hist = accord(
                    S,
                    X_init=initial,
                    Omega_star=self.Omega_star,
                    lam1=lam1,
                    lam2=lam2,
                    split=self.split,
                    stepsize_multiplier=self.stepsize_multiplier,
                    constant_stepsize=self.constant_stepsize,
                    backtracking=self.backtracking,
                    epstol=self.epstol,
                    maxitr=self.maxitr,
                    penalize_diag=self.penalize_diag,
                    logging_interval=self.logging_interval,
                )
                if hist[-1][2] > self.epstol:
                    print(
                        f'[WARNING]{datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")} The result does not converge.'
                    )

                epBIC_value = compute_epBIC(omega.toarray(), S, self.gamma)
                self.epbic_values.append(epBIC_value)
                if epBIC_value < best_epBIC:
                    best_epBIC = epBIC_value
                    best_lam1 = lam1
                    best_lam2 = lam2
                    self.omega_ = omega
                    self.hist_ = hist

        print(
            f"[LOG] Selected lam1: {best_lam1}, lam2: {best_lam2}, epBIC: {best_epBIC}"
        )

        end_time = time.time()
        # Get final process memory
        mem_after = process.memory_info().rss

        print(f"[LOG] Execution Time: {(end_time - start_time):.2f} seconds")
        print(f"[LOG] Memory Usage: {(mem_after - mem_before):,} bytes")

        return self
