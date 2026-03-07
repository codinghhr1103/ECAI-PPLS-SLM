"""Model selection utilities for PPLS-SLM.

This module provides:
- observed-data Gaussian log-likelihood computation for the PPLS model
- BIC computation for selecting latent dimension r
- K-fold CV selection using prediction MSE (via conditional mean)

We keep the implementation conservative and numerically stable (Cholesky-based
trace and log-det where possible) and return +inf BIC when the covariance is
invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from ppls_slm.ppls_model import PPLSModel
from ppls_slm.apps.prediction import predict_conditional_mean


def _get(params: Mapping, *keys: str):
    for k in keys:
        if k in params:
            return params[k]
    raise KeyError(f"Missing keys {keys} in params (available={list(params.keys())})")


def _as_diag_matrix(x: np.ndarray) -> np.ndarray:
    """Accept either a diagonal vector (r,) or a diagonal matrix (r,r)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return np.diag(x)
    return x


def _center_joint(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Z = np.hstack([X, Y])
    Zc = Z - Z.mean(axis=0, keepdims=True)
    p = X.shape[1]
    return Zc[:, :p], Zc[:, p:]


def compute_observed_log_likelihood(
    X: np.ndarray,
    Y: np.ndarray,
    params: Mapping,
    *,
    assume_centered: bool = False,
) -> float:
    r"""Compute the full observed-data Gaussian log-likelihood ln L_N(\hat\Theta).

    Uses the full form (including (p+q) ln(2\pi)) so values are comparable across r.


    Parameters
    - X: (N,p)
    - Y: (N,q)
    - params: dict with keys W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2

    Returns
    - log_lik: float (can be -inf if covariance is invalid)
    """

    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    if not assume_centered:
        Xc, Yc = _center_joint(X, Y)
    else:
        Xc, Yc = X, Y

    N, p = Xc.shape
    q = Yc.shape[1]

    W = np.asarray(_get(params, "W"))
    C = np.asarray(_get(params, "C"))
    B = _as_diag_matrix(np.asarray(_get(params, "B")))
    Sigma_t = _as_diag_matrix(np.asarray(_get(params, "Sigma_t")))

    sigma_e2 = float(_get(params, "sigma_e2", "sigma2_e", "sigma_e2"))
    sigma_f2 = float(_get(params, "sigma_f2", "sigma2_f", "sigma_f2"))
    sigma_h2 = float(_get(params, "sigma_h2", "sigma2_h", "sigma_h2"))

    r = W.shape[1]
    model = PPLSModel(p, q, r)

    Sigma = model.compute_covariance_matrix(W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2)

    # Sample second moment about the mean
    Zc = np.hstack([Xc, Yc])
    S = (Zc.T @ Zc) / float(N)

    sign, log_det = np.linalg.slogdet(Sigma)
    if sign <= 0 or (not np.isfinite(log_det)):
        return -np.inf

    # trace(S Sigma^{-1}) via Cholesky solves when possible
    try:
        L = np.linalg.cholesky(Sigma)
        tmp = np.linalg.solve(L, S)
        trace_term = float(np.trace(np.linalg.solve(L.T, tmp)))
    except np.linalg.LinAlgError:
        try:
            trace_term = float(np.trace(S @ np.linalg.inv(Sigma)))
        except np.linalg.LinAlgError:
            return -np.inf

    log_lik = -float(N) / 2.0 * (
        (p + q) * float(np.log(2.0 * np.pi)) + float(log_det) + float(trace_term)
    )
    return float(log_lik)


def effective_num_parameters_fixed_noise(*, p: int, q: int, r: int) -> int:
    """Effective free parameters k(r) under fixed (sigma_e^2, sigma_f^2).

    Using orthonormality constraints for W and C (Stiefel manifolds), and diagonal
    B, Sigma_t, plus scalar sigma_h^2:

        k(r) = (p + q - r + 1) r + 1

    This matches the derivation in the paper's model selection discussion.
    """

    if r < 1:
        raise ValueError("r must be >= 1")
    if r > min(p, q):
        raise ValueError("r must be <= min(p,q)")
    return int((p + q - r + 1) * r + 1)


def compute_bic(
    X: np.ndarray,
    Y: np.ndarray,
    params: Mapping,
    *,
    r: int,
    assume_centered: bool = False,
) -> Tuple[float, float]:
    """Compute BIC(r) and log-likelihood for fitted params.

    Returns (bic, log_lik). If the covariance is invalid, returns (+inf, -inf).
    """

    N, p = X.shape
    q = Y.shape[1]

    log_lik = compute_observed_log_likelihood(X, Y, params, assume_centered=assume_centered)
    if not np.isfinite(log_lik):
        return float("inf"), float("-inf")

    k_r = effective_num_parameters_fixed_noise(p=p, q=q, r=int(r))
    bic = -2.0 * float(log_lik) + float(k_r) * float(np.log(N))
    return float(bic), float(log_lik)


@dataclass(frozen=True)
class BICResult:
    best_r: int
    bic: Dict[int, float]
    log_likelihood: Dict[int, float]


def select_r_by_bic(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r_candidates: Iterable[int],
    fit_fn: Callable[[np.ndarray, np.ndarray, int], Mapping],
    assume_centered: bool = False,
) -> BICResult:
    r"""Select r by minimising BIC(r).

    For each candidate r, we fit the model (via fit_fn) and compute

        BIC(r) = -2 ln L_N(\hat\Theta(r)) + k(r) ln N,


    where k(r) = (p + q - r + 1) r + 1 under fixed (sigma_e^2, sigma_f^2).

    Returns +inf BIC for failed fits / invalid covariance.
    """

    X = np.asarray(X)
    Y = np.asarray(Y)

    bic: Dict[int, float] = {}
    loglik: Dict[int, float] = {}

    for r in [int(r) for r in r_candidates]:
        try:
            params = fit_fn(X, Y, int(r))
            b, ll = compute_bic(X, Y, params, r=int(r), assume_centered=assume_centered)
        except Exception:
            b, ll = float("inf"), float("-inf")

        bic[int(r)] = float(b)
        loglik[int(r)] = float(ll)

    best_r = min(bic, key=bic.get)
    return BICResult(best_r=int(best_r), bic=bic, log_likelihood=loglik)


@dataclass(frozen=True)
class CVResult:
    best_r: int
    cv_mse: Dict[int, float]
    cv_mse_std: Dict[int, float]



def select_r_by_cv_prediction_mse(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r_candidates: Iterable[int],
    fit_fn: Callable[[np.ndarray, np.ndarray, int], Mapping],
    n_folds: int = 5,
    random_state: int = 42,
) -> CVResult:
    """Select r by K-fold CV using prediction MSE.

    Standardization is done *within each fold* (fit on train, apply to test).
    Predictions are made using the PPLS conditional mean.
    """

    from sklearn.model_selection import KFold

    X = np.asarray(X)
    Y = np.asarray(Y)

    kf = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(random_state))

    cv_mse: Dict[int, float] = {}
    cv_mse_std: Dict[int, float] = {}

    r_list = [int(r) for r in r_candidates]

    for r in r_list:
        fold_mses: List[float] = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Standardize within fold (avoid leakage)
            X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
            Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0)
            X_std[X_std == 0] = 1.0
            Y_std[Y_std == 0] = 1.0

            X_train_s = (X_train - X_mean) / X_std
            X_test_s = (X_test - X_mean) / X_std
            Y_train_s = (Y_train - Y_mean) / Y_std

            try:
                params = fit_fn(X_train_s, Y_train_s, r)
                Y_pred_s = predict_conditional_mean(X_test_s, dict(params))
                Y_pred = Y_pred_s * Y_std + Y_mean
                mse = float(np.mean((Y_test - Y_pred) ** 2))
            except Exception:
                mse = float("inf")

            fold_mses.append(mse)

        cv_mse[r] = float(np.mean(fold_mses))
        cv_mse_std[r] = float(np.std(fold_mses))

    best_r = min(cv_mse, key=cv_mse.get)
    return CVResult(best_r=int(best_r), cv_mse=cv_mse, cv_mse_std=cv_mse_std)
