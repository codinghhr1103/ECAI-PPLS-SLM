"""Baseline predictors for the prediction experiments (Section 8.3).

This module provides two non-PPLS baselines:
- Classical PLS regression (PLSR): sklearn.cross_decomposition.PLSRegression
- Ridge regression with cross-validated regularisation: sklearn.linear_model.RidgeCV

All helpers here follow the paper's evaluation protocol:
- Fit standardisation on the training fold only
- Apply to the test fold
- Evaluate metrics on the original (inverse-transformed) Y scale
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RegressionMetrics:
    mse: float
    mae: float
    r2_per_dim: np.ndarray
    r2_mean: float


def compute_regression_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> RegressionMetrics:
    """Compute multi-output regression metrics.

    - MSE/MAE are averaged over all entries.
    - R2 is computed per output dimension and then averaged.

    Notes:
    - For constant targets (zero variance), R2 is defined as 0.0 for that dim.
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    mse = float(np.mean((Y_true - Y_pred) ** 2))
    mae = float(np.mean(np.abs(Y_true - Y_pred)))

    q = Y_true.shape[1]
    r2_per_dim = np.zeros(q, dtype=float)
    for j in range(q):
        ss_res = float(np.sum((Y_true[:, j] - Y_pred[:, j]) ** 2))
        ss_tot = float(np.sum((Y_true[:, j] - np.mean(Y_true[:, j])) ** 2))
        r2_per_dim[j] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return RegressionMetrics(
        mse=mse,
        mae=mae,
        r2_per_dim=r2_per_dim,
        r2_mean=float(np.mean(r2_per_dim)),
    )


def _standardize_train_test(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
):
    """Standardise X and Y using training-set statistics only."""
    from sklearn.preprocessing import StandardScaler

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    Y_train_s = scaler_y.fit_transform(Y_train)
    X_test_s = scaler_x.transform(X_test)
    Y_test_s = scaler_y.transform(Y_test)

    return X_train_s, Y_train_s, X_test_s, Y_test_s, scaler_x, scaler_y


def run_plsr_prediction(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    n_components: int,
) -> Dict:
    """Classical PLS regression baseline (multi-output).

    Returns a dict with keys: Y_pred, metrics (RegressionMetrics).
    """
    from sklearn.cross_decomposition import PLSRegression

    X_train_s, Y_train_s, X_test_s, _Y_test_s, _sx, sy = _standardize_train_test(
        X_train, Y_train, X_test, Y_test
    )

    plsr = PLSRegression(n_components=int(n_components), scale=False)
    plsr.fit(X_train_s, Y_train_s)

    Y_pred_s = plsr.predict(X_test_s)
    Y_pred = sy.inverse_transform(Y_pred_s)

    metrics = compute_regression_metrics(Y_test, Y_pred)

    return {
        "Y_pred": Y_pred,
        "metrics": metrics,
    }


def run_ridge_prediction(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    *,
    alphas: Optional[Sequence[float]] = None,
    cv: int = 5,
) -> Dict:
    """Ridge regression baseline (multi-output via per-dimension RidgeCV).

    Parameters
    ----------
    alphas:
        Candidate regularisation strengths. If None, defaults to logspace(-4,4,50).
    cv:
        Inner CV folds used by RidgeCV.

    Returns a dict with keys: Y_pred, metrics (RegressionMetrics), alpha_per_dim.
    """
    from sklearn.linear_model import RidgeCV

    if alphas is None:
        alphas = np.logspace(-4, 4, 50)

    X_train_s, Y_train_s, X_test_s, _Y_test_s, _sx, sy = _standardize_train_test(
        X_train, Y_train, X_test, Y_test
    )

    q = Y_train_s.shape[1]
    Y_pred_s = np.zeros((X_test_s.shape[0], q), dtype=float)
    alpha_per_dim = np.zeros(q, dtype=float)

    for j in range(q):
        ridge = RidgeCV(alphas=np.asarray(list(alphas), dtype=float), cv=int(cv))
        ridge.fit(X_train_s, Y_train_s[:, j])
        Y_pred_s[:, j] = ridge.predict(X_test_s)
        try:
            alpha_per_dim[j] = float(ridge.alpha_)
        except Exception:
            alpha_per_dim[j] = np.nan

    Y_pred = sy.inverse_transform(Y_pred_s)
    metrics = compute_regression_metrics(Y_test, Y_pred)

    return {
        "Y_pred": Y_pred,
        "metrics": metrics,
        "alpha_per_dim": alpha_per_dim,
    }
