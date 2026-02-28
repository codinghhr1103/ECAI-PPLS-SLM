"""
PPLS Model Core Implementation
==============================

Core classes for the Probabilistic Partial Least Squares (PPLS) model:

    x = t W^T + e,    t ~ N(0, Sigma_t)
    y = u C^T + f,    u = t B + h
    e ~ N(0, sigma_e^2 I_p),  f ~ N(0, sigma_f^2 I_q),  h ~ N(0, sigma_h^2 I_r)

Classes:
    PPLSModel        – covariance matrix, log-likelihood, sampling
    PPLSObjective    – scalar-form log-likelihood for optimisation
    PPLSConstraints  – orthonormality and bound constraints
    NoiseEstimator   – closed-form pre-estimation of sigma_e^2, sigma_f^2
"""

import numpy as np
from typing import Tuple, Dict


class PPLSModel:
    """
    Core PPLS model: covariance computation, log-likelihood, and data generation.

    The joint covariance of (x, y) is

        Sigma = [ W Sigma_t W' + sigma_e^2 I        W Sigma_t B C'                              ]
                [ C B Sigma_t W'                      C (B^2 Sigma_t + sigma_h^2 I) C' + sigma_f^2 I ]

    where B = diag(b), Sigma_t = diag(theta_t^2).
    """

    def __init__(self, p: int, q: int, r: int):
        self.p = p
        self.q = q
        self.r = r
        if r > min(p, q):
            raise ValueError(f"r ({r}) must be <= min(p, q) = {min(p, q)}")

    def compute_covariance_matrix(
        self, W: np.ndarray, C: np.ndarray, B: np.ndarray,
        Sigma_t: np.ndarray, sigma_e2: float, sigma_f2: float, sigma_h2: float
    ) -> np.ndarray:
        """Return the (p+q) x (p+q) joint covariance matrix."""
        b = np.diag(B)
        theta_t2 = np.diag(Sigma_t)

        Sigma_xx = W @ Sigma_t @ W.T + sigma_e2 * np.eye(self.p)
        Sigma_xy = W @ Sigma_t @ B @ C.T
        B2_Sigma_t = np.diag(b ** 2 * theta_t2)
        Sigma_yy = C @ (B2_Sigma_t + sigma_h2 * np.eye(self.r)) @ C.T + sigma_f2 * np.eye(self.q)

        return np.block([[Sigma_xx, Sigma_xy],
                         [Sigma_xy.T, Sigma_yy]])

    def log_likelihood_matrix(self, S: np.ndarray, Sigma: np.ndarray) -> float:
        """
        Negative profile log-likelihood (up to constants):
            L = ln det(Sigma) + tr(S Sigma^{-1})
        Minimise this to maximise the likelihood.
        """
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return np.inf
        try:
            L = np.linalg.cholesky(Sigma)
            Z = np.linalg.solve(L, S)
            trace_term = np.trace(np.linalg.solve(L.T, Z))
        except np.linalg.LinAlgError:
            try:
                trace_term = np.trace(S @ np.linalg.inv(Sigma))
            except np.linalg.LinAlgError:
                return np.inf
        return logdet + trace_term

    def sample(
        self, n_samples: int,
        W: np.ndarray, C: np.ndarray, B: np.ndarray, Sigma_t: np.ndarray,
        sigma_e2: float, sigma_f2: float, sigma_h2: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate (X, Y) from the PPLS generative model."""
        theta_t = np.sqrt(np.diag(Sigma_t))
        T = np.random.randn(n_samples, self.r) @ np.diag(theta_t)
        H = np.sqrt(sigma_h2) * np.random.randn(n_samples, self.r)
        U = T @ B + H
        E = np.sqrt(sigma_e2) * np.random.randn(n_samples, self.p)
        F = np.sqrt(sigma_f2) * np.random.randn(n_samples, self.q)
        X = T @ W.T + E
        Y = U @ C.T + F
        return X, Y


class PPLSObjective:
    """
    Scalar-form log-likelihood for interior-point optimisation.

    Avoids forming and inverting the full (p+q) x (p+q) covariance by
    computing ln det(Sigma) and tr(S Sigma^{-1}) component-wise over the
    r latent dimensions.
    """

    def __init__(self, p: int, q: int, r: int, S: np.ndarray):
        self.p, self.q, self.r = p, q, r
        self.S = S
        self.S_xx = S[:p, :p]
        self.S_xy = S[:p, p:]
        self.S_yy = S[p:, p:]
        self.sigma_e2 = 0.01
        self.sigma_f2 = 0.01

    def scalar_log_likelihood(self, theta: np.ndarray) -> float:
        """Scalar-form negative log-likelihood for minimisation."""
        W, C, B, Sigma_t, sigma_h2 = self._theta_to_params(theta)
        se2, sf2 = self.sigma_e2, self.sigma_f2

        theta_t2 = np.diag(Sigma_t)
        b = np.diag(B)
        if np.any(theta_t2 <= 0) or np.any(b <= 0) or sigma_h2 <= 0 or se2 <= 0 or sf2 <= 0:
            return 1e10

        ln_det = self._compute_ln_det(W, C, b, theta_t2, se2, sf2, sigma_h2)
        trace = self._compute_trace(W, C, b, theta_t2, se2, sf2, sigma_h2)
        result = ln_det + trace
        return result if np.isfinite(result) else 1e10

    def _compute_ln_det(self, W, C, b, theta_t2, se2, sf2, sh2) -> float:
        """
        ln det(Sigma) = (p-r) ln(se2) + (q-r) ln(sf2) + sum_i ln(D_i)
        where D_i = (sf2 + sh2)(theta_t2_i + se2) + b_i^2 theta_t2_i sf2
        """
        val = (self.p - self.r) * np.log(se2) + (self.q - self.r) * np.log(sf2)
        for i in range(self.r):
            D_i = (sf2 + sh2) * (theta_t2[i] + se2) + b[i] ** 2 * theta_t2[i] * sf2
            if D_i <= 0:
                return 1e10
            val += np.log(D_i)
        return val

    def _compute_trace(self, W, C, b, theta_t2, se2, sf2, sh2) -> float:
        """
        tr(S Sigma^{-1}) = tr(S_xx)/se2 + tr(S_yy)/sf2
            - sum_i [K2_i M2_i + K4_i M4_i + K6_i M6_i]
        """
        val = np.trace(self.S_xx) / se2 + np.trace(self.S_yy) / sf2
        for i in range(self.r):
            D_i = (sf2 + sh2) * (theta_t2[i] + se2) + b[i] ** 2 * theta_t2[i] * sf2
            if abs(D_i) < 1e-15:
                return 1e10
            M2 = (sf2 + sh2) * theta_t2[i] / D_i
            M4 = (sh2 * (theta_t2[i] + se2) + b[i] ** 2 * theta_t2[i] * se2) / D_i
            M6 = b[i] * theta_t2[i] / D_i

            wi = W[:, i:i + 1]
            ci = C[:, i:i + 1]
            K2 = (wi.T @ self.S_xx @ wi).item() / se2
            K4 = (ci.T @ self.S_yy @ ci).item() / sf2
            K6 = 2.0 * (wi.T @ self.S_xy @ ci).item()
            val -= K2 * M2 + K4 * M4 + K6 * M6
        return val

    def _theta_to_params(self, theta):
        idx = 0
        W = theta[idx:idx + self.p * self.r].reshape(self.p, self.r); idx += self.p * self.r
        C = theta[idx:idx + self.q * self.r].reshape(self.q, self.r); idx += self.q * self.r
        Sigma_t = np.diag(theta[idx:idx + self.r]); idx += self.r
        B = np.diag(theta[idx:idx + self.r]); idx += self.r
        sigma_h2 = theta[idx]
        return W, C, B, Sigma_t, sigma_h2

    def _params_to_theta(self, W, C, B, Sigma_t, sigma_h2):
        return np.concatenate([W.flatten(), C.flatten(),
                               np.diag(Sigma_t), np.diag(B), [sigma_h2]])


class PPLSConstraints:
    """
    Optimisation constraints for PPLS:
        W'W = I_r,  C'C = I_r,  b_i > 0,  theta_ti^2 > 0,  sigma_h^2 > 0
    """

    @staticmethod
    def get_bounds(p: int, q: int, r: int) -> list:
        bounds = [(None, None)] * (p * r + q * r)     # W, C
        bounds += [(1e-6, None)] * r                   # theta_t^2
        bounds += [(1e-6, None)] * r                   # b
        bounds.append((1e-6, None))                    # sigma_h^2
        return bounds

    @staticmethod
    def get_inequality_constraints(p: int, q: int, r: int, slack: float = 1e-3):
        """
        Vectorised orthonormality constraints for scipy trust-constr.

        Returns a NonlinearConstraint enforcing per-element:
            -slack <= (W'W)_{ij} - delta_{ij} <= slack
            -slack <= (C'C)_{ij} - delta_{ij} <= slack
        for upper-triangle indices (i <= j).
        """
        from scipy.optimize import NonlinearConstraint

        n_tri = r * (r + 1) // 2

        def constraint_fun(theta):
            W = theta[:p * r].reshape(p, r)
            C = theta[p * r:(p + q) * r].reshape(q, r)
            WtW = W.T @ W
            CtC = C.T @ C
            vals = np.empty(2 * n_tri)
            k = 0
            for i in range(r):
                for j in range(i, r):
                    target = 1.0 if i == j else 0.0
                    vals[k] = WtW[i, j] - target
                    vals[n_tri + k] = CtC[i, j] - target
                    k += 1
            return vals

        lb = -slack * np.ones(2 * n_tri)
        ub = slack * np.ones(2 * n_tri)
        return NonlinearConstraint(constraint_fun, lb, ub)

    @staticmethod
    def inequality_constraints(p, q, r, slack=1e-3):
        """Legacy wrapper."""
        return PPLSConstraints.get_inequality_constraints(p, q, r, slack)


class NoiseEstimator:
    """
    Closed-form pre-estimation of observation noise variances:

        sigma_e^2 = (1/N) sum_i ||x_i - x_bar||^2
        sigma_f^2 = (1/N) sum_i ||y_i - y_bar||^2

    These are total-variance estimators that serve as initial estimates for
    the SLM optimisation.  The theoretical error bound is given by Theorem 5
    in the paper.
    """

    @staticmethod
    def estimate_noise_variances(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        Parameters
        ----------
        X : (N, p) input data
        Y : (N, q) output data

        Returns
        -------
        sigma_e2, sigma_f2 : estimated noise variances
        """
        N = X.shape[0]

        X_centered = X - np.mean(X, axis=0, keepdims=True)
        Y_centered = Y - np.mean(Y, axis=0, keepdims=True)

        sigma_e2 = np.sum(X_centered ** 2) / N
        sigma_f2 = np.sum(Y_centered ** 2) / N

        return max(sigma_e2, 1e-6), max(sigma_f2, 1e-6)