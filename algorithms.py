"""
SLM, EM, and ECM Algorithm Implementations
==========================================

Algorithm classes for PPLS parameter estimation:
    ScalarLikelihoodMethod  – interior-point optimisation of scalar likelihood
    EMAlgorithm             – full EM with closed-form E- and M-steps
    ECMAlgorithm            – ECM with re-computed E-step after each CM sub-step

Supporting classes:
    PPLSAlgorithm           – abstract base with sign-alignment and metrics
    InitialPointGenerator   – shared random starting points for multi-start
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import orth
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import warnings
import pickle
import os
import json
from datetime import datetime

from ppls_model import PPLSModel, PPLSObjective, PPLSConstraints, NoiseEstimator


# ======================================================================
#  Abstract base
# ======================================================================
class PPLSAlgorithm(ABC):
    """Common interface and utilities for SLM / EM / ECM."""

    def __init__(self, p: int, q: int, r: int):
        self.p, self.q, self.r = p, q, r
        self.model = PPLSModel(p, q, r)

    @abstractmethod
    def fit(self, X, Y, starting_points, experiment_dir=None, trial_id=None) -> Dict:
        ...

    def align_signs(self, W_est, C_est, B_est,
                    W_true=None, C_true=None, B_true=None):
        """Resolve column sign indeterminacy by aligning with ground truth."""
        W, C, B = W_est.copy(), C_est.copy(), B_est.copy()
        if W_true is not None and C_true is not None:
            for i in range(self.r):
                if np.corrcoef(W[:, i], W_true[:, i])[0, 1] < 0:
                    W[:, i] *= -1
                if np.corrcoef(C[:, i], C_true[:, i])[0, 1] < 0:
                    C[:, i] *= -1
        b = np.diag(B)
        signs = np.sign(b); signs[signs == 0] = 1
        B = np.diag(np.abs(b))
        for i in range(self.r):
            if signs[i] < 0:
                C[:, i] *= -1
        return W, C, B

    def compute_metrics(self, params_est: Dict, params_true: Dict) -> Dict:
        W, C, B = self.align_signs(
            params_est['W'], params_est['C'], params_est['B'],
            params_true['W'], params_true['C'], params_true['B'])
        return {
            'mse_W': np.mean((W - params_true['W']) ** 2),
            'mse_C': np.mean((C - params_true['C']) ** 2),
            'mse_B': np.mean((np.diag(B) - np.diag(params_true['B'])) ** 2),
            'mse_Sigma_t': np.mean((np.diag(params_est['Sigma_t']) - np.diag(params_true['Sigma_t'])) ** 2),
            'mse_sigma_h2': (params_est['sigma_h2'] - params_true['sigma_h2']) ** 2,
        }


# ======================================================================
#  Starting point generator
# ======================================================================
class InitialPointGenerator:
    """
    Generate shared random starting points for multi-start optimisation.
    All parameters (including diagonal and scalar ones) are randomised so
    that the multi-start strategy explores the parameter space effectively.
    """

    def __init__(self, p: int, q: int, r: int, n_starts: int = 32,
                 random_seed: int = 42):
        self.p, self.q, self.r = p, q, r
        self.n_starts = n_starts
        self.random_seed = random_seed

    def generate_starting_points(self) -> List[np.ndarray]:
        rng = np.random.RandomState(self.random_seed)
        return [self._generate_single_point(rng) for _ in range(self.n_starts)]

    def _generate_single_point(self, rng: np.random.RandomState) -> np.ndarray:
        # Random orthonormal W and C
        W = orth(rng.randn(self.p, self.r))
        C = orth(rng.randn(self.q, self.r))
        if W.shape[1] < self.r:
            W = rng.randn(self.p, self.r)
            W, _ = np.linalg.qr(W); W = W[:, :self.r]
        if C.shape[1] < self.r:
            C = rng.randn(self.q, self.r)
            C, _ = np.linalg.qr(C); C = C[:, :self.r]

        # Random diagonal parameters (log-uniform spread)
        theta_t = np.sort(np.exp(rng.uniform(-1.0, 1.0, self.r)))[::-1]
        b = np.sort(np.exp(rng.uniform(-0.5, 1.0, self.r)))[::-1]
        sigma_h2 = np.exp(rng.uniform(-3.0, -0.5))

        return np.concatenate([W.flatten(), C.flatten(), theta_t, b, [sigma_h2]])

    def save_starting_points(self, starting_points, experiment_dir, algorithm_name="common"):
        d = os.path.join(experiment_dir, "initial_points"); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{algorithm_name}_starting_points.pkl"), 'wb') as f:
            pickle.dump(starting_points, f)


# ======================================================================
#  SLM
# ======================================================================
class ScalarLikelihoodMethod(PPLSAlgorithm):
    """
    Scalar Likelihood Method: interior-point optimisation of the
    scalar-form log-likelihood with per-element orthonormality constraints.
    """

    def __init__(self, p, q, r, optimizer='trust-constr', max_iter=100,
                 use_noise_preestimation=True):
        super().__init__(p, q, r)
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.use_noise_preestimation = use_noise_preestimation

    def fit(self, X, Y, starting_points, experiment_dir=None, trial_id=None):
        N = X.shape[0]
        XY = np.hstack([X, Y])
        XY_c = XY - XY.mean(axis=0)
        S = (XY_c.T @ XY_c) / N

        if self.use_noise_preestimation:
            sigma_e2, sigma_f2 = NoiseEstimator.estimate_noise_variances(X, Y)
        else:
            sigma_e2, sigma_f2 = 0.01, 0.01

        objective = PPLSObjective(self.p, self.q, self.r, S)
        objective.sigma_e2 = sigma_e2
        objective.sigma_f2 = sigma_f2

        constraints = PPLSConstraints.get_inequality_constraints(self.p, self.q, self.r)
        bounds = PPLSConstraints.get_bounds(self.p, self.q, self.r)

        solutions = []
        for theta0 in starting_points:
            try:
                res = self._optimize_single_start(theta0, objective, constraints, bounds)
                solutions.append(res)
            except Exception as e:
                warnings.warn(f"SLM optimisation failed for one start: {e}")

        best = self._select_best_solution(solutions)
        W, C, B, Sigma_t, sigma_h2 = objective._theta_to_params(best['x'])
        results = {
            'W': W, 'C': C, 'B': B, 'Sigma_t': Sigma_t,
            'sigma_e2': sigma_e2, 'sigma_f2': sigma_f2, 'sigma_h2': sigma_h2,
            'objective_value': best['fun'],
            'n_iterations': best.get('nit', 0),
            'success': best['success'],
        }
        if experiment_dir:
            self._save_results(results, experiment_dir, "SLM", trial_id)
        return results

    def _optimize_single_start(self, theta0, objective, constraints, bounds):
        options = {
            'maxiter': self.max_iter, 'gtol': 1e-3, 'xtol': 1e-3,
            'barrier_tol': 1e-3, 'initial_constr_penalty': 1.0,
        }
        return minimize(objective.scalar_log_likelihood, theta0,
                        method=self.optimizer, constraints=constraints,
                        bounds=bounds, options=options)

    def _select_best_solution(self, solutions):
        if not solutions:
            raise ValueError("No SLM solutions available")
        valid = [s for s in solutions if s['success']]
        pool = valid if valid else solutions
        return min(pool, key=lambda s: s['fun'])

    def _save_results(self, results, experiment_dir, name, trial_id):
        d = os.path.join(experiment_dir, "estimates", name); os.makedirs(d, exist_ok=True)
        prefix = f"trial_{trial_id:03d}" if trial_id is not None else "estimated"
        with open(os.path.join(d, f"{prefix}_results.pkl"), 'wb') as f:
            pickle.dump(results, f)
        for k in ['W', 'C', 'B', 'Sigma_t']:
            if k in results:
                np.save(os.path.join(d, f"{prefix}_{k}.npy"), results[k])


# ======================================================================
#  Shared E-step helper
# ======================================================================
def _ppls_e_step(X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2, r):
    """
    Full joint-posterior E-step for the PPLS model.

    Constructs the joint distribution of (t, u, x, y) and conditions on
    (x, y) to obtain the exact posterior moments.  This avoids the
    sequential approximation E[t|x] then E[u|y,t] which loses the coupling
    between t and u through y.

    Returns: E_T (N,r), E_U (N,r), Cov_T (r,r), Cov_U (r,r), Cov_TU (r,r)
    """
    N = X.shape[0]
    p, q = W.shape[0], C.shape[0]
    reg = 1e-8

    b = np.diag(B)
    theta_t2 = np.diag(Sigma_t)

    # Prior covariance of latent (t, u)  — block (2r x 2r)
    Sigma_uu_prior = np.diag(b ** 2 * theta_t2 + sigma_h2)
    Sigma_tu_prior = Sigma_t @ B                             # diag(theta_t2 * b)
    Sigma_zz = np.block([
        [Sigma_t,          Sigma_tu_prior],
        [Sigma_tu_prior.T, Sigma_uu_prior]
    ])

    # Observation model:  [x; y] = A @ [t; u] + noise
    A = np.zeros((p + q, 2 * r))
    A[:p, :r] = W
    A[p:, r:] = C

    Sigma_noise = np.empty(p + q)
    Sigma_noise[:p] = sigma_e2
    Sigma_noise[p:] = sigma_f2

    # Marginal covariance of (x, y) and posterior gain
    AS = Sigma_zz @ A.T                         # (2r, p+q)
    M = A @ AS + np.diag(Sigma_noise) + reg * np.eye(p + q)

    try:
        L_M = np.linalg.cholesky(M)
        Gain_T = np.linalg.solve(L_M, A @ Sigma_zz)
        Gain = np.linalg.solve(L_M.T, Gain_T).T
    except np.linalg.LinAlgError:
        Gain = AS @ np.linalg.pinv(M)

    # Posterior expectations
    XY = np.hstack([X, Y])
    E_Z = XY @ Gain.T
    E_T = E_Z[:, :r]
    E_U = E_Z[:, r:]

    # Posterior covariance (constant across samples)
    Cov_z = Sigma_zz - Gain @ A @ Sigma_zz
    Cov_z = (Cov_z + Cov_z.T) / 2 + reg * np.eye(2 * r)

    return E_T, E_U, Cov_z[:r, :r], Cov_z[r:, r:], Cov_z[:r, r:]


# ======================================================================
#  EM Algorithm
# ======================================================================
class EMAlgorithm(PPLSAlgorithm):
    """
    Full Expectation-Maximisation for PPLS.

    E-step: joint posterior of (t, u) given (x, y).
    M-step: closed-form updates for W, C, B, Sigma_t, sigma_h^2,
            sigma_e^2, sigma_f^2.
    """

    def __init__(self, p, q, r, max_iter=1000, tolerance=1e-4):
        super().__init__(p, q, r)
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X, Y, starting_points, experiment_dir=None, trial_id=None):
        solutions = []
        for theta0 in starting_points:
            try:
                solutions.append(self._run_single_em(X, Y, theta0))
            except Exception as e:
                warnings.warn(f"EM failed for one start: {e}")
        best = self._select_best(solutions)
        if experiment_dir:
            self._save_results(best, experiment_dir, "EM", trial_id)
        return best

    def _run_single_em(self, X, Y, theta0):
        p, q, r = self.p, self.q, self.r
        N = X.shape[0]

        W, C, theta_t, b, sigma_h2 = self._unpack(theta0)
        Sigma_t = np.diag(theta_t)
        B = np.diag(b)
        sigma_e2, sigma_f2 = 0.1, 0.1

        for iteration in range(self.max_iter):
            old = self._snapshot(W, C, B, Sigma_t, sigma_h2, sigma_e2, sigma_f2)

            E_T, E_U, Cov_T, Cov_U, Cov_TU = _ppls_e_step(
                X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2, r)

            W, C, B, Sigma_t, sigma_h2, sigma_e2, sigma_f2 = \
                self._m_step(X, Y, E_T, E_U, Cov_T, Cov_U, Cov_TU, N)

            new = self._snapshot(W, C, B, Sigma_t, sigma_h2, sigma_e2, sigma_f2)
            if self._converged(old, new):
                break

        ll = self._compute_ll(X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2, N)
        return {
            'W': W, 'C': C, 'B': B, 'Sigma_t': Sigma_t,
            'sigma_e2': sigma_e2, 'sigma_f2': sigma_f2, 'sigma_h2': sigma_h2,
            'log_likelihood': ll, 'n_iterations': iteration + 1,
        }

    def _m_step(self, X, Y, E_T, E_U, Cov_T, Cov_U, Cov_TU, N):
        r = self.r
        reg = 1e-8

        sum_ET_ET = E_T.T @ E_T + N * Cov_T
        sum_EU_EU = E_U.T @ E_U + N * Cov_U
        sum_EU_ET = E_U.T @ E_T + N * Cov_TU.T

        # W: polar factor of X' E_T  (closed-form under W'W = I)
        XET = X.T @ E_T
        Uw, _, Vwt = np.linalg.svd(XET, full_matrices=False)
        W = Uw[:, :r] @ Vwt[:r, :]

        # C: polar factor of Y' E_U
        YEU = Y.T @ E_U
        Uc, _, Vct = np.linalg.svd(YEU, full_matrices=False)
        C = Uc[:, :r] @ Vct[:r, :]

        # B: diagonal
        d_ETET = np.diag(sum_ET_ET)
        d_EUET = np.diag(sum_EU_ET)
        b = d_EUET / (d_ETET + reg)
        b = np.maximum(b, 1e-6)
        B = np.diag(b)

        # Sigma_t: diagonal
        theta_t2 = np.diag(sum_ET_ET) / N
        theta_t2 = np.maximum(theta_t2, 1e-6)
        Sigma_t = np.diag(theta_t2)

        # sigma_h^2
        sigma_h2 = (np.trace(sum_EU_EU) - 2 * np.sum(b * d_EUET) +
                     np.sum(b ** 2 * d_ETET)) / (N * r)
        sigma_h2 = max(sigma_h2, 1e-6)

        # sigma_e^2  (with trace correction for posterior uncertainty)
        Xr = X - E_T @ W.T
        sigma_e2 = (np.sum(Xr ** 2) + np.trace(Cov_T @ (W.T @ W))) / (N * self.p)
        sigma_e2 = max(sigma_e2, 1e-6)

        # sigma_f^2
        Yr = Y - E_U @ C.T
        sigma_f2 = (np.sum(Yr ** 2) + np.trace(Cov_U @ (C.T @ C))) / (N * self.q)
        sigma_f2 = max(sigma_f2, 1e-6)

        return W, C, B, Sigma_t, sigma_h2, sigma_e2, sigma_f2

    def _unpack(self, theta0):
        p, q, r = self.p, self.q, self.r
        idx = 0
        W = theta0[idx:idx + p * r].reshape(p, r); idx += p * r
        C = theta0[idx:idx + q * r].reshape(q, r); idx += q * r
        theta_t = theta0[idx:idx + r]; idx += r
        b = theta0[idx:idx + r]; idx += r
        sigma_h2 = theta0[idx]
        return W, C, theta_t, b, sigma_h2

    def _snapshot(self, W, C, B, Sigma_t, sigma_h2, sigma_e2, sigma_f2):
        return {'W': W.copy(), 'C': C.copy(), 'B': B.copy(),
                'Sigma_t': Sigma_t.copy(), 'sigma_h2': sigma_h2,
                'sigma_e2': sigma_e2, 'sigma_f2': sigma_f2}

    def _converged(self, old, new):
        tol = self.tolerance
        for k in ['W', 'C']:
            rel = np.linalg.norm(new[k] - old[k], 'fro') / (np.linalg.norm(old[k], 'fro') + 1e-10)
            if rel > tol:
                return False
        for k in ['B', 'Sigma_t']:
            if np.max(np.abs(np.diag(new[k]) - np.diag(old[k]))) > tol:
                return False
        for k in ['sigma_h2', 'sigma_e2', 'sigma_f2']:
            if abs(new[k] - old[k]) / (abs(old[k]) + 1e-10) > tol:
                return False
        return True

    def _compute_ll(self, X, Y, W, C, B, Sigma_t, se2, sf2, sh2, N):
        XY = np.hstack([X, Y])
        XY_c = XY - XY.mean(axis=0)
        S = (XY_c.T @ XY_c) / N
        Sigma = self.model.compute_covariance_matrix(W, C, B, Sigma_t, se2, sf2, sh2)
        return self.model.log_likelihood_matrix(S, Sigma)

    def _select_best(self, solutions):
        if not solutions:
            raise ValueError("No valid EM solutions")
        return min(solutions, key=lambda s: s.get('log_likelihood', np.inf))

    def _save_results(self, results, experiment_dir, name, trial_id):
        d = os.path.join(experiment_dir, "estimates", name); os.makedirs(d, exist_ok=True)
        prefix = f"trial_{trial_id:03d}" if trial_id is not None else "estimated"
        with open(os.path.join(d, f"{prefix}_results.pkl"), 'wb') as f:
            pickle.dump(results, f)
        for k in ['W', 'C', 'B', 'Sigma_t']:
            if k in results:
                np.save(os.path.join(d, f"{prefix}_{k}.npy"), results[k])


# ======================================================================
#  ECM Algorithm
# ======================================================================
class ECMAlgorithm(PPLSAlgorithm):
    """
    Expectation Conditional-Maximisation for PPLS.

    Three CM sub-steps per iteration, each preceded by a fresh E-step:
        CM-1: update W, C
        CM-2: update B, Sigma_t
        CM-3: update sigma_e^2, sigma_f^2, sigma_h^2
    """

    def __init__(self, p, q, r, max_iter=1000, tolerance=1e-4):
        super().__init__(p, q, r)
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X, Y, starting_points, experiment_dir=None, trial_id=None):
        solutions = []
        for theta0 in starting_points:
            try:
                solutions.append(self._run_single_ecm(X, Y, theta0))
            except Exception as e:
                warnings.warn(f"ECM failed for one start: {e}")
        best = self._select_best(solutions)
        if experiment_dir:
            self._save_results(best, experiment_dir, "ECM", trial_id)
        return best

    def _run_single_ecm(self, X, Y, theta0):
        p, q, r = self.p, self.q, self.r
        N = X.shape[0]
        reg = 1e-8

        W, C, theta_t, b, sigma_h2 = self._unpack(theta0)
        Sigma_t = np.diag(theta_t); B = np.diag(b)
        sigma_e2, sigma_f2 = 0.1, 0.1

        for iteration in range(self.max_iter):
            old = self._snapshot(W, C, B, Sigma_t, sigma_h2, sigma_e2, sigma_f2)

            # ---- CM-1: update W, C ----
            E_T, E_U, Cov_T, Cov_U, Cov_TU = _ppls_e_step(
                X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2, r)
            Uw, _, Vwt = np.linalg.svd(X.T @ E_T, full_matrices=False)
            W = Uw[:, :r] @ Vwt[:r, :]
            Uc, _, Vct = np.linalg.svd(Y.T @ E_U, full_matrices=False)
            C = Uc[:, :r] @ Vct[:r, :]

            # ---- CM-2: update B, Sigma_t ----
            E_T, E_U, Cov_T, Cov_U, Cov_TU = _ppls_e_step(
                X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2, r)
            sum_ETET = E_T.T @ E_T + N * Cov_T
            sum_EUET = E_U.T @ E_T + N * Cov_TU.T
            d_ETET = np.diag(sum_ETET)
            b = np.maximum(np.diag(sum_EUET) / (d_ETET + reg), 1e-6)
            B = np.diag(b)
            Sigma_t = np.diag(np.maximum(d_ETET / N, 1e-6))

            # ---- CM-3: update noise variances ----
            E_T, E_U, Cov_T, Cov_U, Cov_TU = _ppls_e_step(
                X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2, r)
            sum_ETET = E_T.T @ E_T + N * Cov_T
            sum_EUEU = E_U.T @ E_U + N * Cov_U
            sum_EUET = E_U.T @ E_T + N * Cov_TU.T

            Xr = X - E_T @ W.T
            sigma_e2 = max((np.sum(Xr ** 2) + np.trace(Cov_T @ (W.T @ W))) / (N * p), 1e-6)
            Yr = Y - E_U @ C.T
            sigma_f2 = max((np.sum(Yr ** 2) + np.trace(Cov_U @ (C.T @ C))) / (N * q), 1e-6)
            sigma_h2 = max(
                (np.trace(sum_EUEU) - 2 * np.sum(b * np.diag(sum_EUET)) +
                 np.sum(b ** 2 * np.diag(sum_ETET))) / (N * r), 1e-6)

            new = self._snapshot(W, C, B, Sigma_t, sigma_h2, sigma_e2, sigma_f2)
            if self._converged(old, new):
                break

        ll = self._compute_ll(X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2, N)
        return {
            'W': W, 'C': C, 'B': B, 'Sigma_t': Sigma_t,
            'sigma_e2': sigma_e2, 'sigma_f2': sigma_f2, 'sigma_h2': sigma_h2,
            'log_likelihood': ll, 'n_iterations': iteration + 1,
        }

    _unpack = EMAlgorithm._unpack
    _snapshot = EMAlgorithm._snapshot
    _converged = EMAlgorithm._converged
    _compute_ll = EMAlgorithm._compute_ll
    _select_best = EMAlgorithm._select_best
    _save_results = EMAlgorithm._save_results