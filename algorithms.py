"""
SLM, EM, and ECM Algorithm Implementations
==========================================

This module implements the Scalar Likelihood Method (SLM), Expectation-Maximization (EM), 
and Expectation-Conditional Maximization (ECM) algorithms for PPLS parameter estimation. 
All algorithms use identical multi-start strategies for fair comparison, as described in 
Section 7.1 of the paper.

Architecture Overview:
---------------------
The module provides:
1. PPLSAlgorithm: Abstract base class defining common interface
2. InitialPointGenerator: Unified starting point generation for all algorithms
3. ScalarLikelihoodMethod: SLM implementation with interior-point optimization
4. EMAlgorithm: EM algorithm with closed-form updates
5. ECMAlgorithm: ECM algorithm with conditional maximization steps

Function List:
--------------
PPLSAlgorithm (Abstract Base):
    - fit(X, Y, starting_points): Abstract method for parameter estimation
    - align_signs(W_est, C_est, B_est, W_true, C_true, B_true): Align signs of loadings
    - compute_metrics(params_est, params_true): Calculate estimation quality metrics
    - set_initial_points(starting_points): Set identical starting points

InitialPointGenerator:
    - __init__(p, q, r, n_starts, random_seed): Initialize generator
    - generate_starting_points(): Generate identical starting points for all algorithms
    - _generate_single_point(): Create one random starting point (Algorithm 1)
    - _orthonormalize_loadings(W, C): Ensure orthonormal columns
    - save_starting_points(filepath): Save points for reproducibility

ScalarLikelihoodMethod:
    - __init__(p, q, r, optimizer_settings): Initialize SLM
    - fit(X, Y, starting_points): Main SLM fitting with multi-start
    - _optimize_single_start(theta0, objective, constraints, bounds): Run single optimization
    - _setup_constraints(): Setup orthonormality and bound constraints
    - _select_best_solution(solutions): Select best from multi-start results

EMAlgorithm:
    - __init__(p, q, r, max_iter, tolerance): Initialize EM
    - fit(X, Y, starting_points): Main EM fitting
    - _e_step(X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2): E-step
    - _m_step(X, Y, E_T, E_U, Cov_T, Cov_U): M-step with closed-form updates
    - _run_single_em(X, Y, theta0): Run EM from single initialization
    - _check_convergence(params_old, params_new): Monitor convergence
    - _update_W_C(X, Y, E_T, E_U): Update loading matrices with orthogonalization

ECMAlgorithm:
    - __init__(p, q, r, max_iter, tolerance): Initialize ECM
    - fit(X, Y, starting_points): Main ECM fitting
    - _run_single_ecm(X, Y, theta0): Run ECM from single initialization
    - _e_step_vectorized(X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2): E-step
    - _cm_step_loadings(X, Y, E_T, E_U): CM-step for loading matrices
    - _cm_step_latent_params(E_T, E_U, Cov_T, Cov_U): CM-step for latent parameters
    - _cm_step_noise_params(X, Y, E_T, E_U): CM-step for noise parameters

Call Relationships:
------------------
ScalarLikelihoodMethod.fit() → InitialPointGenerator.generate_starting_points()
ScalarLikelihoodMethod.fit() → NoiseEstimator.estimate_noise_variances()
ScalarLikelihoodMethod._optimize_single_start() → PPLSObjective.scalar_log_likelihood()
ScalarLikelihoodMethod._setup_constraints() → PPLSConstraints.inequality_constraints()
EMAlgorithm.fit() → InitialPointGenerator.generate_starting_points()
EMAlgorithm.fit() → NoiseEstimator.estimate_noise_variances()
EMAlgorithm._run_single_em() → EMAlgorithm._e_step()
EMAlgorithm._run_single_em() → EMAlgorithm._m_step()
ECMAlgorithm.fit() → InitialPointGenerator.generate_starting_points()
ECMAlgorithm._run_single_ecm() → ECMAlgorithm._e_step_vectorized()
ECMAlgorithm._run_single_ecm() → ECMAlgorithm._cm_step_loadings()
ECMAlgorithm._run_single_ecm() → ECMAlgorithm._cm_step_latent_params()
ECMAlgorithm._run_single_ecm() → ECMAlgorithm._cm_step_noise_params()
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import orth, inv, sqrtm, eigh
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import warnings
from joblib import Parallel, delayed
import pickle
import os
import json
from datetime import datetime

from ppls_model import PPLSModel, PPLSObjective, PPLSConstraints, NoiseEstimator


class PPLSAlgorithm(ABC):
    """
    Abstract base class for PPLS parameter estimation algorithms.
    Provides common interface and utilities for SLM, EM, and ECM algorithms.
    """
    
    def __init__(self, p: int, q: int, r: int):
        """Initialize with model dimensions."""
        self.p = p
        self.q = q
        self.r = r
        self.model = PPLSModel(p, q, r)
        
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, 
            starting_points: List[np.ndarray],
            experiment_dir: Optional[str] = None,
            trial_id: Optional[int] = None) -> Dict:
        """
        Fit PPLS model to data using provided starting points.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, p)
            Input data
        Y : np.ndarray of shape (N, q)
            Output data
        starting_points : List[np.ndarray]
            List of identical starting points for multi-start
        experiment_dir : str, optional
            Directory to save results
        trial_id : int, optional
            Trial number for Monte Carlo experiments
            
        Returns:
        --------
        results : dict
            Estimated parameters and diagnostics
        """
        pass
    
    def align_signs(self, W_est: np.ndarray, C_est: np.ndarray, B_est: np.ndarray,
                    W_true: Optional[np.ndarray] = None, 
                    C_true: Optional[np.ndarray] = None,
                    B_true: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align signs of estimated loadings with ground truth to handle sign indeterminacy.
        
        Parameters:
        -----------
        W_est, C_est, B_est : estimated parameters
        W_true, C_true, B_true : true parameters (if available)
        
        Returns:
        --------
        W_aligned, C_aligned, B_aligned : sign-aligned parameters
        """
        W_aligned = W_est.copy()
        C_aligned = C_est.copy()
        B_aligned = B_est.copy()
        
        if W_true is not None and C_true is not None:
            for i in range(self.r):
                # Check correlation with true loadings
                w_corr = np.corrcoef(W_est[:, i], W_true[:, i])[0, 1]
                c_corr = np.corrcoef(C_est[:, i], C_true[:, i])[0, 1]
                
                # Flip signs if negative correlation
                if w_corr < 0:
                    W_aligned[:, i] *= -1
                if c_corr < 0:
                    C_aligned[:, i] *= -1
                    
        # Ensure B diagonal elements are positive (identifiability constraint)
        b_diag = np.diag(B_aligned)
        sign_flips = np.sign(b_diag)
        sign_flips[sign_flips == 0] = 1
        B_aligned = np.diag(np.abs(b_diag))
        
        # Apply corresponding sign flips to C
        for i in range(self.r):
            if sign_flips[i] < 0:
                C_aligned[:, i] *= -1
                
        return W_aligned, C_aligned, B_aligned
    
    def compute_metrics(self, params_est: Dict, params_true: Dict) -> Dict:
        """
        Calculate estimation quality metrics (MSE, bias, variance).
        
        Parameters:
        -----------
        params_est : dict
            Estimated parameters
        params_true : dict
            True parameters
            
        Returns:
        --------
        metrics : dict
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Align signs first
        W_aligned, C_aligned, B_aligned = self.align_signs(
            params_est['W'], params_est['C'], params_est['B'],
            params_true['W'], params_true['C'], params_true['B']
        )
        
        # Compute MSE for each parameter
        metrics['mse_W'] = np.mean((W_aligned - params_true['W'])**2)
        metrics['mse_C'] = np.mean((C_aligned - params_true['C'])**2)
        metrics['mse_B'] = np.mean((np.diag(B_aligned) - np.diag(params_true['B']))**2)
        metrics['mse_Sigma_t'] = np.mean((np.diag(params_est['Sigma_t']) - np.diag(params_true['Sigma_t']))**2)
        metrics['mse_sigma_h2'] = (params_est['sigma_h2'] - params_true['sigma_h2'])**2
        
        return metrics


class InitialPointGenerator:
    """
    Generates identical starting points for both SLM and EM algorithms.
    Implements Algorithm 1 from the paper for fair comparison.
    """
    
    def __init__(self, p: int, q: int, r: int, n_starts: int = 32, 
                 random_seed: int = 42):
        """
        Initialize the starting point generator.
        
        Parameters:
        -----------
        p, q, r : int
            Model dimensions
        n_starts : int
            Number of starting points to generate
        random_seed : int
            Random seed for reproducibility
        """
        self.p = p
        self.q = q
        self.r = r
        self.n_starts = n_starts
        self.random_seed = random_seed
        
    def generate_starting_points(self) -> List[np.ndarray]:
        """
        Generate identical starting points for multi-start optimization.
        
        Returns:
        --------
        starting_points : List[np.ndarray]
            List of starting parameter vectors
        """
        np.random.seed(self.random_seed)
        starting_points = []
        
        for _ in range(self.n_starts):
            theta0 = self._generate_single_point()
            starting_points.append(theta0)
            
        return starting_points
    
    def _generate_single_point(self) -> np.ndarray:
        """
        Generate a single starting point according to Algorithm 1.
        
        Returns:
        --------
        theta0 : np.ndarray
            Flattened parameter vector
        """
        # Step 1: Generate random W and C with better initialization
        W = np.random.randn(self.p, self.r) * 0.5  # Smaller initial values
        C = np.random.randn(self.q, self.r) * 0.5
        
        # Step 2: Orthonormalize columns
        W, C = self._orthonormalize_loadings(W, C)
        
        # Step 3: Initialize other parameters with better values
        # Initialize Sigma_t diagonal with decreasing values (identifiability)
        theta_t = np.linspace(1.0, 0.3, self.r)  # Decreasing pattern
        
        # Initialize B diagonal with decreasing values
        b = np.linspace(1.5, 0.5, self.r)  # Decreasing pattern
        
        # Initialize sigma_h2 with reasonable value
        sigma_h2 = 0.05
        
        # Step 4: Flatten to parameter vector
        theta0 = np.concatenate([
            W.flatten(),
            C.flatten(),
            theta_t,
            b,
            [sigma_h2]
        ])
        
        return theta0
    
    def _orthonormalize_loadings(self, W: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure W and C have orthonormal columns using Gram-Schmidt.
        
        Parameters:
        -----------
        W : np.ndarray of shape (p, r)
        C : np.ndarray of shape (q, r)
        
        Returns:
        --------
        W_orth, C_orth : orthonormalized matrices
        """
        W_orth = orth(W)
        C_orth = orth(C)
        
        # Handle case where orth returns fewer columns
        if W_orth.shape[1] < self.r:
            W_orth = W / np.linalg.norm(W, axis=0, keepdims=True)
        if C_orth.shape[1] < self.r:
            C_orth = C / np.linalg.norm(C, axis=0, keepdims=True)
            
        return W_orth, C_orth
    
    def save_starting_points(self, starting_points: List[np.ndarray], 
                           experiment_dir: str,
                           algorithm_name: str = "common"):
        """
        Save starting points to experiment directory.
        
        Parameters:
        -----------
        starting_points : List[np.ndarray]
            List of starting parameter vectors
        experiment_dir : str
            Path to experiment directory
        algorithm_name : str
            Name to identify which algorithm
        """
        init_dir = os.path.join(experiment_dir, "initial_points")
        os.makedirs(init_dir, exist_ok=True)
        
        # Save as pickle
        filename = f"{algorithm_name}_starting_points.pkl"
        with open(os.path.join(init_dir, filename), 'wb') as f:
            pickle.dump(starting_points, f)
            
        # Save summary
        summary = {
            "n_starts": len(starting_points),
            "parameter_vector_length": len(starting_points[0]) if starting_points else 0,
            "algorithm": algorithm_name,
            "random_seed": self.random_seed,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(init_dir, f"{algorithm_name}_starting_points_info.json"), 'w') as f:
            json.dump(summary, f, indent=4)


class ScalarLikelihoodMethod(PPLSAlgorithm):
    """
    Scalar Likelihood Method (SLM) implementation using interior-point optimization.
    Based on Section 4.3 of the paper.
    """
    
    def __init__(self, p: int, q: int, r: int, 
                 optimizer: str = 'trust-constr',
                 max_iter: int = 100,
                 use_noise_preestimation: bool = True):
        """
        Initialize SLM algorithm.
        
        Parameters:
        -----------
        p, q, r : int
            Model dimensions
        optimizer : str
            Optimization method ('trust-constr' for interior-point)
        max_iter : int
            Maximum iterations per optimization
        use_noise_preestimation : bool
            Whether to pre-estimate noise variances
        """
        super().__init__(p, q, r)
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.use_noise_preestimation = use_noise_preestimation
        
    def fit(self, X: np.ndarray, Y: np.ndarray, 
            starting_points: List[np.ndarray],
            experiment_dir: Optional[str] = None,
            trial_id: Optional[int] = None) -> Dict:
        """
        Fit PPLS model using SLM with multi-start optimization.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, p)
            Input data
        Y : np.ndarray of shape (N, q)  
            Output data
        starting_points : List[np.ndarray]
            Identical starting points for multi-start
        experiment_dir : str, optional
            Directory to save results
        trial_id : int, optional
            Trial number for Monte Carlo experiments
            
        Returns:
        --------
        results : dict
            Best estimated parameters and optimization info
        """
        # Compute sample covariance matrix S
        N = X.shape[0]
        XY = np.hstack([X, Y])
        XY_centered = XY - np.mean(XY, axis=0)
        S = (XY_centered.T @ XY_centered) / N
        
        # Pre-estimate noise variances if requested
        if self.use_noise_preestimation:
            sigma_e2, sigma_f2 = NoiseEstimator.estimate_noise_variances(X, Y)
        else:
            sigma_e2, sigma_f2 = 0.01, 0.01
            
        # Setup objective function
        objective = PPLSObjective(self.p, self.q, self.r, S)
        objective.sigma_e2 = sigma_e2
        objective.sigma_f2 = sigma_f2
        
        # Setup constraints and bounds
        constraints = self._setup_constraints()
        bounds = PPLSConstraints.get_bounds(self.p, self.q, self.r)
        
        # Run multi-start optimization
        solutions = []
        for theta0 in starting_points:
            try:
                result = self._optimize_single_start(
                    theta0, objective, constraints, bounds
                )
                solutions.append(result)
            except Exception as e:
                warnings.warn(f"Optimization failed for one starting point: {e}")
                continue
                
        # Select best solution
        best_solution = self._select_best_solution(solutions)
        
        # Extract parameters from best solution
        W, C, B, Sigma_t, sigma_h2 = objective._theta_to_params(best_solution['x'])
        
        results = {
            'W': W,
            'C': C,
            'B': B,
            'Sigma_t': Sigma_t,
            'sigma_e2': sigma_e2,
            'sigma_f2': sigma_f2,
            'sigma_h2': sigma_h2,
            'objective_value': best_solution['fun'],
            'n_iterations': best_solution.get('nit', 0),
            'success': best_solution['success']
        }
        
        # Save results if directory provided
        if experiment_dir:
            self._save_results(results, experiment_dir, "SLM", trial_id)
            
        return results
    
    def _save_results(self, results: Dict, experiment_dir: str, 
                     algorithm_name: str, trial_id: Optional[int] = None):
        """Save estimation results to experiment directory."""
        est_dir = os.path.join(experiment_dir, "estimates", algorithm_name)
        os.makedirs(est_dir, exist_ok=True)
        
        # Determine filename
        if trial_id is not None:
            prefix = f"trial_{trial_id:03d}"
        else:
            prefix = "estimated"
            
        # Save full results
        with open(os.path.join(est_dir, f"{prefix}_results.pkl"), 'wb') as f:
            pickle.dump(results, f)
            
        # Save parameters separately
        for param in ['W', 'C', 'B', 'Sigma_t']:
            if param in results:
                np.save(os.path.join(est_dir, f"{prefix}_{param}.npy"), results[param])
                
        # Save summary
        summary = {
            "algorithm": algorithm_name,
            "trial_id": trial_id,
            "success": results.get('success', False),
            "objective_value": float(results.get('objective_value', np.inf)),
            "n_iterations": results.get('n_iterations', 0),
            "sigma_e2": results.get('sigma_e2'),
            "sigma_f2": results.get('sigma_f2'),
            "sigma_h2": results.get('sigma_h2'),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(est_dir, f"{prefix}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
    
    def _optimize_single_start(self, theta0: np.ndarray, objective: PPLSObjective,
                              constraints: List, bounds: List) -> Dict:
        """
        Run optimization from a single starting point.
        
        Parameters:
        -----------
        theta0 : np.ndarray
            Starting parameter vector
        objective : PPLSObjective
            Objective function instance
        constraints : list
            Constraint specifications
        bounds : list
            Parameter bounds
            
        Returns:
        --------
        result : dict
            Optimization result from scipy.optimize
        """
        # Optimization options optimized for speed and stability
        options = {
            'maxiter': self.max_iter,
            'gtol': 1e-2,
            'xtol': 1e-2,
            'barrier_tol': 1e-2,
            'initial_constr_penalty': 1.0
        }
        
        # Run optimization
        result = minimize(
            fun=objective.scalar_log_likelihood,
            x0=theta0,
            method=self.optimizer,
            constraints=constraints,
            bounds=bounds,
            options=options
        )
        
        return result
    
    def _setup_constraints(self) -> List[Dict]:
        """
        Setup optimization constraints.
        
        Returns:
        --------
        constraints : list
            List of constraint dictionaries for scipy.optimize
        """
        constraints = []
        
        # Simplified constraint approach: use penalty method instead of strict equality
        def combined_constraint(theta):
            """Combined constraint function with tolerance."""
            W = theta[:self.p * self.r].reshape(self.p, self.r)
            C = theta[self.p * self.r:(self.p + self.q) * self.r].reshape(self.q, self.r)
            
            # Compute constraint violations
            W_gram = W.T @ W
            C_gram = C.T @ C
            
            # Sum of squared deviations from identity (should be small)
            W_violation = np.sum((W_gram - np.eye(self.r))**2)
            C_violation = np.sum((C_gram - np.eye(self.r))**2)
            
            # Return total violation (should be <= tolerance)
            return 1e-3 - W_violation - C_violation
        
        constraints.append({
            'type': 'ineq',
            'fun': combined_constraint
        })
        
        return constraints
    
    def _select_best_solution(self, solutions: List[Dict]) -> Dict:
        """
        Select best solution from multi-start results.
        
        Parameters:
        -----------
        solutions : List[Dict]
            List of optimization results
            
        Returns:
        --------
        best_solution : dict
            Solution with lowest objective value
        """
        if not solutions:
            raise ValueError("No solutions provided")
            
        valid_solutions = [s for s in solutions if s['success']]
        
        if not valid_solutions:
            # Return best unsuccessful solution if no successful ones
            return min(solutions, key=lambda s: s['fun'])
            
        # Return solution with lowest objective value
        return min(valid_solutions, key=lambda s: s['fun'])


class EMAlgorithm(PPLSAlgorithm):
    """
    Expectation-Maximization (EM) algorithm for PPLS parameter estimation.
    Implements closed-form updates with all parameters estimated in the EM loop.
    """
    
    def __init__(self, p: int, q: int, r: int,
                 max_iter: int = 1000,
                 tolerance: float = 1e-4):
        """
        Initialize EM algorithm.
        
        Parameters:
        -----------
        p, q, r : int
            Model dimensions
        max_iter : int
            Maximum EM iterations
        tolerance : float
            Convergence tolerance
        """
        super().__init__(p, q, r)
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def fit(self, X: np.ndarray, Y: np.ndarray,
            starting_points: List[np.ndarray],
            experiment_dir: Optional[str] = None,
            trial_id: Optional[int] = None) -> Dict:
        """
        Fit PPLS model using EM algorithm with identical starting points.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, p)
            Input data
        Y : np.ndarray of shape (N, q)
            Output data
        starting_points : List[np.ndarray]
            Identical starting points for multi-start
        experiment_dir : str, optional
            Directory to save results
        trial_id : int, optional
            Trial number for Monte Carlo experiments
            
        Returns:
        --------
        results : dict
            Best estimated parameters
        """
        # Run EM from each starting point (no pre-estimation needed)
        solutions = []
        for theta0 in starting_points:
            try:
                result = self._run_single_em(X, Y, theta0)
                solutions.append(result)
            except Exception as e:
                warnings.warn(f"EM failed for one starting point: {e}")
                continue
                
        # Select best solution based on likelihood
        best_solution = self._select_best_solution(X, Y, solutions)
        
        # Save results if directory provided
        if experiment_dir:
            self._save_results(best_solution, experiment_dir, "EM", trial_id)
            
        return best_solution
    
    def _save_results(self, results: Dict, experiment_dir: str,
                     algorithm_name: str, trial_id: Optional[int] = None):
        """Save estimation results to experiment directory."""
        est_dir = os.path.join(experiment_dir, "estimates", algorithm_name)
        os.makedirs(est_dir, exist_ok=True)
        
        # Determine filename
        if trial_id is not None:
            prefix = f"trial_{trial_id:03d}"
        else:
            prefix = "estimated"
            
        # Save full results
        with open(os.path.join(est_dir, f"{prefix}_results.pkl"), 'wb') as f:
            pickle.dump(results, f)
            
        # Save parameters separately
        for param in ['W', 'C', 'B', 'Sigma_t']:
            if param in results:
                np.save(os.path.join(est_dir, f"{prefix}_{param}.npy"), results[param])
                
        # Save summary
        summary = {
            "algorithm": algorithm_name,
            "trial_id": trial_id,
            "log_likelihood": float(results.get('log_likelihood', -np.inf)),
            "n_iterations": results.get('n_iterations', 0),
            "sigma_e2": results.get('sigma_e2'),
            "sigma_f2": results.get('sigma_f2'),
            "sigma_h2": results.get('sigma_h2'),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(est_dir, f"{prefix}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
    
    def _run_single_em(self, X: np.ndarray, Y: np.ndarray, theta0: np.ndarray) -> Dict:
        """
        Run EM algorithm from single initialization.
        
        Parameters:
        -----------
        X, Y : data matrices
        theta0 : initial parameter vector
        
        Returns:
        --------
        results : dict
            Estimated parameters
        """
        # Extract initial parameters
        W = theta0[:self.p * self.r].reshape(self.p, self.r)
        C = theta0[self.p * self.r:(self.p + self.q) * self.r].reshape(self.q, self.r)
        theta_t = theta0[(self.p + self.q) * self.r:(self.p + self.q + 1) * self.r]
        b = theta0[(self.p + self.q + 1) * self.r:(self.p + self.q + 2) * self.r]
        sigma_h2 = theta0[-1]
        
        Sigma_t = np.diag(theta_t)
        B = np.diag(b)
        
        # Initialize noise variances (will be estimated in M-step)
        sigma_e2 = 0.1
        sigma_f2 = 0.1
        
        # Pre-compute for efficiency
        N = X.shape[0]
        eye_p = np.eye(self.p)
        eye_q = np.eye(self.q)
        eye_r = np.eye(self.r)
        
        # EM iterations
        for iteration in range(self.max_iter):
            # Store old parameters for convergence check
            params_old = {
                'W': W.copy(), 'C': C.copy(), 'B': B.copy(),
                'Sigma_t': Sigma_t.copy(), 'sigma_h2': sigma_h2,
                'sigma_e2': sigma_e2, 'sigma_f2': sigma_f2
            }
            
            # E-step: compute posterior expectations (vectorized)
            E_T, E_U, Cov_T, Cov_U = self._e_step_vectorized(
                X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2,
                eye_p, eye_q, eye_r
            )
            
            # M-step: update parameters
            W, C, B, Sigma_t, sigma_h2, sigma_e2, sigma_f2 = self._m_step_optimized(
                X, Y, E_T, E_U, Cov_T, Cov_U, N
            )
            
            # Check convergence
            params_new = {
                'W': W, 'C': C, 'B': B,
                'Sigma_t': Sigma_t, 'sigma_h2': sigma_h2,
                'sigma_e2': sigma_e2, 'sigma_f2': sigma_f2
            }
            
            if self._check_convergence_fast(params_old, params_new):
                break
                
        # Compute final log-likelihood
        XY = np.hstack([X, Y])
        XY_centered = XY - np.mean(XY, axis=0)
        S = (XY_centered.T @ XY_centered) / N
        
        Sigma = self.model.compute_covariance_matrix(
            W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2
        )
        log_likelihood = self.model.log_likelihood_matrix(S, Sigma)
        
        results = {
            'W': W,
            'C': C,
            'B': B,
            'Sigma_t': Sigma_t,
            'sigma_e2': sigma_e2,
            'sigma_f2': sigma_f2,
            'sigma_h2': sigma_h2,
            'log_likelihood': log_likelihood,
            'n_iterations': iteration + 1
        }
        
        return results
    
    def _e_step_vectorized(self, X: np.ndarray, Y: np.ndarray,
                          W: np.ndarray, C: np.ndarray, B: np.ndarray,
                          Sigma_t: np.ndarray, sigma_e2: float, 
                          sigma_f2: float, sigma_h2: float,
                          eye_p: np.ndarray, eye_q: np.ndarray, eye_r: np.ndarray) -> Tuple:
        """
        E-step: Compute posterior expectations of latent variables (vectorized).
        
        Returns:
        --------
        E_T : np.ndarray of shape (N, r)
            E[t_i | x_i, y_i]
        E_U : np.ndarray of shape (N, r)
            E[u_i | x_i, y_i]
        Cov_T : np.ndarray of shape (r, r)
            Posterior covariance of t
        Cov_U : np.ndarray of shape (r, r)
            Posterior covariance of u
        """
        N = X.shape[0]
        
        # Add regularization for numerical stability
        reg = 1e-8
        
        # Step 1: Compute E[t | x]
        WSigma_t = W @ Sigma_t  # (p, r)
        Cov_x = WSigma_t @ W.T + sigma_e2 * eye_p  # (p, p)
        
        try:
            # Add regularization to diagonal
            Cov_x_reg = Cov_x + reg * eye_p
            L_x = np.linalg.cholesky(Cov_x_reg)
            v = np.linalg.solve(L_x.T, np.linalg.solve(L_x, WSigma_t))  # (p, r)
        except np.linalg.LinAlgError:
            try:
                Cov_x_inv = np.linalg.pinv(Cov_x + reg * eye_p)
                v = Cov_x_inv @ WSigma_t  # (p, r)
            except:
                # Fallback: use simple approximation
                v = WSigma_t / (sigma_e2 + reg)
            
        E_T = X @ v  # (N, p) @ (p, r) = (N, r)
        
        # Step 2: Compute E[u | y, t] using simplified approach
        # Start with E[u | t] = B @ E[t]
        E_U_base = E_T @ B  # (N, r)
        
        # Compute residual and correction
        Y_pred = E_U_base @ C.T  # (N, r) @ (r, q) = (N, q)
        Y_residual = Y - Y_pred  # (N, q)
        
        # Simple correction based on residual
        if sigma_f2 > reg:
            correction = (Y_residual @ C) / (sigma_f2 + sigma_h2 + reg)  # (N, r)
            E_U = E_U_base + correction
        else:
            E_U = E_U_base
        
        # Step 3: Compute posterior covariances with regularization
        Cov_T = Sigma_t - Sigma_t @ W.T @ v  # (r, r)
        Cov_T = (Cov_T + Cov_T.T) / 2 + reg * eye_r  # Ensure symmetry and positive definiteness
        
        # Simplified covariance for U
        Cov_U = B @ Cov_T @ B.T + sigma_h2 * eye_r + reg * eye_r
        
        return E_T, E_U, Cov_T, Cov_U
    
    def _m_step_optimized(self, X: np.ndarray, Y: np.ndarray,
                         E_T: np.ndarray, E_U: np.ndarray,
                         Cov_T: np.ndarray, Cov_U: np.ndarray, N: int) -> Tuple:
        """
        M-step: Update parameters using closed-form solutions (optimized).
        
        Returns:
        --------
        W, C, B, Sigma_t, sigma_h2, sigma_e2, sigma_f2 : updated parameters
        """
        reg = 1e-8
        
        # Update W using stabilized QR decomposition
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        E_T_centered = E_T - np.mean(E_T, axis=0, keepdims=True)
        
        # Compute cross-covariance
        XTcov = X_centered.T @ E_T_centered / N  # (p, r)
        
        # Use SVD for more stable orthogonalization
        try:
            U_w, s_w, Vt_w = np.linalg.svd(XTcov, full_matrices=False)
            W_new = U_w[:, :self.r] @ Vt_w[:self.r, :]
        except:
            # Fallback to QR
            W_new, _ = np.linalg.qr(XTcov)
            W_new = W_new[:, :self.r]
            
        # Ensure we have r columns
        if W_new.shape[1] < self.r:
            # Pad with random orthogonal vectors
            remaining = self.r - W_new.shape[1]
            random_part = np.random.randn(self.p, remaining)
            # Orthogonalize against existing
            for i in range(remaining):
                for j in range(W_new.shape[1]):
                    random_part[:, i] -= np.dot(random_part[:, i], W_new[:, j]) * W_new[:, j]
                # Normalize
                random_part[:, i] /= (np.linalg.norm(random_part[:, i]) + reg)
            W_new = np.hstack([W_new, random_part])
            
        # Update C similarly
        Y_centered = Y - np.mean(Y, axis=0, keepdims=True)
        E_U_centered = E_U - np.mean(E_U, axis=0, keepdims=True)
        
        YUcov = Y_centered.T @ E_U_centered / N  # (q, r)
        
        try:
            U_c, s_c, Vt_c = np.linalg.svd(YUcov, full_matrices=False)
            C_new = U_c[:, :self.r] @ Vt_c[:self.r, :]
        except:
            C_new, _ = np.linalg.qr(YUcov)
            C_new = C_new[:, :self.r]
            
        if C_new.shape[1] < self.r:
            remaining = self.r - C_new.shape[1]
            random_part = np.random.randn(self.q, remaining)
            for i in range(remaining):
                for j in range(C_new.shape[1]):
                    random_part[:, i] -= np.dot(random_part[:, i], C_new[:, j]) * C_new[:, j]
                random_part[:, i] /= (np.linalg.norm(random_part[:, i]) + reg)
            C_new = np.hstack([C_new, random_part])
            
        # Update B with numerical stability
        E_T_E_T = E_T.T @ E_T + N * Cov_T  # (r, r)
        E_U_E_T = E_U.T @ E_T  # (r, r)
        
        # Use only diagonal elements for diagonal B
        diag_E_T_E_T = np.diag(E_T_E_T)
        diag_E_U_E_T = np.diag(E_U_E_T)
        
        # Avoid division by very small numbers
        b_new = diag_E_U_E_T / (diag_E_T_E_T + reg)
        
        # Ensure positive and reasonable bounds
        b_new = np.clip(np.abs(b_new), 0.01, 10.0)
        B_new = np.diag(b_new)
        
        # Update Sigma_t with bounds
        sigma_t_diag = diag_E_T_E_T / N
        sigma_t_diag = np.clip(sigma_t_diag, 0.01, 10.0)
        Sigma_t_new = np.diag(sigma_t_diag)
        
        # Update sigma_h2 with bounds
        E_U_U = E_U.T @ E_U + N * Cov_U
        
        # Simplified calculation for diagonal B
        trace_1 = np.trace(E_U_U)
        trace_2 = np.sum(b_new * diag_E_U_E_T)
        trace_3 = np.sum(b_new**2 * diag_E_T_E_T)
        
        sigma_h2_new = (trace_1 - 2 * trace_2 + trace_3) / (N * self.r)
        sigma_h2_new = max(sigma_h2_new, 0.01)  # Lower bound
        sigma_h2_new = min(sigma_h2_new, 1.0)   # Upper bound
        
        # Update sigma_e2 (noise variance for X)
        X_residual = X - E_T @ W_new.T
        sigma_e2_new = np.sum(X_residual**2) / (N * self.p)
        sigma_e2_new = max(sigma_e2_new, 0.01)  # Lower bound
        sigma_e2_new = min(sigma_e2_new, 1.0)   # Upper bound
        
        # Update sigma_f2 (noise variance for Y)  
        Y_residual = Y - E_U @ C_new.T
        sigma_f2_new = np.sum(Y_residual**2) / (N * self.q)
        sigma_f2_new = max(sigma_f2_new, 0.01)  # Lower bound
        sigma_f2_new = min(sigma_f2_new, 1.0)   # Upper bound
        
        return W_new, C_new, B_new, Sigma_t_new, sigma_h2_new, sigma_e2_new, sigma_f2_new
    
    def _check_convergence_fast(self, params_old: Dict, params_new: Dict) -> bool:
        """
        Check convergence based on parameter changes (fast version).
        
        Parameters:
        -----------
        params_old, params_new : dict
            Old and new parameter values
            
        Returns:
        --------
        converged : bool
            True if converged
        """
        # Use relative changes with more relaxed tolerance
        tol = self.tolerance
        
        # For matrices, use Frobenius norm with relative scaling
        change_W = np.linalg.norm(params_new['W'] - params_old['W'], 'fro') / (np.linalg.norm(params_old['W'], 'fro') + 1e-10)
        change_C = np.linalg.norm(params_new['C'] - params_old['C'], 'fro') / (np.linalg.norm(params_old['C'], 'fro') + 1e-10)
        
        # For diagonal matrices, check diagonal elements with absolute tolerance
        change_B = np.max(np.abs(np.diag(params_new['B']) - np.diag(params_old['B'])))
        change_Sigma_t = np.max(np.abs(np.diag(params_new['Sigma_t']) - np.diag(params_old['Sigma_t'])))
        
        # For scalar parameters, use relative change
        change_h = abs(params_new['sigma_h2'] - params_old['sigma_h2']) / (params_old['sigma_h2'] + 1e-10)
        change_e = abs(params_new['sigma_e2'] - params_old['sigma_e2']) / (params_old['sigma_e2'] + 1e-10)
        change_f = abs(params_new['sigma_f2'] - params_old['sigma_f2']) / (params_old['sigma_f2'] + 1e-10)
        
        # Use a combination of relative and absolute criteria
        rel_converged = max(change_W, change_C, change_h, change_e, change_f) < tol
        abs_converged = max(change_B, change_Sigma_t) < tol * 10  # More relaxed for diagonal elements
        
        return rel_converged and abs_converged
    
    def _select_best_solution(self, X: np.ndarray, Y: np.ndarray, 
                            solutions: List[Dict]) -> Dict:
        """
        Select best solution based on likelihood.
        
        Parameters:
        -----------
        X, Y : data matrices
        solutions : list of EM results
        
        Returns:
        --------
        best_solution : dict
            Solution with highest likelihood
        """
        if not solutions:
            raise ValueError("No valid solutions found")
            
        # Return solution with lowest negative log-likelihood
        return min(solutions, key=lambda s: -s.get('log_likelihood', float('inf')))


class ECMAlgorithm(PPLSAlgorithm):
    """
    Expectation-Conditional Maximization (ECM) algorithm for PPLS parameter estimation.
    Implements TRUE ECM where E-step is performed after EACH conditional maximization step.
    """
    
    def __init__(self, p: int, q: int, r: int,
                 max_iter: int = 1000,
                 tolerance: float = 1e-4):
        """
        Initialize ECM algorithm.
        
        Parameters:
        -----------
        p, q, r : int
            Model dimensions
        max_iter : int
            Maximum ECM iterations
        tolerance : float
            Convergence tolerance
        """
        super().__init__(p, q, r)
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def fit(self, X: np.ndarray, Y: np.ndarray,
            starting_points: List[np.ndarray],
            experiment_dir: Optional[str] = None,
            trial_id: Optional[int] = None) -> Dict:
        """
        Fit PPLS model using ECM algorithm with identical starting points.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, p)
            Input data
        Y : np.ndarray of shape (N, q)
            Output data
        starting_points : List[np.ndarray]
            Identical starting points for multi-start
        experiment_dir : str, optional
            Directory to save results
        trial_id : int, optional
            Trial number for Monte Carlo experiments
            
        Returns:
        --------
        results : dict
            Best estimated parameters
        """
        # Run ECM from each starting point
        solutions = []
        for theta0 in starting_points:
            try:
                result = self._run_single_ecm(X, Y, theta0)
                solutions.append(result)
            except Exception as e:
                warnings.warn(f"ECM failed for one starting point: {e}")
                continue
                
        # Select best solution based on likelihood
        best_solution = self._select_best_solution(X, Y, solutions)
        
        # Save results if directory provided
        if experiment_dir:
            self._save_results(best_solution, experiment_dir, "ECM", trial_id)
            
        return best_solution
    
    def _save_results(self, results: Dict, experiment_dir: str,
                     algorithm_name: str, trial_id: Optional[int] = None):
        """Save estimation results to experiment directory."""
        est_dir = os.path.join(experiment_dir, "estimates", algorithm_name)
        os.makedirs(est_dir, exist_ok=True)
        
        # Determine filename
        if trial_id is not None:
            prefix = f"trial_{trial_id:03d}"
        else:
            prefix = "estimated"
            
        # Save full results
        with open(os.path.join(est_dir, f"{prefix}_results.pkl"), 'wb') as f:
            pickle.dump(results, f)
            
        # Save parameters separately
        for param in ['W', 'C', 'B', 'Sigma_t']:
            if param in results:
                np.save(os.path.join(est_dir, f"{prefix}_{param}.npy"), results[param])
                
        # Save summary
        summary = {
            "algorithm": algorithm_name,
            "trial_id": trial_id,
            "log_likelihood": float(results.get('log_likelihood', -np.inf)),
            "n_iterations": results.get('n_iterations', 0),
            "sigma_e2": results.get('sigma_e2'),
            "sigma_f2": results.get('sigma_f2'),
            "sigma_h2": results.get('sigma_h2'),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(est_dir, f"{prefix}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
    
    def _run_single_ecm(self, X: np.ndarray, Y: np.ndarray, theta0: np.ndarray) -> Dict:
        """
        Run TRUE ECM algorithm from single initialization.
        Key difference from EM: E-step is performed after EACH CM-step.
        
        Parameters:
        -----------
        X, Y : data matrices
        theta0 : initial parameter vector
        
        Returns:
        --------
        results : dict
            Estimated parameters
        """
        # Initialize parameters (same as EM)
        W = theta0[:self.p * self.r].reshape(self.p, self.r)
        C = theta0[self.p * self.r:(self.p + self.q) * self.r].reshape(self.q, self.r)
        theta_t = theta0[(self.p + self.q) * self.r:(self.p + self.q + 1) * self.r]
        b = theta0[(self.p + self.q + 1) * self.r:(self.p + self.q + 2) * self.r]
        sigma_h2 = theta0[-1]
        
        Sigma_t = np.diag(theta_t)
        B = np.diag(b)
        
        # Initialize noise variances (will be estimated in CM-steps)
        sigma_e2 = 0.1
        sigma_f2 = 0.1
        
        # Pre-compute for efficiency
        N = X.shape[0]
        eye_p = np.eye(self.p)
        eye_q = np.eye(self.q)
        eye_r = np.eye(self.r)
        
        # TRUE ECM iterations
        for iteration in range(self.max_iter):
            # Store old parameters for convergence check
            params_old = {
                'W': W.copy(), 'C': C.copy(), 'B': B.copy(),
                'Sigma_t': Sigma_t.copy(), 'sigma_h2': sigma_h2,
                'sigma_e2': sigma_e2, 'sigma_f2': sigma_f2
            }
            
            # ========== CM-STEP 1: Update W and C ==========
            # E-step with current parameters
            E_T, E_U, Cov_T, Cov_U = self._e_step_vectorized(
                X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2,
                eye_p, eye_q, eye_r
            )
            
            # Update only W and C while keeping other parameters fixed
            W, C = self._cm_step_loadings(X, Y, E_T, E_U, N)
            
            # ========== CM-STEP 2: Update B and Sigma_t ==========
            # Re-compute E-step with updated W and C
            E_T, E_U, Cov_T, Cov_U = self._e_step_vectorized(
                X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2,
                eye_p, eye_q, eye_r
            )
            
            # Update only B and Sigma_t while keeping other parameters fixed
            B, Sigma_t = self._cm_step_latent_params(E_T, E_U, Cov_T, Cov_U, N)
            
            # ========== CM-STEP 3: Update noise parameters ==========
            # Re-compute E-step with updated W, C, B, and Sigma_t
            E_T, E_U, Cov_T, Cov_U = self._e_step_vectorized(
                X, Y, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2,
                eye_p, eye_q, eye_r
            )
            
            # Update only noise parameters while keeping other parameters fixed
            sigma_e2, sigma_f2, sigma_h2 = self._cm_step_noise_params(
                X, Y, E_T, E_U, W, C, B, Cov_T, Cov_U, N
            )
            
            # Check convergence
            params_new = {
                'W': W, 'C': C, 'B': B,
                'Sigma_t': Sigma_t, 'sigma_h2': sigma_h2,
                'sigma_e2': sigma_e2, 'sigma_f2': sigma_f2
            }
            
            if self._check_convergence_fast(params_old, params_new):
                break
                
        # Compute final log-likelihood
        XY = np.hstack([X, Y])
        XY_centered = XY - np.mean(XY, axis=0)
        S = (XY_centered.T @ XY_centered) / N
        
        Sigma = self.model.compute_covariance_matrix(
            W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2
        )
        log_likelihood = self.model.log_likelihood_matrix(S, Sigma)
        
        results = {
            'W': W,
            'C': C,
            'B': B,
            'Sigma_t': Sigma_t,
            'sigma_e2': sigma_e2,
            'sigma_f2': sigma_f2,
            'sigma_h2': sigma_h2,
            'log_likelihood': log_likelihood,
            'n_iterations': iteration + 1
        }
        
        return results
    
    def _e_step_vectorized(self, X: np.ndarray, Y: np.ndarray,
                          W: np.ndarray, C: np.ndarray, B: np.ndarray,
                          Sigma_t: np.ndarray, sigma_e2: float, 
                          sigma_f2: float, sigma_h2: float,
                          eye_p: np.ndarray, eye_q: np.ndarray, eye_r: np.ndarray) -> Tuple:
        """
        E-step: Compute posterior expectations of latent variables (same as EM).
        
        Returns:
        --------
        E_T : np.ndarray of shape (N, r)
            E[t_i | x_i, y_i]
        E_U : np.ndarray of shape (N, r)
            E[u_i | x_i, y_i]
        Cov_T : np.ndarray of shape (r, r)
            Posterior covariance of t
        Cov_U : np.ndarray of shape (r, r)
            Posterior covariance of u
        """
        N = X.shape[0]
        
        # Add regularization for numerical stability
        reg = 1e-8
        
        # Step 1: Compute E[t | x]
        WSigma_t = W @ Sigma_t  # (p, r)
        Cov_x = WSigma_t @ W.T + sigma_e2 * eye_p  # (p, p)
        
        try:
            # Add regularization to diagonal
            Cov_x_reg = Cov_x + reg * eye_p
            L_x = np.linalg.cholesky(Cov_x_reg)
            v = np.linalg.solve(L_x.T, np.linalg.solve(L_x, WSigma_t))  # (p, r)
        except np.linalg.LinAlgError:
            try:
                Cov_x_inv = np.linalg.pinv(Cov_x + reg * eye_p)
                v = Cov_x_inv @ WSigma_t  # (p, r)
            except:
                # Fallback: use simple approximation
                v = WSigma_t / (sigma_e2 + reg)
            
        E_T = X @ v  # (N, p) @ (p, r) = (N, r)
        
        # Step 2: Compute E[u | y, t] using simplified approach
        # Start with E[u | t] = B @ E[t]
        E_U_base = E_T @ B  # (N, r)
        
        # Compute residual and correction
        Y_pred = E_U_base @ C.T  # (N, r) @ (r, q) = (N, q)
        Y_residual = Y - Y_pred  # (N, q)
        
        # Simple correction based on residual
        if sigma_f2 > reg:
            correction = (Y_residual @ C) / (sigma_f2 + sigma_h2 + reg)  # (N, r)
            E_U = E_U_base + correction
        else:
            E_U = E_U_base
        
        # Step 3: Compute posterior covariances with regularization
        Cov_T = Sigma_t - Sigma_t @ W.T @ v  # (r, r)
        Cov_T = (Cov_T + Cov_T.T) / 2 + reg * eye_r  # Ensure symmetry and positive definiteness
        
        # Simplified covariance for U
        Cov_U = B @ Cov_T @ B.T + sigma_h2 * eye_r + reg * eye_r
        
        return E_T, E_U, Cov_T, Cov_U
    
    def _cm_step_loadings(self, X: np.ndarray, Y: np.ndarray,
                         E_T: np.ndarray, E_U: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        CM-step 1: Update loading matrices W and C while keeping other parameters fixed.
        
        Returns:
        --------
        W_new, C_new : updated loading matrices
        """
        reg = 1e-8
        
        # Update W using stabilized QR decomposition
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        E_T_centered = E_T - np.mean(E_T, axis=0, keepdims=True)
        
        # Compute cross-covariance
        XTcov = X_centered.T @ E_T_centered / N  # (p, r)
        
        # Use SVD for more stable orthogonalization
        try:
            U_w, s_w, Vt_w = np.linalg.svd(XTcov, full_matrices=False)
            W_new = U_w[:, :self.r] @ Vt_w[:self.r, :]
        except:
            # Fallback to QR
            W_new, _ = np.linalg.qr(XTcov)
            W_new = W_new[:, :self.r]
            
        # Ensure we have r columns
        if W_new.shape[1] < self.r:
            # Pad with random orthogonal vectors
            remaining = self.r - W_new.shape[1]
            random_part = np.random.randn(self.p, remaining)
            # Orthogonalize against existing
            for i in range(remaining):
                for j in range(W_new.shape[1]):
                    random_part[:, i] -= np.dot(random_part[:, i], W_new[:, j]) * W_new[:, j]
                # Normalize
                random_part[:, i] /= (np.linalg.norm(random_part[:, i]) + reg)
            W_new = np.hstack([W_new, random_part])
            
        # Update C similarly
        Y_centered = Y - np.mean(Y, axis=0, keepdims=True)
        E_U_centered = E_U - np.mean(E_U, axis=0, keepdims=True)
        
        YUcov = Y_centered.T @ E_U_centered / N  # (q, r)
        
        try:
            U_c, s_c, Vt_c = np.linalg.svd(YUcov, full_matrices=False)
            C_new = U_c[:, :self.r] @ Vt_c[:self.r, :]
        except:
            C_new, _ = np.linalg.qr(YUcov)
            C_new = C_new[:, :self.r]
            
        if C_new.shape[1] < self.r:
            remaining = self.r - C_new.shape[1]
            random_part = np.random.randn(self.q, remaining)
            for i in range(remaining):
                for j in range(C_new.shape[1]):
                    random_part[:, i] -= np.dot(random_part[:, i], C_new[:, j]) * C_new[:, j]
                random_part[:, i] /= (np.linalg.norm(random_part[:, i]) + reg)
            C_new = np.hstack([C_new, random_part])
            
        return W_new, C_new
    
    def _cm_step_latent_params(self, E_T: np.ndarray, E_U: np.ndarray,
                              Cov_T: np.ndarray, Cov_U: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        CM-step 2: Update latent parameters B and Sigma_t while keeping other parameters fixed.
        
        Returns:
        --------
        B_new, Sigma_t_new : updated latent parameters
        """
        reg = 1e-8
        
        # Update B with numerical stability
        E_T_E_T = E_T.T @ E_T + N * Cov_T  # (r, r)
        E_U_E_T = E_U.T @ E_T  # (r, r)
        
        # Use only diagonal elements for diagonal B
        diag_E_T_E_T = np.diag(E_T_E_T)
        diag_E_U_E_T = np.diag(E_U_E_T)
        
        # Avoid division by very small numbers
        b_new = diag_E_U_E_T / (diag_E_T_E_T + reg)
        
        # Ensure positive and reasonable bounds
        b_new = np.clip(np.abs(b_new), 0.01, 10.0)
        B_new = np.diag(b_new)
        
        # Update Sigma_t with bounds
        sigma_t_diag = diag_E_T_E_T / N
        sigma_t_diag = np.clip(sigma_t_diag, 0.01, 10.0)
        Sigma_t_new = np.diag(sigma_t_diag)
        
        return B_new, Sigma_t_new
    
    def _cm_step_noise_params(self, X: np.ndarray, Y: np.ndarray,
                             E_T: np.ndarray, E_U: np.ndarray,
                             W: np.ndarray, C: np.ndarray, B: np.ndarray,
                             Cov_T: np.ndarray, Cov_U: np.ndarray, N: int) -> Tuple[float, float, float]:
        """
        CM-step 3: Update noise parameters sigma_e2, sigma_f2, sigma_h2 while keeping other parameters fixed.
        
        Returns:
        --------
        sigma_e2_new, sigma_f2_new, sigma_h2_new : updated noise parameters
        """
        # Update sigma_e2 (noise variance for X)
        X_residual = X - E_T @ W.T
        sigma_e2_new = np.sum(X_residual**2) / (N * self.p)
        sigma_e2_new = max(sigma_e2_new, 0.01)  # Lower bound
        sigma_e2_new = min(sigma_e2_new, 1.0)   # Upper bound
        
        # Update sigma_f2 (noise variance for Y)  
        Y_residual = Y - E_U @ C.T
        sigma_f2_new = np.sum(Y_residual**2) / (N * self.q)
        sigma_f2_new = max(sigma_f2_new, 0.01)  # Lower bound
        sigma_f2_new = min(sigma_f2_new, 1.0)   # Upper bound
        
        # Update sigma_h2 with bounds
        E_U_U = E_U.T @ E_U + N * Cov_U
        b = np.diag(B)
        
        # Extract diagonal elements for calculation
        E_T_E_T = E_T.T @ E_T + N * Cov_T
        E_U_E_T = E_U.T @ E_T
        
        diag_E_T_E_T = np.diag(E_T_E_T)
        diag_E_U_E_T = np.diag(E_U_E_T)
        
        # Simplified calculation for diagonal B
        trace_1 = np.trace(E_U_U)
        trace_2 = np.sum(b * diag_E_U_E_T)
        trace_3 = np.sum(b**2 * diag_E_T_E_T)
        
        sigma_h2_new = (trace_1 - 2 * trace_2 + trace_3) / (N * self.r)
        sigma_h2_new = max(sigma_h2_new, 0.01)  # Lower bound
        sigma_h2_new = min(sigma_h2_new, 1.0)   # Upper bound
        
        return sigma_e2_new, sigma_f2_new, sigma_h2_new
    
    def _check_convergence_fast(self, params_old: Dict, params_new: Dict) -> bool:
        """
        Check convergence based on parameter changes (same as EM).
        
        Parameters:
        -----------
        params_old, params_new : dict
            Old and new parameter values
            
        Returns:
        --------
        converged : bool
            True if converged
        """
        # Use relative changes with more relaxed tolerance
        tol = self.tolerance
        
        # For matrices, use Frobenius norm with relative scaling
        change_W = np.linalg.norm(params_new['W'] - params_old['W'], 'fro') / (np.linalg.norm(params_old['W'], 'fro') + 1e-10)
        change_C = np.linalg.norm(params_new['C'] - params_old['C'], 'fro') / (np.linalg.norm(params_old['C'], 'fro') + 1e-10)
        
        # For diagonal matrices, check diagonal elements with absolute tolerance
        change_B = np.max(np.abs(np.diag(params_new['B']) - np.diag(params_old['B'])))
        change_Sigma_t = np.max(np.abs(np.diag(params_new['Sigma_t']) - np.diag(params_old['Sigma_t'])))
        
        # For scalar parameters, use relative change
        change_h = abs(params_new['sigma_h2'] - params_old['sigma_h2']) / (params_old['sigma_h2'] + 1e-10)
        change_e = abs(params_new['sigma_e2'] - params_old['sigma_e2']) / (params_old['sigma_e2'] + 1e-10)
        change_f = abs(params_new['sigma_f2'] - params_old['sigma_f2']) / (params_old['sigma_f2'] + 1e-10)
        
        # Use a combination of relative and absolute criteria
        rel_converged = max(change_W, change_C, change_h, change_e, change_f) < tol
        abs_converged = max(change_B, change_Sigma_t) < tol * 10  # More relaxed for diagonal elements
        
        return rel_converged and abs_converged
    
    def _select_best_solution(self, X: np.ndarray, Y: np.ndarray, 
                            solutions: List[Dict]) -> Dict:
        """
        Select best solution based on likelihood (same as EM).
        
        Parameters:
        -----------
        X, Y : data matrices
        solutions : list of ECM results
        
        Returns:
        --------
        best_solution : dict
            Solution with highest likelihood
        """
        if not solutions:
            raise ValueError("No valid solutions found")
            
        # Return solution with lowest negative log-likelihood
        return min(solutions, key=lambda s: -s.get('log_likelihood', float('inf')))