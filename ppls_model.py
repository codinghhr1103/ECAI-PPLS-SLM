"""
PPLS Model Core Implementation
==============================

This module implements the fundamental PPLS (Probabilistic Partial Least Squares) model structure,
objective functions, and utility classes based on the paper "Scalar Likelihood Method for 
Probabilistic Partial Least Squares Model with Rank n Update".

Architecture Overview:
---------------------
The module provides four main classes:
1. PPLSModel: Core PPLS model implementation with covariance matrix computation
2. PPLSObjective: Optimization objective functions in both matrix and scalar forms
3. PPLSConstraints: Constraint handling for optimization
4. NoiseEstimator: Pre-estimation of noise parameters

Function List:
--------------
PPLSModel:
    - __init__(p, q, r): Initialize model dimensions
    - compute_covariance_matrix(W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2): Compute joint covariance matrix Σ
    - log_likelihood_matrix(S, Sigma): Matrix form log-likelihood
    - sample(n_samples, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2): Generate samples from PPLS model

PPLSObjective:
    - __init__(p, q, r, S): Initialize objective function with dimensions and sample covariance
    - scalar_log_likelihood(theta): Scalar form log-likelihood (Theorem A)
    - _compute_ln_det(W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2): Compute ln(det Σ)
    - _compute_trace(W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2): Compute tr(SΣ^-1)
    - _theta_to_params(theta): Convert flattened parameters to matrices
    - _params_to_theta(W, C, B, Sigma_t, sigma_h2): Convert matrices to flattened parameters

PPLSConstraints:
    - orthonormality_constraint(theta, p, q, r): Implement W'W = I and C'C = I constraints
    - get_bounds(p, q, r): Define parameter bounds for optimization
    - inequality_constraints(p, q, r, slack): Convert equality to inequality constraints

NoiseEstimator:
    - estimate_noise_variances(X, Y): Closed-form noise variance estimation
    - _center_data(data): Center data for noise estimation

Call Relationships:
------------------
PPLSObjective._compute_ln_det() → PPLSModel.compute_covariance_matrix()
PPLSObjective.scalar_log_likelihood() → PPLSObjective._theta_to_params()
PPLSObjective.scalar_log_likelihood() → PPLSObjective._compute_ln_det()
PPLSObjective.scalar_log_likelihood() → PPLSObjective._compute_trace()
algorithms.py → NoiseEstimator.estimate_noise_variances()
algorithms.py → PPLSConstraints.orthonormality_constraint()
algorithms.py → PPLSConstraints.get_bounds()
algorithms.py → PPLSConstraints.inequality_constraints()
"""

import numpy as np
from scipy.linalg import orth
from typing import Tuple, Dict, Optional


class PPLSModel:
    """
    Core PPLS model implementation based on Equation (1) in the paper:
    x = tW^T + e
    y = uC^T + f  
    u = tB + h
    
    where:
    - x ∈ R^p, y ∈ R^q are observed variables
    - t ∈ R^r, u ∈ R^r are latent variables
    - W ∈ R^(p×r), C ∈ R^(q×r) are loading matrices
    - B = diag(b1, ..., br) is diagonal connection matrix
    - e ~ N(0, σ²_e I_p), f ~ N(0, σ²_f I_q), h ~ N(0, σ²_h I_r) are noise terms
    - t ~ N(0, Σ_t) where Σ_t = diag(θ²_t1, ..., θ²_tr)
    """
    
    def __init__(self, p: int, q: int, r: int):
        """
        Initialize PPLS model with dimensions.
        
        Parameters:
        -----------
        p : int
            Dimension of x (input variables)
        q : int
            Dimension of y (output variables)
        r : int
            Number of latent variables
        """
        self.p = p
        self.q = q
        self.r = r
        
        # Validate dimensions
        if r > min(p, q):
            raise ValueError(f"r ({r}) must be less than min(p, q) = {min(p, q)}")
            
    def compute_covariance_matrix(self, W: np.ndarray, C: np.ndarray, B: np.ndarray,
                                  Sigma_t: np.ndarray, sigma_e2: float, 
                                  sigma_f2: float, sigma_h2: float) -> np.ndarray:
        """
        Compute the joint covariance matrix Σ of (x, y) according to Equation (3).
        
        Σ = [W*Σ_t*W' + σ²_e*I_p     W*Σ_t*B*C'                ]
            [C*B*Σ_t*W'             C*(B²*Σ_t + σ²_h*I_r)*C' + σ²_f*I_q]
            
        Returns:
        --------
        Sigma : np.ndarray of shape (p+q, p+q)
            Joint covariance matrix
        """
        # Compute blocks of the covariance matrix
        Sigma_xx = W @ Sigma_t @ W.T + sigma_e2 * np.eye(self.p)
        Sigma_xy = W @ Sigma_t @ B @ C.T
        Sigma_yx = Sigma_xy.T
        Sigma_yy = C @ (B @ B @ Sigma_t + sigma_h2 * np.eye(self.r)) @ C.T + sigma_f2 * np.eye(self.q)
        
        # Assemble full covariance matrix
        Sigma = np.block([[Sigma_xx, Sigma_xy],
                          [Sigma_yx, Sigma_yy]])
        
        return Sigma
    
    def log_likelihood_matrix(self, S: np.ndarray, Sigma: np.ndarray) -> float:
        """
        Compute the log-likelihood in matrix form according to Equation (4).
        
        ln L_N(θ) = -N/2 * [(p+q)ln(2π) + ln det Σ + tr(SΣ^-1)]
        
        Parameters:
        -----------
        S : np.ndarray of shape (p+q, p+q)
            Sample covariance matrix
        Sigma : np.ndarray of shape (p+q, p+q)
            Model covariance matrix
            
        Returns:
        --------
        log_likelihood : float
            Log-likelihood value (without the constant term)
        """
        # Compute log determinant
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return -np.inf
            
        # Compute trace(S * Sigma^-1)
        try:
            Sigma_inv = np.linalg.inv(Sigma)
            trace_term = np.trace(S @ Sigma_inv)
        except np.linalg.LinAlgError:
            return -np.inf
            
        # Return negative log-likelihood (for minimization)
        return logdet + trace_term
    
    def sample(self, n_samples: int, W: np.ndarray, C: np.ndarray, B: np.ndarray,
               Sigma_t: np.ndarray, sigma_e2: float, sigma_f2: float, 
               sigma_h2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples from the PPLS model according to Equation (1).
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        W, C, B, Sigma_t : model parameters
        sigma_e2, sigma_f2, sigma_h2 : noise variances
        
        Returns:
        --------
        X : np.ndarray of shape (n_samples, p)
            Generated x samples
        Y : np.ndarray of shape (n_samples, q)
            Generated y samples
        """
        # Generate latent variables t ~ N(0, Σ_t)
        theta_t = np.sqrt(np.diag(Sigma_t))
        T = np.random.randn(n_samples, self.r) @ np.diag(theta_t)
        
        # Generate noise h ~ N(0, σ²_h I_r)
        H = np.sqrt(sigma_h2) * np.random.randn(n_samples, self.r)
        
        # Compute u = tB + h
        U = T @ B + H
        
        # Generate observation noise
        E = np.sqrt(sigma_e2) * np.random.randn(n_samples, self.p)
        F = np.sqrt(sigma_f2) * np.random.randn(n_samples, self.q)
        
        # Generate observations
        X = T @ W.T + E
        Y = U @ C.T + F
        
        return X, Y


class PPLSObjective:
    """
    Implementation of the scalar likelihood function for PPLS parameter estimation.
    Based on Theorem A in the paper, which expands the likelihood from matrix to scalar form.
    """
    
    def __init__(self, p: int, q: int, r: int, S: np.ndarray):
        """
        Initialize the objective function.
        
        Parameters:
        -----------
        p, q, r : int
            Model dimensions
        S : np.ndarray of shape (p+q, p+q)
            Sample covariance matrix computed from data
        """
        self.p = p
        self.q = q
        self.r = r
        self.S = S
        
        # Partition S according to Equation (11)
        self.S_xx = S[:p, :p]
        self.S_xy = S[:p, p:]
        self.S_yx = S[p:, :p]
        self.S_yy = S[p:, p:]
        
    def scalar_log_likelihood(self, theta: np.ndarray) -> float:
        """
        Compute the scalar form of log-likelihood according to Theorem A.
        
        L(θ) = ln(det Σ) + tr(SΣ^-1)
        
        Parameters:
        -----------
        theta : np.ndarray
            Flattened parameter vector
            
        Returns:
        --------
        L : float
            Objective function value
        """
        # Convert flattened parameters to matrices
        W, C, B, Sigma_t, sigma_h2 = self._theta_to_params(theta)
        
        # Pre-estimated noise variances (will be set by NoiseEstimator)
        sigma_e2 = self.sigma_e2 if hasattr(self, 'sigma_e2') else 0.01
        sigma_f2 = self.sigma_f2 if hasattr(self, 'sigma_f2') else 0.01
        
        # Compute ln(det Σ) using Equation (12)
        ln_det = self._compute_ln_det(W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2)
        
        # Compute tr(SΣ^-1) using Equation (13)
        trace = self._compute_trace(W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2)
        
        return ln_det + trace
    
    def _compute_ln_det(self, W: np.ndarray, C: np.ndarray, B: np.ndarray,
                        Sigma_t: np.ndarray, sigma_e2: float, sigma_f2: float,
                        sigma_h2: float) -> float:
        """
        Compute ln(det Σ) according to Equation (12).
        
        ln(det Σ) = (p-r)ln σ²_e + (q-r)ln σ²_f + 
                    Σᵢ ln[(σ²_f + σ²_h)(θ²_tᵢ + σ²_e) + b²ᵢθ²_tᵢσ²_e]
        """
        ln_det = (self.p - self.r) * np.log(sigma_e2) + (self.q - self.r) * np.log(sigma_f2)
        
        theta_t2 = np.diag(Sigma_t)  # θ²_tᵢ values
        b = np.diag(B)  # bᵢ values
        
        for i in range(self.r):
            D_i = (sigma_f2 + sigma_h2) * (theta_t2[i] + sigma_e2) + b[i]**2 * theta_t2[i] * sigma_e2
            ln_det += np.log(D_i)
            
        return ln_det
    
    def _compute_trace(self, W: np.ndarray, C: np.ndarray, B: np.ndarray,
                       Sigma_t: np.ndarray, sigma_e2: float, sigma_f2: float,
                       sigma_h2: float) -> float:
        """
        Compute tr(SΣ^-1) according to Equation (13).
        
        tr(SΣ^-1) = (1/σ²_e)tr(S_xx) + (1/σ²_f)tr(S_yy) - 
                    Σᵢ[K₂(i)M₂(i) + K₄(i)M₄(i) + K₆(i)M₆(i)]
        """
        trace = np.trace(self.S_xx) / sigma_e2 + np.trace(self.S_yy) / sigma_f2
        
        theta_t2 = np.diag(Sigma_t)
        b = np.diag(B)
        
        for i in range(self.r):
            # Compute D_i
            D_i = (sigma_f2 + sigma_h2) * (theta_t2[i] + sigma_e2) + b[i]**2 * theta_t2[i] * sigma_e2
            
            # Compute M values
            M2_i = (sigma_f2 + sigma_h2) * theta_t2[i] / D_i
            M4_i = (sigma_h2 * (theta_t2[i] + sigma_e2) + b[i]**2 * theta_t2[i] * sigma_e2) / D_i
            M6_i = b[i] * theta_t2[i] / D_i
            
            # Compute K values
            W_i = W[:, i:i+1]  # i-th column of W
            C_i = C[:, i:i+1]  # i-th column of C
            
            K2_i = (W_i.T @ self.S_xx @ W_i).item() / sigma_e2
            K4_i = (C_i.T @ self.S_yy @ C_i).item() / sigma_f2
            K6_i = 2 * (W_i.T @ self.S_xy @ C_i).item()
            
            trace -= K2_i * M2_i + K4_i * M4_i + K6_i * M6_i
            
        return trace
    
    def _theta_to_params(self, theta: np.ndarray) -> Tuple:
        """
        Convert flattened parameter vector to individual matrices.
        
        Parameters:
        -----------
        theta : np.ndarray
            Flattened parameter vector of length (p+q+2)*r + 1
            
        Returns:
        --------
        W, C, B, Sigma_t, sigma_h2 : tuple of arrays
        """
        idx = 0
        
        # Extract W (p × r)
        W = theta[idx:idx + self.p * self.r].reshape(self.p, self.r)
        idx += self.p * self.r
        
        # Extract C (q × r)
        C = theta[idx:idx + self.q * self.r].reshape(self.q, self.r)
        idx += self.q * self.r
        
        # Extract diagonal of Sigma_t (r values)
        theta_t2 = theta[idx:idx + self.r]
        Sigma_t = np.diag(theta_t2)
        idx += self.r
        
        # Extract diagonal of B (r values)
        b = theta[idx:idx + self.r]
        B = np.diag(b)
        idx += self.r
        
        # Extract sigma_h2
        sigma_h2 = theta[idx]
        
        return W, C, B, Sigma_t, sigma_h2
    
    def _params_to_theta(self, W: np.ndarray, C: np.ndarray, B: np.ndarray,
                         Sigma_t: np.ndarray, sigma_h2: float) -> np.ndarray:
        """
        Convert individual matrices to flattened parameter vector.
        """
        theta = np.concatenate([
            W.flatten(),
            C.flatten(),
            np.diag(Sigma_t),
            np.diag(B),
            [sigma_h2]
        ])
        return theta


class PPLSConstraints:
    """
    Handle constraints for PPLS parameter optimization.
    Main constraints:
    1. W'W = I_r (orthonormal columns of W)
    2. C'C = I_r (orthonormal columns of C)
    3. b_i > 0 for all i
    4. θ²_ti * b_i strictly decreasing
    """
    
    @staticmethod
    def orthonormality_constraint(theta: np.ndarray, p: int, q: int, r: int) -> np.ndarray:
        """
        Compute the orthonormality constraint violations.
        
        Returns:
        --------
        violations : np.ndarray
            Array of constraint violations (should be zero when satisfied)
        """
        # Extract W and C from theta
        W = theta[:p*r].reshape(p, r)
        C = theta[p*r:(p+q)*r].reshape(q, r)
        
        # Compute W'W - I and C'C - I
        W_violation = W.T @ W - np.eye(r)
        C_violation = C.T @ C - np.eye(r)
        
        # Flatten violations
        violations = np.concatenate([
            W_violation.flatten(),
            C_violation.flatten()
        ])
        
        return violations
    
    @staticmethod
    def get_bounds(p: int, q: int, r: int) -> list:
        """
        Get parameter bounds for optimization.
        
        Returns:
        --------
        bounds : list of tuples
            Lower and upper bounds for each parameter
        """
        bounds = []
        
        # W and C: no explicit bounds
        bounds.extend([(None, None)] * (p * r + q * r))
        
        # theta_t2: positive values
        bounds.extend([(1e-6, None)] * r)
        
        # b: positive values
        bounds.extend([(1e-6, None)] * r)
        
        # sigma_h2: positive value
        bounds.append((1e-6, None))
        
        return bounds
    
    @staticmethod
    def inequality_constraints(p: int, q: int, r: int, slack: float = 1e-2) -> Dict:
        """
        Convert equality constraints to inequality constraints for optimization.
        According to Equations (15) and (16) in the paper.
        
        Returns:
        --------
        constraints : dict
            Constraint dictionary for scipy.optimize
        """
        def constraint_fun(theta):
            """Combined constraint function for W'W = I and C'C = I."""
            W = theta[:p*r].reshape(p, r)
            C = theta[p*r:(p+q)*r].reshape(q, r)
            
            # Compute constraint violations
            W_gram = W.T @ W
            C_gram = C.T @ C
            
            # Sum of squared deviations from identity
            W_violation = np.sum((W_gram - np.eye(r))**2) + np.sum(W_gram**2) - r
            C_violation = np.sum((C_gram - np.eye(r))**2) + np.sum(C_gram**2) - r
            
            # Should be <= slack
            return slack - W_violation - C_violation
        
        constraints = {
            'type': 'ineq',
            'fun': constraint_fun
        }
        
        return constraints


class NoiseEstimator:
    """
    Pre-estimation of noise variances σ²_e and σ²_f using closed-form solutions.
    Based on Equations (19) and (20) in the paper.
    """
    
    @staticmethod
    def estimate_noise_variances(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        Estimate noise variances using maximum likelihood estimation.
        
        σ²_e = (1/N) Σᵢ ||xᵢ - x̄||²
        σ²_f = (1/N) Σᵢ ||yᵢ - ȳ||²
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, p)
            Input data matrix
        Y : np.ndarray of shape (N, q)
            Output data matrix
            
        Returns:
        --------
        sigma_e2 : float
            Estimated noise variance for X
        sigma_f2 : float
            Estimated noise variance for Y
        """
        # Center the data
        X_centered = NoiseEstimator._center_data(X)
        Y_centered = NoiseEstimator._center_data(Y)
        
        # Compute variances
        N = X.shape[0]
        sigma_e2 = np.sum(X_centered**2) / N
        sigma_f2 = np.sum(Y_centered**2) / N
        
        # Ensure positive values
        sigma_e2 = max(sigma_e2, 1e-6)
        sigma_f2 = max(sigma_f2, 1e-6)
        
        return sigma_e2, sigma_f2
    
    @staticmethod
    def _center_data(data: np.ndarray) -> np.ndarray:
        """
        Center data by subtracting the mean.
        
        Parameters:
        -----------
        data : np.ndarray of shape (N, d)
            Data matrix
            
        Returns:
        --------
        centered_data : np.ndarray
            Centered data matrix
        """
        mean = np.mean(data, axis=0, keepdims=True)
        return data - mean