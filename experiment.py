"""
Experiment Runner and Analysis for PPLS Parameter Estimation
===========================================================

This module orchestrates Monte Carlo experiments ensuring fair comparison between SLM, EM, 
and ECM algorithms with identical starting points and fixed ground truth parameters. Uses a 
simplified directory structure with all data loaded from the data directory and results saved 
to the results directory.

Architecture Overview:
---------------------
The module provides:
1. PPLSExperiment: Main experiment coordinator for running comparative studies
2. PerformanceMetrics: Calculate estimation quality metrics (MSE, bias, variance)
3. ParameterRecovery: Assess parameter recovery quality and identifiability

Function List:
--------------
PPLSExperiment:
    - __init__(config, base_dir, results_dir): Initialize experiment with simplified structure
    - run_monte_carlo(): Execute multiple trials with parallel processing
    - _run_single_trial(trial_id, X, Y, true_params): Execute one complete trial with all algorithms
    - _compare_algorithms(X, Y, starting_points, true_params): Compare SLM, EM, and ECM
    - _run_algorithm_with_stats(algorithm, X, Y, starting_points, algorithm_name): Run algorithm and collect stats
    - analyze_results(trial_results): Compute summary statistics across trials
    - save_experiment_summary(results): Save overall experiment results

PerformanceMetrics:
    - __init__(true_params): Initialize with ground truth
    - compute_mse(estimated_params): Mean squared error calculation
    - compute_bias(estimated_params_list): Bias of parameter estimates
    - compute_variance(estimated_params_list): Variance of estimates
    - compute_correlation(W_est, C_est, W_true, C_true): Loading correlations
    - generate_summary_table(slm_metrics_list, em_metrics_list, ecm_metrics_list): Create results table
    - save_metrics(metrics, filename): Save computed metrics

ParameterRecovery:
    - align_estimated_params(params_est, params_true): Handle sign indeterminacy
    - compute_loading_errors(W_est, C_est, W_true, C_true): W and C errors
    - compute_diagonal_errors(B_est, Sigma_t_est, B_true, Sigma_t_true): B and Σt errors
    - assess_identifiability(params): Check identifiability constraints
    - compute_recovery_rate(estimated_list, true_params, threshold): Recovery success rate

Call Relationships:
------------------
PPLSExperiment.run_monte_carlo() → PPLSExperiment._run_single_trial()
PPLSExperiment._run_single_trial() → PPLSExperiment._compare_algorithms()
PPLSExperiment._compare_algorithms() → ScalarLikelihoodMethod.fit()
PPLSExperiment._compare_algorithms() → EMAlgorithm.fit()
PPLSExperiment._compare_algorithms() → ECMAlgorithm.fit()
PPLSExperiment.analyze_results() → PerformanceMetrics.compute_mse()
PPLSExperiment.analyze_results() → PerformanceMetrics.generate_summary_table()
PerformanceMetrics.compute_mse() → ParameterRecovery.align_estimated_params()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import pickle
import os
import time
from datetime import datetime
import warnings

from algorithms import InitialPointGenerator, ScalarLikelihoodMethod, EMAlgorithm, ECMAlgorithm, NoiseEstimator
from ppls_model import PPLSModel


class PPLSExperiment:
    """
    Main experiment coordinator for comparing SLM, EM, and ECM algorithms.
    Ensures fair comparison by using identical datasets and starting points.
    """
    
    def __init__(self, config: Dict, base_dir: str, results_dir: str):
        """
        Initialize experiment with configuration and simplified directory structure.
        
        Parameters:
        -----------
        config : dict
            Experiment configuration
        base_dir : str
            Base directory containing data
        results_dir : str
            Directory to save results
        """
        self.config = config
        self.base_dir = base_dir
        self.results_dir = results_dir
        
        # Model parameters
        self.p = config['model']['p']
        self.q = config['model']['q']
        self.r = config['model']['r']
        self.n_samples = config['model']['n_samples']
        
        # Algorithm settings
        self.n_starts = config['algorithms']['common']['n_starts']
        self.algorithm_seed = config['algorithms']['common']['random_seed']
        
        # Experiment settings
        self.n_trials = config['experiment']['n_trials']
        self.experiment_seed = config['experiment']['random_seed']
        
        # Load data and ground truth
        self._load_data_and_ground_truth()
        
    def _load_data_and_ground_truth(self):
        """Load experimental data and ground truth parameters from data directory."""
        data_dir = os.path.join(self.base_dir, 'data')
        
        # Load data arrays
        self.X_trials = np.load(os.path.join(data_dir, 'X_trials.npy'))
        self.Y_trials = np.load(os.path.join(data_dir, 'Y_trials.npy'))
        
        # Load ground truth parameters
        with open(os.path.join(data_dir, 'ground_truth.pkl'), 'rb') as f:
            self.true_params = pickle.load(f)
            
        # Validate data dimensions
        expected_shape = (self.n_trials, self.n_samples)
        if self.X_trials.shape[:2] != expected_shape:
            raise ValueError(f"X data shape mismatch: expected {expected_shape}, got {self.X_trials.shape[:2]}")
        if self.Y_trials.shape[:2] != expected_shape:
            raise ValueError(f"Y data shape mismatch: expected {expected_shape}, got {self.Y_trials.shape[:2]}")
            
        print(f"Loaded data for {self.n_trials} trials")
        print(f"Data dimensions: X{self.X_trials.shape}, Y{self.Y_trials.shape}")
        
    def run_monte_carlo(self) -> Dict:
        """
        Execute Monte Carlo experiment with all three algorithms.
        
        Returns:
        --------
        results : dict
            Dictionary containing all trial results and analysis
        """
        print(f"\n{'='*60}")
        print(f"STARTING MONTE CARLO EXPERIMENT")
        print(f"{'='*60}")
        print(f"Total trials: {self.n_trials}")
        print(f"Model dimensions: p={self.p}, q={self.q}, r={self.r}")
        print(f"Sample size: {self.n_samples}")
        print(f"Starting points per algorithm: {self.n_starts}")
        print(f"Algorithms: SLM, EM, ECM")
        print(f"Ground truth: Fixed for all trials")
        print(f"{'='*60}\n")
        
        # Track timing
        start_time = time.time()
        
        # Generate starting points once for all trials
        init_generator = InitialPointGenerator(
            p=self.p, q=self.q, r=self.r,
            n_starts=self.n_starts,
            random_seed=self.algorithm_seed
        )
        starting_points = init_generator.generate_starting_points()
        print(f"Generated {len(starting_points)} identical starting points for all trials")
        
        # Run trials sequentially
        print("Running trials sequentially...")
        trial_results = []
        
        for trial_id in range(self.n_trials):
            print(f"\nRunning Trial {trial_id+1}/{self.n_trials}...")
            
            # Get data for this trial
            X = self.X_trials[trial_id]
            Y = self.Y_trials[trial_id]
            
            result = self._run_single_trial(trial_id, X, Y, self.true_params, starting_points)
            
            if result is not None:
                trial_results.append(result)
                print(f"✓ Trial {trial_id+1} completed successfully")
            else:
                print(f"✗ Trial {trial_id+1} failed")
        
        # Filter out failed trials
        valid_results = [r for r in trial_results if r is not None]
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETED")
        print(f"{'='*60}")
        print(f"Successful trials: {len(valid_results)}/{self.n_trials}")
        print(f"Success rate: {len(valid_results)/self.n_trials*100:.1f}%")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Average time per trial: {elapsed_time/self.n_trials:.1f} seconds")
        
        # Add runtime statistics summary if available
        if valid_results and 'statistics' in valid_results[0]:
            total_slm_runtime = sum(r['statistics']['slm']['runtime'] for r in valid_results)
            total_em_runtime = sum(r['statistics']['em']['runtime'] for r in valid_results)
            total_ecm_runtime = sum(r['statistics']['ecm']['runtime'] for r in valid_results)
            avg_slm_convergence = np.mean([r['statistics']['slm']['convergence_rate'] for r in valid_results])
            avg_em_convergence = np.mean([r['statistics']['em']['convergence_rate'] for r in valid_results])
            avg_ecm_convergence = np.mean([r['statistics']['ecm']['convergence_rate'] for r in valid_results])
            
            print(f"\nAlgorithm Performance Summary:")
            print(f"  SLM: {total_slm_runtime:.1f}s total, {avg_slm_convergence*100:.1f}% convergence")
            print(f"  EM:  {total_em_runtime:.1f}s total, {avg_em_convergence*100:.1f}% convergence")
            print(f"  ECM: {total_ecm_runtime:.1f}s total, {avg_ecm_convergence*100:.1f}% convergence")
        
        print(f"{'='*60}\n")
        
        if len(valid_results) == 0:
            raise RuntimeError("All trials failed. Please check your configuration and data.")
        
        # Analyze results
        print("Analyzing results...")
        analysis = self.analyze_results(valid_results)
        
        # Save experiment summary
        experiment_results = {
            'config': self.config,
            'n_trials_completed': len(valid_results),
            'trial_results': valid_results,
            'analysis': analysis,
            'timing': {
                'total_time': elapsed_time,
                'avg_time_per_trial': elapsed_time / self.n_trials
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ground_truth_params': self.true_params
        }
        
        self.save_experiment_summary(experiment_results)
        print("✓ Experiment summary saved\n")
        
        return experiment_results
        
    def _run_single_trial(self, trial_id: int, X: np.ndarray, Y: np.ndarray,
                         true_params: Dict, starting_points: List[np.ndarray]) -> Optional[Dict]:
        """
        Execute one complete trial with all three algorithms.
        
        Parameters:
        -----------
        trial_id : int
            Trial identifier
        X, Y : np.ndarray
            Data matrices for this trial
        true_params : dict
            Ground truth parameters
        starting_points : List[np.ndarray]
            Starting points for multi-start optimization
            
        Returns:
        --------
        trial_result : dict or None
            Results from this trial, or None if failed
        """
        try:
            print(f"  Data shape: X{X.shape}, Y{Y.shape}")
            
            # Run all three algorithms
            comparison_results = self._compare_algorithms(
                X, Y, starting_points, true_params, trial_id
            )
            
            # Combine results
            trial_result = {
                'trial_id': trial_id,
                'true_params': true_params,
                'slm_results': comparison_results['slm'],
                'em_results': comparison_results['em'],
                'ecm_results': comparison_results['ecm'],
                'data_shape': {'X': X.shape, 'Y': Y.shape},
                'statistics': comparison_results.get('statistics', {})
            }
            
            return trial_result
            
        except Exception as e:
            print(f"  ERROR in Trial {trial_id+1}: {str(e)}")
            warnings.warn(f"Trial {trial_id} failed: {str(e)}")
            
            # Save error information
            error_info = {
                'trial_id': trial_id,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            error_file = os.path.join(self.results_dir, f"trial_{trial_id:03d}_error.json")
            try:
                with open(error_file, 'w') as f:
                    json.dump(error_info, f, indent=4)
            except:
                pass
                
            return None
            
    def _compare_algorithms(self, X: np.ndarray, Y: np.ndarray,
                          starting_points: List[np.ndarray],
                          true_params: Dict, trial_id: int) -> Dict:
        """
        Compare SLM, EM, and ECM algorithms.
        
        Parameters:
        -----------
        X, Y : np.ndarray
            Data matrices
        starting_points : List[np.ndarray]
            Identical starting points for all algorithms
        true_params : dict
            Ground truth parameters
        trial_id : int
            Trial identifier
            
        Returns:
        --------
        results : dict
            Results from all three algorithms
        """
        # Initialize all three algorithms
        slm = ScalarLikelihoodMethod(
            p=self.p, q=self.q, r=self.r,
            optimizer=self.config['algorithms']['slm']['optimizer'],
            max_iter=self.config['algorithms']['slm']['max_iter'],
            use_noise_preestimation=self.config['algorithms']['slm']['use_noise_preestimation']
        )
        
        em = EMAlgorithm(
            p=self.p, q=self.q, r=self.r,
            max_iter=self.config['algorithms']['em']['max_iter'],
            tolerance=self.config['algorithms']['em']['tolerance']
        )
        
        ecm = ECMAlgorithm(
            p=self.p, q=self.q, r=self.r,
            max_iter=self.config['algorithms']['ecm']['max_iter'],
            tolerance=self.config['algorithms']['ecm']['tolerance']
        )
        
        print(f"\n{'='*60}")
        print(f"Trial {trial_id+1}: Algorithm Comparison")
        print(f"{'='*60}")
        
        # Run SLM first
        print(f"Running SLM with {len(starting_points)} starting points...")
        slm_start_time = time.time()
        slm_results, slm_stats = self._run_algorithm_with_stats(
            slm, X, Y, starting_points, trial_id, "SLM"
        )
        slm_time = time.time() - slm_start_time
        
        print(f"SLM completed: {slm_time:.2f}s, convergence: {slm_stats['convergence_rate']*100:.1f}%")
        
        # Run EM second
        print(f"Running EM with {len(starting_points)} starting points...")
        em_start_time = time.time()
        em_results, em_stats = self._run_algorithm_with_stats(
            em, X, Y, starting_points, trial_id, "EM"
        )
        em_time = time.time() - em_start_time
        
        print(f"EM completed: {em_time:.2f}s, convergence: {em_stats['convergence_rate']*100:.1f}%")
        
        # Run ECM third
        print(f"Running ECM with {len(starting_points)} starting points...")
        ecm_start_time = time.time()
        ecm_results, ecm_stats = self._run_algorithm_with_stats(
            ecm, X, Y, starting_points, trial_id, "ECM"
        )
        ecm_time = time.time() - ecm_start_time
        
        print(f"ECM completed: {ecm_time:.2f}s, convergence: {ecm_stats['convergence_rate']*100:.1f}%")
        
        print(f"{'='*60}")
        
        # Compile statistics
        stats_summary = {
            'trial_id': trial_id,
            'slm': {
                'runtime': slm_time,
                'avg_time_per_start': slm_time/len(starting_points),
                'converged': slm_stats['converged'],
                'failed': slm_stats['failed'],
                'convergence_rate': slm_stats['converged']/len(starting_points),
                'best_objective': slm_stats['best_objective'],
                'avg_iterations': slm_stats['avg_iterations']
            },
            'em': {
                'runtime': em_time,
                'avg_time_per_start': em_time/len(starting_points),
                'converged': em_stats['converged'],
                'failed': em_stats['failed'],
                'convergence_rate': em_stats['converged']/len(starting_points),
                'best_likelihood': em_stats['best_likelihood'],
                'avg_iterations': em_stats['avg_iterations']
            },
            'ecm': {
                'runtime': ecm_time,
                'avg_time_per_start': ecm_time/len(starting_points),
                'converged': ecm_stats['converged'],
                'failed': ecm_stats['failed'],
                'convergence_rate': ecm_stats['converged']/len(starting_points),
                'best_likelihood': ecm_stats['best_likelihood'],
                'avg_iterations': ecm_stats['avg_iterations']
            }
        }
        
        # Save statistics
        stats_file = os.path.join(self.results_dir, f"trial_{trial_id:03d}_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats_summary, f, indent=4)
        
        return {
            'slm': slm_results,
            'em': em_results,
            'ecm': ecm_results,
            'statistics': stats_summary
        }
        
    def _run_algorithm_with_stats(self, algorithm, X: np.ndarray, Y: np.ndarray,
                                 starting_points: List[np.ndarray],
                                 trial_id: int, algorithm_name: str) -> Tuple[Dict, Dict]:
        """
        Run algorithm and collect detailed statistics.
        
        Parameters:
        -----------
        algorithm : PPLSAlgorithm
            Algorithm instance (SLM, EM, or ECM)
        X, Y : np.ndarray
            Data matrices
        starting_points : List[np.ndarray]
            Starting points for multi-start
        trial_id : int
            Trial identifier
        algorithm_name : str
            Name of algorithm ("SLM", "EM", or "ECM")
            
        Returns:
        --------
        results : dict
            Best estimated parameters
        stats : dict
            Running statistics
        """
        # Use the algorithm's fit method directly
        results = algorithm.fit(X, Y, starting_points)
        
        # Compile basic statistics
        if algorithm_name == "SLM":
            stats = {
                'converged': 1 if results.get('success', False) else 0,
                'failed': 0 if results.get('success', False) else 1,
                'convergence_rate': 1.0 if results.get('success', False) else 0.0,
                'avg_iterations': results.get('n_iterations', 0),
                'best_objective': results.get('objective_value', np.inf)
            }
        else:  # EM or ECM
            stats = {
                'converged': 1 if results.get('log_likelihood', -np.inf) > -np.inf else 0,
                'failed': 0 if results.get('log_likelihood', -np.inf) > -np.inf else 1,
                'convergence_rate': 1.0 if results.get('log_likelihood', -np.inf) > -np.inf else 0.0,
                'avg_iterations': results.get('n_iterations', 0),
                'best_likelihood': results.get('log_likelihood', -np.inf)
            }
        
        return results, stats
        
    def analyze_results(self, trial_results: List[Dict]) -> Dict:
        """
        Compute summary statistics and performance metrics across all trials.
        
        Parameters:
        -----------
        trial_results : List[Dict]
            Results from all successful trials
            
        Returns:
        --------
        analysis : dict
            Summary statistics and performance comparison
        """
        # Initialize metrics calculators
        slm_metrics_list = []
        em_metrics_list = []
        ecm_metrics_list = []
        
        # Collect runtime statistics
        slm_runtime_stats = []
        em_runtime_stats = []
        ecm_runtime_stats = []
        
        # Collect metrics for each trial
        for trial in trial_results:
            # Create performance metric calculator
            metrics_calc = PerformanceMetrics(trial['true_params'])
            
            # Compute metrics for SLM
            slm_metrics = metrics_calc.compute_mse(trial['slm_results'])
            slm_metrics_list.append(slm_metrics)
            
            # Compute metrics for EM
            em_metrics = metrics_calc.compute_mse(trial['em_results'])
            em_metrics_list.append(em_metrics)
            
            # Compute metrics for ECM
            ecm_metrics = metrics_calc.compute_mse(trial['ecm_results'])
            ecm_metrics_list.append(ecm_metrics)
            
            # Collect runtime statistics if available
            if 'statistics' in trial:
                slm_runtime_stats.append(trial['statistics']['slm'])
                em_runtime_stats.append(trial['statistics']['em'])
                ecm_runtime_stats.append(trial['statistics']['ecm'])
        
        # Aggregate metrics
        analysis = {
            'slm': self._aggregate_metrics(slm_metrics_list),
            'em': self._aggregate_metrics(em_metrics_list),
            'ecm': self._aggregate_metrics(ecm_metrics_list),
            'comparison': self._compare_methods(slm_metrics_list, em_metrics_list, ecm_metrics_list)
        }
        
        # Add runtime statistics
        if slm_runtime_stats and em_runtime_stats and ecm_runtime_stats:
            analysis['runtime_statistics'] = {
                'slm': self._aggregate_runtime_stats(slm_runtime_stats),
                'em': self._aggregate_runtime_stats(em_runtime_stats),
                'ecm': self._aggregate_runtime_stats(ecm_runtime_stats),
                'overall': {
                    'total_runtime_slm': sum(s['runtime'] for s in slm_runtime_stats),
                    'total_runtime_em': sum(s['runtime'] for s in em_runtime_stats),
                    'total_runtime_ecm': sum(s['runtime'] for s in ecm_runtime_stats)
                }
            }
        
        # Generate summary table
        if trial_results:
            metrics_calc = PerformanceMetrics(trial_results[0]['true_params'])
            summary_table = metrics_calc.generate_summary_table(
                slm_metrics_list, em_metrics_list, ecm_metrics_list
            )
            analysis['summary_table'] = summary_table
            
        return analysis
        
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across trials."""
        if not metrics_list:
            return {}
            
        # Extract metric names
        metric_names = list(metrics_list[0].keys())
        
        aggregated = {}
        for metric in metric_names:
            values = [m[metric] for m in metrics_list]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
        return aggregated
        
    def _aggregate_runtime_stats(self, stats_list: List[Dict]) -> Dict:
        """Aggregate runtime statistics across trials."""
        if not stats_list:
            return {}
            
        aggregated = {
            'avg_runtime': np.mean([s['runtime'] for s in stats_list]),
            'std_runtime': np.std([s['runtime'] for s in stats_list]),
            'avg_time_per_start': np.mean([s['avg_time_per_start'] for s in stats_list]),
            'avg_convergence_rate': np.mean([s['convergence_rate'] for s in stats_list]),
            'total_converged': sum(s['converged'] for s in stats_list),
            'total_failed': sum(s['failed'] for s in stats_list),
            'avg_iterations': np.mean([s['avg_iterations'] for s in stats_list])
        }
        
        return aggregated
        
    def _compare_methods(self, slm_metrics: List[Dict], em_metrics: List[Dict], 
                        ecm_metrics: List[Dict]) -> Dict:
        """Compare performance between SLM, EM, and ECM."""
        comparison = {}
        
        if not slm_metrics or not em_metrics or not ecm_metrics:
            return comparison
            
        # Compare each metric
        metric_names = list(slm_metrics[0].keys())
        
        for metric in metric_names:
            slm_values = np.array([m[metric] for m in slm_metrics])
            em_values = np.array([m[metric] for m in em_metrics])
            ecm_values = np.array([m[metric] for m in ecm_metrics])
            
            comparison[metric] = {
                'slm_best_count': np.sum((slm_values <= em_values) & (slm_values <= ecm_values)),
                'em_best_count': np.sum((em_values <= slm_values) & (em_values <= ecm_values)),
                'ecm_best_count': np.sum((ecm_values <= slm_values) & (ecm_values <= em_values)),
                'slm_mean': np.mean(slm_values),
                'em_mean': np.mean(em_values),
                'ecm_mean': np.mean(ecm_values)
            }
            
        return comparison
        
    def save_experiment_summary(self, results: Dict):
        """Save overall experiment results."""
        # Save comprehensive experiment results
        with open(os.path.join(self.results_dir, "experiment_results.pkl"), 'wb') as f:
            pickle.dump(results, f)
            
        # Save human-readable experiment summary
        readable_summary = {
            'experiment_info': {
                'n_trials_completed': results['n_trials_completed'],
                'success_rate_percent': round(results['n_trials_completed'] / self.n_trials * 100, 1),
                'total_runtime_minutes': round(results['timing']['total_time'] / 60, 2),
                'timestamp': results['timestamp']
            },
            'algorithm_performance': {},
            'parameter_estimation_quality': {}
        }
        
        # Add algorithm performance comparison
        if 'runtime_statistics' in results['analysis']:
            runtime = results['analysis']['runtime_statistics']
            readable_summary['algorithm_performance'] = {
                'slm': {
                    'avg_time_seconds': round(runtime.get('slm', {}).get('avg_runtime', 0), 2),
                    'avg_convergence_rate_percent': round(runtime.get('slm', {}).get('avg_convergence_rate', 0) * 100, 1)
                },
                'em': {
                    'avg_time_seconds': round(runtime.get('em', {}).get('avg_runtime', 0), 2),
                    'avg_convergence_rate_percent': round(runtime.get('em', {}).get('avg_convergence_rate', 0) * 100, 1)
                },
                'ecm': {
                    'avg_time_seconds': round(runtime.get('ecm', {}).get('avg_runtime', 0), 2),
                    'avg_convergence_rate_percent': round(runtime.get('ecm', {}).get('avg_convergence_rate', 0) * 100, 1)
                }
            }
        
        # Add MSE comparison summary
        for method in ['slm', 'em', 'ecm']:
            if method in results['analysis']:
                method_mse = {}
                for param in ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']:
                    key = f'mse_{param}'
                    if key in results['analysis'][method]:
                        method_mse[param] = {
                            'mean': round(results['analysis'][method][key]['mean'], 6),
                            'std': round(results['analysis'][method][key]['std'], 6)
                        }
                readable_summary['parameter_estimation_quality'][method] = method_mse
        
        with open(os.path.join(self.results_dir, "experiment_summary.json"), 'w') as f:
            json.dump(readable_summary, f, indent=2)
            

class PerformanceMetrics:
    """
    Calculate estimation quality metrics for PPLS parameter recovery.
    """
    
    def __init__(self, true_params: Dict):
        """
        Initialize with ground truth parameters.
        
        Parameters:
        -----------
        true_params : dict
            True parameter values
        """
        self.true_params = true_params
        self.recovery = ParameterRecovery()
        
    def compute_mse(self, estimated_params: Dict) -> Dict:
        """
        Compute mean squared error for all parameters with proper sign alignment.
        
        Parameters:
        -----------
        estimated_params : dict
            Estimated parameter values
            
        Returns:
        --------
        mse_dict : dict
            MSE for each parameter
        """
        # Align parameters to handle sign indeterminacy
        aligned_params = self.recovery.align_estimated_params(
            estimated_params, self.true_params
        )
        
        mse_dict = {}
        
        # MSE for loading matrices
        mse_dict['mse_W'] = np.mean((aligned_params['W'] - self.true_params['W'])**2)
        mse_dict['mse_C'] = np.mean((aligned_params['C'] - self.true_params['C'])**2)
        
        # MSE for diagonal matrices
        mse_dict['mse_B'] = np.mean(
            (np.diag(aligned_params['B']) - np.diag(self.true_params['B']))**2
        )
        mse_dict['mse_Sigma_t'] = np.mean(
            (np.diag(aligned_params['Sigma_t']) - np.diag(self.true_params['Sigma_t']))**2
        )
        
        # MSE for scalar parameters
        mse_dict['mse_sigma_h2'] = (
            aligned_params['sigma_h2'] - self.true_params['sigma_h2']
        )**2
        
        return mse_dict
        
    def generate_summary_table(self, slm_metrics_list: List[Dict],
                             em_metrics_list: List[Dict],
                             ecm_metrics_list: List[Dict]) -> pd.DataFrame:
        """
        Generate summary table for all three algorithms (Table 2 format).
        
        Parameters:
        -----------
        slm_metrics_list : List[Dict]
            SLM metrics from all trials
        em_metrics_list : List[Dict]
            EM metrics from all trials
        ecm_metrics_list : List[Dict]
            ECM metrics from all trials
            
        Returns:
        --------
        table : pd.DataFrame
            Summary statistics table
        """
        # Define parameters to include
        params = ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']
        
        # Initialize table data
        table_data = []
        
        for param in params:
            mse_key = f'mse_{param}'
            
            # SLM statistics
            slm_values = [m[mse_key] for m in slm_metrics_list if mse_key in m]
            slm_mean = np.mean(slm_values) if slm_values else np.nan
            slm_std = np.std(slm_values) if slm_values else np.nan
            
            # EM statistics
            em_values = [m[mse_key] for m in em_metrics_list if mse_key in m]
            em_mean = np.mean(em_values) if em_values else np.nan
            em_std = np.std(em_values) if em_values else np.nan
            
            # ECM statistics
            ecm_values = [m[mse_key] for m in ecm_metrics_list if mse_key in m]
            ecm_mean = np.mean(ecm_values) if ecm_values else np.nan
            ecm_std = np.std(ecm_values) if ecm_values else np.nan
            
            # Format as mean ± std (×10^2)
            slm_str = f"{slm_mean*100:.2f}±{slm_std*100:.2f}"
            em_str = f"{em_mean*100:.2f}±{em_std*100:.2f}"
            ecm_str = f"{ecm_mean*100:.2f}±{ecm_std*100:.2f}"
            
            table_data.append({
                'Parameter': param,
                'SLM': slm_str,
                'EM': em_str,
                'ECM': ecm_str
            })
            
        # Create DataFrame
        table = pd.DataFrame(table_data)
        
        return table


class ParameterRecovery:
    """
    Assess parameter recovery quality and handle identifiability issues.
    """
    
    def align_estimated_params(self, params_est: Dict, 
                              params_true: Dict) -> Dict:
        """
        Align estimated parameters with ground truth to handle sign indeterminacy.
        
        Parameters:
        -----------
        params_est : dict
            Estimated parameters
        params_true : dict
            True parameters
            
        Returns:
        --------
        aligned_params : dict
            Sign-aligned parameters
        """
        aligned_params = params_est.copy()
        
        W_est = params_est['W']
        C_est = params_est['C']
        B_est = params_est['B']
        
        W_true = params_true['W']
        C_true = params_true['C']
        
        r = W_true.shape[1]
        
        # Align each component
        for i in range(r):
            # Check correlation with true loadings
            w_corr = np.corrcoef(W_est[:, i], W_true[:, i])[0, 1]
            c_corr = np.corrcoef(C_est[:, i], C_true[:, i])[0, 1]
            
            # Flip signs if negative correlation
            if w_corr < 0:
                W_est[:, i] *= -1
            if c_corr < 0:
                C_est[:, i] *= -1
                
        # Ensure B diagonal elements are positive
        b_diag = np.diag(B_est)
        B_aligned = np.diag(np.abs(b_diag))
        
        aligned_params['W'] = W_est
        aligned_params['C'] = C_est
        aligned_params['B'] = B_aligned
        
        return aligned_params