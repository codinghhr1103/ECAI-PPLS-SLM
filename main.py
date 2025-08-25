"""
Main Execution Script for PPLS Parameter Estimation Comparison
=============================================================

This script coordinates the entire experiment pipeline from configuration to final results,
implementing a simplified three-stage approach: data generation, parameter estimation, and
visualization. Each stage can be skipped if outputs already exist.

Architecture Overview:
---------------------
The main script orchestrates:
1. Configuration loading and validation
2. Experiment setup and execution
3. Results analysis and visualization
4. Report generation

Function List:
--------------
main(): Main execution function
load_configuration(config_path): Load experiment settings from JSON
setup_logging(base_dir): Configure logging for tracking
run_data_generation_stage(config, base_dir): Execute data generation if needed
run_parameter_estimation_stage(config, base_dir): Execute parameter estimation if needed
run_visualization_stage(config, base_dir): Generate figures if needed
check_directory_status(directory): Check if directory exists and is non-empty
save_results(results, output_dir): Save results and generate reports
validate_configuration(config): Check configuration validity
print_experiment_summary(results, stages_run): Display summary to console

Call Relationships:
------------------
main() → load_configuration()
main() → check_directory_status()
main() → run_data_generation_stage()
main() → run_parameter_estimation_stage()
main() → run_visualization_stage()
run_data_generation_stage() → SineDataGenerator.__init__()
run_parameter_estimation_stage() → PPLSExperiment.__init__()
run_visualization_stage() → PPLSVisualizer.__init__()
"""

import os
import sys
import logging
from datetime import datetime
import json
import numpy as np
from typing import Dict, Optional, Tuple

from data_generator import SineDataGenerator
from experiment import PPLSExperiment
from visualization import PPLSVisualizer


def main():
    """
    Main execution function for PPLS parameter estimation comparison.
    """
    # Load configuration
    print("Loading configuration...")
    config = load_configuration('config.json')
        
    # Validate configuration
    if not validate_configuration(config):
        print("Configuration validation failed. Exiting.")
        return 1
    
    # Set base directory from config (defaults to current directory if not specified)
    base_dir = config.get('output', {}).get('base_dir', os.getcwd())
    base_dir = os.path.abspath(base_dir)
    
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(base_dir)
    
    print("\n" + "="*60)
    print("PPLS Parameter Estimation Comparison Experiment")
    print("="*60)
    print(f"Model dimensions: p={config['model']['p']}, q={config['model']['q']}, r={config['model']['r']}")
    print(f"Number of trials: {config['experiment']['n_trials']}")
    print(f"Base directory: {base_dir}")
    
    # Validate base directory is writable
    if not os.access(base_dir, os.W_OK):
        print(f"Error: Base directory is not writable: {base_dir}")
        return 1
    print("="*60 + "\n")
    
    stages_run = []
    
    try:
        # Stage 1: Data Generation
        data_generated = run_data_generation_stage(config, base_dir)
        if data_generated:
            stages_run.append("Data Generation")
            
        # Stage 2: Parameter Estimation
        estimation_run = run_parameter_estimation_stage(config, base_dir)
        if estimation_run:
            stages_run.append("Parameter Estimation")
            
        # Stage 3: Visualization
        figures_generated = run_visualization_stage(config, base_dir)
        if figures_generated:
            stages_run.append("Visualization")
            
        # Print summary
        print_experiment_summary(config, stages_run, base_dir)
        
        print(f"\nExperiment completed successfully!")
        print(f"Stages executed: {', '.join(stages_run) if stages_run else 'None (all outputs existed)'}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}", exc_info=True)
        print(f"\nExperiment failed with error: {str(e)}")
        return 1


def load_configuration(config_path: str) -> Dict:
    """
    Load experiment configuration from JSON file.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        # Try to find config in common locations
        search_paths = [
            config_path,
            os.path.join(os.path.dirname(__file__), config_path),
            os.path.join(os.path.dirname(__file__), '..', config_path),
            'config.json',
            os.path.join(os.path.dirname(__file__), 'config.json')
        ]
        
        config_found = False
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                config_found = True
                break
                
        if not config_found:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Set default values if not specified
    defaults = {
        'model': {
            'p': 20,
            'q': 20,
            'r': 3,
            'n_samples': 500
        },
        'data_generation': {
            'noise_levels': {
                'low': {'sigma_e2': 0.1, 'sigma_f2': 0.1, 'sigma_h2': 0.05},
                'high': {'sigma_e2': 0.5, 'sigma_f2': 0.5, 'sigma_h2': 0.25}
            },
            'sine_parameters': {
                'frequency': 0.7,
                'magnitude': 0.7
            }
        },
        'algorithms': {
            'common': {
                'n_starts': 32,
                'random_seed': 42
            },
            'slm': {
                'optimizer': 'trust-constr',
                'max_iter': 100,
                'use_noise_preestimation': True
            },
            'em': {
                'max_iter': 1000,
                'tolerance': 1e-6
            }
        },
        'experiment': {
            'n_trials': 100,
            'random_seed': 42
        },
        'output': {
            'save_intermediate': True,
            'base_dir': None,
            'figure_format': 'pdf',
            'force_data_generation': False,
            'force_parameter_estimation': False,
            'force_visualization': False
        }
    }
    
    # Merge defaults with loaded config
    def merge_dicts(default, config):
        for key, value in default.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                merge_dicts(value, config[key])
                
    merge_dicts(defaults, config)
    
    return config


def setup_logging(base_dir: str):
    """
    Configure logging for experiment tracking.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for saving logs
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, 'experiment.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log experiment start
    logging.info("="*60)
    logging.info("PPLS Parameter Estimation Experiment Started")
    logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*60)


def run_data_generation_stage(config: Dict, base_dir: str) -> bool:
    """
    Execute data generation stage if needed.
    
    Parameters:
    -----------
    config : dict
        Experiment configuration
    base_dir : str
        Base directory
        
    Returns:
    --------
    generated : bool
        True if data was generated, False if skipped
    """
    data_dir = os.path.join(base_dir, 'data')
    force = config.get('output', {}).get('force_data_generation', False)
    
    # Check if data already exists
    if not force and check_directory_status(data_dir):
        print("Data directory exists and is non-empty. Skipping data generation.")
        logging.info("Skipping data generation - data already exists")
        return False
        
    print("Stage 1: Generating experimental data...")
    logging.info("Starting data generation stage")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize data generator
    generator = SineDataGenerator(
        p=config['model']['p'],
        q=config['model']['q'],
        r=config['model']['r'],
        n_samples=config['model']['n_samples'],
        random_seed=config['experiment']['random_seed'],
        output_dir=data_dir
    )
    
    # Get noise levels from config
    noise_config = config.get('data_generation', {}).get('noise_levels', {}).get('low', {})
    sigma_e2 = noise_config.get('sigma_e2', 0.1)
    sigma_f2 = noise_config.get('sigma_f2', 0.1)
    sigma_h2 = noise_config.get('sigma_h2', 0.05)
    
    # Generate ground truth parameters
    true_params = generator.generate_true_parameters(
        sigma_e2=sigma_e2,
        sigma_f2=sigma_f2,
        sigma_h2=sigma_h2
    )
    
    # Generate data for all trials
    all_X = []
    all_Y = []
    
    for trial_id in range(config['experiment']['n_trials']):
        # Use different random seed for each trial's data generation
        data_seed = config['experiment']['random_seed'] + 1000 + trial_id
        np.random.seed(data_seed)
        
        # Generate data with fixed parameters
        from ppls_model import PPLSModel
        model = PPLSModel(config['model']['p'], config['model']['q'], config['model']['r'])
        
        X, Y = model.sample(
            n_samples=config['model']['n_samples'],
            W=true_params['W'],
            C=true_params['C'],
            B=true_params['B'],
            Sigma_t=true_params['Sigma_t'],
            sigma_e2=true_params['sigma_e2'],
            sigma_f2=true_params['sigma_f2'],
            sigma_h2=true_params['sigma_h2']
        )
        
        all_X.append(X)
        all_Y.append(Y)
    
    # Save data arrays (核心数据，程序必需)
    np.save(os.path.join(data_dir, 'X_trials.npy'), np.array(all_X))
    np.save(os.path.join(data_dir, 'Y_trials.npy'), np.array(all_Y))
    
    # Save ground truth parameters (核心参数，程序必需)
    import pickle
    with open(os.path.join(data_dir, 'ground_truth.pkl'), 'wb') as f:
        pickle.dump(true_params, f)
        
    # Save human-readable parameter summary (人类可读)
    param_summary = {
        'model_info': {
            'dimensions': {'p': config['model']['p'], 'q': config['model']['q'], 'r': config['model']['r']},
            'n_trials': config['experiment']['n_trials'],
            'n_samples': config['model']['n_samples']
        },
        'noise_parameters': {
            'sigma_e2': float(sigma_e2),
            'sigma_f2': float(sigma_f2), 
            'sigma_h2': float(sigma_h2)
        },
        'loading_matrices': {
            'W_shape': list(true_params['W'].shape),
            'C_shape': list(true_params['C'].shape),
            'W_norm_by_component': [float(np.linalg.norm(true_params['W'][:, i])) for i in range(config['model']['r'])],
            'C_norm_by_component': [float(np.linalg.norm(true_params['C'][:, i])) for i in range(config['model']['r'])]
        },
        'diagonal_parameters': {
            'B_diagonal': [float(x) for x in np.diag(true_params['B'])],
            'Sigma_t_diagonal': [float(x) for x in np.diag(true_params['Sigma_t'])],
            'identifiability_products': [float(x) for x in (np.diag(true_params['Sigma_t']) * np.diag(true_params['B']))]
        },
        'generation_info': {
            'random_seed': config['experiment']['random_seed'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    with open(os.path.join(data_dir, 'data_summary.json'), 'w') as f:
        json.dump(param_summary, f, indent=2)
    
    print(f"✓ Data generated for {config['experiment']['n_trials']} trials")
    print(f"✓ Data saved to: {data_dir}")
    logging.info(f"Data generation completed - {config['experiment']['n_trials']} trials")
    
    return True


def run_parameter_estimation_stage(config: Dict, base_dir: str) -> bool:
    """
    Execute parameter estimation stage if needed.
    
    Parameters:
    -----------
    config : dict
        Experiment configuration
    base_dir : str
        Base directory
        
    Returns:
    --------
    estimated : bool
        True if estimation was run, False if skipped
    """
    results_dir = os.path.join(base_dir, 'results')
    data_dir = os.path.join(base_dir, 'data')
    force = config.get('output', {}).get('force_parameter_estimation', False)
    
    # Check if results already exist
    if not force and check_directory_status(results_dir):
        print("Results directory exists and is non-empty. Skipping parameter estimation.")
        logging.info("Skipping parameter estimation - results already exist")
        return False
    
    # Check if data exists
    if not check_directory_status(data_dir):
        raise FileNotFoundError("Data directory is empty or doesn't exist. Run data generation first.")
    
    print("Stage 2: Running parameter estimation...")
    logging.info("Starting parameter estimation stage")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize experiment with simplified directory structure
    experiment = PPLSExperiment(config, base_dir, results_dir)
    
    # Run Monte Carlo experiment
    results = experiment.run_monte_carlo()
    
    # Save core results (程序必需)
    import pickle
    with open(os.path.join(results_dir, 'experiment_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save human-readable summary (人类可读)
    readable_summary = {
        'experiment_overview': {
            'n_trials_completed': results.get('n_trials_completed', 0),
            'success_rate_percent': round(results.get('n_trials_completed', 0) / config['experiment']['n_trials'] * 100, 1),
            'total_runtime_minutes': round(results.get('timing', {}).get('total_time', 0) / 60, 2),
            'avg_time_per_trial_seconds': round(results.get('timing', {}).get('avg_time_per_trial', 0), 2)
        },
        'algorithm_performance': {},
        'parameter_estimation_quality': {},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add algorithm performance comparison
    if 'analysis' in results and 'runtime_statistics' in results['analysis']:
        runtime_stats = results['analysis']['runtime_statistics']
        readable_summary['algorithm_performance'] = {
            'slm': {
                'avg_runtime_seconds': round(runtime_stats.get('slm', {}).get('avg_runtime', 0), 2),
                'avg_convergence_rate_percent': round(runtime_stats.get('slm', {}).get('avg_convergence_rate', 0) * 100, 1)
            },
            'em': {
                'avg_runtime_seconds': round(runtime_stats.get('em', {}).get('avg_runtime', 0), 2),
                'avg_convergence_rate_percent': round(runtime_stats.get('em', {}).get('avg_convergence_rate', 0) * 100, 1)
            },
            'em_vs_slm_speed_ratio': round(runtime_stats.get('overall', {}).get('avg_em_vs_slm_ratio', 1), 2)
        }
    
    # Add parameter estimation quality
    if 'analysis' in results:
        for method in ['slm', 'em']:
            if method in results['analysis']:
                method_quality = {}
                for param in ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']:
                    key = f'mse_{param}'
                    if key in results['analysis'][method]:
                        method_quality[f'{param}_mse_mean'] = round(results['analysis'][method][key]['mean'], 6)
                        method_quality[f'{param}_mse_std'] = round(results['analysis'][method][key]['std'], 6)
                readable_summary['parameter_estimation_quality'][method] = method_quality
    
    with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
        json.dump(readable_summary, f, indent=2)
    
    print(f"✓ Parameter estimation completed for {results.get('n_trials_completed', 0)} trials")
    print(f"✓ Results saved to: {results_dir}")
    logging.info(f"Parameter estimation completed - {results.get('n_trials_completed', 0)} trials")
    
    return True


def run_visualization_stage(config: Dict, base_dir: str) -> bool:
    """
    Execute visualization stage if needed.
    
    Parameters:
    -----------
    config : dict
        Experiment configuration
    base_dir : str
        Base directory
        
    Returns:
    --------
    generated : bool
        True if figures were generated, False if skipped
    """
    figures_dir = os.path.join(base_dir, 'figures')
    results_dir = os.path.join(base_dir, 'results')
    data_dir = os.path.join(base_dir, 'data')
    force = config.get('output', {}).get('force_visualization', False)
    figure_format = config.get('output', {}).get('figure_format', 'pdf')
    
    # Check if figures already exist
    if not force and check_directory_status(figures_dir):
        print("Figures directory exists and is non-empty. Skipping visualization.")
        logging.info("Skipping visualization - figures already exist")
        return False
    
    # Check if prerequisites exist
    if not check_directory_status(results_dir):
        raise FileNotFoundError("Results directory is empty or doesn't exist. Run parameter estimation first.")
    if not check_directory_status(data_dir):
        raise FileNotFoundError("Data directory is empty or doesn't exist. Run data generation first.")
    
    print("Stage 3: Generating visualizations...")
    logging.info("Starting visualization stage")
    
    # Create figures directory
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = PPLSVisualizer(base_dir, figure_format=figure_format)
    
    # Load results
    import pickle
    results_path = os.path.join(results_dir, 'experiment_results.pkl')
    with open(results_path, 'rb') as f:
        experiment_results = pickle.load(f)
    
    # Generate comprehensive summary
    visualizer.create_results_summary(experiment_results)
    
    # Generate loading comparison for first trial if available
    if experiment_results.get('trial_results'):
        first_trial = experiment_results['trial_results'][0]
        
        # Plot all components
        for component_idx in range(config['model']['r']):
            visualizer.plot_loading_comparison(first_trial, component_idx)
    
    # Generate parameter recovery plots
    if 'analysis' in experiment_results:
        visualizer.plot_parameter_recovery(experiment_results['analysis'])
    
    # Generate convergence history
    if 'trial_results' in experiment_results:
        visualizer.plot_convergence_history(experiment_results['trial_results'])
    
    # Save all figures
    visualizer.save_all_figures()
    
    print(f"✓ Visualizations generated")
    print(f"✓ Figures saved to: {figures_dir}")
    logging.info("Visualization completed")
    
    return True


def check_directory_status(directory: str) -> bool:
    """
    Check if directory exists and is non-empty.
    
    Parameters:
    -----------
    directory : str
        Directory path to check
        
    Returns:
    --------
    exists_and_nonempty : bool
        True if directory exists and contains files
    """
    if not os.path.exists(directory):
        return False
    
    try:
        return len(os.listdir(directory)) > 0
    except OSError:
        return False


def validate_configuration(config: Dict) -> bool:
    """
    Check configuration validity.
    
    Parameters:
    -----------
    config : dict
        Configuration to validate
        
    Returns:
    --------
    valid : bool
        True if configuration is valid
    """
    try:
        # Check required sections
        required_sections = ['model', 'algorithms', 'experiment']
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
                
        # Check model parameters
        p = config['model']['p']
        q = config['model']['q']
        r = config['model']['r']
        
        if r > min(p, q):
            print(f"Invalid: r ({r}) must be less than min(p, q) = {min(p, q)}")
            return False
            
        if config['model']['n_samples'] < 10:
            print("Invalid: n_samples must be at least 10")
            return False
            
        # Check algorithm parameters
        if config['algorithms']['common']['n_starts'] < 1:
            print("Invalid: n_starts must be at least 1")
            return False
            
        # Check experiment parameters
        if config['experiment']['n_trials'] < 1:
            print("Invalid: n_trials must be at least 1")
            return False
            
        return True
        
    except Exception as e:
        print(f"Configuration validation error: {str(e)}")
        return False


def print_experiment_summary(config: Dict, stages_run: list, base_dir: str):
    """
    Display experiment summary to console.
    
    Parameters:
    -----------
    config : dict
        Experiment configuration
    stages_run : list
        List of stages that were executed
    base_dir : str
        Base directory path
    """
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Model dimensions: p={config['model']['p']}, q={config['model']['q']}, r={config['model']['r']}")
    print(f"  Sample size: {config['model']['n_samples']}")
    print(f"  Number of trials: {config['experiment']['n_trials']}")
    print(f"  Starting points: {config['algorithms']['common']['n_starts']}")
    
    print(f"\nStages executed: {', '.join(stages_run) if stages_run else 'None (all outputs existed)'}")
    
    print("\nOutput directories:")
    print(f"  Data: {os.path.join(base_dir, 'data')}")
    print(f"  Results: {os.path.join(base_dir, 'results')}")
    print(f"  Figures: {os.path.join(base_dir, 'figures')}")
    print(f"  Logs: {os.path.join(base_dir, 'logs')}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    sys.exit(main())