# PPLS-SLM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code for reproducing experimental results from "Scalar Likelihood Method for Probabilistic Partial Least Squares Model with Rank n Update" (ECAI 2025).

## Installation

```bash
git clone https://github.com/yourusername/PPLS-SLM.git
cd PPLS-SLM
pip install -r requirements.txt
```

## Quick Start

```python
from algorithms import ScalarLikelihoodMethod, EMAlgorithm, ECMAlgorithm, InitialPointGenerator
from data_generator import SineDataGenerator

# Generate synthetic PPLS data
generator = SineDataGenerator(p=20, q=20, r=3, n_samples=500)
true_params = generator.generate_true_parameters()
X, Y = generator.generate_samples(true_params)

# Generate starting points
init_gen = InitialPointGenerator(p=20, q=20, r=3, n_starts=32)
starting_points = init_gen.generate_starting_points()

# Fit SLM model
slm = ScalarLikelihoodMethod(p=20, q=20, r=3)
slm_results = slm.fit(X, Y, starting_points)

# Compare with EM and ECM
em = EMAlgorithm(p=20, q=20, r=3)
em_results = em.fit(X, Y, starting_points)

ecm = ECMAlgorithm(p=20, q=20, r=3)
ecm_results = ecm.fit(X, Y, starting_points)
```

## Run Complete Experiment

### Full Monte Carlo Experiment
```bash
python main.py
```

This will execute all three stages:
1. **Data Generation**: Create synthetic PPLS datasets for all trials
2. **Parameter Estimation**: Run SLM, EM, and ECM algorithms 
3. **Visualization**: Generate figures and tables

### Configuration
Edit `config.json` to modify:
- Model dimensions (p, q, r)
- Number of trials and samples
- Algorithm parameters
- Output settings

## Algorithm Implementations

### SLM Algorithm
- **File**: `algorithms.py` - `ScalarLikelihoodMethod` class
- **Features**: Interior-point optimization with scalar likelihood function
- **Key Innovation**: Rank-n update technique avoiding dimension reduction constraints

### Comparison Methods  
- **EM Algorithm**: `algorithms.py` - `EMAlgorithm` class
- **ECM Algorithm**: `algorithms.py` - `ECMAlgorithm` class  
- **Base Class**: `PPLSAlgorithm` - Abstract base for common functionality
- **Utilities**: `ppls_model.py` - Core PPLS model, objectives, and constraints

## Core Components

### Data Generation
- **File**: `data_generator.py` - `SineDataGenerator` class
- **Features**: Sine function-based loading matrices with orthonormal constraints
- **Model**: Implements PPLS structure: x = tW^T + e, y = uC^T + f, u = tB + h

### Experiment Framework
- **File**: `experiment.py` - `PPLSExperiment` class
- **Features**: Monte Carlo experiments with identical starting points
- **Metrics**: `PerformanceMetrics` and `ParameterRecovery` classes

### Visualization
- **File**: `visualization.py` - `PPLSVisualizer` class
- **Features**: Publication-ready plots and Excel table exports
- **Output**: Loading comparisons, MSE analysis, convergence statistics

## File Structure

```
PPLS-SLM/
├── algorithms.py           # SLM, EM, ECM implementations
├── ppls_model.py          # Core PPLS model and utilities  
├── data_generator.py      # Synthetic data generation
├── experiment.py          # Monte Carlo experiment framework
├── visualization.py       # Results plotting and tables
├── main.py               # Main execution script
├── config.json           # Experiment configuration
├── requirements.txt      # Dependencies
└── README.md
```

## Stage-by-Stage Execution

You can run individual stages by modifying `config.json`:

```json
{
  "output": {
    "force_data_generation": true,
    "force_parameter_estimation": false, 
    "force_visualization": false
  }
}
```

### Stage 1: Data Generation Only
Set `force_data_generation: true`, others `false`

### Stage 2: Parameter Estimation Only  
Set `force_parameter_estimation: true`, others `false`

### Stage 3: Visualization Only
Set `force_visualization: true`, others `false`