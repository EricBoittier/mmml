# DCMNET MCTS Optimization

This directory contains an adapted version of the MCTS (Monte Carlo Tree Search) algorithm specifically designed for optimizing combinations of DCMNET models to minimize ESP (Electrostatic Potential) loss.

## Overview

Instead of playing TicTacToe, this MCTS implementation solves the problem of finding the optimal combination of 7 pre-trained DCMNET models (DCM1-DCM7) that minimizes ESP prediction error.

## Key Components

### 1. `dcmnet_mcts.py` - Core MCTS Implementation
- **DCMNETSelectionEnv**: Environment representing model selection state
- **DCMNET_MCTS**: MCTS algorithm adapted for model combination optimization
- **DCMNETSelectionNet**: Neural network for policy/value prediction (optional)
- **optimize_dcmnet_combination()**: High-level function for optimization

### 2. `dcmnet_ensemble_example.py` - Advanced Usage
- **DCMNETEnsembleOptimizer**: Complete optimization interface
- **evaluate_combination()**: Evaluate specific model combinations
- **compare_all_combinations()**: Exhaustive search for small numbers of models
- **optimize_with_mcts()**: MCTS optimization with multiple restarts

### 3. `run_dcmnet_mcts.py` - Integration Examples
- Quick optimization examples
- Integration guide for existing workflows
- Complete usage demonstrations

## Quick Start

### Basic Usage

```python
from dcmnet_mcts import optimize_dcmnet_combination

# Your molecular data
molecular_data = {
    'atomic_numbers': jnp.array([6, 1, 1, 1, 1]),  # CH4
    'positions': jnp.array([[0.0, 0.0, 0.0], ...]),
    'dst_idx': jnp.array([0, 0, 0, 0]),
    'src_idx': jnp.array([1, 2, 3, 4])
}

# Your ESP target and VdW surface
esp_target = jnp.array([...])  # Your ESP values
vdw_surface = jnp.array([...])  # Your VdW surface points

# Run optimization
best_selection, best_loss = optimize_dcmnet_combination(
    molecular_data=molecular_data,
    esp_target=esp_target,
    vdw_surface=vdw_surface,
    n_simulations=1000,
    temperature=1.0
)

# Results
print(f"Best model selection: {best_selection}")
print(f"Best ESP loss: {best_loss:.6f}")
```

### Advanced Usage

```python
from dcmnet_ensemble_example import DCMNETEnsembleOptimizer

# Initialize optimizer
optimizer = DCMNETEnsembleOptimizer(
    molecular_data=molecular_data,
    esp_target=esp_target,
    vdw_surface=vdw_surface
)

# Run optimization with multiple restarts
best_selection, best_loss = optimizer.optimize_with_mcts(
    n_simulations=1000,
    temperature=1.0,
    n_restarts=5
)

# Get detailed results
optimizer.print_results()
best_models = optimizer.get_best_models()
```

## How It Works

### 1. Problem Formulation
- **State**: Binary vector indicating which of the 7 DCMNET models are selected
- **Actions**: Toggle selection of any model (0-6)
- **Goal**: Find combination that minimizes ESP loss
- **Reward**: Negative ESP loss (lower loss = higher reward)

### 2. MCTS Algorithm
- **Selection**: Use PUCT formula to balance exploration vs exploitation
- **Expansion**: Add new model combinations to the search tree
- **Evaluation**: Calculate ESP loss for new combinations
- **Backup**: Update value estimates up the tree

### 3. Model Combination
- Selected models' predictions are averaged
- ESP is calculated from combined predictions
- Loss is computed against target ESP values

## Key Features

### Intelligent Search
- MCTS explores promising combinations efficiently
- Avoids exhaustive search of all 2^7 = 128 combinations
- Focuses on regions of the search space likely to contain good solutions

### Robust Handling
- Gracefully handles missing models (DCM5-DCM7)
- Provides fallback predictions for unavailable models
- Continues optimization even if some models fail

### Multiple Restarts
- Runs multiple independent optimizations
- Returns the best result across all runs
- Reduces dependence on random initialization

### Detailed Results
- Tracks optimization history
- Provides model names and selection details
- Compares with exhaustive search (for small numbers of models)

## Parameters

### MCTS Parameters
- **n_simulations**: Number of MCTS simulations (default: 1000)
- **temperature**: Action selection temperature (default: 1.0)
- **c_puct**: PUCT exploration constant (default: 1.5)
- **dirichlet_alpha**: Dirichlet noise parameter (default: 0.3)
- **root_noise_frac**: Root noise fraction (default: 0.25)

### Optimization Parameters
- **n_restarts**: Number of independent optimization runs (default: 5)
- **max_models**: Maximum number of models to consider (default: 7)

## Integration with Your Workflow

### 1. Data Preparation
Replace the example data with your actual molecular data:
```python
molecular_data = {
    'atomic_numbers': your_atomic_numbers,
    'positions': your_atom_positions,
    'dst_idx': your_dst_indices,
    'src_idx': your_src_indices
}
esp_target = your_esp_values
vdw_surface = your_vdw_surface_points
```

### 2. Model Loading
Ensure your DCMNET models are available:
```python
from dcmnet.models import DCM1, DCM2, DCM3, DCM4, dcm1_params, dcm2_params, dcm3_params, dcm4_params
# Add DCM5, DCM6, DCM7 when available
```

### 3. Optimization
Run the optimization:
```python
best_selection, best_loss = optimize_dcmnet_combination(
    molecular_data=molecular_data,
    esp_target=esp_target,
    vdw_surface=vdw_surface,
    n_simulations=1000  # Adjust based on your computational budget
)
```

### 4. Use Results
Apply the best combination:
```python
# best_selection is a binary array indicating which models to use
selected_models = [f"DCM{i+1}" for i in range(7) if best_selection[i]]
print(f"Use these models: {selected_models}")
```

## Performance Tips

### Computational Budget
- **Low budget**: n_simulations=200, n_restarts=2
- **Medium budget**: n_simulations=500, n_restarts=3
- **High budget**: n_simulations=1000, n_restarts=5

### Model Availability
- Currently supports DCM1-DCM4
- Add DCM5-DCM7 to the models dictionary when available
- Missing models are handled gracefully with dummy predictions

### Parallelization
- Each restart can be run in parallel
- MCTS simulations within each restart are sequential
- Consider parallelizing across different molecular systems

## Troubleshooting

### Common Issues
1. **Import errors**: Make sure all DCMNET dependencies are installed
2. **Model loading errors**: Check that model parameter files exist
3. **ESP calculation errors**: Verify molecular data format and dimensions
4. **Memory issues**: Reduce n_simulations or use smaller batch sizes

### Debugging
- Enable verbose output to see optimization progress
- Check that all required molecular data fields are present
- Verify ESP target and VdW surface dimensions match

## Example Output

```
=== DCMNET Ensemble Optimization Results ===
Best ESP Loss: 0.123456
Best Model Combination: ['DCM2', 'DCM4']
Number of Models Used: 2

=== Optimization History ===
Run 1: Loss = 0.145678, Models = ['DCM1', 'DCM3']
Run 2: Loss = 0.123456, Models = ['DCM2', 'DCM4']
Run 3: Loss = 0.134567, Models = ['DCM1', 'DCM2', 'DCM3']
```

## Future Enhancements

### Potential Improvements
1. **Neural network guidance**: Train a policy-value network to guide MCTS
2. **Adaptive parameters**: Automatically adjust MCTS parameters based on problem size
3. **Multi-objective optimization**: Optimize for multiple criteria simultaneously
4. **Transfer learning**: Use results from similar molecules to guide search
5. **Parallel MCTS**: Implement parallel MCTS for faster optimization

### Extensions
1. **Different loss functions**: Support for other loss functions beyond ESP
2. **Weighted combinations**: Allow weighted averaging instead of simple averaging
3. **Model pruning**: Automatically remove poor-performing models
4. **Online learning**: Update model combinations during training

## References

- Original MCTS implementation adapted from TicTacToe example
- DCMNET paper and implementation
- AlphaZero algorithm for inspiration
- PUCT algorithm for action selection
