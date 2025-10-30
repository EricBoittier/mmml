# DCMNet Multi-Batch Training System

Comprehensive guide to the new multi-batch training infrastructure with gradient accumulation, advanced monitoring, and experiment tracking.

## 🎯 Overview

The multi-batch training system provides:

1. **Gradient Accumulation** - Train with effectively larger batch sizes without memory constraints
2. **Advanced Configuration Management** - Structured, reproducible experiment configuration
3. **Enhanced Monitoring** - Comprehensive statistics and progress tracking
4. **Better Checkpointing** - Automatic checkpoint management with history
5. **Improved Analysis** - Batch-wise diagnostics and model comparison

## 📁 New Files

### Core Infrastructure
- `training_config.py` - Configuration management (`ExperimentConfig`, `TrainingConfig`, `ModelConfig`)
- `training_multibatch.py` - Multi-batch training with gradient accumulation
- `analysis_multibatch.py` - Enhanced analysis tools
- `train_runner.py` - Convenient training interface (CLI + Python API)

## 🚀 Quick Start

### Option 1: Command Line

```bash
# Basic training
python -m mmml.dcmnet.dcmnet.train_runner \
    --name my_experiment \
    --num-epochs 100 \
    --batch-size 2 \
    --gradient-accumulation-steps 4

# Advanced training with custom settings
python -m mmml.dcmnet.dcmnet.train_runner \
    --name advanced_experiment \
    --num-epochs 200 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-5 \
    --use-lr-schedule \
    --lr-schedule-type cosine \
    --esp-w 1000.0 \
    --chg-w 1.0 \
    --n-dcm 3 \
    --features 32 \
    --num-train 5000 \
    --num-valid 1000
```

### Option 2: Python API

```python
from mmml.dcmnet.dcmnet.train_runner import run_training

# Simple training
params, loss, exp_dir = run_training(
    name="my_experiment",
    num_epochs=100,
    batch_size=2,
    gradient_accumulation_steps=4
)

# Advanced configuration
params, loss, exp_dir = run_training(
    name="advanced_experiment",
    num_epochs=200,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    use_lr_schedule=True,
    lr_schedule_type="cosine",
    esp_w=1000.0,
    chg_w=1.0,
    n_dcm=3,
    features=32,
    num_train=5000,
    num_valid=1000
)
```

### Option 3: Full Control with Config Objects

```python
import jax
from mmml.dcmnet.dcmnet.training_config import ExperimentConfig, TrainingConfig, ModelConfig
from mmml.dcmnet.dcmnet.training_multibatch import train_model_multibatch
from mmml.dcmnet.dcmnet.data import prepare_datasets
from mmml.dcmnet.dcmnet.analysis import create_model

# Create configuration
config = ExperimentConfig(
    name="custom_experiment",
    description="My custom training run"
)

# Configure model
config.model = ModelConfig(
    n_dcm=2,
    features=16,
    max_degree=2,
    num_iterations=2
)

# Configure training
config.training = TrainingConfig(
    num_epochs=100,
    batch_size=2,
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    use_lr_schedule=True,
    lr_schedule_type="cosine",
    esp_w=1000.0,
    chg_w=1.0,
    save_every_n_epochs=10
)

# Load data
key = jax.random.PRNGKey(42)
train_data, valid_data = prepare_datasets(
    key, 
    num_train=1000, 
    num_valid=200,
    filename=["esp2000.npz"]
)

# Create model
model = create_model(
    n_dcm=config.model.n_dcm,
    features=config.model.features
)

# Train
final_params, best_loss = train_model_multibatch(
    key=key,
    model=model,
    train_data=train_data,
    valid_data=valid_data,
    config=config
)
```

## 📊 Features

### 1. Gradient Accumulation

Accumulate gradients over multiple batches before updating parameters. This allows training with effectively larger batch sizes without increased memory usage.

```python
# Effective batch size = batch_size * gradient_accumulation_steps
config.training = TrainingConfig(
    batch_size=1,                      # Physical batch size (memory limited)
    gradient_accumulation_steps=8,     # Accumulate over 8 batches
    # Effective batch size = 1 * 8 = 8
)
```

**Benefits:**
- Train with larger effective batch sizes on limited memory
- More stable gradients and training
- Better generalization

### 2. Learning Rate Schedules

Built-in support for various LR schedules with warmup:

```python
config.training = TrainingConfig(
    learning_rate=1e-4,
    use_lr_schedule=True,
    lr_schedule_type="cosine",  # "cosine", "exponential", or "step"
    warmup_epochs=5,
    min_lr_factor=0.01
)
```

**Schedule Types:**
- **Cosine**: Smooth decay following cosine curve
- **Exponential**: Exponential decay
- **Step**: Constant after warmup

### 3. Enhanced Monitoring

Comprehensive statistics tracked and displayed:

```
==================================================================================
Epoch   5 Summary (Time: 12.34s)
==================================================================================
Metric                          Train           Valid            Diff        % Diff
----------------------------------------------------------------------------------
loss                      1.234567e-03    1.456789e-03    2.222220e-04       18.00%
mono_mae                  5.678900e-05    6.123400e-05    4.445000e-06        7.83%
mono_rmse                 7.890120e-05    8.234560e-05    3.444400e-06        4.37%
mono_mean                 1.234500e-07    1.456700e-07    2.222000e-08       18.00%
mono_std                  9.876540e-02    9.987650e-02    1.111100e-03        1.12%
----------------------------------------------------------------------------------
Avg batch time: 0.0234s
==================================================================================
```

### 4. Automatic Checkpointing

Intelligent checkpoint management:

```
experiments/my_experiment/
├── config.json                      # Full experiment configuration
├── checkpoints/
│   ├── checkpoint_epoch_10.pkl      # Periodic checkpoints
│   ├── checkpoint_epoch_20.pkl
│   ├── checkpoint_epoch_30.pkl
│   ├── checkpoint_best.pkl          # Best validation loss
│   └── checkpoint_latest.pkl        # Latest checkpoint (for resuming)
```

**Features:**
- Automatic periodic saving (configurable interval)
- Best checkpoint tracking
- Latest checkpoint for easy resumption
- Checkpoint pruning (keep top-k)

### 5. Experiment Configuration

Structured configuration with JSON serialization:

```python
from mmml.dcmnet.dcmnet.training_config import ExperimentConfig

# Create config
config = ExperimentConfig(name="my_exp")

# Save config
config.save("experiments/my_exp/config.json")

# Load config
loaded_config = ExperimentConfig.load("experiments/my_exp/config.json")

# Export to dict
config_dict = config.to_dict()
```

## 🔍 Analysis Tools

### Checkpoint Analysis

```python
from mmml.dcmnet.dcmnet.analysis_multibatch import analyze_checkpoint, print_analysis_summary

# Analyze a checkpoint
analysis = analyze_checkpoint(
    checkpoint_path="experiments/my_exp/checkpoints/checkpoint_best.pkl",
    model=model,
    test_data=test_data
)

# Print summary
print_analysis_summary(analysis)

# Output:
# ================================================================================
# Model Analysis Summary
# ================================================================================
# Metric                                Value            Unit
# --------------------------------------------------------------------------------
# Total Loss                      1.234567e-03                
# Loss Std Dev                    5.678901e-04                
# MAE                             2.345678e-05            a.u.
# RMSE                            3.456789e-05            a.u.
# Max Error                       1.234567e-04            a.u.
# Correlation                     9.876543e-01                
# ...
```

### Compare Multiple Checkpoints

```python
from mmml.dcmnet.dcmnet.analysis_multibatch import compare_checkpoints

checkpoints = [
    "experiments/exp1/checkpoints/checkpoint_best.pkl",
    "experiments/exp2/checkpoints/checkpoint_best.pkl",
    "experiments/exp3/checkpoints/checkpoint_best.pkl",
]

comparison_df = compare_checkpoints(
    checkpoint_paths=checkpoints,
    model=model,
    test_data=test_data,
    labels=["Baseline", "Large Model", "With Scheduler"]
)

print(comparison_df)
```

### Training History Analysis

```python
from mmml.dcmnet.dcmnet.analysis_multibatch import analyze_training_history

history = analyze_training_history("experiments/my_exp")

print(f"Best epoch: {history['best_epoch']}")
print(f"Best valid loss: {history['best_valid_loss']}")

# Access full history DataFrame
df = history['history_df']
df.plot(x='epochs', y=['train_loss', 'valid_loss'])
```

### Complete Batch Analysis

```python
from mmml.dcmnet.dcmnet.analysis_multibatch import batch_analysis_summary

results = batch_analysis_summary(
    exp_dir="experiments/my_exp",
    model=model,
    test_data=test_data,
    output_dir="experiments/my_exp/analysis"
)

# Generates:
# - experiments/my_exp/analysis/best_analysis.json
# - experiments/my_exp/analysis/predictions.csv
# - experiments/my_exp/analysis/training_history.csv
```

## 📈 Training Recommendations

### For Small Memory (<16GB GPU)

```python
config.training = TrainingConfig(
    batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8
    use_grad_clip=True,
    grad_clip_norm=1.0
)
```

### For Medium Memory (16-32GB GPU)

```python
config.training = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    use_grad_clip=True,
    grad_clip_norm=2.0
)
```

### For Large Memory (>32GB GPU)

```python
config.training = TrainingConfig(
    batch_size=8,
    gradient_accumulation_steps=2,  # Effective batch size = 16
    use_grad_clip=True,
    grad_clip_norm=2.0
)
```

### For Long Training

```python
config.training = TrainingConfig(
    num_epochs=500,
    use_lr_schedule=True,
    lr_schedule_type="cosine",
    warmup_epochs=20,
    min_lr_factor=0.01,
    save_every_n_epochs=10,
    use_ema=True,
    ema_decay=0.999
)
```

## 🔄 Resuming Training

```bash
# From command line
python -m mmml.dcmnet.dcmnet.train_runner \
    --name my_experiment \
    --restart-checkpoint experiments/my_experiment/checkpoints/checkpoint_latest.pkl
```

```python
# From Python
run_training(
    name="my_experiment",
    restart_checkpoint="experiments/my_experiment/checkpoints/checkpoint_latest.pkl"
)
```

## 🎛️ Configuration Reference

### ExperimentConfig

```python
ExperimentConfig(
    name: str                    # Experiment name
    description: str             # Experiment description
    model: ModelConfig          # Model configuration
    training: TrainingConfig    # Training configuration
    data_files: list            # List of data files
    random_seed: int            # Random seed for reproducibility
    output_dir: str             # Output directory
    created_at: str             # Creation timestamp (auto)
    git_commit: str             # Git commit hash (optional)
    notes: str                  # Additional notes
)
```

### ModelConfig

```python
ModelConfig(
    n_dcm: int = 2                      # Number of distributed multipoles per atom
    features: int = 16                  # Number of features
    max_degree: int = 2                 # Maximum spherical harmonic degree
    num_iterations: int = 2             # Number of message passing iterations
    num_basis_functions: int = 16       # Number of radial basis functions
    cutoff: float = 4.0                 # Interaction cutoff (Angstrom)
    max_atomic_number: int = 36         # Maximum atomic number
    include_pseudotensors: bool = False # Include pseudotensor features
)
```

### TrainingConfig

```python
TrainingConfig(
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_atoms: int = 60
    
    # Loss weights
    esp_w: float = 1.0
    chg_w: float = 0.01
    
    # Optimization
    use_grad_clip: bool = True
    grad_clip_norm: float = 2.0
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Multi-batch training
    gradient_accumulation_steps: int = 1
    use_parallel_batches: bool = False
    
    # Learning rate schedule
    use_lr_schedule: bool = False
    lr_schedule_type: str = "cosine"
    warmup_epochs: int = 5
    min_lr_factor: float = 0.01
    
    # Checkpointing
    save_every_n_epochs: int = 10
    keep_top_k_checkpoints: int = 5
    
    # Logging
    log_every_n_batches: int = 10
    compute_full_stats_every_n_epochs: int = 1
    
    # Data
    num_train: int = 1000
    num_valid: int = 200
)
```

## 🔧 Advanced Usage

### Custom Loss Functions

The system is designed to be extensible. You can implement custom loss functions:

```python
# See training_multibatch.py for compute_gradients function
# Modify to use your custom loss function
```

### Custom Analysis

```python
from mmml.dcmnet.dcmnet.analysis_multibatch import load_checkpoint

# Load checkpoint
checkpoint = load_checkpoint("path/to/checkpoint.pkl")
params = checkpoint['params']
config = checkpoint['config']
metrics = checkpoint['metrics']

# Run custom analysis
# ...
```

## 📝 Notes

1. **Gradient Accumulation**: Effective batch size = `batch_size × gradient_accumulation_steps`
2. **Memory Usage**: Controlled by `batch_size`, not `gradient_accumulation_steps`
3. **EMA**: Exponential Moving Average helps stabilize training and often improves final performance
4. **Checkpoints**: Contain full state for perfect resumption
5. **Configuration**: Always saved with checkpoints for reproducibility

## 🐛 Troubleshooting

### Out of Memory

Reduce `batch_size`:
```python
config.training.batch_size = 1  # Minimum physical batch size
```

### Training Unstable

Enable/adjust gradient clipping:
```python
config.training.use_grad_clip = True
config.training.grad_clip_norm = 1.0  # Try smaller values
```

### Slow Convergence

Try learning rate schedule:
```python
config.training.use_lr_schedule = True
config.training.lr_schedule_type = "cosine"
config.training.warmup_epochs = 10
```

### Overfitting

Use larger effective batch size:
```python
config.training.gradient_accumulation_steps = 8  # or higher
config.training.use_ema = True
```

## 🎉 Comparison: Old vs New

### Old Training

```python
# Old way - basic training
params, loss = train_model(
    key, model, train_data, valid_data,
    num_epochs=100,
    learning_rate=1e-4,
    batch_size=1,
    writer=None,
    ndcm=2
)
```

### New Multi-Batch Training

```python
# New way - full-featured training
params, loss, exp_dir = run_training(
    name="my_experiment",              # Named experiments
    num_epochs=100,
    batch_size=1,
    gradient_accumulation_steps=8,     # Larger effective batch
    learning_rate=1e-4,
    use_lr_schedule=True,              # LR scheduling
    use_grad_clip=True,                # Gradient clipping
    use_ema=True,                      # EMA for stability
    save_every_n_epochs=10,            # Auto checkpointing
    n_dcm=2
)

# Full analysis
from mmml.dcmnet.dcmnet.analysis_multibatch import batch_analysis_summary
results = batch_analysis_summary(exp_dir, model, test_data)
```

## 📚 Examples

See `examples/dcm/` directory for complete training examples and notebooks.

## 🚀 Future Enhancements

Planned features:
- [ ] Parallel batch processing with `vmap`/`pmap`
- [ ] Distributed training across multiple GPUs
- [ ] Hyperparameter tuning integration (Optuna)
- [ ] Weights & Biases integration
- [ ] Real-time training monitoring dashboard
- [ ] Automated experiment comparison reports

---

**Happy Training! 🎯**

