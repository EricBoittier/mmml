# Hyperparameter Scan for PhysNetJax

This directory contains scripts for running automated hyperparameter scans using SLURM array jobs.

## Overview

Two scan scripts are provided:

1. **`slurm_quick_scan.sh`** - Fast scan of 6 model sizes (features: 32-256, iterations: 4-5)
2. **`slurm_model_scan.sh`** - Comprehensive scan of 24 configurations (features, iterations, basis functions, residual blocks, batch sizes)

## Quick Start

### 1. Prepare Environment

Ensure your data is ready:
```bash
ls ../preclassified_data/energies_forces_dipoles_train.npz
ls ../preclassified_data/energies_forces_dipoles_valid.npz
```

Create logs directory:
```bash
mkdir -p logs
```

### 2. Submit Quick Scan (Recommended First)

Test 6 different model sizes:
```bash
sbatch slurm_quick_scan.sh
```

This will run 6 jobs in parallel (array indices 0-5):
- Config 0: 32 features, 4 iterations, max_degree=0 (tiny, scalar only)
- Config 1: 64 features, 4 iterations, max_degree=0 (small, scalar only)
- Config 2: 128 features, 4 iterations, max_degree=0 (medium, scalar only)
- Config 3: 256 features, 4 iterations, max_degree=0 (large, scalar only)
- Config 4: 128 features, 5 iterations, max_degree=2 (medium-deep with vectors)
- Config 5: 256 features, 5 iterations, max_degree=2 (large-deep with vectors)

### 3. Submit Full Scan

For comprehensive hyperparameter exploration:
```bash
sbatch slurm_model_scan.sh
```

This tests 24 configurations ranging from small (32 features) to large (512 features).

## Monitor Progress

Check job status:
```bash
squeue -u $USER
```

Check logs:
```bash
tail -f logs/quick_*.out    # Quick scan
tail -f logs/scan_*.out     # Full scan
```

Check specific job:
```bash
tail -f logs/quick_JOBID_0.out  # Replace JOBID with actual job ID
```

## Configurations Tested

### Quick Scan (6 configs)
| Config | Features | Iterations | Max Degree | Basis | Residual | Batch |
|--------|----------|------------|------------|-------|----------|-------|
| 0      | 32       | 4          | 0          | 64    | 3        | 16    |
| 1      | 64       | 4          | 0          | 64    | 3        | 16    |
| 2      | 128      | 4          | 0          | 64    | 3        | 16    |
| 3      | 256      | 4          | 0          | 64    | 3        | 16    |
| 4      | 128      | 5          | 2          | 64    | 3        | 16    |
| 5      | 256      | 5          | 2          | 64    | 3        | 16    |

### Full Scan (24 configs)
Tests combinations of:
- **Features**: 32, 64, 128, 256, 512
- **Max Degree**: 0 (scalar only), 2 (vectors), 3 (tensors)
- **Iterations**: 3, 4, 5
- **Basis functions**: 32, 64, 128
- **Residual blocks**: 2, 3, 4
- **Batch sizes**: 4, 8, 16, 32 (larger for smaller models)

**Note on max_degree:**
- `max_degree=0`: Scalar features only (fastest, simplest)
- `max_degree=2`: Includes vector features (better for directional properties like forces/dipoles)
- `max_degree=3`: Includes tensor features (most expressive, slowest)

## Analyze Results

After jobs complete, analyze the results:

```bash
python analyze_scan_results.py
```

Or specify custom checkpoint directory:
```bash
python analyze_scan_results.py --checkpoints-dir ~/mmml/checkpoints
```

This will:
- Extract metrics from all experiment checkpoints
- Sort models by performance
- Display top 5 models
- Show statistics by model size
- Save full results to `scan_results.csv`

### Manual Result Checking

Check individual experiment:
```bash
ls ~/mmml/checkpoints/co2_quick_f128_i4/
cat ~/mmml/checkpoints/co2_quick_f128_i4/best_model_metrics.json
```

## Customization

### Modify Hyperparameters

Edit the `CONFIGS` array in either script:

**Quick scan format:**
```bash
declare -a CONFIGS=(
    "FEATURES|ITERATIONS|MAX_DEGREE"
    "64|4|0"    # 64 features, 4 iterations, scalar only
    "128|5|2"   # 128 features, 5 iterations, with vectors
    # Add more...
)
```

**Full scan format:**
```bash
declare -a CONFIGS=(
    "FEATURES|ITERATIONS|BASIS|N_RES|BATCH|MAX_DEGREE"
    "64|4|64|3|16|0"      # All parameters
    "128|5|128|4|8|2"     # Add more...
)
```

### Change Training Settings

Edit the `python trainer.py` command in the script:
- `--epochs`: Training duration
- `--learning-rate`: Initial learning rate
- `--schedule`: LR schedule (constant, warmup_cosine, etc.)
- `--energy-weight`, `--forces-weight`, `--dipole-weight`: Loss weights
- `--atomic-energy-method`: default, linear_regression, or mean
- `--no-energy-bias`: Disable model's atomic energy bias

### Resource Requirements

Adjust SLURM resources based on model size:

```bash
#SBATCH --time=12:00:00        # Increase for larger models
#SBATCH --mem-per-cpu=40G      # Increase for very large models
#SBATCH --partition=a100       # Use A100 for faster training
#SBATCH --gres=gpu:2           # Use 2 GPUs (if model supports it)
```

## Example Workflow

```bash
# 1. Submit quick scan
sbatch slurm_quick_scan.sh

# 2. Monitor
squeue -u $USER
tail -f logs/quick_*.out

# 3. Wait for completion (check with squeue)

# 4. Analyze results
python analyze_scan_results.py

# 5. Check results
cat scan_results.csv
ls ~/mmml/checkpoints/

# 6. Use best model for further training or inference
# The best checkpoint will be saved in:
# ~/mmml/checkpoints/co2_quick_f128_i4/  (or similar)
```

## Tips

1. **Start small**: Run the quick scan first to get a sense of what works
2. **Check early**: Monitor first few jobs to ensure they're running correctly
3. **Resource management**: Don't run too many large models simultaneously
4. **Save results**: Keep the CSV files from analysis for comparison
5. **Extend training**: You can restart from best checkpoint with `--restart` flag

## Troubleshooting

### Jobs not starting
- Check partition availability: `sinfo -p titan`
- Check your job queue: `squeue -u $USER`
- Check SLURM allocation

### Out of memory errors
- Reduce batch size
- Reduce model size (features, basis functions)
- Increase `--mem-per-cpu`

### Poor convergence
- Adjust learning rate
- Try different schedules
- Check loss weights ratio
- Increase epochs

### Jobs failing immediately
- Check logs in `logs/` directory
- Verify data paths are correct
- Ensure virtual environment is activated correctly
- Check CUDA module is loaded

## Output Files

Each experiment creates:
- `~/mmml/checkpoints/EXPERIMENT_NAME/` - Checkpoint directory
  - `params_best.pkl` - Best model parameters
  - `model_attributes.json` - Model configuration
  - `training_metrics.json` - Training history
  - `best_model_metrics.json` - Best validation metrics

- `logs/quick_JOBID_ARRAYID.out` - Standard output
- `logs/quick_JOBID_ARRAYID.err` - Error output

## Next Steps

After identifying the best configuration:
1. Train longer with more epochs
2. Fine-tune learning rate schedule
3. Adjust loss weights for your specific use case
4. Use the model for inference or transfer learning

