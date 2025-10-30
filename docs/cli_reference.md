# CLI Reference

Complete reference for all MMML command-line tools.

## Installation

```bash
cd /home/ericb/mmml
pip install -e .
```

## Main Command

```bash
python -m mmml.cli <command> [options]
```

**Available Commands:**
- `xml2npz` - Convert Molpro XML to NPZ format
- `validate` - Validate NPZ files
- `train` - Train models
- `evaluate` - Evaluate models

## xml2npz

Convert Molpro XML output files to standardized NPZ format.

### Synopsis

```bash
python -m mmml.cli xml2npz INPUT [INPUT ...] -o OUTPUT [OPTIONS]
```

### Arguments

**Required:**
- `INPUT` - Input XML file(s) or directory/directories
- `-o, --output OUTPUT` - Output NPZ file path

**Conversion Options:**
- `--padding N` - Pad to N atoms (default: 60)
- `--no-variables` - Exclude Molpro variables
- `-r, --recursive` - Recursively search directories

**Validation:**
- `--validate` - Validate output (recommended)
- `--no-validate` - Skip validation
- `--strict` - Strict validation mode

**Output:**
- `--summary FILE.json` - Save summary to JSON
- `-q, --quiet` - Suppress output
- `-v, --verbose` - Verbose output

**Advanced:**
- `--continue-on-error` - Don't stop on errors
- `--max-files N` - Limit number of files

### Examples

Basic conversion:
```bash
python -m mmml.cli xml2npz output.xml -o data.npz
```

Batch with validation:
```bash
python -m mmml.cli xml2npz calculations/*.xml \
    -o dataset.npz \
    --validate \
    --summary summary.json
```

Recursive directory:
```bash
python -m mmml.cli xml2npz projects/ \
    -o all_data.npz \
    --recursive
```

Large molecules:
```bash
python -m mmml.cli xml2npz proteins/*.xml \
    -o proteins.npz \
    --padding 200
```

### Output

**Console:**
```
🔍 Finding XML files...
📁 Found 100 XML file(s)

🔄 Converting to NPZ format...
   Output: dataset.npz
   Padding: 60 atoms
   Variables: Yes
Converting XML files: 100%|██████████| 100/100 [00:10<00:00, 9.8it/s]

✓ Validating output...
✓ Validation passed

📊 Dataset Summary:
   Structures: 100
   Atoms: 60
   Properties: R, Z, E, F, D, esp, ...
   Elements: [1, 6, 7, 8]
   Energy range: [-250.5, -85.3] Ha

✅ Conversion complete!
```

**JSON Summary (`--summary`):**
```json
{
  "input_files": 100,
  "output_file": "dataset.npz",
  "padding_atoms": 60,
  "dataset_info": {
    "n_structures": 100,
    "energy_range": {"min": -250.5, "max": -85.3},
    "unique_elements": [1, 6, 7, 8]
  }
}
```

## validate

Validate NPZ files against MMML schema.

### Synopsis

```bash
python -m mmml.cli validate FILE [FILE ...]
```

### Arguments

- `FILE` - NPZ file(s) to validate

### Examples

Single file:
```bash
python -m mmml.cli validate dataset.npz
```

Multiple files:
```bash
python -m mmml.cli validate train.npz valid.npz test.npz
```

### Output

```
============================================================
Validating: dataset.npz
============================================================
✓ NPZ file 'dataset.npz' is valid

Dataset Summary:
  Structures: 1000
  Atoms per structure: 60
  Properties: R, Z, E, F, D, esp
  Elements: [1, 6, 7, 8]
```

### Exit Codes

- `0` - All files valid
- `1` - Validation failed or error

## train

Train DCMNet or PhysNetJAX models.

### Synopsis

```bash
python -m mmml.cli train [OPTIONS]
```

### Arguments

**Model:**
- `--model {dcmnet,physnetjax}` - Model to train (default: dcmnet)

**Data:**
- `--train FILE` - Training NPZ file (required)
- `--valid FILE` - Validation NPZ file (optional)
- `--train-fraction F` - Auto-split fraction (default: 0.8)

**Training:**
- `--batch-size N` - Batch size (default: 32)
- `--max-epochs N` - Maximum epochs (default: 1000)
- `--learning-rate LR` - Learning rate (default: 0.001)
- `--early-stopping N` - Early stopping patience (default: 50)

**Targets:**
- `--targets T [T ...]` - Training targets (default: energy)

**Output:**
- `--output DIR` - Output directory (default: checkpoints)
- `--log-interval N` - Log interval (default: 10)

**Preprocessing:**
- `--center-coords` - Center coordinates
- `--normalize-energy` - Normalize energies

**Configuration:**
- `--config FILE.yaml` - Load config from YAML
- `--save-config FILE.yaml` - Save config and exit

**Options:**
- `--dry-run` - Test setup without training
- `-v, --verbose` - Verbose output
- `-q, --quiet` - Quiet mode

### Examples

Basic training:
```bash
python -m mmml.cli train \
    --model dcmnet \
    --train train.npz \
    --valid valid.npz
```

With config file:
```bash
python -m mmml.cli train --config config.yaml
```

Multiple targets:
```bash
python -m mmml.cli train \
    --model dcmnet \
    --train train.npz \
    --targets energy forces dipole esp
```

Create config template:
```bash
python -m mmml.cli train \
    --model dcmnet \
    --train train.npz \
    --batch-size 64 \
    --save-config my_config.yaml
```

Test setup:
```bash
python -m mmml.cli train \
    --model dcmnet \
    --train train.npz \
    --dry-run
```

### Configuration File

```yaml
# config.yaml
model: dcmnet

train_file: train.npz
valid_file: valid.npz

batch_size: 32
max_epochs: 1000
learning_rate: 0.001
early_stopping: 50

targets: [energy, forces, dipole]
loss_weights:
  energy: 1.0
  forces: 100.0
  dipole: 10.0

output_dir: checkpoints/dcmnet
log_interval: 10

center_coordinates: false
normalize_energy: false

model_params:
  features: 128
  max_degree: 2
  num_iterations: 3
  cutoff: 5.0
```

## evaluate

Evaluate trained models on test data.

### Synopsis

```bash
python -m mmml.cli evaluate --model MODEL [MODEL ...] --data DATA [OPTIONS]
```

### Arguments

**Required:**
- `--model MODEL [MODEL ...]` - Model checkpoint(s)
- `--data FILE` - Test data NPZ file

**Evaluation:**
- `--properties P [P ...]` - Properties to evaluate (default: energy)
- `--batch-size N` - Batch size (default: 32)

**Output:**
- `--output DIR` - Output directory (default: results)
- `--report` - Generate markdown report
- `--save-predictions` - Save predictions to file

**Options:**
- `-v, --verbose` - Verbose output
- `-q, --quiet` - Quiet mode

### Examples

Basic evaluation:
```bash
python -m mmml.cli evaluate \
    --model checkpoint.pkl \
    --data test.npz
```

Multiple properties:
```bash
python -m mmml.cli evaluate \
    --model checkpoint.pkl \
    --data test.npz \
    --properties energy forces dipole
```

With report:
```bash
python -m mmml.cli evaluate \
    --model checkpoint.pkl \
    --data test.npz \
    --report \
    --output results/
```

Compare models:
```bash
python -m mmml.cli evaluate \
    --model model1.pkl model2.pkl model3.pkl \
    --data test.npz \
    --report
```

### Output

**JSON Results (`evaluation_results.json`):**
```json
{
  "model": "checkpoint",
  "metrics": {
    "energy": {
      "mae": 0.001234,
      "rmse": 0.002345,
      "max_error": 0.012345,
      "r2": 0.9987
    },
    "forces": {
      "mae": 0.0123,
      "rmse": 0.0234,
      "max_error": 0.1234,
      "r2": 0.9876
    }
  }
}
```

**Markdown Report (`evaluation_report.md`):**
```markdown
# Model Evaluation Report

Generated: 2025-10-30 12:00:00

## Dataset Statistics
- Structures: 1000
- Atoms: 60
- Elements: [1, 6, 7, 8]

## Evaluation Metrics

| Property | MAE | RMSE | Max Error | R² |
|----------|-----|------|-----------|----| 
| Energy | 0.001234 | 0.002345 | 0.012345 | 0.9987 |
| Forces | 0.012300 | 0.023400 | 0.123400 | 0.9876 |
```

## Common Workflows

### Complete Pipeline

```bash
# 1. Convert data
python -m mmml.cli xml2npz calculations/*.xml -o dataset.npz --validate

# 2. Train model
python -m mmml.cli train \
    --model dcmnet \
    --train dataset.npz \
    --train-fraction 0.8 \
    --output checkpoints/

# 3. Evaluate
python -m mmml.cli evaluate \
    --model checkpoints/best.pkl \
    --data test.npz \
    --report
```

### Model Comparison

```bash
# Evaluate multiple models
python -m mmml.cli evaluate \
    --model checkpoints/dcmnet.pkl checkpoints/physnet.pkl \
    --data test.npz \
    --properties energy forces \
    --report \
    --output comparison/
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple experiments
for exp in exp_*; do
    python -m mmml.cli xml2npz "${exp}"/*.xml \
        -o "datasets/${exp}.npz" \
        --validate
    
    python -m mmml.cli train \
        --model dcmnet \
        --train "datasets/${exp}.npz" \
        --output "checkpoints/${exp}/"
done
```

## Exit Codes

All commands use standard exit codes:
- `0` - Success
- `1` - Error (see stderr for details)

## Environment Variables

Currently none. Future versions may support:
- `MMML_DATA_DIR` - Default data directory
- `MMML_CONFIG` - Default configuration
- `MMML_CACHE_DIR` - Cache directory

## Shell Completion

Future feature: Tab completion for bash/zsh.

## See Also

- [Data Pipeline](data_pipeline.md) - Complete pipeline guide
- [Configuration](configuration.md) - Config file format
- [API Reference](api/data.md) - Python API

---

**Version:** 0.1.0  
**Last Updated:** October 30, 2025

