# MMML CLI Reference

Complete reference for the MMML command-line interface.

## Installation

```bash
cd /home/ericb/mmml
pip install -e .
```

## Usage

```bash
python -m mmml.cli <command> [options]
```

Or create an alias:
```bash
alias mmml='python -m mmml.cli'
```

## Commands

### `xml2npz` - Convert Molpro XML to NPZ

Convert Molpro XML output files to the standardized NPZ format used for training.

#### Basic Usage

```bash
# Single file
python -m mmml.cli xml2npz output.xml -o data.npz

# Multiple files
python -m mmml.cli xml2npz file1.xml file2.xml file3.xml -o dataset.npz

# Directory (all XML files)
python -m mmml.cli xml2npz molpro_outputs/ -o dataset.npz

# Recursive directory search
python -m mmml.cli xml2npz data/ -o dataset.npz --recursive
```

#### Options

**Input/Output:**
- `inputs` (required): Input XML file(s) or directory/directories
- `-o, --output` (required): Output NPZ file path

**Conversion Options:**
- `--padding N`: Number of atoms to pad to (default: 60)
- `--no-variables`: Exclude Molpro internal variables from output
- `--recursive, -r`: Recursively search directories for XML files

**Validation:**
- `--validate`: Validate output NPZ file against schema (recommended)
- `--no-validate`: Skip validation (faster but not recommended)
- `--strict`: Use strict validation (fail on warnings)

**Output Options:**
- `--summary FILE.json`: Save conversion summary to JSON file
- `--quiet, -q`: Suppress progress output
- `--verbose, -v`: Verbose output

**Advanced:**
- `--continue-on-error`: Continue processing even if some files fail
- `--max-files N`: Maximum number of files to process (for testing)

#### Examples

**Convert with validation:**
```bash
python -m mmml.cli xml2npz molpro_outputs/*.xml \
    -o dataset.npz \
    --validate \
    --summary summary.json
```

**Larger molecules:**
```bash
python -m mmml.cli xml2npz proteins/*.xml \
    -o protein_dataset.npz \
    --padding 200
```

**Quick test run:**
```bash
python -m mmml.cli xml2npz data/*.xml \
    -o test.npz \
    --max-files 10 \
    --quiet
```

**Recursive search:**
```bash
python -m mmml.cli xml2npz molpro_calculations/ \
    -o all_data.npz \
    --recursive \
    --validate
```

#### Output Summary

The `--summary` option saves a JSON file with:
```json
{
  "input_files": 100,
  "output_file": "dataset.npz",
  "padding_atoms": 60,
  "include_variables": true,
  "dataset_info": {
    "n_structures": 100,
    "n_atoms": 60,
    "properties": ["R", "Z", "E", "F", "D", ...],
    "unique_elements": [1, 6, 7, 8],
    "energy_range": {
      "min": -187.6,
      "max": -50.2,
      "mean": -120.5,
      "std": 25.3
    }
  }
}
```

---

### `validate` - Validate NPZ Files

Validate NPZ files against the MMML schema.

#### Basic Usage

```bash
# Validate single file
python -m mmml.cli validate dataset.npz

# Validate multiple files
python -m mmml.cli validate train.npz valid.npz test.npz
```

#### Output

```
============================================================
Validating: dataset.npz
============================================================
✓ NPZ file 'dataset.npz' is valid

Dataset Summary:
  Structures: 1000
  Atoms per structure: 60
  Properties: R, Z, E, F, D, esp, ...
  Elements: [1, 6, 7, 8]
```

#### Exit Codes

- `0`: All files valid
- `1`: One or more files invalid or errors occurred

---

### `train` - Train Models (Coming Soon)

Unified training interface for DCMNet and PhysNetJAX.

**Status:** Phase 2.2 (Planned)

**Planned Usage:**
```bash
python -m mmml.cli train \
    --model dcmnet \
    --config config.yaml \
    --train train.npz \
    --valid valid.npz \
    --output checkpoints/
```

**For now, use the Python API:**
```python
from mmml.data import load_npz
from mmml.data.adapters import prepare_dcmnet_batches

data = load_npz('train.npz')
batches = prepare_dcmnet_batches(data, batch_size=32)
# ... train model ...
```

---

### `evaluate` - Evaluate Models (Coming Soon)

Model evaluation and comparison interface.

**Status:** Phase 2.3 (Planned)

**Planned Usage:**
```bash
python -m mmml.cli evaluate \
    --model checkpoints/best.pkl \
    --data test.npz \
    --output results/ \
    --properties E,F,D
```

---

## Common Workflows

### Workflow 1: Single Calculation

```bash
# 1. Convert XML to NPZ
python -m mmml.cli xml2npz output.xml -o data.npz --validate

# 2. Validate (optional, already done above)
python -m mmml.cli validate data.npz

# 3. Use in Python
python
>>> from mmml.data import load_npz
>>> data = load_npz('data.npz')
```

### Workflow 2: Batch Processing

```bash
# 1. Convert all calculations
python -m mmml.cli xml2npz calculations/*.xml \
    -o full_dataset.npz \
    --validate \
    --summary dataset_summary.json

# 2. Review summary
cat dataset_summary.json | python -m json.tool

# 3. Validate again if needed
python -m mmml.cli validate full_dataset.npz

# 4. Split and train (Python)
python
>>> from mmml.data import load_npz, train_valid_split
>>> data = load_npz('full_dataset.npz')
>>> train, valid = train_valid_split(data, train_fraction=0.8)
```

### Workflow 3: Incremental Dataset Building

```bash
# Convert in batches
python -m mmml.cli xml2npz batch1/*.xml -o dataset_batch1.npz
python -m mmml.cli xml2npz batch2/*.xml -o dataset_batch2.npz
python -m mmml.cli xml2npz batch3/*.xml -o dataset_batch3.npz

# Combine in Python
python
>>> from mmml.data import load_multiple_npz
>>> data = load_multiple_npz([
...     'dataset_batch1.npz',
...     'dataset_batch2.npz',
...     'dataset_batch3.npz'
... ], combine=True)
>>> # Save combined dataset
>>> import numpy as np
>>> np.savez_compressed('full_dataset.npz', **data)
```

### Workflow 4: Quality Control

```bash
# Convert with strict validation
python -m mmml.cli xml2npz calculations/*.xml \
    -o dataset.npz \
    --strict \
    --summary qc_report.json

# Check for issues
cat qc_report.json | jq '.dataset_info.unique_elements'
cat qc_report.json | jq '.dataset_info.energy_range'

# Validate final output
python -m mmml.cli validate dataset.npz
```

---

## Exit Codes

All MMML CLI commands use standard exit codes:

- `0`: Success
- `1`: Error (invalid arguments, processing failure, validation failure)

---

## Environment Variables

Currently none. Future versions may support:

- `MMML_DATA_DIR`: Default data directory
- `MMML_CONFIG`: Default configuration file
- `MMML_VERBOSE`: Default verbosity level

---

## Troubleshooting

### Problem: "No XML files found"

**Solution:** Check your input paths and use `--recursive` if needed:
```bash
python -m mmml.cli xml2npz data/ -o output.npz --recursive
```

### Problem: Validation warnings about unknown keys

**Cause:** The NPZ contains additional properties not in the standard schema (e.g., `orbital_energies`, `metadata`).

**Solution:** These warnings are usually harmless. The schema allows optional keys. To suppress:
```bash
python -m mmml.cli xml2npz input.xml -o output.npz --no-validate
```

### Problem: Memory errors with large datasets

**Solutions:**
1. Process in batches:
```bash
python -m mmml.cli xml2npz batch/*.xml -o batch1.npz --max-files 100
```

2. Reduce padding:
```bash
python -m mmml.cli xml2npz input.xml -o output.npz --padding 30
```

### Problem: Slow conversion

**Solutions:**
1. Use `--no-variables` to skip Molpro variable extraction:
```bash
python -m mmml.cli xml2npz input.xml -o output.npz --no-variables
```

2. Skip validation during conversion (validate separately):
```bash
python -m mmml.cli xml2npz input.xml -o output.npz --no-validate
python -m mmml.cli validate output.npz
```

---

## Tips & Best Practices

### 1. Always Validate

```bash
python -m mmml.cli xml2npz input.xml -o output.npz --validate
```

### 2. Save Summaries for Records

```bash
python -m mmml.cli xml2npz calculations/*.xml \
    -o dataset.npz \
    --summary dataset_$(date +%Y%m%d).json
```

### 3. Test with Small Batches First

```bash
python -m mmml.cli xml2npz data/*.xml \
    -o test.npz \
    --max-files 10
```

### 4. Use Appropriate Padding

- Small molecules (< 30 atoms): `--padding 30`
- Medium molecules (30-60 atoms): `--padding 60` (default)
- Large molecules (60-150 atoms): `--padding 150`
- Proteins: `--padding 500` or more

### 5. Keep Original XMLs

The conversion is lossless, but keep XMLs for:
- Debugging conversion issues
- Extracting additional properties later
- Regenerating with different settings

---

## Advanced Usage

### Batch Scripts

**Convert all experiments:**
```bash
#!/bin/bash
for exp in exp_*; do
    python -m mmml.cli xml2npz "${exp}"/*.xml \
        -o "datasets/${exp}.npz" \
        --validate \
        --summary "datasets/${exp}_summary.json"
done
```

**Parallel processing:**
```bash
#!/bin/bash
find molpro_outputs/ -name "*.xml" | \
    parallel -j 8 "python -m mmml.cli xml2npz {} -o datasets/{/.}.npz"
```

### Integration with Workflows

**Make conversion:**
```makefile
dataset.npz: calculations/*.xml
	python -m mmml.cli xml2npz $^ -o $@ --validate

train.npz valid.npz: dataset.npz
	python scripts/split_dataset.py dataset.npz
```

**Python workflow:**
```python
import subprocess
import sys

# Convert
result = subprocess.run([
    sys.executable, '-m', 'mmml.cli', 'xml2npz',
    'calculations/*.xml',
    '-o', 'dataset.npz',
    '--validate'
], check=True)

# Validate
result = subprocess.run([
    sys.executable, '-m', 'mmml.cli', 'validate',
    'dataset.npz'
], check=True)
```

---

## Version History

- **v0.1.0** (2025-10-30): Initial CLI release
  - `xml2npz` command with full options
  - `validate` command
  - Batch processing support
  - Progress bars and summaries

---

## See Also

- `QUICKSTART.md` - Quick start guide
- `PIPELINE_PLAN.md` - Architecture and future plans
- `mmml/data/npz_schema.py` - NPZ format specification
- `mmml/parse_molpro/README.md` - Molpro XML parser details

---

**Last Updated:** October 30, 2025  
**Status:** Phase 2.1 Complete ✓

