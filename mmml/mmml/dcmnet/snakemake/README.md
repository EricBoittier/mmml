# Snakemake Workflow for dcmnet Training

This folder contains a Snakemake workflow for running dcmnet training experiments using the CLI in `dcmnet/main.py`.

## Files
- `Snakefile`: Defines the workflow rules for running training jobs.
- `config.yaml`: Lists experiments, their data files, and CLI arguments.

## Usage

1. **Edit `config.yaml`**
   - Add or modify experiments under the `experiments` key.
   - For each experiment, specify:
     - `data_files`: List of data file paths (relative to project root or absolute).
     - `extra_args`: Any extra command-line arguments for `main.py` (e.g., `--type dipole`).

2. **Run Snakemake**
   ```bash
   cd snakemake
   snakemake -j 1  # or -j N for N parallel jobs
   ```
   This will run all experiments defined in `config.yaml`.

3. **Results**
   - Logs and checkpoints for each experiment are saved in `results/<exp_name>/logs` and `results/<exp_name>/checkpoints`.
   - A `done.txt` file is created in each experiment's results folder upon successful completion.

## Customization
- Add new experiments by copying and editing an entry in `config.yaml`.
- Change the CLI arguments in `extra_args` to use different training types, hyperparameters, etc.
- You can add more rules to the `Snakefile` for downstream analysis, evaluation, or plotting.

## Example config.yaml
```yaml
experiments:
  exp_default:
    data_files:
      - data/qm9-esp-dip-40000-0.npz
      - data/qm9-esp-dip-40000-1.npz
      - data/qm9-esp-dip-40000-2.npz
    extra_args: "--type default --num_epochs 100 --batch_size 8 --esp_w 10000.0"

  exp_dipole:
    data_files:
      - data/qm9-esp-dip-40000-0.npz
      - data/qm9-esp-dip-40000-1.npz
      - data/qm9-esp-dip-40000-2.npz
    extra_args: "--type dipole --num_epochs 100 --batch_size 8 --esp_w 10000.0"
```

## Requirements
- Python with Snakemake installed (`pip install snakemake`)
- All dcmnet dependencies (see project root README)

---

For more advanced usage, see the main project README and the comments in the Snakefile. 