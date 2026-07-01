# `mmml downstream`

Downstream analysis utilities.


## Usage

```bash
mmml downstream --help
```

## Options

```text
usage: mmml downstream [-h] --dataset DATASET --checkpoint-dcm CHECKPOINT_DCM
                       --checkpoint-noneq CHECKPOINT_NONEQ
                       [--sample-index SAMPLE_INDEX]
                       [--mode {check,quick,full}] [--output-dir OUTPUT_DIR]
                       [--temperature TEMPERATURE] [--md-steps MD_STEPS]
                       [--timestep TIMESTEP] [--ir-delta IR_DELTA]
                       [--freq-delta FREQ_DELTA] [--opt-fmax OPT_FMAX]
                       [--opt-steps OPT_STEPS] [--raman]
                       [--raman-delta RAMAN_DELTA] [--raman-field RAMAN_FIELD]

Run harmonic / MD downstream analyses on an MMML dataset

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to dataset NPZ file
  --checkpoint-dcm CHECKPOINT_DCM
                        Equivariant checkpoint (best_params.pkl)
  --checkpoint-noneq CHECKPOINT_NONEQ
                        Non-equivariant checkpoint
  --sample-index SAMPLE_INDEX
                        Configuration index in dataset
  --mode {check,quick,full}
                        Analysis mode: 'check' (single-point), 'quick'
                        (harmonic), 'full' (+MD)
  --output-dir OUTPUT_DIR
  --temperature TEMPERATURE
                        MD temperature in Kelvin
  --md-steps MD_STEPS   Number of MD steps for full mode
  --timestep TIMESTEP   MD timestep in femtoseconds
  --ir-delta IR_DELTA   Displacement for IR finite differences (Å)
  --freq-delta FREQ_DELTA
                        Displacement for numerical Hessian (Å)
  --opt-fmax OPT_FMAX   Geometry optimisation force threshold (eV/Å)
  --opt-steps OPT_STEPS
                        Maximum optimisation steps
  --raman               Compute Raman spectrum using finite-field
                        polarizability
  --raman-delta RAMAN_DELTA
                        Displacement for Raman derivatives (Å)
  --raman-field RAMAN_FIELD
                        Electric field strength for Raman finite-field (V/Å)
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
