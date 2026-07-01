# `mmml compare-npz`

Reference vs model NPZ plots.


## Usage

```bash
mmml compare-npz --help
```

## Options

```text
usage: mmml compare-npz [-h] [--reference REFERENCE] [--predictions PREDICTIONS] [--checkpoint CHECKPOINT] [--data DATA] [-o OUTPUT_DIR] [--max-frames MAX_FRAMES] [--stride STRIDE] [--cutoff CUTOFF] [--use-dcmnet-dipole] [--energy-unit ENERGY_UNIT] [--force-unit FORCE_UNIT]
                        [--no-plots] [--save-predictions]

Compare reference and model NPZ data (metrics + plots).

options:
  -h, --help            show this help message and exit
  --reference REFERENCE
                        Reference NPZ (PySCF / QM labels)
  --predictions PREDICTIONS
                        Model prediction NPZ (E, F, D, ...)
  --checkpoint CHECKPOINT
                        Model checkpoint JSON/pkl/dir; run inference on --data
  --data DATA           Labeled NPZ for --checkpoint mode (R,Z,E,F,...)
  -o, --output-dir OUTPUT_DIR
                        Output directory for metrics and plots
  --max-frames MAX_FRAMES
                        Max structures to compare (default: all)
  --stride STRIDE       Frame stride for --checkpoint mode (default: 1)
  --cutoff CUTOFF       Model cutoff override for checkpoint inference
  --use-dcmnet-dipole   Use DCMNet dipole from joint checkpoint
  --energy-unit ENERGY_UNIT
                        Energy unit label for plots (default: infer from reference NPZ)
  --force-unit FORCE_UNIT
                        Force unit label for plots (default: eV/Å)
  --no-plots            Skip matplotlib plots
  --save-predictions    With --checkpoint, save inference NPZ to output dir

Compare reference (PySCF/QM) and model NPZ trajectories with metrics and plots.

Modes
-----
1. Two NPZ files (reference labels vs model predictions):
       mmml compare-npz --reference ref.npz --predictions pred.npz -o out/

2. Checkpoint inference against labeled NPZ (same file holds R,Z,E,F,...):
       mmml compare-npz --checkpoint params.json --data test.npz -o out/ --max-frames 200

Issue #12: per-atom / per-element force analysis and richer validation plots.
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
