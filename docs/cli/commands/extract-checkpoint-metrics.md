# `mmml extract-checkpoint-metrics`

Plot training metrics from Orbax checkpoints.


## Usage

```bash
mmml extract-checkpoint-metrics --help
```

## Options

```text
usage: mmml extract-checkpoint-metrics [-h] -o OUTPUT [--log-loss] [--quiet]
                                       [--stride STRIDE]
                                       [--max-epochs MAX_EPOCHS]
                                       [--metrics-json METRICS_JSON]
                                       [--ef-only]
                                       [--plot-style {dark,google,mpl_classic,nature,science,tron,xmgrace}]
                                       [--individual-dir INDIVIDUAL_DIR]
                                       checkpoint_dir

Extract and plot training metrics from Orbax checkpoints

positional arguments:
  checkpoint_dir        Checkpoint directory containing epoch-* subdirectories

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output plot file (PNG)
  --log-loss            Use log scale for loss axes (recommended)
  --quiet               Suppress output
  --stride STRIDE       Read every Nth epoch checkpoint (default: 1 = all).
                        Use for large runs.
  --max-epochs MAX_EPOCHS
                        Cap the number of epoch checkpoints read after stride
                        (default: no cap).
  --metrics-json METRICS_JSON
                        Optional path to write extracted metrics as JSON
                        arrays.
  --ef-only             Plot energy/forces panels only (omit dipole inset from
                        main layout).
  --plot-style {dark,google,mpl_classic,nature,science,tron,xmgrace}
                        Matplotlib style preset (default: google). Options:
                        nature, xmgrace, google, tron, mpl_classic.
  --individual-dir INDIVIDUAL_DIR
                        If set, write one PNG per metric into this directory.

Examples:
  # Plot glycol training with log scale
  python -m mmml.cli.extract_checkpoint_metrics \
      examples/glycol/checkpoints/glycol_production/glycol_production-*/ \
      --output glycol_training.png \
      --log-loss
  
  # Without log scale
  python -m mmml.cli.extract_checkpoint_metrics \
      checkpoints/run/run-uuid/ \
      --output training.png
```


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
