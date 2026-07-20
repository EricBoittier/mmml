# `mmml lambda-mbar`

MBAR post-processing for lambda TI.


## Usage

```bash
mmml lambda-mbar --help
```

## Options

```text
usage: mmml lambda-mbar [-h] --run-dir RUN_DIR [--checkpoint CHECKPOINT]
                        [--temperature-K TEMPERATURE_K]
                        [--couple-residues COUPLE_RESIDUES]
                        [--ml-cutoff ML_CUTOFF] [--mm-switch-on MM_SWITCH_ON]
                        [--mm-cutoff MM_CUTOFF] [--mbar-verbose] [--no-plots]

MBAR analysis for a completed lambda-dynamics run. Reads lambda_ti_snapshots.npz
from --run-dir and updates lambda_ti_summary.json.

Input & configuration:
  --checkpoint CHECKPOINT
                        Override checkpoint (default: read from summary args if
                        present).
  --couple-residues COUPLE_RESIDUES
                        Override 1-based coupled residue numbers (default: read
                        from snapshots).

Scientific model:
  --temperature-K TEMPERATURE_K
                        kT for reduced potentials (default: from summary or 100
                        K).
  --ml-cutoff ML_CUTOFF
  --mm-switch-on MM_SWITCH_ON
  --mm-cutoff MM_CUTOFF

Output & artifacts:
  --no-plots            Skip updating diagnostic plots.

Diagnostics & safety:
  -h, --help            show this help message and exit
  --mbar-verbose

Other options:
  --run-dir RUN_DIR     Output directory from mmml md-system --setup lambda_ti
                        or scripts/meoh_dimer_lambda_ti.py
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
