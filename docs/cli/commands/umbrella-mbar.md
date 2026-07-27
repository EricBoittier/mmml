# `mmml umbrella-mbar`

MBAR post-processing for umbrella-sample runs.


## Usage

```bash
mmml umbrella-mbar --help
```

## Options

```text
usage: mmml umbrella-mbar [-h] --run-dir RUN_DIR [--checkpoint CHECKPOINT]
                          [--temperature-K TEMPERATURE_K] [--mbar-verbose]
                          [--ml-batch-size ML_BATCH_SIZE]

MBAR analysis for a completed umbrella-sample run. Reads umbrella_snapshots.npz
from --run-dir and updates umbrella_summary.json.

Input & configuration:
  --checkpoint CHECKPOINT
                        Override checkpoint for U_ML re-evaluation (default:
                        from snapshots/summary).

Scientific model:
  --temperature-K TEMPERATURE_K
                        kT for reduced potentials (default: from
                        snapshots/summary).

Execution:
  --ml-batch-size ML_BATCH_SIZE
                        Reserved for batched U_ML re-eval (default: 32)

Diagnostics & safety:
  -h, --help            show this help message and exit
  --mbar-verbose        Verbose pymbar output

Other options:
  --run-dir RUN_DIR     Output directory from mmml umbrella-sample

CLI for MBAR post-processing of umbrella sampling runs. Usage: mmml umbrella-
mbar --run-dir out/umbrella [--checkpoint PATH] [--temperature-K 300]
```


## Related docs

- [Batched umbrella sampling](../../umbrella.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
