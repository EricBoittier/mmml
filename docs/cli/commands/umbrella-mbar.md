# `mmml umbrella-mbar`

MBAR post-processing for `mmml umbrella-sample` runs.


## Usage

```bash
mmml umbrella-mbar --help
```

## Options

```text
usage: mmml umbrella-mbar [-h] --run-dir RUN_DIR [--checkpoint CHECKPOINT]
                          [--temperature-K TEMPERATURE_K] [--mbar-verbose]
                          [--ml-batch-size ML_BATCH_SIZE]
```

MBAR analysis for a completed umbrella-sample run. Reads
`umbrella_snapshots.npz` from `--run-dir` and updates `umbrella_summary.json`.

### Inputs

- `--run-dir` — Output directory from `mmml umbrella-sample` (required)
- `--checkpoint` — Override checkpoint for \(U_{\mathrm{ML}}\) re-evaluation
- `--temperature-K` — Override \(T\) for reduced potentials

### MBAR

- `--mbar-verbose` — Verbose pymbar output
- `--ml-batch-size` — Reserved for batched \(U_{\mathrm{ML}}\) re-eval

Requires `pymbar>=4` (`uv sync --extra mbar`).

## See also

- Overview: [Batched umbrella sampling](../umbrella.md)
- Sampling: [`mmml umbrella-sample`](umbrella-sample.md)
- Alchemical MBAR: [`mmml lambda-mbar`](lambda-mbar.md)
