# `mmml kernnn-train`

Train KerNN kernel Softplus MLP (E/F).


## Usage

```bash
mmml kernnn-train --help
```

## Options

```text
usage: mmml kernnn-train [-h] [--data DATA] [--workdir WORKDIR]
                         [--ntrain NTRAIN] [--nvalid NVALID] [--seed SEED]
                         [--n-hidden N_HIDDEN] [--batch-size BATCH_SIZE]
                         [--learning-rate LEARNING_RATE] [--f-weight F_WEIGHT]
                         [--epochs EPOCHS] [--patience PATIENCE]
                         [--ema-decay EMA_DECAY] [--kernel KERNEL]

Train KerNN (kernel Softplus MLP) on NPZ (R, E, F)

Input & configuration:
  --data DATA           NPZ with R, E, F

Execution:
  --seed SEED           RNG seed for split/init
  --batch-size BATCH_SIZE
  --epochs EPOCHS

Output & artifacts:
  --workdir WORKDIR     Output directory

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --ntrain NTRAIN       Training set size
  --nvalid NVALID       Validation set size
  --n-hidden N_HIDDEN   Hidden layer width
  --learning-rate LEARNING_RATE
  --f-weight F_WEIGHT   Force loss weight
  --patience PATIENCE   Early-stop after this many non-improving validation
                        epochs
  --ema-decay EMA_DECAY
  --kernel KERNEL       1D kernel name (default k33)
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
