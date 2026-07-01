# `mmml physnet-evaluate`

Evaluate PhysNet checkpoint.


## Usage

```bash
mmml physnet-evaluate --help
```

## Options

```text
usage: mmml physnet-evaluate [-h] --checkpoint CHECKPOINT --data DATA
                             [-o OUTPUT_DIR] [--natoms NATOMS]
                             [--batch-size BATCH_SIZE] [--seed SEED]
                             [--num-samples NUM_SAMPLES]
                             [--subtract-atom-energies] [--subtract-mean]
                             [--plots] [--no-save-npz]

Evaluate PhysNetJAX checkpoint on NPZ (energies, forces, dipoles).

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        PhysNet checkpoint root (directory containing epoch-*
                        orbax runs), same as mmml physnet-md --checkpoint
  --data DATA           NPZ with R, Z, N, E, F (and optionally D / Dxyz /
                        dipole if model predicts dipoles)
  -o, --output-dir OUTPUT_DIR
                        Directory for metrics.json and optional plots
                        (default: ./physnet_evaluate_out)
  --natoms NATOMS       Padded atom count (must match training). Default:
                        inferred from NPZ Z/R width.
  --batch-size BATCH_SIZE
                        Batch size for inference (default: 16). Remainder
                        samples are skipped.
  --seed SEED           PRNG seed for batch shuffling (default: 0).
  --num-samples NUM_SAMPLES
                        If set, evaluate at most this many structures (after
                        shuffle split).
  --subtract-atom-energies
                        Subtract atomic reference energies from E (same option
                        as training data prep).
  --subtract-mean       Subtract mean energy from E (training-style).
  --plots               Write parity plots (requires matplotlib).
  --no-save-npz         Do not write predictions.npz (default: save).

Evaluate a trained PhysNet (PhysNetJAX) checkpoint on an NPZ dataset.

Runs real model inference (orbax checkpoint + EF forward), reports energy / force /
dipole errors in kcal/mol (and eV where noted), optional parity plots.

Usage:
    mmml physnet-evaluate --checkpoint out/ckpts/run --data splits/test.npz -o eval_out/
    mmml physnet-evaluate --checkpoint out/ckpts/run --data splits/test.npz \
        --natoms 64 --batch-size 32 --plots --num-samples 500
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
