# `mmml compare-charmm-ml`

CHARMM PSF charges vs joint ML dipoles/ESP.


## Usage

```bash
mmml compare-charmm-ml --help
```

## Options

```text
usage: mmml compare-charmm-ml [-h] --checkpoint CHECKPOINT --valid-efd VALID_EFD
                              --valid-esp VALID_ESP --pdb PDB
                              [--n-samples N_SAMPLES] [--out-dir OUT_DIR]
                              [--cutoff CUTOFF] [--subtract-atom-energies]

Compare CHARMM vs ML dipoles and ESPs

Input & configuration:
  --checkpoint CHECKPOINT
                        Path to train_joint checkpoint (dir with best_params.pkl
                        or path to best_params.pkl)
  --pdb PDB             PDB path for CHARMM setup (same molecule as validation
                        data)

Scientific model:
  --cutoff CUTOFF       Cutoff for prepare_batch_data

Output & artifacts:
  --out-dir OUT_DIR     Output directory for plots

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --valid-efd VALID_EFD
                        Validation energies/forces/dipoles NPZ file
  --valid-esp VALID_ESP
                        Validation ESP grids NPZ file
  --n-samples N_SAMPLES
                        Number of validation samples to evaluate
  --subtract-atom-energies
                        Subtract reference atomic energies (match training)
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
