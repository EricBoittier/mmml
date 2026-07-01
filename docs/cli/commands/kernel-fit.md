# `mmml kernel-fit`

Kernel fitting utilities.


## Usage

```bash
mmml kernel-fit --help
```

## Options

```text
usage: mmml kernel-fit [-h] [--out-dir OUT_DIR] [--natmk NATMK]
                       [--out-h5 OUT_H5] [--out-mdcm OUT_MDCM]
                       [--out-kmdcm OUT_KMDCM] [--residue-name RESIDUE_NAME]
                       [--nkfr NKFR] [--optimize]
                       [--train-frames TRAIN_FRAMES] [--lam LAM]
                       [--sigma SIGMA] [--base-name BASE_NAME]
                       h5

Fit kernel ridge: distance matrix -> (AQ,BQ,CQ). Write CHARMM kernel files and
optional H5.

positional arguments:
  h5                    charmm_ml_comparison.h5 or similar

options:
  -h, --help            show this help message and exit
  --out-dir, -o OUT_DIR
                        Directory for x_fit.txt, coefs*.txt
  --natmk NATMK         Number of atoms for distance matrix (default: from H5)
  --out-h5 OUT_H5       If set, evaluate and write H5 for GUI
  --out-mdcm OUT_MDCM   Write .mdcm file (default: out_dir/RESIDUE.mdcm)
  --out-kmdcm OUT_KMDCM
                        Write .kmdcm kernel file (default:
                        out_dir/RESIDUE.kmdcm)
  --residue-name RESIDUE_NAME
                        Residue name for mdcm header and default filenames
  --nkfr NKFR           NKFR for kmdcm (default: number of frames)
  --optimize            Optimize (AQ,BQ,CQ) per frame before fitting
  --train-frames TRAIN_FRAMES
                        Comma-separated frame indices for training (default:
                        all)
  --lam LAM             Kernel ridge regularization
  --sigma SIGMA         RBF kernel width
  --base-name BASE_NAME
                        Prefix for x_fit/coefs filenames
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
