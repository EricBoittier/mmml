# `mmml prepare-mm-dataset`

Assign CGenFF types/charges to a dimer NPZ (hybrid ML/MM).


## Usage

```bash
mmml prepare-mm-dataset --help
```

## Options

```text
usage: mmml prepare-mm-dataset [-h] [--config CONFIG] [-i DATA] [-o OUTPUT]
                               [--prm-path PRM_PATH] [--rtf-path RTF_PATH]
                               [--num-workers NUM_WORKERS]
                               [--max-structures MAX_STRUCTURES]
                               [--no-mm-baseline] [--strict]
                               [--save-config SAVE_CONFIG] [--quiet]

Assign CGenFF atom types / charges to a dimer training NPZ.

Input & configuration:
  --config CONFIG       YAML config seeding the flags below
  -i, --data DATA       Input dense NPZ (R/Z/N/...)
  --max-structures MAX_STRUCTURES
                        Process only the first N frames
  --save-config SAVE_CONFIG
                        Write the resolved config to this YAML path

Execution:
  --num-workers NUM_WORKERS
                        Multiprocessing pool size (1 = serial)

Output & artifacts:
  -o, --output OUTPUT   Output enriched NPZ

Diagnostics & safety:
  -h, --help            show this help message and exit
  --strict              Error on the first unassignable frame instead of
                        dropping it
  --quiet               Suppress progress output

Other options:
  --prm-path PRM_PATH   CGenFF parameter (.prm) file
  --rtf-path RTF_PATH   CGenFF topology (.rtf) file
  --no-mm-baseline      Skip the E_cgenff_mm / F_cgenff_mm inter-monomer
                        baseline
```

## Visual examples

![MM baseline decomposition](../../images/prepare-mm-dataset/mm_baseline_decomposition.png)

![Force-field assignment](../../images/prepare-mm-dataset/acodcm_assignment.png)

![Force validation](../../images/prepare-mm-dataset/force_validation.png)


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
