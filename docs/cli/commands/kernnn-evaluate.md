# `mmml kernnn-evaluate`

Evaluate KerNN checkpoint.


## Usage

```bash
mmml kernnn-evaluate --help
```

## Options

```text
usage: mmml kernnn-evaluate [-h] [--checkpoint CHECKPOINT] [--data DATA]
                            [--output-dir OUTPUT_DIR]
                            [--split {train,valid,test,all}] [--seed SEED]
                            [--ntrain NTRAIN] [--nvalid NVALID]
                            [--batch-size BATCH_SIZE] [--split-json SPLIT_JSON]

Evaluate KerNN checkpoint (E/F metrics)

Input & configuration:
  --checkpoint CHECKPOINT
  --data DATA           NPZ with R, E, F

Execution:
  --seed SEED
  --batch-size BATCH_SIZE

Output & artifacts:
  --output-dir OUTPUT_DIR
  --split-json SPLIT_JSON
                        Optional data_split.json from training (overrides
                        seed/ntrain/nvalid)

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --split {train,valid,test,all}
                        Which split to evaluate (seed/ntrain/nvalid define the
                        split)
  --ntrain NTRAIN
  --nvalid NVALID
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
