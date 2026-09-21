# `mmml label-acquire`

Select structures for expensive labels (activation / Jacobian / teacher-gradient).


## Usage

```bash
mmml label-acquire --help
```

## Options

```text
usage: mmml label-acquire [-h] --config CONFIG [--output OUTPUT]
                          [{prepare-pool,fingerprint-models,extract,fit-pca,select,label,train-eval,report,all}]

Compare structure-selection methods for acquiring expensive reference labels.
Teacher potentials are cheap surrogates, not ground truth. Use the Snakemake
workflow for the full DAG.

positional arguments:
  {prepare-pool,fingerprint-models,extract,fit-pca,select,label,train-eval,report,all}
                        Pipeline stage (default: all)

options:
  -h, --help            show this help message and exit
  --config, -c CONFIG   YAML config
  --output, -o OUTPUT   Override output_root from the config
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
