# `mmml pes-design`

Bayesian physical/diverse PES subset design + validation plots.


## Usage

```bash
mmml pes-design --help
```

## Options

```text
usage: mmml pes-design [-h] --input INPUT --output OUTPUT
                       [--report-dir REPORT_DIR] --n-select N_SELECT
                       [--descriptor {pair-rdf,soap,combined}] [--cutoff CUTOFF]
                       [--rdf-bins RDF_BINS] [--type-hash-bins TYPE_HASH_BINS]
                       [--pca-components PCA_COMPONENTS]
                       [--prior-precision PRIOR_PRECISION]
                       [--uncertainty-power UNCERTAINTY_POWER]
                       [--temperatures TEMPERATURES]
                       [--min-distance MIN_DISTANCE] [--max-force MAX_FORCE]
                       [--max-relative-energy MAX_RELATIVE_ENERGY] [--seed SEED]

Select a physical, Bayesian D-optimal, descriptor-diverse PES subset and
validate it against equally sized random sampling.

Input & configuration:
  --input, -i INPUT

Scientific model:
  --cutoff CUTOFF
  --temperatures TEMPERATURES
                        Boltzmann-mixture temperatures in K
  --min-distance MIN_DISTANCE
  --max-force MAX_FORCE
                        Reject frames above max |F| (dataset units)
  --max-relative-energy MAX_RELATIVE_ENERGY
                        Reject frames above group minimum + this value

Execution:
  --seed SEED

Output & artifacts:
  --output, -o OUTPUT
  --report-dir REPORT_DIR

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --n-select N_SELECT
  --descriptor {pair-rdf,soap,combined}
  --rdf-bins RDF_BINS
  --type-hash-bins TYPE_HASH_BINS
  --pca-components PCA_COMPONENTS
  --prior-precision PRIOR_PRECISION
  --uncertainty-power UNCERTAINTY_POWER
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
