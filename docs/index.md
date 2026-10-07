<div class="karml-hero" markdown>
![KARML](images/karml.svg)
</div>

# KARML

Molecular mechanics workflows and machine-learned force fields, on JAX.
Everything runs through one command: `karml`.

<!-- KARML_TOP_HELP_START -->
```console
$ karml -h
usage: karml [-h] <command> ...

KARML: Machine Learning for Molecular Modeling

Subcommands (71 total). Common:
  md-system      mixed-composition MD (YAML + campaigns)
  physnet-train  train PhysNetJAX from NPZ
  configure      interactive config / Snakemake wizard
  env            find resolved/bundled checkpoints and CHARMM paths
  liquid-box     build periodic liquid boxes

Browse:   karml commands
Setup:    karml configure
Examples: karml examples
Flags:    karml <command> --help

Tab completion (bash/zsh/fish):
  pip install 'karml[cli]'
  eval "$(register-python-argcomplete karml)"

options:
  -h, --help  show this help message and exit
```
<!-- KARML_TOP_HELP_END -->

These docs are laid out the same way. The sections along the top are the task
groups from `karml commands`, and each one holds its guides next to the reference
page for every command in that group.

## Start here

<div class="grid cards" markdown>

-   __Install & first run__

    Set up with `uv`, check the machine with `karml doctor`, run something small.

    [→ Getting started](getting-started.md)

-   __How the CLI is organized__

    The four help layers — `-h`, `commands`, `examples`, `<cmd> --help` — and
    what each is for.

    [→ CLI overview](cli/index.md)

-   __Examples__

    Copy-paste invocations, mirroring `karml examples`.

    [→ Examples](examples.md)

-   __Tab completion__

    Per-shell setup for bash, zsh, and fish.

    [→ Completion](cli/completion.md)

</div>

## Browse by task

| Section | Covers | Commands |
|---|---|---|
| [Structure & boxes](sections/structure-boxes.md) | residues, packing, crystals, liquid boxes | `make-res`, `make-box`, `build-crystal`, `liquid-box` |
| [MD & campaigns](sections/md-campaigns.md) | mixed MM/ML dynamics, umbrella sampling, cluster runs | `md-system`, `md-embedding`, `umbrella-sample`, `health-check` |
| [QM & data](sections/qm-data.md) | reference calculations, scans, dataset prep | `pyscf-dft`, `dimer-scan`, `ic-scan`, `fix-and-split` |
| [Hybrid ML/MM potentials](sections/hybrid-potentials.md) | regions, charges, LJ scales, long-range solvers | — (assembly is configured, not commanded) |
| [Training & sampling](sections/training-sampling.md) | PhysNet / EF / KerNN training, NEB, DMC | `physnet-train`, `neb`, `dmc`, `efield-train` |
| [Environment and HPCs](sections/environment-clusters.md) | checkpoints, MPI, threading, SciCORE, profiling | `env`, `configure`, `doctor`, `mpi-launch` |

Each section opens with what the area covers and links into the guides.

## Reference & policy

- [Package architecture](package-architecture.md) — module layout and import graph
- [Calculator capability matrix](calculator-capabilities.md) — what each calculator supports
- [Units summary](UNITS_SUMMARY.md) — conventions and conversions
- [Training NPZ contract](training-npz-contract.md) — keys, train units, total vs interaction
- [API reference](api.md) — generated from docstrings
- [Scientific code policy](scientific-code.md) — reproducibility, provenance, review checklist
- [Contributor guide](development.md) — tests, linting, docs builds

Design notes, audits, tool inventories, and results reports live under
**Internals & reports**. They are engineering records, not user guides.

## Elsewhere

- [Repository](https://github.com/EricBoittier/karml)
- [Issue tracker](https://github.com/EricBoittier/karml/issues)

## CDO: Mr Connor Brandes (Chief Dog Officer)

![CDO: Mr Connor Brandes (Chief Dog Officer)](images/staff/connor.jpeg){ width="320" }
