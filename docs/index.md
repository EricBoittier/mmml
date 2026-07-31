# MMML

Molecular mechanics workflows and machine-learned force fields, on JAX.
Everything runs through one command: `mmml`.

<!-- MMML_TOP_HELP_START -->

```console

$ mmml -h

usage: mmml [-h] <command> ...


MMML: Machine Learning for Molecular Modeling


Subcommands (63 total). Common:

  md-system      mixed-composition MD (YAML + campaigns)

  physnet-train  train PhysNetJAX from NPZ

  configure      interactive config / Snakemake wizard

  env            find resolved/bundled checkpoints and CHARMM paths

  liquid-box     build periodic liquid boxes


Browse:   mmml commands

Setup:    mmml configure

Examples: mmml examples

Flags:    mmml <command> --help


Tab completion (bash/zsh/fish):

  pip install 'mmml[cli]'

  eval "$(register-python-argcomplete mmml)"


options:

  -h, --help  show this help message and exit

```

<!-- MMML_TOP_HELP_END -->

These docs are laid out the same way. The sections along the top are the task
groups from `mmml commands`, and each one holds its guides next to the reference
page for every command in that group.

## Start here

<div class="grid cards" markdown>

-   __Install & first run__

    Set up with `uv`, check the machine with `mmml doctor`, run something small.

    [→ Getting started](getting-started.md)

-   __How the CLI is organized__

    The four help layers — `-h`, `commands`, `examples`, `<cmd> --help` — and
    what each is for.

    [→ CLI overview](cli/index.md)

-   __Examples__

    Copy-paste invocations, mirroring `mmml examples`.

    [→ Examples](examples.md)

-   __Tab completion__

    Per-shell setup for bash, zsh, and fish.

    [→ Completion](cli/completion.md)

</div>

## Browse by task

| Section | Covers | Commands |
|---|---|---|
| [Structure & boxes](packmol-placement.md) | residues, packing, crystals, liquid boxes | `make-res`, `make-box`, `build-crystal`, `liquid-box` |
| [MD & campaigns](md-system-configs.md) | mixed MM/ML dynamics, umbrella sampling, cluster runs | `md-system`, `md-embedding`, `umbrella-sample`, `health-check` |
| [QM & data](qc-cross-check.md) | reference calculations, scans, dataset prep | `pyscf-dft`, `dimer-scan`, `ic-scan`, `fix-and-split` |
| [ML training & MD](hybrid-potential-regions.md) | PhysNet / EF / KerNN training, hybrid potentials, sampling | `physnet-train`, `neb`, `dmc`, `efield-train` |
| [Environment & clusters](scicore.md) | checkpoints, MPI, threading, SciCORE | `env`, `configure`, `doctor`, `mpi-launch` |

## Reference & policy

- [Package architecture](package-architecture.md) — module layout and import graph
- [Calculator capability matrix](calculator-capabilities.md) — what each calculator supports
- [Units summary](UNITS_SUMMARY.md) — conventions and conversions
- [API reference](api.md) — generated from docstrings
- [Scientific code policy](scientific-code.md) — reproducibility, provenance, review checklist
- [Contributor guide](development.md) — tests, linting, docs builds

Design notes, audits, tool inventories, and results reports live under
**Internals & reports**. They are engineering records, not user guides.

## Elsewhere

- [Repository](https://github.com/EricBoittier/mmml)
- [Issue tracker](https://github.com/EricBoittier/mmml/issues)
