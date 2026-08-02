<p align="center">
  <img src="https://raw.githubusercontent.com/EricBoittier/mmml/main/docs/images/mmml.svg" alt="MMML" width="380">
</p>

<h1 align="center">mmml</h1>

[![CI](https://github.com/EricBoittier/mmml/workflows/CI/badge.svg)](https://github.com/EricBoittier/mmml/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/EricBoittier/mmml/branch/main/graph/badge.svg)](https://codecov.io/gh/EricBoittier/mmml/branch/main)
[![Docs](https://readthedocs.org/projects/mmml/badge/?version=latest)](https://mmml.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

**Molecular Mechanics + Machine-Learned Force-Field Toolkit**

MMML combines CHARMM/OpenMM workflows with JAX-based neural models for electrostatics and force prediction, for building and running hybrid ML/MM condensed-phase simulations.

## Status

MMML is in **alpha**. The core calculator, CLI, and `md-system` YAML workflow are usable today, but interfaces are still settling and may change without notice ahead of a first tagged release. Feedback and issues are welcome — see [Getting Help](#getting-help).

## Quick Installation

Requires **Python 3.13**. Prefer [`uv`](https://docs.astral.sh/uv/).

### Using `uv` (recommended)

```bash
git clone https://github.com/EricBoittier/mmml.git
cd mmml
uv sync

# Optional extras
uv sync --extra cli      # shell tab completion (argcomplete)
uv sync --extra md-cpu   # Vesin NL + MDAnalysis (CPU MD smokes)
uv sync --extra dev      # pytest, MkDocs, ruff
make install-gpu         # JAX CUDA 13 + CuPy (GPU nodes; SM 7.5+)
# make install-gpu-cuda12  # older GPUs / CUDA 12
```

For PyCHARMM / Packmol (native libs, not installed by `uv`):

```bash
make install-native   # builds libcharmm + packmol under setup/charmm
make doctor           # mmml doctor — env / CHARMM readiness
```

Or from a fresh clone: `make install-full` (`uv sync` + native build).

### Jupyter kernel (required for the example notebooks)

Register the project venv as its own kernel **once**, and select it in the
notebook (`Kernel → Change Kernel → mmml-venv`):

```bash
.venv/bin/python -m ipykernel install --user --name mmml-venv --display-name "mmml venv"
```

Without this, Jupyter's default `python3` kernel may start a different
interpreter: the kernelspec `uv` installs uses a bare `"python"` in its `argv`,
so it resolves against `PATH` and picks up an active conda environment instead of
`.venv`. The symptom is an immediate

```
TypeError: 'type' object is not subscriptable
```

on the first `import mmml...`, because that interpreter is too old to parse
`tuple[float, ...]` annotations. It looks like broken code but is purely kernel
selection. Check with `import sys; print(sys.executable)` — it must point inside
`.venv`.

### Using Conda / micromamba

```bash
# Conda
conda env create -f setup/environment.yml
conda activate mmml

# Or Makefile micromamba targets (preferred on clusters)
make micromamba-create
make micromamba-create-gpu        # CUDA 12 env file
make micromamba-create-gpu-cuda13
```

GPU env files: `setup/environment-gpu.yml`, `setup/environment-gpu-cuda13.yml`.

### Using Docker

Dockerfile and Compose live under [`devtools/docker/`](devtools/docker/):

```bash
cd devtools/docker
docker compose up -d mmml-cpu
docker compose exec mmml-cpu bash
# GPU: docker compose up -d mmml-gpu
```

### CLI tab completion

```bash
uv sync --extra cli
eval "$(register-python-argcomplete mmml)"
# or: eval "$(mmml completion bash)"
```

## CLI quick start

```bash
mmml -h                 # compact top-level help
mmml commands           # all subcommands by category
mmml examples           # copy-paste invocations
mmml configure          # interactive YAML / Snakemake wizard
mmml env                # checkpoints + CHARMM paths
mmml md-system --help   # condensed-phase MD flags
mmml doctor             # environment health check
```

Condensed-phase campaigns: start from
[`mmml/cli/run/md_system.example.yaml`](mmml/cli/run/md_system.example.yaml)
and the [`md-system` YAML config guide](docs/md-system-configs.md).

CPU MD smokes (no CUDA; bundled DESdimers JSON checkpoint):

```bash
make install-md-cpu
source examples/md_cpu/_env.sh
bash examples/md_cpu/run_all.sh
```

See [`examples/md_cpu/README.md`](examples/md_cpu/README.md).

## Quick Example

ML-only energy/forces via ASE using the bundled ACO/DESdimers checkpoint
(`examples/ckpts_json/DESdimers_params.json`):

```python
from pathlib import Path

import ase
import numpy as np
from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

ATOMS_PER_MONOMER = 10
N_MONOMERS = 2
Z = np.array([6, 1, 1, 1, 6, 1, 1, 1, 8, 1] * N_MONOMERS, dtype=np.int32)
R = np.zeros((ATOMS_PER_MONOMER * N_MONOMERS, 3), dtype=np.float64)
# Place monomers apart so the dimer is non-overlapping
R[ATOMS_PER_MONOMER:, 0] = 5.0

ckpt = Path("examples/ckpts_json/DESdimers_params.json")
factory = setup_calculator(
    ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
    N_MONOMERS=N_MONOMERS,
    doML=True,
    doMM=False,
    model_restart_path=str(ckpt),
    MAX_ATOMS_PER_SYSTEM=ATOMS_PER_MONOMER * N_MONOMERS,
    defer_xla_gpu_warmup=True,
    verbose=False,
)
calc, _, _ = unpack_factory_result(
    factory(atomic_numbers=Z, atomic_positions=R, n_monomers=N_MONOMERS)
)
atoms = ase.Atoms(numbers=Z, positions=R)
atoms.calc = calc
print("Energy (kcal/mol):", atoms.get_potential_energy())
```

For a geometry-aware version of the same path, run
`uv run python examples/md_cpu/02_ml_energy_ase.py`.

## Documentation

Published site: [Read the Docs](https://mmml.readthedocs.io/en/latest/).  
Serve the current MkDocs tree locally: `uv sync --extra dev && make docs-serve` → http://127.0.0.1:8000.

Highlights (in-repo):

- [Getting started](docs/getting-started.md) — install, CLI, local docs
- [CLI overview](docs/cli/index.md) — `mmml commands`, examples, tab completion
- [`md-system` YAML configs](docs/md-system-configs.md) — campaigns and condensed-phase builders
- [Calculator capability matrix](docs/calculator-capabilities.md) — calculators, hybrid assembly, LR solvers
- [PyCHARMM + MM/ML checklist](docs/md-cg-capabilities-checklist.md) — status, examples, diagrams
- [MLpot settings](docs/mlpot-settings.md) — COM handoff, medium PBC, spatial MPI

## Getting Help

- **Documentation**: [Read the Docs](https://mmml.readthedocs.io/en/latest/)
- **Issues**: [GitHub Issues](https://github.com/EricBoittier/mmml/issues)

## License

MIT License, Copyright (c) 2025, Eric Boittier. See [LICENSE](LICENSE) for details.

## Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
