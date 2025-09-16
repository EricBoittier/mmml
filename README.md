mmml
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/EricBoittier/mmml/workflows/CI/badge.svg)](https://github.com/EricBoittier/mmml/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/EricBoittier/mmml/branch/main/graph/badge.svg)](https://codecov.io/gh/EricBoittier/mmml/branch/main)

### Overview

- **mmml** is a molecular mechanics + machine-learned force-field toolkit that combines CHARMM/OpenMM
  workflows with JAX-based neural models for electrostatics and force prediction.

- Key components:
  - Docs and tutorials: `docs/`
  - Examples: `examples/`
  - Workflows and scripts: `bin/`, `scripts/`
  - Tests: `tests/`


### Quickstart

Prereqs: CUDA 12 toolchain and a working CHARMM/OpenMM stack (on clusters: `module load cudnn` and `module load charmm`).

Install and run with `uv` (no manual venv needed):

```bash
# From repo root
uv run python -c "print('mmml quickstart OK')"
```

Run tests (will auto-skip heavy/optional tests if deps or data are missing):

```bash
# Minimal constant test
uv run -m pytest -q tests/functionality/mmml/test_mmml_calc.py::test_ev2kcalmol_constant

# Optional data-driven test (set paths if you have them)
export MMML_DATA=/home/ericb/mmml/mmml/data/fixed-acetone-only_MP2_21000.npz
export MMML_CKPT=/home/ericb/mmml/mmml/physnetjax/ckpts
uv run -m pytest -q tests/functionality/mmml/test_mmml_calc.py::test_ml_energy_matches_reference_when_data_available
```

Install project dependencies permanently (optional):

```bash
uv sync
```

If you prefer the legacy installer:

```bash
bash setup/install.sh
```

### Using the calculator (minimal example)

```python
import numpy as np
from pathlib import Path
from mmml.pycharmmInterface.mmml_calculator import setup_calculator, ev2kcalmol
import ase

# Two monomers with 10 atoms each (toy positions and atomic numbers)
ATOMS_PER_MONOMER = 10
N_MONOMERS = 2
Z = np.array([6, 1, 1, 1, 6, 1, 1, 1, 8, 1] * N_MONOMERS, dtype=int)
R = np.zeros((ATOMS_PER_MONOMER * N_MONOMERS, 3), dtype=float)

# Point to your checkpoints if available (tests skip otherwise)
ckpt = Path("mmml/physnetjax/ckpts")

factory = setup_calculator(
    ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
    N_MONOMERS=N_MONOMERS,
    doML=True,
    doMM=False,
    model_restart_path=ckpt,
    MAX_ATOMS_PER_SYSTEM=ATOMS_PER_MONOMER * N_MONOMERS,
    ml_energy_conversion_factor=ev2kcalmol,
    ml_force_conversion_factor=ev2kcalmol,
)

calc, _ = factory(atomic_numbers=Z, atomic_positions=R, n_monomers=N_MONOMERS)
atoms = ase.Atoms(Z, R)
atoms.calc = calc
print("Energy (kcal/mol):", atoms.get_potential_energy())
print("Forces (kcal/mol/Å):", atoms.get_forces().shape)
```

### Documentation

- See `docs/` for Sphinx sources and tutorials. If hosted, start at the project docs home.

### Notes

- Paths like `MMML_DATA` and `MMML_CKPT` are optional environment variables used by some tests and
  examples. When unset, related tests are skipped.
- For development, see packaging/CI metadata in `pyproject.toml`, `setup.cfg`, and GitHub workflows under
  `.github/`.

### Copyright

Copyright (c) 2025, Eric Boittier


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
