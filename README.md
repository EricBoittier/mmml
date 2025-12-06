# mmml

[![GitHub Actions Build Status](https://github.com/EricBoittier/mmml/workflows/CI/badge.svg)](https://github.com/EricBoittier/mmml/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/EricBoittier/mmml/branch/main/graph/badge.svg)](https://codecov.io/gh/EricBoittier/mmml/branch/main)

**Molecular Mechanics + Machine-Learned Force-Field Toolkit**

MMML combines CHARMM/OpenMM workflows with JAX-based neural models for electrostatics and force prediction.

## 📚 Documentation

**For complete documentation, tutorials, and guides, please visit:**

**[Read the Docs](https://mmml.readthedocs.io/en/latest/)**

The documentation includes:
- Installation instructions
- Quick start guides
- API reference
- Tutorials and examples
- Troubleshooting guides

## Quick Installation

### Using `uv` (Recommended)

```bash
git clone https://github.com/EricBoittier/mmml.git
cd mmml
uv sync

# For GPU support
make install-gpu
```

### Using Conda

```bash
conda env create -f environment.yml
conda activate mmml

# For GPU support
conda env create -f environment-gpu.yml
conda activate mmml-gpu
```

### Using Docker

```bash
docker-compose up -d mmml-cpu
docker-compose exec mmml-cpu bash
```

## Quick Example

```python
import numpy as np
from pathlib import Path
from mmml.pycharmmInterface.mmml_calculator import setup_calculator, ev2kcalmol
import ase

ATOMS_PER_MONOMER = 10
N_MONOMERS = 2
Z = np.array([6, 1, 1, 1, 6, 1, 1, 1, 8, 1] * N_MONOMERS, dtype=int)
R = np.zeros((ATOMS_PER_MONOMER * N_MONOMERS, 3), dtype=float)

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
```

## Getting Help

- **Full Documentation**: [Read the Docs](https://mmml.readthedocs.io/en/latest/)
- **Issues**: [GitHub Issues](https://github.com/EricBoittier/mmml/issues)

## License

Copyright (c) 2025, Eric Boittier

## Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
