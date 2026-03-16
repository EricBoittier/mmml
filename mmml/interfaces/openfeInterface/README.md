# OpenFE Interface

Absolute binding free energy (ABFE) calculations via [OpenFE](https://openfree.energy/).

## Installation

**OpenFE is distributed via conda-forge only** (not PyPI). The `openfe`, `gufe`, and `openff-units` packages cannot be installed with pip.

### Recommended: micromamba / mamba / conda

```bash
# Create a dedicated environment with OpenFE
micromamba create -c conda-forge -n openfe openfe=1.9
micromamba activate openfe
```

To use with mmml, either:

1. **Install mmml into the openfe env** (if Python versions are compatible):
   ```bash
   micromamba activate openfe
   pip install -e /path/to/mmml
   ```

2. **Run the ABFE script from the openfe env** (standalone):
   ```bash
   micromamba activate openfe
   python mmml/interfaces/openfeInterface/abfe_script.py --sdf ligand.sdf --pdb protein.pdb
   ```

### Reproducible: conda-lock

```bash
curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/latest/download/openfe-conda-lock.yml
micromamba create -n openfe --file openfe-conda-lock.yml
micromamba activate openfe
```

### Single-file installer (Linux x86_64)

See [OpenFE installation docs](https://docs.openfree.energy/en/stable/installation.html) for the standalone installer.

## Python version note

mmml requires Python 3.13. OpenFE from conda-forge typically uses Python 3.11 or 3.12. To use both, run the ABFE interface from a separate OpenFE conda environment, or install mmml into that environment if you can relax the Python constraint.
