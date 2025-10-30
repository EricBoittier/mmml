mmml
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/EricBoittier/mmml/workflows/CI/badge.svg)](https://github.com/EricBoittier/mmml/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/EricBoittier/mmml/branch/main/graph/badge.svg)](https://codecov.io/gh/EricBoittier/mmml/branch/main)


[Read the Docs](https://mmml.readthedocs.io/en/latest/)

### Overview

- **mmml** is a molecular mechanics + machine-learned force-field toolkit that combines CHARMM/OpenMM
  workflows with JAX-based neural models for electrostatics and force prediction.

- Key components:
  - Docs and tutorials: `docs/`
  - Examples: `examples/`
  - Workflows and scripts: `bin/`, `scripts/`
  - Tests: `tests/`


### Installation

> 🚀 **New to MMML?** Check out the [QUICKSTART.md](QUICKSTART.md) guide to get running in 5 minutes!
>
> 🎮 **Installing with GPU?** See [GPU_INSTALL.md](GPU_INSTALL.md) for GPU-specific installation and troubleshooting.
>
> 📚 **Need detailed help?** See [INSTALL.md](INSTALL.md) for comprehensive installation instructions, troubleshooting, and platform-specific guidance.

MMML can be installed in three different ways depending on your needs:

#### Option 1: Using `uv` (Recommended for Development)

Fastest installation using the modern `uv` package manager. Requires CUDA 12 toolchain for GPU support.

```bash
# Clone the repository
git clone https://github.com/EricBoittier/mmml.git
cd mmml

# Run without installing (uv manages everything)
uv run python -c "print('mmml quickstart OK')"

# Or install dependencies permanently
uv sync

# For GPU support (CUDA 12), use the make command which handles JAX properly
make install-gpu

# Or manually:
uv pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12]" "jaxlib[cuda12]"
uv sync --extra gpu

# For full installation with all optional features
uv sync --extra all
```

**Run tests:**
```bash
# Minimal constant test
uv run -m pytest -q tests/functionality/mmml/test_mmml_calc.py::test_ev2kcalmol_constant

# Optional data-driven test (set paths if you have them)
export MMML_DATA=/path/to/mmml/data/fixed-acetone-only_MP2_21000.npz
export MMML_CKPT=/path/to/mmml/physnetjax/ckpts
uv run -m pytest -q tests/functionality/mmml/test_mmml_calc.py::test_ml_energy_matches_reference_when_data_available
```

#### Option 2: Using Conda (Recommended for HPC/Cluster)

Conda provides better control over system libraries and works well on HPC clusters.

**CPU-only installation:**
```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate mmml
```

**GPU installation (CUDA 12):**
```bash
# Create and activate the GPU environment
conda env create -f environment-gpu.yml
conda activate mmml-gpu
```

**Full installation with all features:**
```bash
# Create environment with all optional dependencies
conda env create -f environment-full.yml
conda activate mmml-full
```

**Installing optional features individually:**
```bash
# After creating the base environment, you can install optional groups
conda activate mmml

# For GPU support, install JAX with CUDA first
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12]" "jaxlib[cuda12]"
pip install -e ".[gpu]"          # GPU support

# Other optional features (no special installation needed)
pip install -e ".[quantum]"      # Quantum chemistry
pip install -e ".[ml]"           # Extra ML tools
pip install -e ".[viz]"          # Visualization tools
pip install -e ".[all]"          # Everything
```

#### Option 3: Using Docker (Recommended for Reproducibility)

Docker provides isolated, reproducible environments.

**CPU version:**
```bash
# Build and run CPU container
docker build --target runtime-cpu -t mmml:cpu .
docker run -it --rm -v $(pwd):/workspace/mmml mmml:cpu
```

**GPU version (requires NVIDIA Docker runtime):**
```bash
# Build and run GPU container
docker build --target runtime-gpu -t mmml:gpu .
docker run -it --rm --gpus all -v $(pwd):/workspace/mmml mmml:gpu
```

**Using Docker Compose (easier):**
```bash
# CPU version
docker-compose up -d mmml-cpu
docker-compose exec mmml-cpu bash

# GPU version
docker-compose up -d mmml-gpu
docker-compose exec mmml-gpu bash

# Jupyter Lab with GPU support
docker-compose up -d mmml-jupyter
# Then open http://localhost:8888 in your browser
```

#### Post-Installation Setup

Set up CHARMM environment variables (if not using Docker):
```bash
# Source the CHARMM setup file
source CHARMMSETUP

# Or manually set the variables
export CHARMM_HOME=/path/to/mmml/setup/charmm
export CHARMM_LIB_DIR=/path/to/mmml/setup/charmm
```

Run the CHARMM setup script (if needed):
```bash
bash setup/install.sh
```

### Optional Dependency Groups

The package has been reorganized with modular optional dependencies. Install only what you need:

| Group | Description | Install with |
|-------|-------------|--------------|
| `gpu` | CUDA 12 support for JAX | `pip install -e ".[gpu]"` |
| `quantum` | Quantum chemistry (PySCF) | `pip install -e ".[quantum]"` |
| `quantum-gpu` | GPU-accelerated quantum chemistry | `pip install -e ".[quantum-gpu]"` |
| `ml` | Extra ML tools (PyTorch, TorchANI, E3NN) | `pip install -e ".[ml]"` |
| `viz` | Visualization tools (Plotly, Seaborn, etc.) | `pip install -e ".[viz]"` |
| `md` | Molecular dynamics analysis (MDAnalysis) | `pip install -e ".[md]"` |
| `chem` | Additional chemistry tools | `pip install -e ".[chem]"` |
| `data` | Data processing (Polars, Parquet) | `pip install -e ".[data]"` |
| `notebooks` | Jupyter notebook support | `pip install -e ".[notebooks]"` |
| `experiments` | Experiment tracking (W&B, Optuna) | `pip install -e ".[experiments]"` |
| `dev` | Development and testing tools | `pip install -e ".[dev]"` |
| `charmm-interface` | Bundled CHARMM Python bindings (requires compiled CHARMM shared library) | `pip install -e ".[charmm-interface]"` |
| `all` | Everything including GPU | `pip install -e ".[all]"` |
| `all-cpu` | Everything except GPU | `pip install -e ".[all-cpu]"` |

You can combine multiple groups:
```bash
# For GPU, always install JAX with CUDA first
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12]" "jaxlib[cuda12]"
pip install -e ".[gpu,quantum,viz,notebooks]"

# Or use the Makefile which handles this automatically
make install-gpu
```

> ⚠️ **Using the CHARMM interface?**  
> Build CHARMM as a shared library (e.g. `bash setup/install.sh` or your preferred workflow) and export `CHARMM_HOME` / `CHARMM_LIB_DIR` so the bundled `pycharmm` package can locate `libcharmm`.  
> The `pycharmm` sources are included under `pycharmm/LICENSE` (GPLv3).

### Quick Commands with Make

For convenience, common tasks are available via `make`:

```bash
make install          # Install with uv
make install-gpu      # Install with GPU support
make conda-create-gpu # Create conda environment with GPU
make docker-build-gpu # Build GPU Docker image
make test             # Run tests
make help             # Show all available commands
```

## FAQs

Common errors are not having CHARMM_HOME and CHARMM_LIB_DIR set. source CHARMMSETUP

```bash
>>> import mmml 
>>> from mmml.pycharmmInterface import import_pycharmm
/pchem-data/meuwly/boittier/home/mmml/mmml/data/top_all36_cgenff.rtf
/pchem-data/meuwly/boittier/home/mmml/mmml/data/par_all36_cgenff.prm
CHARMM_HOME /pchem-data/meuwly/boittier/home/mmml/setup/charmm
CHARMM_LIB_DIR /pchem-data/meuwly/boittier/home/mmml/setup/charmm
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/pchem-data/meuwly/boittier/home/mmml/mmml/pycharmmInterface/import_pycharmm.py", line 48, in <module>
    import pycharmm
  File "/pchem-data/meuwly/boittier/home/mmml/.venv/lib/python3.12/site-packages/pycharmm/__init__.py", line 17, in <module>
    from .atom_info import get_atom_table
  File "/pchem-data/meuwly/boittier/home/mmml/.venv/lib/python3.12/site-packages/pycharmm/atom_info.py", line 24, in <module>
    import pycharmm.coor as coor
  File "/pchem-data/meuwly/boittier/home/mmml/.venv/lib/python3.12/site-packages/pycharmm/coor.py", line 31, in <module>
    import pycharmm.lib as lib  
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/pchem-data/meuwly/boittier/home/mmml/.venv/lib/python3.12/site-packages/pycharmm/lib.py", line 67, in <module>
    charmm_lib = CharmmLib(os.environ.get('CHARMM_LIB_DIR', ''))
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/pchem-data/meuwly/boittier/home/mmml/.venv/lib/python3.12/site-packages/pycharmm/lib.py", line 48, in __init__
    self.init_charmm()
  File "/pchem-data/meuwly/boittier/home/mmml/.venv/lib/python3.12/site-packages/pycharmm/lib.py", line 58, in init_charmm
    self.lib = ctypes.CDLL(self.charmm_lib_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/pchem-data/meuwly/boittier/home/.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/python3.12/ctypes/__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /pchem-data/meuwly/boittier/home/mmml/setup/charmm/libcharmm.so)
>>> exit()
Exception ignored in: <function CharmmLib.__del__ at 0x149de9f237e0>
Traceback (most recent call last):
  File "/pchem-data/meuwly/boittier/home/mmml/.venv/lib/python3.12/site-packages/pycharmm/lib.py", line 54, in __del__
    self.del_charmm()
  File "/pchem-data/meuwly/boittier/home/mmml/.venv/lib/python3.12/site-packages/pycharmm/lib.py", line 62, in del_charmm
    self.lib.del_charmm()  # initiates 'normal stop'
    ^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'del_charmm'
(mmml) (base) boittier@gpu23:~/mmml$ module load gcc
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


MMML_DATA=~/mmml/mmml/data/fixed-acetone-only_MP2_21000.npz MMML_CKPT=~/mmml/mmml/physnetjax/ckpts/test-70821ae3-d06a-4c87-9a2b-f5889c298376/ python demo.py --mm
