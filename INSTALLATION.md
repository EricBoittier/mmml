# MMML Installation Guide

## Quick Start

### Minimal Installation (Core Training)

For basic training with PhysNetJax:
```bash
pip install -e .
```

This includes all core dependencies needed for:
- ✅ Model training
- ✅ Loading/saving checkpoints
- ✅ Making predictions
- ✅ Basic progress tables
- ✅ Data preprocessing

## Optional Features

### Plotting Support

For ASCII chart plotting in training progress:
```bash
pip install -e ".[plotting]"
```

Includes:
- `asciichartpy` - ASCII charts
- `polars` - DataFrame operations

### TensorBoard Logging

For TensorBoard integration and log analysis:
```bash
pip install -e ".[tensorboard]"
```

Includes:
- `tensorboard` - Event logging
- `tensorflow` - Log reading
- `polars` - Log processing

### GPU Support

For CUDA-accelerated training:
```bash
pip install -e ".[gpu]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Quantum Chemistry

For quantum chemistry calculations (PySCF):
```bash
pip install -e ".[quantum]"
```

For GPU-accelerated quantum chemistry:
```bash
pip install -e ".[quantum-gpu]"
```

### Other Optional Groups

**Machine Learning extras**:
```bash
pip install -e ".[ml]"
```

**Visualization**:
```bash
pip install -e ".[viz]"
```

**Molecular Dynamics**:
```bash
pip install -e ".[md]"
```

**Data Processing**:
```bash
pip install -e ".[data]"
```

**Jupyter Notebooks**:
```bash
pip install -e ".[notebooks]"
```

**Experiment Tracking**:
```bash
pip install -e ".[experiments]"
```

## Combined Installations

### Common Combinations

**Training with plotting and TensorBoard**:
```bash
pip install -e ".[plotting,tensorboard]"
```

**GPU training with plotting**:
```bash
pip install -e ".[gpu,plotting,tensorboard]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Complete quantum chemistry setup**:
```bash
pip install -e ".[quantum,gpu,plotting]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Full Installations

**Everything (including GPU)**:
```bash
pip install -e ".[all]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Everything (CPU only)**:
```bash
pip install -e ".[all-cpu]"
```

## Checking What's Installed

```python
# Check optional dependencies
from mmml.physnetjax.physnetjax.utils.pretty_printer import (
    HAS_ASCIICHARTPY, 
    HAS_POLARS
)
from mmml.physnetjax.physnetjax.logger.tensorboard_interface import (
    HAS_TENSORBOARD,
    HAS_TENSORFLOW
)

print(f"Plotting: asciichartpy={HAS_ASCIICHARTPY}, polars={HAS_POLARS}")
print(f"TensorBoard: tensorboard={HAS_TENSORBOARD}, tensorflow={HAS_TENSORFLOW}")
```

## Verification

Test your installation:
```bash
python -c "
from mmml.physnetjax.physnetjax.models.model import EF
from mmml.data import DataConfig

model = EF(features=64, natoms=3)
config = DataConfig()
print('✅ MMML core installation successful!')
"
```

## Troubleshooting

### Import Errors

**"No module named 'asciichartpy'"**
```bash
pip install -e ".[plotting]"
```

**"No module named 'tensorboard'" or "'tensorflow'"**
```bash
pip install -e ".[tensorboard]"
```

**"No module named 'polars'"**
```bash
# For plotting support
pip install -e ".[plotting]"
# OR for data processing
pip install -e ".[data]"
```

### GPU Issues

**JAX not finding CUDA**:
```bash
# Reinstall with proper index
pip uninstall jax jaxlib
pip install jax[cuda12] jaxlib[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Check CUDA version**:
```bash
nvidia-smi
```

### Python Version

MMML requires Python 3.11 or 3.12:
```bash
python --version
```

If using conda:
```bash
conda create -n mmml python=3.12
conda activate mmml
pip install -e ".[plotting,tensorboard]"
```

## Development Installation

For development with testing:
```bash
pip install -e ".[dev,plotting,tensorboard]"
```

## Using uv (Fast Alternative)

If you have `uv` installed:
```bash
# Basic installation
uv pip install -e .

# With optional features
uv pip install -e ".[plotting,tensorboard]"

# Full installation
uv pip install -e ".[all-cpu]"
```

## Environment Examples

### Basic Training Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Full Research Environment
```bash
python -m venv venv-full
source venv-full/bin/activate
pip install -e ".[all-cpu]"
```

### GPU Training Environment
```bash
python -m venv venv-gpu
source venv-gpu/bin/activate
pip install -e ".[gpu,plotting,tensorboard]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Summary of Optional Dependency Groups

| Group | Purpose | Key Packages |
|-------|---------|--------------|
| `plotting` | Training progress visualization | asciichartpy, polars |
| `tensorboard` | TensorBoard logging/analysis | tensorboard, tensorflow, polars |
| `gpu` | CUDA acceleration | jax[cuda12], cupy |
| `quantum` | Quantum chemistry | pyscf, basis-set-exchange |
| `quantum-gpu` | GPU quantum chemistry | gpu4pyscf |
| `ml` | Extra ML tools | torch, e3nn-jax |
| `viz` | Advanced visualization | plotly, seaborn |
| `md` | Molecular dynamics | mdanalysis |
| `data` | Data processing | polars, pyarrow |
| `notebooks` | Jupyter support | ipykernel, jupyter |
| `experiments` | Experiment tracking | wandb, optuna |
| `dev` | Development/testing | pytest |
| `all` | Everything + GPU | All above |
| `all-cpu` | Everything (no GPU) | All except GPU |

## Updating

To update MMML and dependencies:
```bash
pip install -e . --upgrade
```

To update optional dependencies:
```bash
pip install -e ".[plotting,tensorboard]" --upgrade
```

## Uninstalling

```bash
pip uninstall mmml
```

Note: This only uninstalls MMML, not its dependencies. To remove everything:
```bash
# Remove the virtual environment entirely
deactivate
rm -rf venv/  # or venv-full/, venv-gpu/, etc.
```

