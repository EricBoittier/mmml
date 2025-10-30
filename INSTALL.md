# MMML Installation Guide

This guide provides detailed installation instructions for MMML across different platforms and use cases.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation Methods](#installation-methods)
   - [Using uv](#using-uv-recommended-for-development)
   - [Using Conda](#using-conda-recommended-for-hpc)
   - [Using Docker](#using-docker-recommended-for-reproducibility)
3. [Dependency Groups](#dependency-groups)
4. [Platform-Specific Instructions](#platform-specific-instructions)
5. [Troubleshooting](#troubleshooting)

## Quick Start

### For Development (Local Machine)
```bash
git clone https://github.com/EricBoittier/mmml.git
cd mmml
uv sync --extra all
```

### For HPC/Cluster with GPU
```bash
conda env create -f environment-gpu.yml
conda activate mmml-gpu
```

### For Docker
```bash
docker-compose up -d mmml-gpu
docker-compose exec mmml-gpu bash
```

## Installation Methods

### Using `uv` (Recommended for Development)

`uv` is a fast Python package manager that simplifies dependency management.

#### Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and install MMML**:
```bash
git clone https://github.com/EricBoittier/mmml.git
cd mmml
uv sync                    # CPU version
uv sync --extra gpu        # With GPU support
uv sync --extra all        # Everything
```

3. **Run commands**:
```bash
# Without permanent installation
uv run python script.py

# After uv sync, activate the virtual environment
source .venv/bin/activate
python script.py
```

#### Pros
- ✅ Extremely fast dependency resolution
- ✅ Automatic virtual environment management
- ✅ Great for development
- ✅ Lock file for reproducibility

#### Cons
- ❌ Less common in HPC environments
- ❌ Requires modern Python (3.11+)

### Using Conda (Recommended for HPC)

Conda is widely used in scientific computing and HPC clusters.

#### Installation

1. **Install Conda/Mamba** (if not already installed):
```bash
# Miniconda (recommended)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or use mamba for faster dependency resolution
conda install mamba -n base -c conda-forge
```

2. **Choose your environment**:

**CPU-only** (minimal installation):
```bash
conda env create -f environment.yml
conda activate mmml
```

**GPU** (CUDA 12):
```bash
conda env create -f environment-gpu.yml
conda activate mmml-gpu
```

**Full** (all features):
```bash
conda env create -f environment-full.yml
conda activate mmml-full
```

3. **Verify installation**:
```bash
python -c "import mmml; print('MMML installed successfully')"
```

#### Adding Optional Dependencies

After creating the base environment, you can add optional features:
```bash
conda activate mmml
pip install -e ".[quantum]"      # Add quantum chemistry
pip install -e ".[viz]"          # Add visualization tools
pip install -e ".[experiments]"  # Add W&B and Optuna
```

#### Pros
- ✅ Excellent for HPC and cluster environments
- ✅ Better control over system libraries
- ✅ Works well with module systems
- ✅ Handles binary dependencies well

#### Cons
- ❌ Slower dependency resolution
- ❌ Larger environment size
- ❌ Can have conflicts between conda and pip packages

### Using Docker (Recommended for Reproducibility)

Docker provides isolated, reproducible environments that work identically across all systems.

#### Prerequisites

Install Docker and (for GPU support) NVIDIA Container Toolkit:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# For GPU support, install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Using Docker Directly

**CPU version**:
```bash
docker build --target runtime-cpu -t mmml:cpu .
docker run -it --rm -v $(pwd):/workspace/mmml mmml:cpu
```

**GPU version**:
```bash
docker build --target runtime-gpu -t mmml:gpu .
docker run -it --rm --gpus all -v $(pwd):/workspace/mmml mmml:gpu
```

#### Using Docker Compose (Easier)

Docker Compose simplifies multi-container management:

**Start CPU container**:
```bash
docker-compose up -d mmml-cpu
docker-compose exec mmml-cpu bash
```

**Start GPU container**:
```bash
docker-compose up -d mmml-gpu
docker-compose exec mmml-gpu bash
```

**Start Jupyter Lab with GPU**:
```bash
docker-compose up -d mmml-jupyter
# Navigate to http://localhost:8888
```

**Stop containers**:
```bash
docker-compose down
```

#### Pros
- ✅ Perfect reproducibility
- ✅ Isolated from host system
- ✅ Easy to share and deploy
- ✅ Includes all system dependencies

#### Cons
- ❌ Requires Docker installation
- ❌ Larger disk space requirements
- ❌ Some performance overhead

## Dependency Groups

MMML is organized into modular dependency groups. Install only what you need:

### Core Dependencies (Always Installed)
- NumPy, SciPy, Pandas, Matplotlib
- JAX ecosystem (CPU by default)
- ASE (Atomic Simulation Environment)
- Hydra for configuration management

### Optional Groups

| Group | Packages | Use Case |
|-------|----------|----------|
| **gpu** | JAX[cuda12], CuPy, cuTENSOR | GPU acceleration with CUDA 12 |
| **quantum** | PySCF, basis-set-exchange, dscribe | Quantum chemistry calculations |
| **quantum-gpu** | gpu4pyscf | GPU-accelerated quantum chemistry |
| **ml** | PyTorch, TorchANI, E3NN | Additional ML frameworks |
| **viz** | Plotly, Seaborn, py3dmol | Visualization and plotting |
| **md** | MDAnalysis | Molecular dynamics analysis |
| **chem** | RDKit, chemcoord, pyxtal | Chemistry utilities |
| **data** | Polars, PyArrow | Fast data processing |
| **notebooks** | Jupyter, IPython | Interactive notebooks |
| **experiments** | W&B, Optuna | Experiment tracking & hyperparameter tuning |
| **dev** | pytest | Development tools |
| **all** | Everything above | Complete installation |
| **all-cpu** | Everything except GPU | Full CPU-only installation |

### Installing Multiple Groups

```bash
# Combine groups with commas (for GPU, install JAX first)
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12]" "jaxlib[cuda12]"
pip install -e ".[gpu,quantum,viz,notebooks]"

# Or use uv (recommended - handles JAX properly)
make install-gpu  # Includes proper JAX CUDA installation
uv sync --extra quantum --extra viz --extra notebooks
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

1. **System dependencies**:
```bash
sudo apt-get update
sudo apt-get install -y build-essential gfortran cmake git curl
```

2. **For GPU support**:
```bash
# Install CUDA 12 toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# Verify CUDA installation
nvidia-smi
nvcc --version
```

3. **Proceed with any installation method above**

### macOS

1. **System dependencies**:
```bash
brew install gcc cmake git
```

2. **Note**: GPU support is not available on macOS (use CPU version)

3. **Proceed with uv or conda installation**

### Windows (WSL2)

MMML is best run on Linux. On Windows, use WSL2:

1. **Install WSL2**:
```powershell
wsl --install
```

2. **Inside WSL2, follow Linux instructions**

3. **For GPU support**, ensure you have:
   - Windows 11 or Windows 10 (version 21H2 or higher)
   - NVIDIA GPU with latest drivers
   - CUDA support in WSL2 (automatically configured in recent Windows versions)

### HPC Clusters

Many HPC clusters have specific requirements:

1. **Load required modules**:
```bash
module load gcc/11.2.0
module load cuda/12.2.0
module load python/3.11
```

2. **Use conda for better compatibility**:
```bash
conda env create -f environment-gpu.yml -p $HOME/envs/mmml-gpu
conda activate $HOME/envs/mmml-gpu
```

3. **Set CHARMM environment variables**:
```bash
export CHARMM_HOME=/path/to/mmml/setup/charmm
export CHARMM_LIB_DIR=/path/to/mmml/setup/charmm
```

4. **Submit jobs using SLURM/PBS**:
```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source activate $HOME/envs/mmml-gpu
python your_script.py
```

## Post-Installation Setup

### CHARMM Configuration

Set up CHARMM environment variables:

```bash
# Option 1: Source the provided setup file
source CHARMMSETUP

# Option 2: Set manually
export CHARMM_HOME=/path/to/mmml/setup/charmm
export CHARMM_LIB_DIR=/path/to/mmml/setup/charmm
export LD_LIBRARY_PATH=$CHARMM_LIB_DIR:$LD_LIBRARY_PATH

# Add to your ~/.bashrc or ~/.zshrc for persistence
echo 'source /path/to/mmml/CHARMMSETUP' >> ~/.bashrc
```

### Verify Installation

Run the test suite:

```bash
# Quick test
python -c "import mmml; print('Success!')"

# Run unit tests
pytest tests/functionality/mmml/test_mmml_calc.py::test_ev2kcalmol_constant

# With data (if available)
export MMML_DATA=/path/to/data.npz
export MMML_CKPT=/path/to/checkpoints
pytest tests/functionality/mmml/
```

## Troubleshooting

### Common Issues

#### 1. CHARMM Library Not Found

**Error**: `OSError: libcharmm.so: cannot open shared object file`

**Solution**:
```bash
export CHARMM_HOME=/path/to/mmml/setup/charmm
export CHARMM_LIB_DIR=/path/to/mmml/setup/charmm
export LD_LIBRARY_PATH=$CHARMM_LIB_DIR:$LD_LIBRARY_PATH
```

#### 2. JAX Not Detecting GPU / CUDA Issues

**Error**: `CUDA version mismatch` or `An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed`

**Solution**:
```bash
# Check your CUDA version
nvidia-smi
nvcc --version

# Uninstall CPU-only JAX
pip uninstall jax jaxlib -y

# Reinstall JAX with CUDA support from the correct index
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "jax[cuda12]>=0.4.20" "jaxlib[cuda12]>=0.4.20"

# Verify GPU is detected
python -c "import jax; print('GPUs:', jax.devices())"

# Or use the verify script
python verify_install.py
```

**For comprehensive GPU troubleshooting, see [GPU_INSTALL.md](GPU_INSTALL.md)**

#### 3. Out of Memory (GPU)

**Error**: `CUDA out of memory`

**Solution**:
```python
# Reduce batch size or use CPU fallback
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
```

#### 4. Conda Environment Conflicts

**Error**: Package conflicts during conda install

**Solution**:
```bash
# Use mamba for better dependency resolution
conda install mamba -c conda-forge
mamba env create -f environment-gpu.yml

# Or create a minimal environment and add packages incrementally
conda create -n mmml python=3.11
conda activate mmml
pip install -e .
```

#### 5. Docker Permission Denied

**Error**: `permission denied while trying to connect to the Docker daemon`

**Solution**:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Getting Help

- **GitHub Issues**: https://github.com/EricBoittier/mmml/issues
- **Documentation**: https://mmml.readthedocs.io/
- **Email**: eric.boittier@icloud.com

## Performance Tips

1. **Use GPU when available**: Install with `[gpu]` extra for 10-100x speedup
2. **Use `uv` for faster installs**: Much faster than pip
3. **Use mamba instead of conda**: Faster dependency resolution
4. **Pin versions in production**: Use `uv.lock` or `conda env export`
5. **Cache compiled code**: JAX compiles on first run; subsequent runs are faster

## Migration from Old Setup

If you're upgrading from an older version of MMML:

1. **Backup your old environment**:
```bash
conda env export > old-environment.yml
```

2. **Create a new environment**:
```bash
conda env create -f environment-gpu.yml
```

3. **Reinstall any custom packages**:
```bash
conda activate mmml-gpu
pip install your-custom-package
```

4. **Update your scripts** if API has changed (check CHANGELOG.md)

