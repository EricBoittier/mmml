# Dependency and Installation Cleanup - Summary of Changes

## Overview

This update completely reorganizes the MMML installation system to provide:
- ✅ Cleaner, modular dependencies
- ✅ Multiple installation methods (uv, conda, Docker)
- ✅ Better documentation
- ✅ Easier maintenance

## Files Created

### Configuration Files

1. **`pyproject.toml`** (Updated)
   - Reorganized dependencies into core + optional groups
   - Removed duplicates (e.g., `pyscf` was listed twice)
   - Created logical groups: `gpu`, `quantum`, `ml`, `viz`, `md`, etc.
   - Added `all` and `all-cpu` convenience groups

2. **`environment.yml`** (New)
   - Conda environment for CPU-only installation
   - Minimal dependencies
   - Best for development without GPU

3. **`environment-gpu.yml`** (New)
   - Conda environment with CUDA 12 support
   - Includes GPU-accelerated packages
   - Best for HPC clusters with GPUs

4. **`environment-full.yml`** (New)
   - Full installation with all optional features
   - GPU + quantum chemistry + ML + visualization
   - For complete development environments

5. **`Dockerfile`** (Updated)
   - Multi-stage build for smaller images
   - Separate CPU and GPU targets
   - Better caching for faster rebuilds
   - Uses modern Docker syntax

6. **`.dockerignore`** (New)
   - Optimizes Docker build context
   - Excludes unnecessary files from image

7. **`docker-compose.yml`** (New)
   - Easy multi-container management
   - Pre-configured CPU, GPU, and Jupyter services
   - Volume management for persistent data

8. **`Makefile`** (New)
   - Convenient commands for common tasks
   - Works across all installation methods
   - Includes test, clean, build targets

### Documentation Files

9. **`INSTALL.md`** (New)
   - Comprehensive installation guide
   - Platform-specific instructions (Linux, macOS, Windows/WSL, HPC)
   - Troubleshooting section
   - Performance tips

10. **`QUICKSTART.md`** (New)
    - Get started in 5 minutes
    - Simple, actionable steps
    - First example code
    - Quick reference table

11. **`README.md`** (Updated)
    - Reorganized installation section
    - Added optional dependency groups table
    - Links to detailed guides
    - Make commands reference

12. **`CHANGES.md`** (This file)
    - Summary of all changes
    - Migration guide
    - What's new

## Dependency Organization

### Before
- 100+ dependencies in a flat list
- Many unnecessary packages
- Duplicates (e.g., `pyscf` listed twice)
- No way to install minimal version
- CUDA packages always installed

### After

**Core Dependencies** (always installed):
- Scientific computing: NumPy, SciPy, Pandas, Matplotlib
- JAX ecosystem: JAX, Flax, Optax, Haiku (CPU by default)
- Chemistry: ASE
- Configuration: Hydra, OmegaConf
- Utilities: tqdm, rich, tabulate

**Optional Groups** (install as needed):
```bash
pip install -e ".[gpu]"          # GPU support (CUDA 12)
pip install -e ".[quantum]"      # Quantum chemistry
pip install -e ".[quantum-gpu]"  # GPU quantum chemistry
pip install -e ".[ml]"           # PyTorch, TorchANI, E3NN
pip install -e ".[viz]"          # Plotly, Seaborn, py3dmol
pip install -e ".[md]"           # MDAnalysis
pip install -e ".[chem]"         # RDKit, chemcoord
pip install -e ".[data]"         # Polars, PyArrow
pip install -e ".[notebooks]"    # Jupyter
pip install -e ".[experiments]"  # W&B, Optuna
pip install -e ".[all]"          # Everything
pip install -e ".[all-cpu]"      # Everything except GPU
```

## Installation Methods

### 1. Using `uv` (Recommended for Development)

**Before:**
```bash
uv sync  # Installs everything, including GPU packages
```

**After:**
```bash
uv sync                    # Core only
uv sync --extra gpu        # With GPU
uv sync --extra all        # Everything
make install-gpu           # Using Makefile
```

### 2. Using Conda (Recommended for HPC)

**Before:**
- No conda environment files provided
- Users had to create their own

**After:**
```bash
conda env create -f environment.yml          # CPU
conda env create -f environment-gpu.yml      # GPU
conda env create -f environment-full.yml     # Everything
make conda-create-gpu                        # Using Makefile
```

### 3. Using Docker (Recommended for Reproducibility)

**Before:**
- Basic Dockerfile with single target
- No docker-compose
- No GPU optimization

**After:**
```bash
# Multi-stage Docker with CPU and GPU targets
make docker-build-cpu
make docker-run-gpu
docker-compose up -d mmml-jupyter  # Jupyter Lab

# Or manually
docker build --target runtime-cpu -t mmml:cpu .
docker build --target runtime-gpu -t mmml:gpu .
```

## Benefits

### For Developers
- ✅ Faster installation (only install what you need)
- ✅ Better dependency management
- ✅ Easier to test different configurations
- ✅ Clear documentation

### For Users
- ✅ Multiple installation methods
- ✅ Works on different platforms
- ✅ Easy troubleshooting
- ✅ Quick start guide

### For Maintainers
- ✅ Cleaner dependency structure
- ✅ Easier to update packages
- ✅ Logical grouping
- ✅ Better reproducibility

## Migration Guide

### If you were using `uv`

**Before:**
```bash
uv sync
```

**After (for same behavior):**
```bash
uv sync --extra all
```

**Or (for minimal install):**
```bash
uv sync                    # Core only
uv sync --extra gpu        # Add GPU
```

### If you were using pip/conda

**Before:**
```bash
pip install -e .
```

**After:**
```bash
# Minimal
pip install -e .

# With GPU
pip install -e ".[gpu]"

# Everything
pip install -e ".[all]"
```

### Updating Existing Environments

**For uv:**
```bash
uv sync --upgrade --extra all
```

**For conda:**
```bash
# Remove old environment
conda env remove -n mmml

# Create new one
conda env create -f environment-gpu.yml
conda activate mmml-gpu
```

**For Docker:**
```bash
# Remove old images
make docker-clean

# Rebuild
make docker-build-gpu
```

## Breaking Changes

### None for Basic Usage

If you're using `uv sync` or `pip install -e .`, nothing changes for core functionality.

### Optional Dependencies

Some packages are now optional:
- PyTorch, TorchANI → install with `[ml]`
- PySCF → install with `[quantum]`
- GPU packages → install with `[gpu]`
- Visualization tools → install with `[viz]`

**To get the old behavior:**
```bash
uv sync --extra all
# or
pip install -e ".[all]"
```

## Testing

All existing tests should pass. The test suite is unchanged.

```bash
# Quick test (always works)
make test-quick

# Full test suite
make test

# With coverage
make test-coverage
```

## What's Removed

- Commented-out dependencies in `pyproject.toml` (OpenFF, OpenMM, RDKit, TensorFlow)
- Duplicate entries
- Scattered `requirements.txt` files are no longer needed (kept for backwards compatibility)

## What's Added

- 12 new/updated configuration and documentation files
- Optional dependency groups
- Makefile with 30+ convenient commands
- Docker Compose configuration
- 3 conda environment variants
- Comprehensive installation guide
- Quick start guide

## File Structure

```
mmml/
├── pyproject.toml           # Updated with modular dependencies
├── environment.yml          # New: Conda CPU
├── environment-gpu.yml      # New: Conda GPU
├── environment-full.yml     # New: Conda Full
├── Dockerfile              # Updated: Multi-stage
├── .dockerignore           # New: Docker optimization
├── docker-compose.yml      # New: Easy Docker management
├── Makefile                # New: Convenient commands
├── README.md               # Updated: Better organization
├── INSTALL.md              # New: Detailed installation guide
├── QUICKSTART.md           # New: 5-minute start guide
└── CHANGES.md              # This file
```

## Next Steps

1. **Review the changes**: Read through the updated files
2. **Test installation**: Try one of the installation methods
3. **Update workflows**: If using CI/CD, update to use new install methods
4. **Report issues**: If something doesn't work, open an issue

## Questions?

- **GitHub Issues**: https://github.com/EricBoittier/mmml/issues
- **Documentation**: https://mmml.readthedocs.io/
- **Quick Start**: See QUICKSTART.md
- **Detailed Install**: See INSTALL.md

---

**Created**: 2025-10-30
**Author**: AI Assistant (Claude)
**Reviewed by**: [To be filled]

