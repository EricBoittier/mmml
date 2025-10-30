# MMML Quick Start Guide

Get MMML up and running in 5 minutes!

## Choose Your Installation Method

### 🚀 Fastest: Using `uv` (Recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install MMML
git clone https://github.com/EricBoittier/mmml.git
cd mmml

# CPU version
make install

# Or GPU version (requires CUDA 12 - this properly installs JAX with CUDA)
make install-gpu

# Run a quick test
make test-quick
```

**Done!** You can now use MMML:
```bash
uv run python -c "import mmml; print('MMML ready!')"
```

### 🐍 Using Conda (For HPC/Clusters)

```bash
git clone https://github.com/EricBoittier/mmml.git
cd mmml

# CPU version
make conda-create
conda activate mmml

# Or GPU version
make conda-create-gpu
conda activate mmml-gpu

# Test it
python -c "import mmml; print('MMML ready!')"
```

### 🐳 Using Docker (Most Isolated)

```bash
git clone https://github.com/EricBoittier/mmml.git
cd mmml

# Build and run (CPU)
make docker-build-cpu
make docker-run-cpu

# Or GPU version
make docker-build-gpu
make docker-run-gpu

# Or use Docker Compose
docker-compose up -d mmml-gpu
docker-compose exec mmml-gpu bash
```

## First Steps

### 1. Set up CHARMM (if needed)

```bash
source CHARMMSETUP
# Or run the setup script
bash setup/install.sh
```

### 2. Run Your First Calculation

Create a file `test_mmml.py`:

```python
import numpy as np
from mmml.pycharmmInterface.mmml_calculator import setup_calculator, ev2kcalmol
import ase

# Define a simple system (2 water molecules)
ATOMS_PER_MONOMER = 10
N_MONOMERS = 2
Z = np.array([6, 1, 1, 1, 6, 1, 1, 1, 8, 1] * N_MONOMERS, dtype=int)
R = np.random.randn(ATOMS_PER_MONOMER * N_MONOMERS, 3) * 0.1

# Create calculator (ML only, no CHARMM needed for this test)
factory = setup_calculator(
    ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
    N_MONOMERS=N_MONOMERS,
    doML=False,  # Set to True if you have ML checkpoints
    doMM=False,
    MAX_ATOMS_PER_SYSTEM=ATOMS_PER_MONOMER * N_MONOMERS,
)

calc, _ = factory(atomic_numbers=Z, atomic_positions=R, n_monomers=N_MONOMERS)
atoms = ase.Atoms(Z, R)
atoms.calc = calc

print(f"Energy: {atoms.get_potential_energy():.4f} kcal/mol")
print(f"Forces shape: {atoms.get_forces().shape}")
```

Run it:
```bash
# With uv
uv run python test_mmml.py

# Or if you used conda/have an activated venv
python test_mmml.py
```

### 3. Explore Examples

```bash
# Check out the examples directory
cd examples/dcm/

# Run an example configuration
uv run python train.py
```

## Common Commands

```bash
# Show all available make commands
make help

# Run tests
make test              # All tests
make test-quick        # Quick test only
make test-coverage     # With coverage report

# Clean up
make clean             # Remove build artifacts
make clean-all         # Remove everything including venv

# Docker commands
make docker-build-gpu  # Build GPU Docker image
make docker-jupyter    # Start Jupyter Lab
make docker-clean      # Remove all Docker artifacts

# Development
make dev-setup         # Set up development environment
make lint              # Run linter
make format            # Format code
```

## What's Next?

### 📚 Learn More
- **[INSTALL.md](INSTALL.md)** - Detailed installation guide
- **[README.md](README.md)** - Full documentation
- **[Examples](examples/)** - Example scripts and configs

### 🔧 Optional Features

Install only what you need:

```bash
# GPU support (10-100x faster)
pip install -e ".[gpu]"

# Quantum chemistry calculations
pip install -e ".[quantum]"

# Visualization tools
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"
```

### 🐛 Troubleshooting

**CHARMM not found?**
```bash
export CHARMM_HOME=$(pwd)/setup/charmm
export CHARMM_LIB_DIR=$(pwd)/setup/charmm
source CHARMMSETUP
```

**CUDA issues?**
```bash
# Check CUDA version
nvidia-smi

# Ensure JAX can see GPU
python -c "import jax; print(jax.devices())"
```

**Import errors?**
```bash
# Verify installation
python -c "import mmml; print('OK')"

# Check installed packages
uv pip list  # for uv
conda list   # for conda
```

### 💡 Tips

1. **Use GPU when available** - 10-100x faster for large systems
2. **Start with examples** - Check `examples/` directory
3. **Use Makefile** - Simplifies common tasks (`make help`)
4. **Docker for reproducibility** - Guarantees it works the same everywhere
5. **Conda for HPC** - Best compatibility with cluster modules

## Getting Help

- **Issues**: https://github.com/EricBoittier/mmml/issues
- **Docs**: https://mmml.readthedocs.io/
- **Email**: eric.boittier@icloud.com

## Quick Reference

| Task | Command |
|------|---------|
| Install (uv) | `make install` |
| Install GPU | `make install-gpu` |
| Conda env | `make conda-create-gpu` |
| Docker GPU | `make docker-run-gpu` |
| Run tests | `make test` |
| Clean | `make clean` |
| Help | `make help` |

---

**Happy computing! 🚀**

