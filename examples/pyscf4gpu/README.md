# GPU-accelerated DFT with PySCF/gpu4pyscf

Examples using `mmml.interfaces.pyscf4gpuInterface` for GPU-accelerated quantum chemistry.

**Requirements:** cupy, gpu4pyscf, pyscf. Install with:
```bash
uv sync --extra quantum-gpu   # or --extra all
# Or: make micromamba-create-full
```
If `uv` is not in PATH (e.g. HPC): use full path, e.g. `~/micromamba/bin/uv sync --extra all`.

## Example script

```bash
# Run water energy + gradient
make pyscf-example

# Or directly:
uv run python examples/pyscf4gpu/water_energy.py
```

## CLI

```bash
# Energy only
make pyscf-dft

# Or with custom molecule and options:
uv run mmml pyscf-dft --mol "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0" --energy --output results.pkl

# Energy + gradient + hessian + harmonic analysis
uv run mmml pyscf-dft --mol water.xyz --energy --gradient --hessian --harmonic --thermo
```

## Programmatic usage

```python
from mmml.interfaces.pyscf4gpuInterface.calcs import setup_mol, compute_dft, get_dummy_args
from mmml.interfaces.pyscf4gpuInterface.enums import CALCS

mol_str = "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0"
args = get_dummy_args(mol_str, [CALCS.ENERGY, CALCS.GRADIENT])
args.basis = "def2-tzvp"
args.xc = "PBE0"

output = compute_dft(args, [CALCS.ENERGY, CALCS.GRADIENT])
print(f"Energy: {output['energy']} Hartree")
```

## Troubleshooting

### `CUDA_ERROR_NO_BINARY_FOR_GPU: no kernel image is available for execution on the device`

This means CuPy was built for different GPU architectures than your machine. Common causes:

1. **New GPU (e.g. RTX 5080 / Blackwell sm_120)** – Needs CUDA 12.8+ and a matching CuPy. Check for conda/CUDA version conflicts: `conda list | grep cuda`; ensure CuPy links to the same CUDA toolkit as `nvcc --version`.

2. **Clear CuPy cache** – Stale JIT cache can cause mismatches:
   ```bash
   rm -rf ~/.cupy/kernel_cache
   # or set: export CUPY_CACHE_DIR=/tmp/cupy_cache
   ```

3. **Check GPU and CuPy compatibility:**
   ```bash
   nvidia-smi  # GPU model and driver
   python -c "import cupy; p=cupy.cuda.runtime.getDeviceProperties(0); print(f'Compute: {p.major}.{p.minor}')"
   ```

4. **Upgrade CuPy** – Newer wheels support more architectures:
   ```bash
   pip install --upgrade cupy-cuda13x
   ```

5. **HPC clusters** – Load the correct CUDA module before installing:
   ```bash
   module load cuda/12.2  # or your cluster's version
   pip install cupy-cuda13x
   ```
