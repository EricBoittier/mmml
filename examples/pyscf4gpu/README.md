# GPU-accelerated DFT with PySCF/gpu4pyscf

Examples using `mmml.interfaces.pyscf4gpuInterface` for GPU-accelerated quantum chemistry.

**Requirements:** cupy, gpu4pyscf, pyscf. Install with:
```bash
uv sync --extra quantum-gpu   # or --extra all
# Or: make micromamba-create-full
```
If `uv` is not in PATH (e.g. HPC): use full path, e.g. `~/micromamba/bin/uv sync --extra all`.

## Example scripts

Example scripts live in `examples/mmml_tutorial/` (numbered by README order):

| # | CLI | Programmatic |
|---|-----|--------------|
| 03 | `03_pyscf_dft_cli.sh` | `03_pyscf_dft_programmatic.py` |
| 04 | `04_pyscf_dft_cli_full.sh` | `04_pyscf_dft_programmatic.py` |
| 05 | `05_pyscf_mp2_cli.sh` | `05_pyscf_mp2_programmatic.py` |

Run from project root, e.g.:
```bash
bash examples/mmml_tutorial/03_pyscf_dft_cli.sh
uv run python examples/mmml_tutorial/03_pyscf_dft_programmatic.py
```

Or use the legacy water example:
```bash
make pyscf-example
uv run python examples/pyscf4gpu/water_energy.py
```

## CLI

```bash
# 01: Energy only
make pyscf-dft
uv run mmml pyscf-dft --mol "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0" --energy --output results

# 02: Energy + gradient + hessian + harmonic analysis
uv run mmml pyscf-dft --mol water.xyz --energy --gradient --hessian --harmonic --thermo

# 03: MP2 (post-HF, not DFT)
uv run mmml pyscf-mp2 --mol "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0" --energy --gradient
```

## Programmatic usage

```python
from mmml.interfaces.pyscf4gpuInterface.calcs import compute_dft, get_dummy_args, save_pyscf_results
from mmml.interfaces.pyscf4gpuInterface.enums import CALCS

mol_str = "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0"
args = get_dummy_args(mol_str, [CALCS.ENERGY, CALCS.GRADIENT])
args.basis = "def2-TZVP"  # default is def2-SVP
args.xc = "PBE0"

output = compute_dft(args, [CALCS.ENERGY, CALCS.GRADIENT])
print(f"Energy: {output['energy']} Hartree")
save_pyscf_results("output", output)
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
