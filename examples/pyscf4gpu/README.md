# GPU-accelerated DFT with PySCF/gpu4pyscf

Examples using `mmml.interfaces.pyscf4gpuInterface` for GPU-accelerated quantum chemistry.

**Requirements:** gpu4pyscf, pyscf (install via `make micromamba-create-full` or `uv sync --extra quantum-gpu`)

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
