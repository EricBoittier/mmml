# MCP examples — builds and hybrid calculators

Runnable templates for the MCP `build_smoke` recipe and direct `mmml` use.
Configs here are copied into `artifacts/mcp_runs/<run_id>/configs/` on `configure_run`.

## MCP workflow (`build_smoke`)

```bash
cd /path/to/mmml
source examples/md_cpu/_env.sh   # JAX_PLATFORMS=cpu, MMML_CKPT, …
export CHARMM_HOME=... CHARMM_LIB_DIR=...   # required for make-res / liquid-box / hybrid

# 1. Configure run (writes manifest + hybrid YAML templates)
uv run python -c "
from mmml.mcp.recipes import configure_run
print(configure_run('build001', recipe='build_smoke', mode='smoke'))
"

# 2. Stages (dry-run first)
uv run python -c "
from mmml.mcp.recipes import run_recipe_stage
for stage in ['make_res','box_build','hybrid_md_ase','hybrid_md_jaxmd','hybrid_md_pycharmm']:
    print(stage, run_recipe_stage('build001', stage, mode='smoke', dry_run=True)['state'])
"

# 3. Execute (requires PyCHARMM + Packmol)
uv run python -c "from mmml.mcp.recipes import run_recipe_stage; run_recipe_stage('build001','make_res',mode='smoke')"
uv run python -c "from mmml.mcp.recipes import run_recipe_stage; run_recipe_stage('build001','box_build',mode='smoke',background=True)"
uv run python -c "from mmml.mcp.recipes import run_recipe_stage; run_recipe_stage('build001','hybrid_md_jaxmd',mode='smoke')"
```

Or use the shell driver:

```bash
bash examples/mcp/run_build_smoke.sh build001
DRY_RUN=1 bash examples/mcp/run_build_smoke.sh build001   # print commands only
```

## Hybrid calculator — three backends

All use `setup_calculator()` in `mmml_calculator.py` (ML PhysNet + CHARMM MM).

| Backend | Module | ASE API | JAX API |
|---------|--------|---------|---------|
| **ASE** | `md_pbc_suite/ase.py` | `atoms.calc = calc` | — |
| **JAX-MD** | `md_pbc_suite/jaxmd.py` | — | `spherical_cutoff_calculator` |
| **PyCHARMM** | `md_pbc_suite/pycharmm_mlpot.py` | MLpot registration | CHARMM dynamics |

### Direct `mmml md-system` (vacuum NVE smoke)

```bash
source examples/md_cpu/_env.sh

# ASE — VelocityVerlet + hybrid calculator
mmml md-system --config mmml/mcp/examples/hybrid_ase.yaml --run-all

# JAX-MD — jax-md integrator + JAX forces
mmml md-system --config mmml/mcp/examples/hybrid_jaxmd.yaml --run-all

# PyCHARMM — MLpot / CHARMM dynamics
mmml md-system --config mmml/mcp/examples/hybrid_pycharmm.yaml --run-all
```

Set `MMML_CKPT` or edit `defaults.checkpoint` in each YAML.

**Packmol placement:** vacuum `free_nve` jobs need `packmol_radius` (or `box_size`)
in `defaults` so cluster geometry can be built. JAX-MD `free_nve` omits `--box-size`
from its subprocess argv; `packmol_radius` alone is sufficient.

**JAX-MD dependency:** if `import jax_md` fails with `mutable_array` from `jax._src.core`,
your `jax` / `flax` / `jax-md` pins are incompatible. ASE and PyCHARMM hybrid backends
are unaffected. Re-sync the project venv (`uv sync`) or use the git-pinned `jax-md`
from `pyproject.toml`.

### Programmatic hybrid calculator (ASE)

```python
from pathlib import Path
import numpy as np
import ase
from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

ckpt = Path("examples/ckpts_json/DESdimers_params.json")
n_mono, n_atoms = 2, 10
z = np.array([6, 1, 1, 17, 17] * n_mono, dtype=int)
r = np.random.default_rng(0).normal(size=(n_mono * n_atoms, 3))

factory = setup_calculator(
    ATOMS_PER_MONOMER=n_atoms // n_mono,
    N_MONOMERS=n_mono,
    model_restart_path=ckpt,
    doML=True,
    doMM=True,
    cell=False,
)
calc, jax_fn = factory(atomic_numbers=z, atomic_positions=r, n_monomers=n_mono)
atoms = ase.Atoms(z, r)
atoms.calc = calc
print("E (eV):", atoms.get_potential_energy())
print("μ (e·Å):", calc.results.get("dipole"))
# jax_fn — use with jax-md for GPU/CPU MD
```

### Programmatic hybrid calculator (JAX-MD)

After `setup_calculator`, the returned `jax_fn` is a JIT-friendly energy/force routine
for use with `jax-md` integrators:

```python
from pathlib import Path
import jax.numpy as jnp
from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

ckpt = Path("examples/ckpts_json/DESdimers_params.json")
z = jnp.array([6, 1, 1, 17, 17] * 2, dtype=jnp.int32)
r = jnp.zeros((20, 3))

factory = setup_calculator(
    ATOMS_PER_MONOMER=5,
    N_MONOMERS=2,
    model_restart_path=ckpt,
    doML=True,
    doMM=True,
    cell=False,
)
calc, jax_fn = factory(atomic_numbers=z, atomic_positions=r, n_monomers=2)
e, f = jax_fn(r, z)
print("E (eV):", float(e))
```

Run dynamics via `mmml md-system --config mmml/mcp/examples/hybrid_jaxmd.yaml --run-all`
or MCP stage `hybrid_md_jaxmd`.

### Programmatic hybrid calculator (PyCHARMM MLpot)

```python
# Prefer the campaign YAML + mmml md-system for production:
# mmml md-system --config mmml/mcp/examples/hybrid_pycharmm.yaml --run-all
#
# MPI-linked libcharmm may require:
# MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system --config ... --run-all
```

### PBC from certified `liquid-box`

After `mmml liquid-box` (or MCP `box_build` stage):

```bash
mmml md-system --config mmml/mcp/examples/hybrid_pbc_jaxmd.yaml --run-all
```

## Geometry commands (allowlisted for MCP `submit_mmml_command`)

```bash
mmml make-res --res DCM --skip-energy-show
mmml liquid-box --composition DCM:12 -o boxes/liquid --profile standard --box-size 24
mmml make-box --res DCM --n 12 --side_length 24
```

## See also

- [`examples/md_cpu/README.md`](../../examples/md_cpu/README.md) — CPU smoke matrix
- [`docs/liquid-box-workflow.md`](../../docs/liquid-box-workflow.md) — two-phase liquid workflow
- [`mmml/mcp/README.md`](../README.md) — MCP server tools
