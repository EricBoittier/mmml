# CPU MD examples (no CUDA)

Runnable smoke workflows for **ASE**, **jax-md**, and **neighbor-list** paths on CPU,
using the bundled DESdimers JSON checkpoint (`examples/ckpts_json/DESdimers_params.json`,
ACO dimer, 20 atoms).

## Install

```bash
cd /path/to/mmml
make install-md-cpu    # or: uv sync --extra md-cpu
source examples/md_cpu/_env.sh
```

`md-cpu` adds **Vesin** (NL reference) and **MDAnalysis** (trajectory helpers). Core JAX,
jax-md, and ASE are already in the base package.

## Environment

```bash
source examples/md_cpu/_env.sh
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `JAX_PLATFORMS` | `cpu` | Force CPU JAX |
| `JAX_ENABLE_X64` | `1` | Double precision (recommended for MD) |
| `MMML_CKPT` | `examples/ckpts_json/DESdimers_params.json` | PhysNet DESdimers weights |
| `MMML_MM_NL_BACKEND` | `auto` | MM pair builder: `auto`, `vesin`, `cell_list`, `jax_md` |

Artifacts land under `artifacts/md_cpu/`.

## Quick run

```bash
bash examples/md_cpu/run_all.sh
```

## Examples

| Step | Script | Backend / focus | PyCHARMM |
|------|--------|-----------------|----------|
| 0 | `00_check_env.py` | deps, JAX CPU, checkpoint | no |
| 1 | `01_neighbor_list_parity.sh` | Vesin / jax-md / ASE / cell-list | no |
| 2 | `02_ml_energy_ase.py` | ASE ML energy/forces | no |
| 3 | `03_ml_energy_jaxmd.py` | JAX-MD vs ASE (ML-only) | no |
| 4 | `05_free_nve_ase_smoke.py` | ASE VelocityVerlet NVE smoke | no |
| 5 | `06_free_nve_jaxmd_smoke.py` | JAX-MD NVE smoke | no |
| 6 | `07_nl_backend_matrix.sh` | NL cutoffs + `MMML_MM_NL_BACKEND` | no |
| 7 | `04_md_system_evaluate_ase.sh` | `md-system --evaluate-npz` | **yes** |
| 8 | `05_md_system_free_nve_ase.sh` | `md-system` vacuum NVE | **yes** |
| 9 | `06_md_system_free_nve_jaxmd.sh` | `md-system` vacuum NVE | **yes** |

Steps 0–6 run without PyCHARMM. Steps 7–9 need `CHARMM_HOME` / `CHARMM_LIB_DIR` and are
skipped automatically by `run_all.sh` when PyCHARMM is unavailable.

### Neighbor lists

`01` and `07` wrap
[`tests/functionality/neighbor_lists/05_compare_nl_backends.py`](../../tests/functionality/neighbor_lists/05_compare_nl_backends.py).
With PyCHARMM:

```bash
RUN_CHARMM_NL=1 bash examples/md_cpu/07_nl_backend_matrix.sh
# or
uv run python tests/functionality/neighbor_lists/05_compare_nl_backends.py \
  --composition ACO:2 --backends vesin,jax_md,ase,pycharmm
```

### MD backends (ML-only, no CHARMM)

```bash
uv run python examples/md_cpu/05_free_nve_ase_smoke.py
uv run python examples/md_cpu/06_free_nve_jaxmd_smoke.py
```

### Full `md-system` (PyCHARMM + hybrid MM)

```bash
export CHARMM_HOME=... CHARMM_LIB_DIR=...
bash examples/md_cpu/05_md_system_free_nve_ase.sh
bash examples/md_cpu/06_md_system_free_nve_jaxmd.sh
```

## Related

- [`tests/functionality/neighbor_lists/README.md`](../../tests/functionality/neighbor_lists/README.md)
- [`tests/functionality/mlpot/README.md`](../../tests/functionality/mlpot/README.md)
- [`tests/functionality/pyxtal/README.md`](../../tests/functionality/pyxtal/README.md)
