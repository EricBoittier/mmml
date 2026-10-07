# Medium PBC dense liquids (500–2000 monomers)

Workflow for single-rank GPU throughput with global sparse dimers before spatial MPI decomposition is available.

## Prerequisites

- Launch via [`scripts/karml-charmm-mpirun.sh`](https://github.com/EricBoittier/karml/blob/main/scripts/karml-charmm-mpirun.sh) with **`KARML_MPI_NP=1`** (recommended).
- Default cutoffs: `extended_mm5` (8 / 5 / 1.5 Å) — see [MLpot Settings](mlpot-settings.md).

## Sparse dimer cap validation (required before production)

After minimization / equilibration, validate the sparse ML dimer cap on the **equilibrated CRD**:

```bash
# Example: 1000 monomers, 10 atoms each, 40 Å cubic box
python scripts/validate_mlpot_sparse_dimers.py \
  --crd artifacts/pycharmm_mlpot/my_run/mini_full_mlpot_TAG.crd \
  --n-monomers 1000 --atoms-per-monomer 10 --box-size 40 \
  --ml-max-active-dimers "$CAP_FROM_SETUP_REPORT"
```

Or audit an output directory:

```bash
python scripts/audit_mlpot_cluster.py --output-dir artifacts/pycharmm_mlpot/my_run
```

**Exit code 0** means the tested cap covers all near dimers (COM distance &lt; `mm_switch_on`). **Exit code 1** means the cap is saturated — raise `--ml-max-active-dimers` or enlarge the box; do not proceed silently.

### Cap policy

`md-system` / MLpot registration sizes the sparse cap differently for PBC and
free-space runs:

- **PBC with known box volume:** expected in-range dimer pairs from density and
  `active_radius = mm_switch_on`, multiplied by a 1.4 safety margin, with a
  floor of `max(4005, 6 * n_monomers)` and a ceiling of all unique dimers.
- **Free-space clusters:** all `n(n-1)/2` unique dimers; explicit lower caps are
  promoted so pairs are not dropped.
- **Explicit overrides:** `--ml-max-active-dimers` or
  `KARML_MLPOT_MAX_ACTIVE_DIMERS` set the PBC cap, but a step that exceeds it now
  fails closed with `SparseDimerCapOverflow` instead of truncating forces.

The standalone validator tests the cap you pass; if omitted, it reports its
fallback cap policy. For production checks, compare against the
`max_active_dimers` value printed in the MLpot setup report; pass that value
with `--ml-max-active-dimers` when you want the validator to match a specific
run.

| n_monomers | Minimum PBC floor | Minimum padded PhysNet systems/step |
|------------|------------------:|-------------------------------------:|
| 500 | 4005 | 4505 |
| 1000 | 6000 | 7000 |
| 2000 | 12000 | 14000 |

Dense boxes can exceed these floors; the live setup report is authoritative for
the run's actual cap.

## Recommended `ml_batch_size` (single GPU)

| Regime | Start | OOM / compile RAM | Underutilized GPU |
|--------|-------|-------------------|-------------------|
| 500–2000 monomers | `256` | `128` or `64` | `512` if memory allows |

```bash
export KARML_MLPOT_ML_BATCH_SIZE=256
karml md-system ... --ml-batch-size 256
```

Multi-GPU on one node (still `np=1`): `--ml-gpu-count N` with `--ml-batch-size 128–256`.

### Dual-GPU pmap (recommended on one 2-GPU node)

Use **one MPI rank** and let JAX `pmap` spread PhysNet chunks across both GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
KARML_MPI_NP=1 ./scripts/karml-charmm-mpirun.sh md-system ... \
  --ml-batch-size 128 --ml-gpu-count 2
```

`ml_batch_size` must be small enough that `ceil(systems_per_step / ml_batch_size) >= 2` so both GPUs receive chunks (see `effective_ml_gpu_count` in [`mlpot_gpu_policy.py`](https://github.com/EricBoittier/karml/blob/main/karml/interfaces/pycharmmInterface/mlpot/mlpot_gpu_policy.py)).

Benchmark guidance:

```bash
python scripts/benchmark_mlpot_ml_batch.py --checkpoint path/to/ckpt --n-monomers 90 \
  --batch-sizes 64 128 256 --ml-gpu-count 2
```

### Spatial MPI (`np=2`, experimental)

Per-rank ML decomposition with one GPU per rank — see [Spatial ML MPI](mlpot-spatial-mpi.md):

```bash
export KARML_MLPOT_SPATIAL_MPI=1
KARML_MPI_NP=2 ./scripts/karml-charmm-mpirun.sh md-system ... \
  --ml-spatial-mpi --ml-gpu-count 1 --ml-batch-size 256
```

Do **not** combine `np>1` with `--ml-gpu-count 2` on a 2-GPU node without explicit per-rank GPU binding.

## Staged workflow

1. **Build / minimize / heat** — PyCHARMM MLpot (`md-system`).
2. **Validate sparse cap** — `validate_mlpot_sparse_dimers.py` on equilibrated
   CRD, using the same `mm_switch_on` and cap shown in the setup report.
3. **Long production** — JAX-MD (`run_sim.py`) after ASE/JAX-MD consistency tests pass on the target geometry.

## MPI note

- **Production:** `KARML_MPI_NP=1` with optional `--ml-gpu-count 2` for dual-GPU pmap.
- **Experimental:** `KARML_MPI_NP=2` with `--ml-spatial-mpi` for per-rank ML decomposition (see [Spatial ML MPI](mlpot-spatial-mpi.md)).
- **Do not** use `np>1` with rank-0 bridge for performance; use spatial MPI or stay on `np=1`.

## Python API

```python
from karml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
    suggest_medium_pbc_sizing,
    validate_medium_pbc_geometry,
    workflow_checklist,
)

print(suggest_medium_pbc_sizing(1000))
for line in workflow_checklist(1000):
    print(line)
```
