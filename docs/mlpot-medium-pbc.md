# Medium PBC dense liquids (500–2000 monomers)

Workflow for single-rank GPU throughput with global sparse dimers before spatial MPI decomposition is available.

## Prerequisites

- Launch via [`scripts/mmml-charmm-mpirun.sh`](../scripts/mmml-charmm-mpirun.sh) with **`MMML_MPI_NP=1`** (recommended).
- Default cutoffs: `extended_mm5` (8 / 5 / 1.5 Å) — see [MLpot Settings](mlpot-settings.md).

## Sparse dimer cap validation (required before production)

After minimization / equilibration, validate the sparse ML dimer cap on the **equilibrated CRD**:

```bash
# Example: 1000 monomers, 10 atoms each, 40 Å cubic box
python scripts/validate_mlpot_sparse_dimers.py \
  --crd artifacts/pycharmm_mlpot/my_run/mini_full_mlpot_TAG.crd \
  --n-monomers 1000 --atoms-per-monomer 10 --box-size 40
```

Or audit an output directory:

```bash
python scripts/audit_mlpot_cluster.py --output-dir artifacts/pycharmm_mlpot/my_run
```

**Exit code 0** means the default cap covers all near dimers (COM distance &lt; `mm_switch_on`). **Exit code 1** means the cap is saturated — raise `--ml-max-active-dimers` or enlarge the box; do not proceed silently.

### Default PBC caps

| n_monomers | Cap `max(1000, 6n)` | PhysNet systems/step (upper bound) |
|------------|---------------------|-------------------------------------|
| 500 | 3000 | ≤ 3500 |
| 1000 | 6000 | ≤ 7000 |
| 2000 | 12000 | ≤ 14000 |

## Recommended `ml_batch_size` (single GPU)

| Regime | Start | OOM / compile RAM | Underutilized GPU |
|--------|-------|-------------------|-------------------|
| 500–2000 monomers | `256` | `128` or `64` | `512` if memory allows |

```bash
export MMML_MLPOT_ML_BATCH_SIZE=256
mmml md-system ... --ml-batch-size 256
```

Multi-GPU on one node (still `np=1`): `--ml-gpu-count N` with `--ml-batch-size 128–256`.

### Dual-GPU pmap (recommended on one 2-GPU node)

Use **one MPI rank** and let JAX `pmap` spread PhysNet chunks across both GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system ... \
  --ml-batch-size 128 --ml-gpu-count 2
```

`ml_batch_size` must be small enough that `ceil(systems_per_step / ml_batch_size) >= 2` so both GPUs receive chunks (see `effective_ml_gpu_count` in [`mlpot_gpu_policy.py`](../mmml/interfaces/pycharmmInterface/mlpot/mlpot_gpu_policy.py)).

Benchmark guidance:

```bash
python scripts/benchmark_mlpot_ml_batch.py --checkpoint path/to/ckpt --n-monomers 90 \
  --batch-sizes 64 128 256 --ml-gpu-count 2
```

### Spatial MPI (`np=2`, experimental)

Per-rank ML decomposition with one GPU per rank — see [Spatial ML MPI](mlpot-spatial-mpi.md):

```bash
export MMML_MLPOT_SPATIAL_MPI=1
MMML_MPI_NP=2 ./scripts/mmml-charmm-mpirun.sh md-system ... \
  --ml-spatial-mpi --ml-gpu-count 1 --ml-batch-size 256
```

Do **not** combine `np>1` with `--ml-gpu-count 2` on a 2-GPU node without explicit per-rank GPU binding.

## Staged workflow

1. **Build / minimize / heat** — PyCHARMM MLpot (`md-system`).
2. **Validate sparse cap** — `validate_mlpot_sparse_dimers.py` on equilibrated CRD.
3. **Long production** — JAX-MD (`run_sim.py`) after ASE/JAX-MD consistency tests pass on the target geometry.

## MPI note

- **Production:** `MMML_MPI_NP=1` with optional `--ml-gpu-count 2` for dual-GPU pmap.
- **Experimental:** `MMML_MPI_NP=2` with `--ml-spatial-mpi` for per-rank ML decomposition (see [Spatial ML MPI](mlpot-spatial-mpi.md)).
- **Do not** use `np>1` with rank-0 bridge for performance; use spatial MPI or stay on `np=1`.

## Python API

```python
from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
    suggest_medium_pbc_sizing,
    validate_medium_pbc_geometry,
    workflow_checklist,
)

print(suggest_medium_pbc_sizing(1000))
for line in workflow_checklist(1000):
    print(line)
```
