# Spatial MPI + DOMDEC coexistence (Tier 3 spike)

Manual smoke for aligning CHARMM domain decomposition with spatial MLpot.
**Not run in CI** — requires MPI-linked CHARMM, GPU, and equilibrated PBC cluster.

## Goal

Mode C from [Spatial ML MPI](../../docs/mlpot-spatial-mpi.md): `np>1`, domdec **on**, spatial ML per rank, one GPU per rank. Both CHARMM integration and PhysNet should scale.

## Current blockers (June 2026)

1. MLpot paths call `disable_charmm_domdec()` before SD/dynamics ([`import_pycharmm.py`](../../mmml/interfaces/pycharmmInterface/import_pycharmm.py)).
2. JAX GPU warmup + domdec can segfault (`send_coord_to_recip` / `PMPI_Free_mem`).
3. PyCHARMM exposes no per-rank local/ghost atom API ([`domdec_info.py`](../../mmml/interfaces/pycharmmInterface/mlpot/mpi_spatial/domdec_info.py)).

Tier 2 (`--ml-spatial-mpi`) parallelizes **ML only** with domdec still off; CHARMM integration remains replicated per rank.

## Spike procedure (user-run)

### A. Baseline Tier 2 (spatial ML, domdec off)

```bash
export MMML_MLPOT_SPATIAL_MPI=1
MMML_MPI_NP=2 ./scripts/mmml-charmm-mpirun.sh md-system \
  --composition DCM:20 --box-size 32 \
  --ml-spatial-mpi --ml-gpu-count 1 --ml-batch-size 128 \
  --md-stage mini --ps-heat 0
```

**Pass:** minimization completes; energies finite on all ranks.

### B. Domdec on (do not disable before SD)

1. Temporarily comment out `disable_charmm_domdec()` / `ensure_domdec_off_for_mlpot_energy` in the MLpot dynamics path (local experiment only).
2. Launch with `MMML_MPI_NP=2`, `--ml-spatial-mpi`, short SD (10 steps).
3. Record: segfault yes/no, whether `send_coord_to_recip` appears in backtrace.

### C. ctypes / PyCHARMM survey

- Inspect `ldd libcharmm.so` for `domdec_common` symbols.
- Check whether `pycharmm` exposes atom group / domain queries (none found in June 2026 survey).

## Success criteria (Tier 3)

- SD with domdec on + spatial ML completes without segfault.
- Integration wall time decreases with `n_ranks` on fixed box size.
- Python `SpatialDomainGrid` slab partition matches CHARMM domdec cell ownership (or is replaced by Fortran metadata).

## Fallback

Until Tier 3 passes, use **Tier 1** for one-node 2-GPU production:

```bash
export CUDA_VISIBLE_DEVICES=0,1
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system ... \
  --ml-batch-size 128 --ml-gpu-count 2
```
