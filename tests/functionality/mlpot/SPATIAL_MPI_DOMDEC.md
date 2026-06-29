# Spatial MPI + DOMDEC coexistence (Tier 3 spike)

Manual smoke for aligning CHARMM domain decomposition with spatial MLpot.
**Not run in CI** — requires MPI-linked CHARMM, GPU, and equilibrated PBC cluster.

## Goal

Mode C from [Spatial ML MPI](../../../docs/mlpot-spatial-mpi.md): `np>1`, domdec **on**, spatial ML per rank, one GPU per rank. Both CHARMM integration and PhysNet should scale.

## Status (June 2026)

**Resolved blockers:**
- `domdec_atoms.py` reads `natoml` / `loc2glo_ind` / `atoml` from `libcharmm.so` via ctypes. No upstream PyCHARMM change needed.
- `DomdecAlignedGrid` auto-reads NDIR and exposes `get_local_atom_indices()` / `molecules_owned_by_this_rank()`.
- `build_domdec_spatial_batch_indices` in `batch_builder.py` uses DOMDEC atom ownership (not COM slabs) when `domdec_active=True`.
- `calculate_charmm` in `hybrid_mlpot.py` now calls `make_domdec_aligned_grid` (auto-detects DOMDEC state); falls back to COM-slab Tier 2 when DOMDEC is off.

**Remaining open items:**
1. **`np>1` PyCHARMM setup I/O hangs** (confirmed node09, June 2026): even probe-style
   ``lingo.charmm_script`` READ of prebuilt PSF/CRD blocks all ranks inside
   ``eval_charmm_script``. Live ``--charmm-ener`` must run at ``MMML_MPI_NP=1``;
   use the callback-only sub-test at ``np>1`` for spatial MPI decomposition.
2. JAX GPU warmup + active DOMDEC may segfault (`send_coord_to_recip` / `PMPI_Free_mem`) — defer JAX warmup until after MLpot registration.
3. `domdec off` is an opt-in safety hook (`MMML_FORCE_DOMDEC_OFF=1`). Set `MMML_NO_CHARMM_DOMDEC_OFF=1` to keep DOMDEC on during MLpot ENER smoke.

Tier 2 (`--ml-spatial-mpi`) parallelizes **ML only** with domdec still off; CHARMM integration remains replicated per rank.

## Spike procedure (user-run)

### A. Capture Tier 3 survey

Record the current survey output before the live spike:

```bash
mmml mpi-check --tier3 --json | tee tier3_domdec_survey.json
mmml mpi-check --tier3 --strict
```

Expected while blocked: the JSON report has `tier3.blocked=true`, and the strict command exits non-zero.

### B. Baseline Tier 2 (spatial ML, no DOMDEC metadata)

```bash
export MMML_MLPOT_SPATIAL_MPI=1
MMML_MPI_NP=2 ./scripts/mmml-charmm-mpirun.sh md-system \
  --composition DCM:20 --box-size 32 \
  --ml-spatial-mpi --ml-gpu-count 1 --ml-batch-size 128 \
  --md-stage mini --ps-heat 0
```

**Pass:** minimization completes; energies finite on all ranks.

### C. Domdec on (MLpot ENER smoke)

Start with the smallest possible active-DOMDEC check before any SD/dynamics:

```bash
# Print the exact launch line without importing PyCHARMM.
python tests/functionality/mlpot/09_domdec_mlpot_smoke.py --dry-run

# Live active-DOMDEC ENER smoke. Requires DOMDEC-enabled libcharmm.so.
export MMML_CKPT=/path/to/DESdimers_params.json
MMML_MPI_NP=1 MMML_DOMDEC_MLPOT_SMOKE=1 \
  ./scripts/mmml-charmm-mpirun.sh python \
  tests/functionality/mlpot/09_domdec_mlpot_smoke.py \
  --checkpoint "$MMML_CKPT" \
  --residue OCOH --n-molecules 1 --box-side 32
```

Same-script off-control:

```bash
MMML_MPI_NP=1 MMML_DOMDEC_MLPOT_SMOKE=1 \
  ./scripts/mmml-charmm-mpirun.sh python \
  tests/functionality/mlpot/09_domdec_mlpot_smoke.py \
  --checkpoint "$MMML_CKPT" \
  --residue OCOH --n-molecules 1 --box-side 32 \
  --no-domdec-command
```

Record: pass/fail, traceback/segfault yes/no, and whether `send_coord_to_recip` appears in any backtrace.

### D. Observed Tier 3 smoke outcomes

On `pc-bach` with DOMDEC-enabled `libcharmm.so`:

- `np=1` + `domdec on` + `OCOH:1` + MLpot registration + `ENER` passes.
- `np=1` no-DOMDEC-command control passes with matching energies.
- `np=2` with PyCHARMM topology construction does not reach DOMDEC/MLpot. The failure is in CHARMM setup/topology loading itself.
- `ACO`/`MEOH` fail active DOMDEC earlier because their generated atom order is not DOMDEC `groupxfast` compatible.

Conclusion: the next Tier 3 test must start from a CHARMM-native/prebuilt state, not simultaneous Python-side RTF/PSF construction on every rank.

### E. Tier 3 spatial MPI + DOMDEC smoke (`10_domdec_spatial_mpi_smoke.py`)

The canonical two-step procedure for testing the `build_domdec_spatial_batch_indices`
path with a live DCM cluster:

```bash
# Step 1 — build prebuilt PSF/CRD (np=1, once per system size)
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh python \
  tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \
  --prepare-prebuilt-only --residue DCM --n-molecules 20 --box-side 40

# Step 2 — callback-only DOMDEC path check (no checkpoint, no segfault risk)
MMML_MPI_NP=4 MMML_MLPOT_SPATIAL_MPI=1 \
  ./scripts/mmml-charmm-mpirun.sh python \
  tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py

# Step 3 — live CHARMM ENER at np=1 (checkpoint required; np>1 READ hangs)
MMML_MPI_NP=1 MMML_MLPOT_SPATIAL_MPI=1 \
  CUDA_VISIBLE_DEVICES="" MMML_MLPOT_DEVICE=cpu JAX_PLATFORMS=cpu \
  ./scripts/mmml-charmm-mpirun.sh python \
  tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \
  --charmm-ener --checkpoint "$MMML_CKPT" \
  --residue DCM --n-molecules 20 --box-side 40
```

Or run all three steps via the wrapper:

```bash
# Callback-only (no checkpoint):
bash scripts/run_domdec_spatial_mpi_smoke.sh

# Full live ENER:
MMML_CKPT=/path/to/checkpoint.json \
  bash scripts/run_domdec_spatial_mpi_smoke.sh --live
```

**Pass criteria (callback sub-test):** `use_spatial=True`, `domdec_path=True`,
owned monomers partition the system over all ranks, allreduced energy finite.

**Pass criteria (live ENER sub-test):** `energy.show()` completes, TOTE is
finite, `domdec_summary` reports `DOMDEC active: True` and `Symbols found: 8/8`.

### F. ctypes / PyCHARMM survey

```bash
# Symbol probe on cluster (run after sourcing CHARMM env):
python -c "
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import domdec_summary
import mmml.interfaces.pycharmmInterface.import_pycharmm  # loads libcharmm.so
print(domdec_summary())
"
```

## Success criteria (Tier 3)

- Callback smoke: `use_spatial=True`, `domdec_path=True`, owned monomers correctly
  partitioned, energy finite.
- Live ENER: `domdec_summary` reports `DOMDEC active: True` and `Symbols found: 8/8`;
  TOTE is finite; no segfault.
- CHARMM slab partition from `build_domdec_spatial_batch_indices` matches DOMDEC
  atom ownership (confirmed by the callback test comparing ctypes vs COM paths).
- (Stretch) SD and short MD with `domdec on` + spatial MLpot complete without segfault;
  wall time decreases with `n_ranks` on a fixed box.

## Fallback

Until Tier 3 passes, use Tier 2 spatial MPI for ML decomposition or **Tier 1** for one-node 2-GPU production:

```bash
export CUDA_VISIBLE_DEVICES=0,1
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system ... \
  --ml-batch-size 128 --ml-gpu-count 2
```
