# Spatial MPI + DOMDEC coexistence (Tier 3 spike)

Manual smoke for aligning CHARMM domain decomposition with spatial MLpot.
**Not run in CI** — requires MPI-linked CHARMM, GPU, and equilibrated PBC cluster.

## Goal

Mode C from [Spatial ML MPI](../../../docs/mlpot-spatial-mpi.md): `np>1`, domdec **on**, spatial ML per rank, one GPU per rank. Both CHARMM integration and PhysNet should scale.

## Current blockers (June 2026)

1. `np>1` PyCHARMM topology construction is unsupported for this path. On `pc-bach`, concurrent `crystal free`, `DELETE ATOM`, `read rtf`, and even a minimal `MASS` RTF load hang/abort before DOMDEC or MLpot.
2. PyCHARMM exposes no per-rank local/ghost atom API ([`domdec_info.py`](../../../mmml/interfaces/pycharmmInterface/mlpot/mpi_spatial/domdec_info.py)).
3. JAX GPU warmup + active DOMDEC can segfault (`send_coord_to_recip` / `PMPI_Free_mem`).
4. `domdec off` is opt-in via `MMML_FORCE_DOMDEC_OFF=1` for streams that enabled DOMDEC; it is not a Tier 3 integration path ([`import_pycharmm.py`](../../../mmml/interfaces/pycharmmInterface/import_pycharmm.py)).

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

### E. Next path: prebuilt/native CHARMM state

Use the classic DOMDEC model:

1. Build and validate PSF/CRD/restart outside the `np>1` PyCHARMM topology path.
2. Launch `np>1` with a CHARMM-native input/state load that DOMDEC supports.
3. Establish PBC/image state and `domdec on`.
4. Attach MLpot only after CHARMM state exists on all ranks.
5. Then test `ENER`, short SD, and only later dynamics.

Residue/template requirement: PSF atom order must be DOMDEC-compatible (hydrogens adjacent to bonded heavy atoms) and must remain the canonical atom order for MLpot, ASE/JAX helpers, selections, monomer slices, and checkpoint comparisons.

DCM:10 scaffold:

```bash
# Build/certify a DCM:10 PSF/CRD with np=1.
bash scripts/run_domdec_dcm10_smoke.sh prep

# Offline DOMDEC hydrogen-order validation on the generated PSF.
bash scripts/run_domdec_dcm10_smoke.sh validate

# Native CHARMM np>1 DOMDEC ENER from the prebuilt PSF/CRD.
bash scripts/run_domdec_dcm10_smoke.sh tier3
```

### F. ctypes / PyCHARMM survey

- Inspect `ldd libcharmm.so` for `domdec_common` symbols.
- Check whether `pycharmm` exposes atom group / domain queries (none found in June 2026 survey).

## Success criteria (Tier 3)

- SD with domdec on + spatial ML completes without segfault.
- Integration wall time decreases with `n_ranks` on fixed box size.
- Python `SpatialDomainGrid` slab partition matches CHARMM domdec cell ownership (or is replaced by Fortran metadata).

## Fallback

Until Tier 3 passes, use Tier 2 spatial MPI for ML decomposition or **Tier 1** for one-node 2-GPU production:

```bash
export CUDA_VISIBLE_DEVICES=0,1
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system ... \
  --ml-batch-size 128 --ml-gpu-count 2
```
