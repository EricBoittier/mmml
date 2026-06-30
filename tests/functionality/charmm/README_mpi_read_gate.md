# MPI PyCHARMM READ gate (Tier 3 bisect)

Minimal harness to isolate **cooperative** `eval_charmm_script` topology I/O at `np>1`.
No JAX, ASE, Rich, or MLpot.

## Prerequisites

Prebuilt DCM:20 artifacts (once at np=1):

```bash
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh python \
  tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \
  --prepare-prebuilt-only --residue DCM --n-molecules 20 --box-side 40
```

## PyCHARMM matrix (node09)

```bash
cd ~/mmml

# Baseline — must pass (np=1 uses plain Python, not mpirun — see below)
MMML_MPI_NP=1 ./scripts/run_mpi_pycharmm_read_gate.sh

# Bisect modes at np=4 (record last log line before hang)
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode psf-crd
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode stream-inp
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode restart

# Optional: crystal after READ
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode psf-crd --with-crystal
```

**Pass:** `PASS read_gate: mode=... np=N n_atoms=100`

At ``np>1``, ``--mode psf-crd`` **auto-switches to restart** when
``artifacts/.../dcm_20mer.res`` exists (written by ``--prepare-prebuilt-only``).
Restart bootstrap runs **RTF/PRM/PSF + read restart** (no ``UPDATE`` at load;
lists sync later via ``ENER``/``setup_charmm_environment``). Cooperative PyCHARMM
PSF/CRD-only READ at ``np>1`` often leaves ``n_atoms=0`` — bisect with
``MMML_MPI_BOOTSTRAP_FORCE_PSF_CRD=1``.

At ``np>1``, bootstrap uses **shared artifact paths** by default (cooperative MPI
READ). Optional per-rank UUID copies: ``MMML_MPI_BOOTSTRAP_RANK_LOCAL=1`` (each
rank reads independently — usually leaves ``n_atoms=0`` on DOMDEC builds).

**Do not** use ``mpi4py`` barriers between READ sub-commands (``MMML_MPI_BOOTSTRAP_BARRIER=1``
is bisect-only). At ``np>1``, bootstrap writes a shared ``.inp`` and all ranks run
**one** ``stream`` command (Fortran processes the full READ chain — Python must not
split multiline scripts into separate ``eval_charmm_script`` calls or MPI ranks desync).

**Hang:** last line like `[read_psf rank 2/4] begin ...` — that sub-command is the stall point.

**Abort at import (`BLOCK` / `ILLEGAL COMMAND` / `BOMLEV 0`):** import-time
``reset_block`` runs before ``bomlev -2``. Bootstrap sets
``MMML_SKIP_CHARMM_RESET_BLOCK=1`` before ``import_pycharmm``.

**Warning ``Unrecognized command: crys`` at np=1:** plain-Python import runs
``crystal free`` (vacuum init) which poisons CHARMM before READ. Bootstrap sets
``MMML_SKIP_VACUUM_CHARMM_INIT=1``. Do not run deferred BLOCK under ``mpirun`` — it
hangs. At ``np=1`` the read gate uses plain Python, not ``mpirun``.

**node09 bisect (June 2026):** `np=1` passes; `np=2` hangs on the first `read_rtf` step in
`psf-crd` mode. Root cause: per-rank SCRATCH units inside `eval_charmm_script`
(`setup/api/api_eval.F90`) desync cooperative MPI READ. Fix: restore direct
`maincomx` evaluation + line-splitting in vendored `pycharmm/lingo.py`, then
**rebuild** `libcharmm.so` (rebuild script must sync `api_eval.F90` into the tree):

```bash
grep -q "call maincomx" setup/charmm/source/api/api_eval.F90 || echo "MISSING api_eval patch"
bash scripts/rebuild_charmm_mlpot.sh --clean
```

After rebuild, re-run the matrix above. Also try `stream-inp` if rebuilding is delayed:

```bash
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode stream-inp
```

## Native CHARMM control

Proves Fortran READ works under the same `mpirun` wrapper (no PyCHARMM). Requires a
**standalone** `charmm` executable — `libcharmm.so` alone is not enough:

```bash
bash scripts/rebuild_charmm_native_exec.sh --clean   # if setup/charmm/charmm missing
MMML_MPI_NP=2 ./scripts/run_native_charmm_read_gate.sh
MMML_MPI_NP=2 ./scripts/run_native_charmm_read_gate.sh --with-restart
```

The script resolves `setup/charmm/charmm` (or `CHARMM_EXE`) and writes output under
`artifacts/domdec_spatial_smoke/native_read_gate/`.

**Do not** pass an empty `$CHARMM_EXE` to `mmml-charmm-mpirun.sh` — the wrapper falls
through to the `mmml` CLI when the first argument is not an executable file.

### np>1 stream completes but `n_atoms=0`

The stream `.inp` is correct; library-mode cooperative READ still leaves empty state.
Bisect on node09:

```bash
# 1. Validate restart artifact at np=1
MMML_MPI_NP=1 ./scripts/run_mpi_pycharmm_read_gate.sh --mode restart

# 2. Show CHARMM error text (bootstrap defaults to MMML_QUIET=1)
MMML_QUIET=0 MMML_MPI_NP=2 ./scripts/run_mpi_pycharmm_read_gate.sh --mode psf-crd

# 3. Rank-0-only stream eval (workshop I/O pattern)
MMML_MPI_BOOTSTRAP_RANK0_DRIVE=1 MMML_QUIET=0 MMML_MPI_NP=2 \
  ./scripts/run_mpi_pycharmm_read_gate.sh --mode psf-crd

# 4. PSF/CRD via stream (no .res)
MMML_MPI_BOOTSTRAP_FORCE_PSF_CRD=1 MMML_QUIET=0 MMML_MPI_NP=2 \
  ./scripts/run_mpi_pycharmm_read_gate.sh --mode stream-inp

# 5. Native Fortran (requires setup/charmm/charmm executable)
MMML_MPI_NP=2 ./scripts/run_native_charmm_read_gate.sh
MMML_MPI_NP=2 ./scripts/run_native_charmm_read_gate.sh --with-restart
```

Log lines now include per-command `psf=` and `coor=` counts after each READ step.

If native passes and PyCHARMM fails → bug is in library-mode `eval_charmm_script`, not fabric.

## Bootstrap API version

PyCHARMM read gate logs `bootstrap_api=...` at startup. You need **`direct-api-v4.2`**
(RTF/PRM/PSF via ``api_read``; coords from CRD parse + ``coor.set_positions``).
Restart mode loads coords from **CRD** when present (``--prepare-prebuilt-only`` writes
binary ``write restart`` ``.res`` files that the ASCII parser cannot read). np>1 staging
barriers after rank-0 copies before cooperative API READ.

## Implementation

Bootstrap API: [`mmml/interfaces/pycharmmInterface/charmm_mpi.py`](../../../mmml/interfaces/pycharmmInterface/charmm_mpi.py)
(`bootstrap_topology_mpi`, `bootstrap_charmm_step`).

Used by [`10_domdec_spatial_mpi_smoke.py`](../mlpot/10_domdec_spatial_mpi_smoke.py) for `np>1` live ENER load.
